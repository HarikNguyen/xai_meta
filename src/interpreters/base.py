"""
algos/interpreter.py
====================
Post-hoc XAI cho MAML: Feature Saliency Map của support set S
w.r.t. độ lợi thích nghi  ΔM = L_Q(θ₀) − L_Q(φᵢ*(S)).

──────────────────────────────────────────────────────────────────
  ∂ΔM/∂xⱼ = (α/k) Σₘ [∇²_{φ,xⱼ} ℓⱼ(φ^(m-1))]ᵀ λ^(m)      ∈ ℝᴰ

Adjoint:
  λ^(K) = ∇_φ L_Q(φ^(K))
  λ^(m-1) = λ^(m) − α·H^(m-1)·λ^(m)     [H symmetric → no transpose needed]
  H^(m-1)·v via Pearlmutter HVP: O(P), không build ma trận P×P.
"""

import torch
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Dict, List, Optional

from algos.utils import get_loss_n_preds, put_on_device, get_stratified_bootstrap_batches


class MAMLPostHocExplainer:
    """
    Post-hoc explainer for MAML.

    Parameters
    ----------
    maml : MAML
        MAML manager obj (loaded from checkpoint - theta_0 was prepared).
    device : str, optional
        Device to run on.
    """

    def __init__(self, maml, device: Optional[str] = None):
        self.maml    = maml
        self.device  = device or maml.device
        self.alpha   = maml.base_lr   # inner-loop LR at meta-test time
        self.learner = maml.baselearner

    def _loss_j_joint(
        self,
        phi: List[torch.Tensor],
        x_full: torch.Tensor,
        y_full: torch.Tensor,
        j: int,
    ) -> torch.Tensor:
        """
        ℓⱼ(φ; x_full, y_full) = CE của mẫu thứ j, tính qua forward CHUNG.

        y_full là one-hot [k, n_way] (đúng format codebase: calc_accuracy dùng
        torch.max(y, dim=1) → y phải là one-hot).
        """
        preds    = self.learner.forward_weights(x_full, phi)          # [k, n_way]
        log_probs = F.log_softmax(preds, dim=-1)                       # [k, n_way]
        per_ex   = -(y_full * log_probs).sum(dim=-1)                   # [k]
        return per_ex[j]

    def _compute_trajectory(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
    ) -> List[List[torch.Tensor]]:
        """
        Quỹ đạo detached φ^(0)...φ^(T).

        Mỗi φ^(m) là list tensor tươi (detached clone), không lưu đồ thị
        xuyên bước → adjoint có thể bật requires_grad độc lập tại từng bước.
        """
        phis = [[p.detach().clone() for p in self.maml.theta_0]]

        for _ in range(T):
            phi_r = [p.detach().clone().requires_grad_(True) for p in phis[-1]]
            loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
            grads = autograd.grad(loss, phi_r, create_graph=False)
            phi_next = [
                (p - self.alpha * g).detach().clone()
                for p, g in zip(phi_r, grads)
            ]
            phis.append(phi_next)

        return phis  # len = T+1

    def _hvp(
        self,
        phi: List[torch.Tensor],
        v: List[torch.Tensor],
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Hessian-vector product  H^(m)·v  qua Pearlmutter double-backward.
        H = ∇²_φ L_S(φ),  v được detach trước khi dot.
        Chi phí: O(P) — không dựng ma trận P×P.
        """
        phi_r = [p.detach().requires_grad_(True) for p in phi]
        loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
        grads = autograd.grad(loss, phi_r, create_graph=True)
        dot   = sum((g * vi.detach()).sum() for g, vi in zip(grads, v))
        Hv    = autograd.grad(dot, phi_r, retain_graph=False)
        return [hv.detach() for hv in Hv]

    def _compute_lambdas(
        self,
        phis: List[List[torch.Tensor]],
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Adjoint chain λ^(T)...λ^(0).
        Công thức: λ^(T) = ∇_φ L_Q(φ^(T));  λ^(m-1) = λ^(m) − α·H^(m-1)·λ^(m).
        H đối xứng → không cần transpose.  Trả về dict key=0..T.
        """
        # λ^(T)
        phi_T = [p.detach().requires_grad_(True) for p in phis[T]]
        qL, _ = get_loss_n_preds(phi_T, self.learner, que_x, que_y)
        lam_T = autograd.grad(qL, phi_T)

        lambdas: Dict[int, List[torch.Tensor]] = {T: [l.detach() for l in lam_T]}

        # backward recursion m = T, T-1, ..., 1
        for m in range(T, 0, -1):
            Hv = self._hvp(phis[m - 1], lambdas[m], sup_x, sup_y)
            lambdas[m - 1] = [
                (l - self.alpha * hv).detach()
                for l, hv in zip(lambdas[m], Hv)
            ]

        return lambdas

    # ---- internal saliency core (nhận phis+lambdas đã tính) ----

    def _saliency_x_core(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
        phis: List[List[torch.Tensor]],
        lambdas: Dict[int, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Kênh đặc trưng: [∇²_{φ,xⱼ} ℓⱼ]ᵀ λ^(m) via double-backward.

        QUAN TRỌNG (đã xác nhận bằng số):
          - xⱼ phải nằm trong x_full để BatchNorm thấy đúng statistics.
          - autograd.grad(loss_j, phi_r, create_graph=True) giữ link x→phi→loss,
            cho phép backward thứ hai ∂h/∂xⱼ chạy qua.
        """
        k        = sup_x.shape[0]
        saliency = torch.zeros_like(sup_x)

        for j in range(k):
            total_grad = torch.zeros_like(sup_x[j])

            for m in range(1, T + 1):
                lam_m = lambdas[m]

                # --- xⱼ là leaf; nhúng vào batch đầy đủ ---
                xj_leaf = sup_x[j].detach().requires_grad_(True)
                x_full  = torch.cat([
                    sup_x[:j].detach(),
                    xj_leaf.unsqueeze(0),
                    sup_x[j + 1:].detach(),
                ], dim=0)   # [k, ...], xj_leaf tại vị trí j

                # --- first backward: ∇_φ ℓⱼ, giữ đồ thị để backward qua lần 2 ---
                phi_r  = [p.detach().requires_grad_(True) for p in phis[m - 1]]
                loss_j = self._loss_j_joint(phi_r, x_full, sup_y, j)
                g_phi  = autograd.grad(
                    loss_j, phi_r,
                    create_graph=True,   # bắt buộc: đồ thị của g_phi phải tồn tại
                    retain_graph=True,   # giữ để backward tiếp theo
                )

                # --- scalar h = ⟨∇_φ ℓⱼ, λ^(m)⟩ ---
                h = sum((g * l.detach()).sum() for g, l in zip(g_phi, lam_m))

                # --- second backward: ∂h/∂xⱼ = [∇²_{φ,xⱼ} ℓⱼ]ᵀ λ^(m) ---
                grad_xj = autograd.grad(h, xj_leaf, retain_graph=False)[0]

                total_grad = total_grad + (self.alpha / k) * grad_xj.detach()

            saliency[j] = total_grad

        return saliency  # [k, C, H, W] — cùng shape với sup_x

    # =========================================================================
    # PUBLIC: standalone attribution methods
    # =========================================================================

    def compute_adaptation_gain(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        T: int,
    ) -> float:
        """
        ΔM = L_Q(θ₀) − L_Q(φᵢ*(S)) ∈ ℝ.

        Dương → thích nghi có ích; âm → thích nghi phản tác dụng.
        Đây là baseline cho mọi phép quy kết.
        """
        with torch.no_grad():
            pre_loss, _ = get_loss_n_preds(
                self.maml.theta_0, self.learner, que_x, que_y
            )
        phis = self._compute_trajectory(sup_x, sup_y, T)
        with torch.no_grad():
            post_loss, _ = get_loss_n_preds(phis[T], self.learner, que_x, que_y)
        return (pre_loss - post_loss).item()

    def saliency_x(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        T: int,
        num_bootstraps: int = 10,
        samples_per_class: int = 3,
    ) -> torch.Tensor:
        """
        ∂ΔM/∂xⱼ ∈ ℝ^(k×D) — bản đồ saliency đặc trưng (cùng shape sup_x).

            ∂ΔM/∂xⱼ = (α/k) Σ_{m=1}^{K} [∇²_{φ,xⱼ} ℓⱼ(φ^{m-1})]ᵀ λ^{m}

        Tính qua double-backward: không dựng ma trận P×D.
        Kết quả: pixel/đặc trưng nào trong mẫu j quan trọng với hướng thích nghi.
        """
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])
        total_saliency = torch.zeros_like(sup_x)

        phis    = self._compute_trajectory(sup_x, sup_y, T)

        bootstrap_generator = get_stratified_bootstrap_batches(
            que_x, que_y, num_bootstraps, samples_per_class
        )
        for b_que_x, b_que_y in bootstrap_generator:
            lambdas_b = self._compute_lambdas(phis, b_que_x, b_que_y, sup_x, sup_y, T)
            saliency_b = self._saliency_x_core(sup_x, sup_y, T, phis, lambdas_b)
            total_saliency += saliency_b

        return total_saliency / num_bootstraps

