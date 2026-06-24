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

from algos.utils import get_loss_n_preds, put_on_device
from loaders.utils import get_stratified_bootstrap_batches

class MAMLPostHocExplainer:
    """
    Post-hoc explainer for MAML (Highly Optimized Version).
    
    Tối ưu hóa thời gian chạy bằng cách lợi dụng tính tuyến tính của 
    chuỗi Adjoint và khả năng Vector hóa đạo hàm bậc 2 của PyTorch.
    """

    def __init__(self, maml, device: Optional[str] = None):
        self.maml    = maml
        self.device  = device or maml.device
        self.alpha   = maml.base_lr   
        self.learner = maml.baselearner

    def _compute_trajectory(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
    ) -> List[List[torch.Tensor]]:
        """Quỹ đạo detached φ^(0)...φ^(T)."""
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

        return phis

    def _hvp(
        self,
        phi: List[torch.Tensor],
        v: List[torch.Tensor],
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Hessian-vector product H^(m)·v via Pearlmutter double-backward."""
        phi_r = [p.detach().requires_grad_(True) for p in phi]
        loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
        grads = autograd.grad(loss, phi_r, create_graph=True)
        dot   = sum((g * vi.detach()).sum() for g, vi in zip(grads, v))
        Hv    = autograd.grad(dot, phi_r, retain_graph=False)
        return [hv.detach() for hv in Hv]

    def _saliency_x_core_batched(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
        phis: List[List[torch.Tensor]],
        lambdas: Dict[int, List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Kênh đặc trưng: [∇²_{φ,X} L_S]ᵀ λ^(m) via Batched double-backward.
        Tối ưu hóa: Xử lý toàn bộ batch X cùng lúc, loại bỏ vòng lặp for j=1..k.
        """
        saliency = torch.zeros_like(sup_x)

        for m in range(1, T + 1):
            lam_m = lambdas[m]

            # Kích hoạt đạo hàm cho toàn bộ batch đầu vào
            sup_x_leaf = sup_x.detach().requires_grad_(True)
            phi_r = [p.detach().requires_grad_(True) for p in phis[m - 1]]
            
            # Forward pass 1 lần duy nhất cho toàn bộ tập Support
            loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x_leaf, sup_y)
            
            # Đạo hàm bậc 1 của Mean Loss
            g_phi = autograd.grad(loss, phi_r, create_graph=True, retain_graph=True)
            
            # Tích vô hướng với trạng thái Adjoint
            h = sum((g * l.detach()).sum() for g, l in zip(g_phi, lam_m))
            
            # Đạo hàm bậc 2 dội ngược về toàn bộ pixel của batch
            grad_X = autograd.grad(h, sup_x_leaf, retain_graph=False)[0]
            
            # Alpha/k đã được tích hợp sẵn vì loss bên trên là Mean Loss (1/k)
            saliency += self.alpha * grad_X.detach()

        return saliency

    def saliency_x(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        T: int,
        num_bootstraps: int = 100,
        samples_per_class: int = 3,
    ) -> torch.Tensor:
        """
        Phiên bản siêu tối ưu bằng Linearity of Expectation (Tuyến tính kỳ vọng).
        Thay vì chạy N vòng lặp Adjoint Backward, ta chỉ lấy trung bình 
        trạng thái Lambda cuối cùng, rồi chạy Adjoint Backward ĐÚNG 1 LẦN.
        """
        sup_x, sup_y, que_x, que_y = put_on_device(self.device, [sup_x, sup_y, que_x, que_y])

        # 1. Quỹ đạo Forward (chỉ tính 1 lần)
        phis = self._compute_trajectory(sup_x, sup_y, T)

        # 2. Xấp xỉ kỳ vọng E_Q cho Tín hiệu Trạng thái Cuối cùng (Terminal Lambda)
        phi_T = [p.detach().requires_grad_(True) for p in phis[T]]
        expected_lam_T = [torch.zeros_like(p) for p in phi_T]
        
        bootstrap_generator = get_stratified_bootstrap_batches(
            que_x, que_y, num_bootstraps, samples_per_class
        )
        
        for b_que_x, b_que_y in bootstrap_generator:
            # Chỉ tính đạo hàm bậc 1 ở bước K (rất nhẹ)
            qL, _ = get_loss_n_preds(phi_T, self.learner, b_que_x, b_que_y)
            lam_T_b = autograd.grad(qL, phi_T, retain_graph=False)
            
            # Cộng dồn trung bình
            expected_lam_T = [
                l_avg + l_b.detach() / num_bootstraps 
                for l_avg, l_b in zip(expected_lam_T, lam_T_b)
            ]

        # 3. Chuỗi đi lùi Adjoint (CHỈ CHẠY 1 LẦN DUY NHẤT với expected_lam_T)
        lambdas = {T: expected_lam_T}
        for m in range(T, 0, -1):
            Hv = self._hvp(phis[m - 1], lambdas[m], sup_x, sup_y)
            lambdas[m - 1] = [
                (l - self.alpha * hv).detach() 
                for l, hv in zip(lambdas[m], Hv)
            ]

        # 4. Batched Saliency Core (CHỈ CHẠY 1 LẦN DUY NHẤT)
        total_saliency = self._saliency_x_core_batched(sup_x, sup_y, T, phis, lambdas)

        return total_saliency
