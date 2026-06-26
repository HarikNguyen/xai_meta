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
import torch.autograd as autograd
from typing import Dict, List, Optional, Tuple, Union

from algos.utils import get_loss_n_preds, put_on_device
from loaders.utils import get_stratified_bootstrap_batches


class MAMLPostHocExplainer:
    """
    Post-hoc XAI cho MAML - Phiên bản tối ưu & dễ bảo trì.
    
    Tính Adaptation Gain và Feature Saliency Map dựa trên Adjoint Method + Bootstrap.
    """

    def __init__(self, maml, device: Optional[str] = None):
        self.maml = maml
        self.device = device or maml.device
        self.alpha = maml.base_lr
        self.learner = maml.baselearner

    # ====================== CORE HELPERS ======================

    def _compute_trajectory(self, sup_x: torch.Tensor, sup_y: torch.Tensor, T: int) -> List[List[torch.Tensor]]:
        """Tính quỹ đạo tham số φ^(0) → φ^(T) trên support set."""
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

    def _hvp(self, phi: List[torch.Tensor], v: List[torch.Tensor],
             sup_x: torch.Tensor, sup_y: torch.Tensor) -> List[torch.Tensor]:
        """Hessian-Vector Product qua Pearlmutter trick."""
        phi_r = [p.detach().requires_grad_(True) for p in phi]
        loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
        grads = autograd.grad(loss, phi_r, create_graph=True)
        
        dot = sum((g * vi.detach()).sum() for g, vi in zip(grads, v))
        Hv = autograd.grad(dot, phi_r, retain_graph=False)
        
        return [hv.detach() for hv in Hv]

    def _compute_expected_lambda(self, phi_T: List[torch.Tensor], que_x: torch.Tensor, 
                                que_y: torch.Tensor, num_bootstraps: int, 
                                samples_per_class: int) -> List[torch.Tensor]:
        """Tính E_Q[∇_φ L_Q(φ_T)] bằng stratified bootstrap."""
        bootstrap_gen = get_stratified_bootstrap_batches(
            que_x, que_y, num_bootstraps, samples_per_class
        )
        
        expected_lam = [torch.zeros_like(p) for p in phi_T]
        
        for b_que_x, b_que_y in bootstrap_gen:
            phi_T_grad = [p.clone().detach().requires_grad_(True) for p in phi_T]
            q_loss, _ = get_loss_n_preds(phi_T_grad, self.learner, b_que_x, b_que_y)
            lam_b = autograd.grad(q_loss, phi_T_grad, retain_graph=False)
            
            expected_lam = [
                avg + lb.detach() / num_bootstraps
                for avg, lb in zip(expected_lam, lam_b)
            ]
        
        return expected_lam

    def _compute_adaptation_gain(self, theta_0, phi_T, que_x, que_y, 
                               num_bootstraps: int, samples_per_class: int) -> float:
        """Tính ΔM = E[L_Q(θ₀)] - E[L_Q(φ_T)]"""
        bootstrap_gen = get_stratified_bootstrap_batches(
            que_x, que_y, num_bootstraps, samples_per_class
        )
        
        pre_sum = 0.0
        post_sum = 0.0
        
        with torch.no_grad():
            for b_que_x, b_que_y in bootstrap_gen:
                pre, _ = get_loss_n_preds(theta_0, self.learner, b_que_x, b_que_y)
                post, _ = get_loss_n_preds(phi_T, self.learner, b_que_x, b_que_y)
                pre_sum += pre.item()
                post_sum += post.item()
        
        return (pre_sum - post_sum) / num_bootstraps

    def _saliency_core_batched(self, sup_x: torch.Tensor, sup_y: torch.Tensor,
                             phis: List[List[torch.Tensor]],
                             lambdas: Dict[int, List[torch.Tensor]],
                             max_steps: Optional[int] = None) -> torch.Tensor:

        saliency = torch.zeros_like(sup_x)
        T_max = max_steps or len(phis) - 1
        
        for m in range(1, T_max + 1):
            lam_m = lambdas[m]

            phi_r = [p.detach().requires_grad_(True) for p in phis[m - 1]]
            
            features_m = self.learner(sup_x, phi_r, only_features=True)
            features_m_leaf = features_m.detach().requires_grad_(True)
            preds = self.learner.forward_features(features_m_leaf, phi_r)
            loss = self.learner.criterion(preds, sup_y)
            g_phi = autograd.grad(loss, phi_r, create_graph=True, retain_graph=True, allow_unused=True)
            
            h = sum((g * l.detach()).sum() for g, l in zip(g_phi, lam_m) if g is not None)
            grad_features = autograd.grad(h, features_m_leaf, retain_graph=False)[0]

            # Global Average Pooling -> Tính trọng số (alpha) cho từng kênh đặc trưng
            weights = grad_features.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features_m.detach()).sum(dim=1, keepdim=True)
            cam_upsampled = F.interpolate(cam, size=sup_x.shape[-2:], mode='bilinear', align_corners=False)

            saliency += self.alpha * cam_upsampled
        
        return saliency

        # ====================== MAIN PUBLIC METHOD ======================

    # ====================== MAIN PUBLIC METHOD ======================

    def interpret(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        T: int,
        num_bootstraps: int = 100,
        samples_per_class: int = 3,
        return_gain: bool = False,
        return_saliency: bool = True,
        return_trajectory: bool = False,
    ) -> Union[float, torch.Tensor, List[torch.Tensor], Tuple]:
        """
        Hàm chính: Tính Adaptation Gain và/hoặc Saliency Map.
        
        Returns:
            - 1 flag → trả giá trị đơn
            - Nhiều flag → tuple (gain, saliency, trajectory)
        """
        sup_x, sup_y, que_x, que_y = put_on_device(
            self.device, [sup_x, sup_y, que_x, que_y]
        )

        # 1. Forward Trajectory
        phis = self._compute_trajectory(sup_x, sup_y, T)
        phi_T = [p.detach() for p in phis[T]]

        # 2. Bootstrap + Expectation
        expected_lam_T = None
        if return_saliency or return_trajectory:
            expected_lam_T = self._compute_expected_lambda(
                phi_T, que_x, que_y, num_bootstraps, samples_per_class
            )

        adaptation_gain = None
        if return_gain:
            adaptation_gain = self._compute_adaptation_gain(
                self.maml.theta_0, phi_T, que_x, que_y, 
                num_bootstraps, samples_per_class
            )

        if not (return_saliency or return_trajectory):
            return adaptation_gain

        # 3. Adjoint Backward Pass
        lambdas: Dict[int, List[torch.Tensor]] = {T: expected_lam_T}
        for m in range(T, 0, -1):
            Hv = self._hvp(phis[m - 1], lambdas[m], sup_x, sup_y)
            lambdas[m - 1] = [
                (l - self.alpha * hv).detach()
                for l, hv in zip(lambdas[m], Hv)
            ]

        # 4. Saliency Computation
        total_saliency = None
        trajectory_saliencies = None

        if return_trajectory:
            trajectory_saliencies = []
            for m in range(1, T + 1):
                step_saliency = self._saliency_core_batched(
                    sup_x, sup_y, phis, lambdas, max_steps=m
                )
                trajectory_saliencies.append(step_saliency)
            
            if return_saliency:
                total_saliency = sum(trajectory_saliencies)
        elif return_saliency:
            total_saliency = self._saliency_core_batched(
                sup_x, sup_y, phis, lambdas
            )

        # 5. Return logic
        results = []
        if return_gain:
            results.append(adaptation_gain)
        if return_saliency:
            results.append(total_saliency)
        if return_trajectory:
            results.append(trajectory_saliencies)

        return results[0] if len(results) == 1 else tuple(results)
