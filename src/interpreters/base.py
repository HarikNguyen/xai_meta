"""
algos/interpreter.py
====================
Post-hoc XAI for MAML: Feature Saliency Map of the support set S
w.r.t. adaptation gain ΔM = E_{Q~T_i}[L_Q(φᵢ*(S))] - E_{Q~T_i}[L_Q(θ₀)].

──────────────────────────────────────────────────────────────────
  ∂ΔM/∂xⱼ = E_{Q~T_i}[∂L_Q(φᵢ*(S))/∂xⱼ] - E_{Q~T_i}[∂L_Q(θ₀)/∂xⱼ]; xⱼ ∈ S
          = E_{Q~T_i}[∂l_q(φᵢ*(s))/∂xⱼ]
  ∂l_q(φᵢ*(s))/∂xⱼ = -(α/k) Σₘ [∇²_{φ,xⱼ} ℓⱼ(φ^(m-1))]ᵀ λ^(m)      ∈ ℝᴰ

Adjoint:
  λ^(K) = ∇_φ L_Q(φ^(K))
  λ^(m-1) = λ^(m) − α·H^(m-1)·λ^(m)     [H symmetric → no transpose needed]
  H^(m-1)·v via Pearlmutter HVP: O(P).
"""

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from algos.utils import get_loss_n_preds, put_on_device
from loaders.utils import get_stratified_bootstrap_batches


class MAMLPostHocExplainer:
    """
    Post-hoc XAI for MAML
    Compute Adaptation Gain and Feature Saliency Map.
    """

    def __init__(self, maml, device: Optional[str] = None):
        self.maml = maml
        self.device = device or maml.device
        self.alpha = maml.base_lr
        self.learner = maml.baselearner

    # ====================== CORE HELPERS ======================

    def _compute_trajectory(
        self, sup_x: torch.Tensor, sup_y: torch.Tensor, T: int
    ) -> List[List[torch.Tensor]]:
        """Compute the parameter trajectory φ^(0) → φ^(T) on the support set."""
        phis = [[p.detach().clone() for p in self.maml.theta_0]]

        # fast-forward (fast-adaptation)
        for _ in range(T):
            phi_r = [p.detach().clone().requires_grad_(True) for p in phis[-1]]
            loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
            grads = autograd.grad(loss, phi_r, create_graph=False)
            phi_next = [
                (p - self.alpha * g).detach().clone() for p, g in zip(phi_r, grads)
            ]
            phis.append(phi_next)

        # return the full trajectory (from φ^(0) to φ^(T))
        return phis

    def _hvp(
        self,
        phi: List[torch.Tensor],
        v: List[torch.Tensor],
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Hessian-Vector Product via Pearlmutter."""
        phi_r = [p.detach().requires_grad_(True) for p in phi]
        loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
        grads = autograd.grad(loss, phi_r, create_graph=True)

        dot = sum((g * vi.detach()).sum() for g, vi in zip(grads, v))
        Hv = autograd.grad(dot, phi_r, retain_graph=False)

        return [hv.detach() for hv in Hv]

    def _compute_expected_lambda(
        self, phi_T, bootstrap_query, num_bootstraps
    ) -> List[torch.Tensor]:
        """Compute E_Q[∇_φ^t L_Q(φ_T)] by stratified bootstrap."""
        expected_lam = [torch.zeros_like(p) for p in phi_T]  # init with 0
        for b_que_x, b_que_y in bootstrap_query:
            phi_T_grad = [p.clone().detach().requires_grad_(True) for p in phi_T]
            q_loss, _ = get_loss_n_preds(phi_T_grad, self.learner, b_que_x, b_que_y)
            lam_b = autograd.grad(q_loss, phi_T_grad, retain_graph=False)

            expected_lam = [
                avg + lb.detach() / num_bootstraps
                for avg, lb in zip(expected_lam, lam_b)
            ]

        return expected_lam

    def _compute_adaptation_gain(
        self, theta_0, phi_T, bootstrap_query, num_bootstraps
    ) -> float:
        """Tính ΔM = E[L_Q(θ₀)] - E[L_Q(φ_T)]"""
        pre_sum = 0.0
        post_sum = 0.0

        with torch.no_grad():
            for b_que_x, b_que_y in bootstrap_query:
                pre, _ = get_loss_n_preds(theta_0, self.learner, b_que_x, b_que_y)
                post, _ = get_loss_n_preds(phi_T, self.learner, b_que_x, b_que_y)
                pre_sum += pre.item()
                post_sum += post.item()

        pre_loss = pre_sum / num_bootstraps
        post_loss = post_sum / num_bootstraps

        return ((pre_loss - post_loss) / (pre_loss + 1e-8)) * 100.0

    def _saliency_core_batched(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        phis: List[List[torch.Tensor]],
        lambdas: Dict[int, List[torch.Tensor]],
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:

        saliency = torch.zeros_like(sup_x)
        T_max = max_steps or len(phis) - 1

        for m in range(1, T_max + 1):
            lam_m = lambdas[m]

            phi_r = [p.detach().requires_grad_(True) for p in phis[m - 1]]

            features_m = self.learner(sup_x, phi_r, only_features=True)
            features_m_leaf = features_m.detach().requires_grad_(True)
            preds = self.learner.forward_features(features_m_leaf, phi_r)
            loss = self.learner.criterion(preds, sup_y)
            g_phi = autograd.grad(
                loss, phi_r, create_graph=True, retain_graph=True, allow_unused=True
            )

            h = sum(
                (g * l.detach()).sum() for g, l in zip(g_phi, lam_m) if g is not None
            )
            grad_features = autograd.grad(h, features_m_leaf, retain_graph=False)[0]

            # Global Average Pooling -> Tính trọng số (alpha) cho từng kênh đặc trưng
            weights = grad_features.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features_m.detach()).sum(dim=1, keepdim=True)
            cam_upsampled = F.interpolate(
                cam, size=sup_x.shape[-2:], mode="bilinear", align_corners=False
            )

            saliency -= self.alpha * cam_upsampled

        return saliency

    def interpret(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        que_x: torch.Tensor,
        que_y: torch.Tensor,
        T: int,
        num_bootstraps: int = 100,
        samples_per_class: int = 3,
    ):
        # prepare data
        sup_x, sup_y, que_x, que_y = put_on_device(
            self.device, [sup_x, sup_y, que_x, que_y]
        )
        bootstrap_query_gen = get_stratified_bootstrap_batches(
            que_x, que_y, num_bootstraps, samples_per_class
        )
        bootstrap_query = list(bootstrap_query_gen)

        # get forward trajectory
        phis = self._compute_trajectory(sup_x, sup_y, T)

        # compute lambda_T expectation
        phi_T = [p.detach() for p in phis[T]]
        expected_lam_T = self._compute_expected_lambda(
            phi_T, bootstrap_query, num_bootstraps
        )

        # compute adaptation gain
        adaptation_gain = self._compute_adaptation_gain(
            self.maml.theta_0, phi_T, bootstrap_query, num_bootstraps
        )

        # adjoint backward pass
        lambdas = {T: expected_lam_T}
        for m in range(T, 0, -1):
            Hv = self._hvp(phis[m - 1], lambdas[m], sup_x, sup_y)
            lambdas[m - 1] = [
                (l - self.alpha * hv).detach() for l, hv in zip(lambdas[m], Hv)
            ]

        # Saliency Computation
        saliency_map = self._saliency_core_batched(sup_x, sup_y, phis, lambdas)

        return adaptation_gain, saliency_map
