"""
Feature-space Adjoint Meta-Learning Attribution
====================
Post-hoc XAI for MAML: Feature Saliency Map of the support set S
w.r.t. adaptation gain

    ΔM = E_{Q~T_i}[ -(L(φ_T, Q) - L(φ_freeze_T, Q)) ]
       = E_{Q~T_i}[ L(φ_freeze_T, Q) - L(φ_T, Q) ]

where:
    φ_T        = fast-adapted params after T steps (body + head both adapt)
    φ_freeze_T = {θ₀^body, φ_T*^head}  -- body frozen at θ₀, ONLY head is
                 adapted for T steps (φ_T*^head != φ_T^head, they come from
                 two different trajectories)

──────────────────────────────────────────────────────────────────
Gradient decomposition (adjoint form), for X ∈ {φ_T, φ_freeze_T} with its
own trajectory X^(0..T) and its own adjoint λ^(t)(X_T):

    ∂L(X_T, Q)/∂S = - Σ_{t=1}^{T} α · λ^(t)(X_T) · ∂²L(X^(t-1), S)/∂X^(t-1)∂S

    λ^(T)   = ∇_φ L_Q(X^(T))
    λ^(t-1) = λ^(t) − α · H^(t-1) · λ^(t)     [H symmetric → no transpose]
    H^(t-1)·v via Pearlmutter HVP: O(P)

Therefore:

    ∂ΔM/∂S = ∂L(φ_freeze_T,Q)/∂S − ∂L(φ_T,Q)/∂S
           = Σ_t α λ^(t)(φ_T)        · ∂²L(φ^(t-1),S)/∂φ^(t-1)∂S            (saliency_full)
             − Σ_t α λ^(t)(φ_freeze_T) · ∂²L(φ_freeze^(t-1),S)/∂φ_freeze^(t-1)∂S  (saliency_freeze)

           = saliency_full − saliency_freeze
"""

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from algos.utils import get_loss_n_preds, put_on_device
from loaders.utils import get_stratified_bootstrap_batches
from models.utils import get_layer_parameters_map


class FAMAExplainer:
    """
    Post-hoc XAI for MAML-based
    Compute Adaptation Gain and Feature Saliency Map.
    """

    def __init__(self, algo_mgr, device: Optional[str] = None):
        self.algo_mgr = algo_mgr
        self.device = device or self.algo_mgr.device
        self.learner = self.algo_mgr.baselearner
        self.theta_0 = self.algo_mgr.theta_0
        self.base_lr = self.algo_mgr.base_lr

    # ------------------------------------------------------------------
    # Head / body split
    # ------------------------------------------------------------------
    def _get_head_mask(self) -> List[bool]:
        """
        Boolean mask aligned with self.theta_0: True  -> parameter belongs
        to the task HEAD (the only part that is adapted in φ_freeze),
        False -> parameter belongs to the BODY (frozen at θ₀ in φ_freeze).

        By default we assume the LAST parametrized module returned by
        `get_layer_parameters_map` is the task head (e.g. the final
        classifier / fc layer). Adjust here if your architecture defines
        the head differently (e.g. multiple last layers).
        """
        layer_map = get_layer_parameters_map(self.learner, self.theta_0)
        if not layer_map:
            # fallback: nothing frozen, behaves like full adaptation
            return [True] * len(self.theta_0)

        head_layer = layer_map[-1]
        head_param_ids = {id(p) for p in head_layer["params"]}
        return [id(p) in head_param_ids for p in self.theta_0]

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------
    def _compute_trajectory(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
        adapt_mask: Optional[List[bool]] = None,
    ) -> List[List[torch.Tensor]]:
        """
        Compute the parameter trajectory φ^(0) → φ^(T) on the support set.

        If `adapt_mask` is given, parameters with adapt_mask[i] == False are
        reset back to their θ₀ value after every step (i.e. never actually
        adapted) -- this produces the φ_freeze trajectory where only the
        head evolves and the body stays pinned at θ₀^body.
        """
        if adapt_mask is None:
            adapt_mask = [True] * len(self.theta_0)

        phis = [[p.detach().clone() for p in self.theta_0]]

        for _ in range(T):
            phi_r = [p.detach().clone().requires_grad_(True) for p in phis[-1]]
            loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
            grads = autograd.grad(loss, phi_r, create_graph=False)
            phi_next_all = self.algo_mgr._fast_weights(phi_r, grads)

            phi_next = [
                (p_next if adapt else p0.detach().clone())
                for p_next, p0, adapt in zip(phi_next_all, self.theta_0, adapt_mask)
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
        """Hessian-Vector Product via Pearlmutter."""
        phi_r = [p.detach().requires_grad_(True) for p in phi]
        loss, _ = get_loss_n_preds(phi_r, self.learner, sup_x, sup_y)
        grads = autograd.grad(loss, phi_r, create_graph=True)

        dot = sum((g * vi.detach()).sum() for g, vi in zip(grads, v))
        Hv = autograd.grad(dot, phi_r, retain_graph=False)

        return [hv.detach() for hv in Hv]

    def _compute_expected_lambda(
        self, phi_X, bootstrap_query, num_bootstraps
    ) -> List[torch.Tensor]:
        """Compute E_Q[∇_φ L_Q(φ_X)] by stratified bootstrap. Works for
        either φ_T or φ_freeze_T, just pass the corresponding params."""
        expected_lam = [torch.zeros_like(p) for p in phi_X]
        for b_que_x, b_que_y in bootstrap_query:
            phi_grad = [p.clone().detach().requires_grad_(True) for p in phi_X]
            q_loss, _ = get_loss_n_preds(phi_grad, self.learner, b_que_x, b_que_y)
            lam_b = autograd.grad(q_loss, phi_grad, retain_graph=False)

            expected_lam = [
                avg + lb.detach() / num_bootstraps
                for avg, lb in zip(expected_lam, lam_b)
            ]

        return expected_lam

    def _compute_adaptation_gain(
        self, phi_freeze_T, phi_T, bootstrap_query, num_bootstraps
    ) -> float:
        """
        ΔM = E_{Q~T_i}[ L(φ_freeze_T, Q) - L(φ_T, Q) ]
        φ_freeze_T = {θ₀^body, φ_T*^head}  (body frozen, head trained
        along its own T-step trajectory -- NOT phi_T's head)

        Gain (%) = ΔM / (E_Q[L(φ_T,Q)] + 1e-8) -- relative gain
        """
        freeze_sum = 0.0
        post_sum = 0.0

        with torch.no_grad():
            for b_que_x, b_que_y in bootstrap_query:
                freeze_l, _ = get_loss_n_preds(phi_freeze_T, self.learner, b_que_x, b_que_y)
                post_l, _ = get_loss_n_preds(phi_T, self.learner, b_que_x, b_que_y)
                freeze_sum += freeze_l.item()
                post_sum += post_l.item()

        freeze_loss = freeze_sum / num_bootstraps
        post_loss = post_sum / num_bootstraps

        return ((freeze_loss - post_loss) / (post_loss + 1e-8)) * 100.0

    # ------------------------------------------------------------------
    # Saliency
    # ------------------------------------------------------------------
    def _saliency_core_batched(
        self,
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        phis: List[List[torch.Tensor]],
        lambdas: Dict[int, List[torch.Tensor]],
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes  Σ_t α · λ^(t) · ∂²L(φ^(t-1),S)/∂φ^(t-1)∂S  (Grad-CAM style,
        upsampled to input resolution) for ONE trajectory (phis, lambdas).
        Call it once for (φ, λ) and once for (φ_freeze, λ_freeze), then
        subtract the two results to get ∂ΔM/∂S.
        """
        saliency = torch.zeros_like(sup_x[:, :1])
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

            # Global Average Pooling
            weights = grad_features.mean(dim=(2, 3), keepdim=True)
            cam = (weights * features_m.detach()).sum(dim=1, keepdim=True)
            cam_upsampled = F.interpolate(
                cam, size=sup_x.shape[-2:], mode="bilinear", align_corners=False
            )

            saliency += self.base_lr * cam_upsampled

        return saliency

    def _adjoint_backward(
        self,
        phis: List[List[torch.Tensor]],
        expected_lam_T: List[torch.Tensor],
        sup_x: torch.Tensor,
        sup_y: torch.Tensor,
        T: int,
    ) -> Dict[int, List[torch.Tensor]]:
        """Run the adjoint recursion λ^(t-1) = λ^(t) - α H^(t-1) λ^(t) over
        one trajectory and return the full {t: λ^(t)} dict."""
        lambdas = {T: expected_lam_T}
        for m in range(T, 0, -1):
            Hv = self._hvp(phis[m - 1], lambdas[m], sup_x, sup_y)
            lambdas[m - 1] = [
                (l - self.base_lr * hv).detach() for l, hv in zip(lambdas[m], Hv)
            ]
        return lambdas

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
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

        head_mask = self._get_head_mask()

        # -------- trajectory 1: full adaptation φ (body + head) --------
        phis = self._compute_trajectory(sup_x, sup_y, T)
        phi_T = [p.detach() for p in phis[T]]

        # -------- trajectory 2: φ_freeze = {θ0^body, φ*^head} ----------
        phis_freeze = self._compute_trajectory(sup_x, sup_y, T, adapt_mask=head_mask)
        phi_freeze_T = [p.detach() for p in phis_freeze[T]]

        # expected lambda_T for each trajectory (each w.r.t its own φ_T)
        expected_lam_T = self._compute_expected_lambda(
            phi_T, bootstrap_query, num_bootstraps
        )
        expected_lam_freeze_T = self._compute_expected_lambda(
            phi_freeze_T, bootstrap_query, num_bootstraps
        )

        # ΔM = E_Q[L(φ_freeze_T,Q) - L(φ_T,Q)]
        adaptation_gain = self._compute_adaptation_gain(
            phi_freeze_T, phi_T, bootstrap_query, num_bootstraps
        )

        # adjoint backward pass for each trajectory
        lambdas = self._adjoint_backward(phis, expected_lam_T, sup_x, sup_y, T)
        lambdas_freeze = self._adjoint_backward(
            phis_freeze, expected_lam_freeze_T, sup_x, sup_y, T
        )

        # Saliency for each term, then take the difference (see docstring)
        saliency_full = self._saliency_core_batched(sup_x, sup_y, phis, lambdas)
        saliency_freeze = self._saliency_core_batched(
            sup_x, sup_y, phis_freeze, lambdas_freeze
        )
        saliency_map = saliency_full - saliency_freeze

        return adaptation_gain, saliency_map
