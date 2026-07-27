import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothMarginLoss(nn.Module):
    """Approximate margin loss: L = softplus_beta(1 - z_y + tau*logsumexp_{y'≠y}(z_{y'}/tau)).
    tau->0 recovers a hard max; beta->∞ recovers ReLU."""

    def __init__(self, tau: float = 0.5, beta: float = 5.0, reduction: str = "mean"):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], y: [B, C] - one-hot vector
        B, C = logits.shape
        
        mask = y.bool() # [B] 
        # Get z_y
        z_y = (logits * y).sum(dim=1)  # [B]

        # Remove the correct class from logits (the competition) by setting it to -inf
        logits_masked = logits.masked_fill(mask, float("-inf"))  # [B, C]

        # soft max of the other classes (logsumexp with temperature tau, numerically stable)
        soft_max_other = self.tau * torch.logsumexp(logits_masked / self.tau, dim=1)  # [B]

        # soft margin: m_tilde = z_y - soft_max_other
        m_tilde = z_y - soft_max_other  # [B]

        # Calc softplus_beta(1 - m_tilde)
        u = 1.0 - m_tilde
        loss = F.softplus(self.beta * u) / self.beta  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
