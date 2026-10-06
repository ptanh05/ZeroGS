"""
Learned Split Policy Network (Section 2 of spec).

Replaces rule-based densification (gradient-threshold / opacity-threshold / heuristic
split-clone) with a learned MLP that outputs per-Gaussian action probabilities:

    πθ(features) → [P(KEEP), P(SPLIT), P(MERGE)]

Features (10-dim per Gaussian):
    position (x,y,z), scale (x,y,z), opacity, gradient magnitude,
    reprojection error, view frequency

Integration point:
    Called from GaussianModel.densify_and_prune() when self.use_policy_network=True.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SplitPolicyNetwork(nn.Module):
    """MLP policy network that learns when to KEEP / SPLIT / MERGE each Gaussian.

    Architecture: 10 → 128 → 64 → 3 (Softmax)

    Parameters
    ----------
    input_dim : int
        Number of input features per Gaussian (default 10).
    hidden_dim_1 : int
        First hidden layer size (default 128).
    hidden_dim_2 : int
        Second hidden layer size (default 64).
    num_actions : int
        Output action count (default 3: KEEP=0, SPLIT=1, MERGE=2).
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        num_actions: int = 3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_2, num_actions),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize with small weights -- start near-uniform."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute action probabilities for each Gaussian.

        Parameters
        ----------
        features : torch.Tensor
            Shape (N, input_dim) feature vectors.

        Returns
        -------
        torch.Tensor
            Shape (N, 3) softmax probabilities over [KEEP, SPLIT, MERGE].
        """
        logits = self.net(features)
        return F.softmax(logits, dim=-1)

    def sample_actions(
        self, probs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample discrete actions from the probability distribution.

        Parameters
        ----------
        probs : torch.Tensor
            Shape (N, 3) action probabilities.
        deterministic : bool
            If True, pick argmax (used for evaluation).

        Returns
        -------
        torch.Tensor
            Shape (N,) integer actions: 0=KEEP, 1=SPLIT, 2=MERGE.
        """
        if deterministic:
            return probs.argmax(dim=-1)
        # Categorical sampling
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def log_prob(self, probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log-probability of taken actions (for REINFORCE-style loss)."""
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)


# --------------------------------------------------------------------------- #
#  Feature construction helpers
# --------------------------------------------------------------------------- #

# Index constants into the 10-dim feature vector
_IDX_POS = slice(0, 3)
_IDX_SCALE = slice(3, 6)
_IDX_OPACITY = 6
_IDX_GRAD_MAG = 7
_IDX_REPROJ_ERR = 8
_IDX_VIEW_FREQ = 9


@torch.no_grad()
def compute_policy_features(
    gaussian_model,
    grads: torch.Tensor,
    radii: torch.Tensor,
    extent: float,
) -> torch.Tensor:
    """Build a 10-dim feature vector per Gaussian for the policy network.

    Parameters
    ----------
    gaussian_model : GaussianModel
        The current GaussianModel instance (must have recent render stats).
    grads : torch.Tensor
        Shape (N, 1) accumulated gradient magnitudes.
    radii : torch.Tensor
        Shape (N,) screen-space radii from last render.
    extent : float
        Scene extent for normalisation.

    Returns
    -------
    torch.Tensor
        Shape (N, 10) normalised feature vectors on the same device.
    """
    N = gaussian_model.get_xyz.shape[0]
    device = gaussian_model.get_xyz.device
    dtype = gaussian_model.get_xyz.dtype

    # 1) Position (normalised by scene extent)
    pos = gaussian_model.get_xyz / (extent + 1e-8)

    # 2) Scale (log-space, normalised)
    scale = gaussian_model.get_scaling / (extent + 1e-8)  # shape (N, 3)

    # 3) Opacity (already [0, 1])
    opacity = gaussian_model.get_opacity  # (N, 1)

    # 4) Gradient magnitude (normalised)
    grad_mag = grads.squeeze(-1)  # (N,)

    # 5) Reprojection error: approximated from gradient magnitude
    #    (a higher gradient implies higher reprojection error in GS)
    reproj_err = torch.sigmoid(grad_mag * 100.0 - 10.0)  # squash to ~[0, 1]

    # 6) View frequency: approximated from current-render visibility
    #    If radius > 0 the Gaussian was visible this frame
    view_freq = (radii > 0).float()  # (N,)

    # Assemble
    features = torch.cat(
        [
            pos,                               # (N, 3)
            scale,                             # (N, 3)
            opacity,                           # (N, 1)
            grad_mag.unsqueeze(-1),            # (N, 1)
            reproj_err.unsqueeze(-1),          # (N, 1)
            view_freq.unsqueeze(-1),           # (N, 1)
        ],
        dim=-1,
    )  # (N, 10)

    # Final normalisation to zero-mean unit-variance per dimension
    # (using running stats would be better, but simple z-score works for now)
    features = (features - features.mean(dim=0, keepdim=True)) / (
        features.std(dim=0, keepdim=True) + 1e-8
    )
    return features


# --------------------------------------------------------------------------- #
#  Policy loss
# --------------------------------------------------------------------------- #


def policy_loss(
    render_loss: torch.Tensor,
    gaussian_count: int,
    lambda_complexity: float = 0.0001,
    n_initial: int = 1,
) -> torch.Tensor:
    """Compute the policy training loss.

    L_policy = render_loss + λ * max(0, count / n_initial - 1)

    The complexity penalty discourages unbounded Gaussian growth.  It activates
    once count surpasses the initial point count.

    Parameters
    ----------
    render_loss : torch.Tensor
        Scalar rendering loss (L1 + SSIM) for current iteration.
    gaussian_count : int
        Current number of Gaussians.
    lambda_complexity : float
        Weight of the complexity penalty.
    n_initial : int
        Initial number of Gaussians (from SfM point cloud).

    Returns
    -------
    torch.Tensor
        Scalar policy loss (differentiable through render_loss).
    """
    complexity_ratio = max(0.0, gaussian_count / max(n_initial, 1) - 1.0)
    return render_loss + lambda_complexity * complexity_ratio


# --------------------------------------------------------------------------- #
#  Fallback detection
# --------------------------------------------------------------------------- #


def policy_diverged(loss: torch.Tensor) -> bool:
    """Check if policy loss has diverged (NaN or inf)."""
    return bool(torch.isnan(loss).any() or torch.isinf(loss).any())
