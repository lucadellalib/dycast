# ==============================================================================
# Copyright 2026 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Negative-binomial model (see https://www.sciencedirect.com/science/article/abs/pii/S0167639309000648)."""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


try:
    from .focalnet import FocalEncoder
except ImportError:
    from focalnet import FocalEncoder


__all__ = ["NegBinModel"]


class NegBinModel(nn.Module):
    """Negative-binomial model that predicts per-token expected durations.

    The model predicts a *free* mean duration `mu_free` (strictly positive) for each
    token, then adds a fixed minimum duration:

        mu = mu_free + min_duration

    This makes the minimum duration constraint explicit and prevents the network from
    "hiding" mass at exactly `min_duration`.

    Sampling uses a Negative Binomial parameterization with global dispersion `alpha`:

        var = mu + mu^2 / alpha

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    hidden_dims:
        Sequence of hidden dimensions in the modulation layers.
    downscale_factors:
        Sequence of downscaling factors for each layer.
    focal_window:
        Size of the initial focal window in the modulation layers.
    focal_level:
        Number of hierarchical focal levels in the modulation layers.
    focal_factor:
        Scaling factor for focal window sizes across levels in the modulation layers.
    dropout:
        Dropout probability applied to the modulation and feed-forward layers.
    use_post_norm:
        If True, apply layer normalization or dynamic tanh after modulation.
    use_layerscale:
        If True, apply layer scaling to modulation and feed-forward layers.
    layerscale_init:
        Initial value for layer scaling parameter.
    tanhscale_init:
        Initial value for tanh scaling parameter.
    normalize_modulator:
        If True, normalize the modulator in the modulation layers for stabilizing training.
    causal:
        Whether the module should be causal.
    window_size:
        Maximum number of past tokens each token can attend to (used only if causal=True).
    min_duration:
        Minimum expected duration of a token in feature frames.
    eps:
        Numerical stability constant.

    """

    def __init__(
        self,
        input_dim: "int" = 32,
        hidden_dims: "Sequence[int]" = (1024, 1024, 1024),
        downscale_factors: "Sequence[int]" = (1, 1, 1),
        focal_window: "int" = 14,
        focal_level: "int" = 2,
        focal_factor: "int" = 4,
        dropout: "float" = 0.0,
        use_post_norm: "bool" = False,
        use_layerscale: "bool" = False,
        layerscale_init: "float" = 1e-4,
        tanhscale_init: "float" = 0.5,
        normalize_modulator: "bool" = False,
        causal: "bool" = False,
        window_size: "int" = 512,
        min_duration: "int" = 1,
        eps: "float" = 1e-4,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.downscale_factors = downscale_factors
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.dropout_ = dropout
        self.use_post_norm = use_post_norm
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.tanhscale_init = tanhscale_init
        self.normalize_modulator = normalize_modulator
        self.causal = causal
        self.window_size = window_size
        self.min_duration = min_duration
        self.eps = eps

        # Modules
        self.mu_head = FocalEncoder(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            downscale_factors=downscale_factors,
            focal_window=focal_window,
            focal_level=focal_level,
            focal_factor=focal_factor,
            dropout=dropout,
            use_post_norm=use_post_norm,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            tanhscale_init=tanhscale_init,
            normalize_modulator=normalize_modulator,
            causal=causal,
            window_size=window_size,
        )

        # Global dispersion alpha (learned scalar via softplus)
        self._alpha_unconstrained = nn.Parameter(torch.tensor(0.0))

    def alpha(self) -> "Tensor":
        """Return positive dispersion parameter alpha (scalar)."""
        return nn.functional.softplus(self._alpha_unconstrained) + self.eps

    def forward(
        self,
        input: "Tensor",
        state: "Tuple" = (),
    ) -> "Tuple[Tensor, Tuple]":
        """Predict per-token expected duration.

        Parameters
        ----------
        input:
            Token-level representations of shape (batch_size, num_chunks, dim).
        state:
            Streaming state.

        Returns
        -------
            - Expected *excess* duration above min_duration of shape (batch_size, num_chunks).
            - updated streaming state.

        """
        B, N, _ = input.shape
        mu_raw, *state = self.mu_head(input, *state)
        mu_raw = mu_raw[..., 0]  # [B, N]
        mu_free = nn.functional.softplus(mu_raw) + self.eps
        return mu_free, state

    @torch.no_grad()
    def decode(
        self,
        mu_free: "Tensor",
        length: "Optional[Tensor]" = None,
        num_frames: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Decode integer chunk durations from predicted mean excess durations.

        This method converts predicted *excess* durations `mu_free` into integer
        chunk durations by adding a fixed minimum duration per chunk and optionally
        enforcing a global frame budget.

        Two decoding modes are supported:

        - Free decoding (`num_frames is None`):
            Each chunk duration is independently decoded as
            ``round(mu_free + min_duration)``, subject to masking.

        - Budgeted decoding (`num_frames is not None`):
            The total number of output frames is constrained to exactly
            ``num_frames`` per batch element. Excess durations are first
            renormalized to match the available frame budget and then converted
            to integers using exact-sum rounding.

        Parameters
        ----------
        mu_free:
            Expected *excess* duration above min_duration of shape (batch_size, num_chunks).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        num_frames:
            Optional target total number of output frames per batch element
            of shape (batch_size,). If provided, budgeted decoding is used; otherwise,
            free decoding is applied.

        Returns
        -------
            Integer chunk durations of shape (batch_size, num_chunks). Durations are
            non-negative, equal to zero for padded chunks, and sum to ``num_frames``
            when budgeted decoding is used.

        """
        device = mu_free.device
        B, N = mu_free.shape

        if length is None:
            mask_valid = torch.ones((B, N), device=device, dtype=torch.bool)
        else:
            abs_length = (length * N).ceil().clamp(0, N).to(dtype=torch.long)
            arange = torch.arange(N, device=device)[None, :].expand(B, N)
            mask_valid = arange < abs_length[:, None]

        # Free decoding
        if num_frames is None:
            d = (
                torch.round(mu_free + self.min_duration)
                .to(dtype=torch.long)
                .clamp_min(self.min_duration)
            )

        # Budget decoding
        else:
            num_frames = num_frames.to(device=device, dtype=torch.long)

            n_valid = mask_valid.sum(dim=1).to(dtype=torch.long)
            T_free = (num_frames - self.min_duration * n_valid).clamp_min(0)  # [B]

            mu_free_scaled = renormalize_mu_to_budget(mu_free, T_free, mask_valid)
            d_free = exact_sum_rounding(mu_free_scaled, T_free, mask_valid)

            d = d_free + (mask_valid.to(dtype=torch.long) * self.min_duration)

        return torch.where(mask_valid, d, torch.zeros_like(d))

    @torch.no_grad()
    def sample(
        self,
        mu_free: "Tensor",
        length: "Optional[Tensor]" = None,
        num_frames: "Optional[Tensor]" = None,
    ) -> "Tensor":
        """Sample integer chunk durations from predicted mean excess durations.

        This method samples *excess* durations above ``min_duration`` from a
        Negative Binomial distribution parameterized by the predicted `mu_free`,
        then adds a fixed minimum duration per chunk and optionally enforces a
        global frame budget.

        Two sampling modes are supported:

        - Free sampling (`num_frames is None`):
            Excess durations are sampled independently and the final durations are
            obtained as ``d = d_free + min_duration``, subject to masking.

        - Budgeted sampling (`num_frames is not None`):
            The total number of output frames is constrained to exactly
            ``num_frames`` per batch element. Sampled excess durations are first
            renormalized to match the available frame budget and then converted
            to integers using exact-sum rounding.

        Parameters
        ----------
        mu_free:
            Expected *excess* duration above min_duration of shape (batch_size, num_chunks).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        num_frames:
            Optional target total number of output frames per batch element
            of shape (batch_size,). If provided, budgeted sampling is used; otherwise,
            free sampling is applied.

        Returns
        -------
            Integer chunk durations of shape (batch_size, num_chunks). Durations are
            non-negative, equal to zero for padded chunks, and sum to ``num_frames``
            when budgeted sampling is used.

        """
        device = mu_free.device
        B, N = mu_free.shape

        if length is None:
            mask_valid = torch.ones((B, N), device=device, dtype=torch.bool)
        else:
            abs_length = (length * N).ceil().clamp(0, N).to(dtype=torch.long)
            arange = torch.arange(N, device=device)[None, :].expand(B, N)
            mask_valid = arange < abs_length[:, None]

        alpha = self.alpha().to(dtype=mu_free.dtype, device=device).clamp_min(self.eps)
        mu = mu_free.clamp_min(self.eps)
        r = (1.0 / alpha).clamp_min(self.eps)
        logits = (mu.log() - r.log()).clamp(-30.0, 30.0)  # Optional clamp
        nb = torch.distributions.NegativeBinomial(total_count=r, logits=logits)

        d_free = nb.sample().to(dtype=torch.long)
        d_free = torch.where(mask_valid, d_free, torch.zeros_like(d_free))

        # Free sampling
        if num_frames is None:
            d = d_free + (mask_valid.to(dtype=torch.long) * self.min_duration)

        # Budget sampling
        else:
            num_frames = num_frames.to(device=device, dtype=torch.long)

            n_valid = mask_valid.sum(dim=1).to(dtype=torch.long)
            T_free = (num_frames - self.min_duration * n_valid).clamp_min(0)

            mu_free_scaled = renormalize_mu_to_budget(
                d_free.to(dtype=mu_free.dtype), T_free, mask_valid
            )
            d_free_budget = exact_sum_rounding(mu_free_scaled, T_free, mask_valid)

            d = d_free_budget + (mask_valid.to(dtype=torch.long) * self.min_duration)

        return torch.where(mask_valid, d, torch.zeros_like(d))


def renormalize_mu_to_budget(
    mu: "Tensor", budget: "Tensor", mask_valid: "Tensor"
) -> "Tensor":
    """Renormalize per-token allocations to match a target budget.

    Scales the non-negative values `mu` so that, for each batch element, the
    sum over valid positions equals the specified `budget`. Invalid positions
    (where ``mask_valid == False``) are ignored and set to zero in the output.

    If the sum of `mu` over valid positions is zero for a given batch element,
    the budget is distributed uniformly across all valid positions.

    Parameters
    ----------
    mu:
        Non-negative real-valued allocations of shape (batch_size, num_chunks).
        Values at invalid positions are ignored.
    budget:
        Target total allocation per batch element of shape (batch_size,).
    mask_valid:
        Boolean mask of shape (batch_size, num_chunks) indicating valid positions.

    Returns
    -------
        Renormalized allocations of shape (batch_size, num_chunks) such that,
        for each batch element, the sum over valid positions equals `budget`.
        Values at invalid positions are zero.

    """
    mu = torch.where(mask_valid, mu, torch.zeros_like(mu))
    budget_f = budget.to(dtype=mu.dtype)

    s = mu.sum(dim=1).clamp_min(1e-12)  # [B]
    scaled = mu * (budget_f[:, None] / s[:, None])

    zero_sum = s <= 1e-11
    if torch.any(zero_sum):
        n_valid = mask_valid.sum(dim=1).clamp_min(1).to(dtype=mu.dtype)  # [B]
        uniform = (budget_f / n_valid)[:, None].expand_as(mu)
        scaled = torch.where(zero_sum[:, None] & mask_valid, uniform, scaled)

    return torch.where(mask_valid, scaled, torch.zeros_like(scaled))


def exact_sum_rounding(
    mu: "Tensor",
    target_sum: "Tensor",
    mask_valid: "Tensor",
) -> "Tensor":
    """Exact-sum rounding of non-negative real allocations.

    Converts non-negative real values `mu` into integer allocations that
    sum **exactly** to `target_sum` over valid positions, using a
    *floor + remainder distribution* strategy:

    1. Take `floor(mu)` as a base allocation.
    2. Compute the remaining budget `target_sum - sum(floor(mu))` per sample.
    3. Distribute the remainder by adding 1 to the positions with the largest
       fractional parts, restricted to valid positions.

    The procedure only **adds** mass (never subtracts). If `sum(floor(mu))`
    already exceeds `target_sum`, the remainder is clamped to zero.

    Parameters
    ----------
    mu:
        Non-negative real-valued allocations of shape (batch_size, num_chunks).
        Values at invalid positions are ignored.
    target_sum:
        Target total integer allocation per batch element of shape (batch_size,).
    mask_valid:
        Boolean mask of shape (batch_size, num_chunks) indicating valid positions
        over which the sum constraint is enforced.

    Returns
    -------
        Integer allocations of shape (batch_size, num_chunks) such that, for each
        batch element ``b``:
        - ``d[b, n] >= 0`` for all ``n``,
        - ``d[b, n] == 0`` where ``mask_valid[b, n] == False``,
        - ``sum_n d[b, n] == target_sum[b]`` (up to clamping when
          ``target_sum[b] < sum(floor(mu[b]))``).

    """
    B, N = mu.shape
    device = mu.device

    target_sum = target_sum.to(device=device, dtype=torch.long)
    mask_valid = mask_valid.to(device=device, dtype=torch.bool)

    # Zero-out padding
    mu = torch.where(mask_valid, mu, torch.zeros_like(mu))

    d0 = torch.floor(mu).to(dtype=torch.long)  # [B, N]
    frac = mu - d0.to(dtype=mu.dtype)  # [B, N] in [0, 1)

    s0 = d0.sum(dim=1)  # [B]
    rem = (target_sum - s0).clamp_min(0)  # [B]

    # Clamp remainder by number of valid tokens (can't add more than that)
    n_valid = mask_valid.sum(dim=1).to(dtype=torch.long)  # [B]
    rem = torch.minimum(rem, n_valid)  # [B]

    d = d0.clone()

    K = int(rem.max().item()) if B > 0 else 0
    if K == 0:
        return torch.where(mask_valid, d, torch.zeros_like(d))

    # Mask invalid positions so they never get selected
    frac_masked = torch.where(
        mask_valid,
        frac,
        torch.full_like(frac, float("-inf")),
    )

    # Take top-K fractional parts per row
    topk = torch.topk(frac_masked, k=K, dim=1, largest=True)
    idx = topk.indices  # [B, K]

    # We want to add 1 to the first rem[b] entries in idx[b]
    add_mask = torch.arange(K, device=device)[None, :] < rem[:, None]  # [B, K]
    add = add_mask.to(dtype=torch.long)  # [B, K]

    # Scatter-add into d along dim=1
    d.scatter_add_(dim=1, index=idx, src=add)

    return torch.where(mask_valid, d, torch.zeros_like(d))


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    N = 40
    D = 32

    model = NegBinModel().to(device)
    model.eval()

    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    # -------------------------------------------------
    # Forward (mu_free, state) smoke test + invariants
    # -------------------------------------------------
    x = torch.randn((B, N, D), device=device)

    with torch.no_grad():
        mu_free, state = model(x)

    assert isinstance(mu_free, torch.Tensor)
    assert isinstance(state, (tuple, list))
    assert mu_free.shape == (B, N)
    assert mu_free.dtype == x.dtype
    assert torch.isfinite(mu_free).all()
    assert (mu_free > 0).all()

    # -------------------------------------------------
    # Decode: free + budget invariants
    # -------------------------------------------------
    length = torch.rand((B,), device=device).clamp(0.0, 1.0)

    d_free = model.decode(mu_free, length=length, num_frames=None)

    assert d_free.dtype == torch.long
    assert d_free.shape == (B, N)
    assert (d_free >= 0).all()

    # Mask for checks
    abs_length = (length * N).ceil().clamp(0, N).to(dtype=torch.long)
    arange = torch.arange(N, device=device)[None, :].expand(B, N)
    mask_valid = arange < abs_length[:, None]

    assert torch.equal(d_free[~mask_valid], torch.zeros_like(d_free[~mask_valid]))
    if bool(mask_valid.any()):
        assert (d_free[mask_valid] >= model.min_duration).all()

    # Budgeted decode should sum exactly to num_frames (per sample)
    n_valid = mask_valid.sum(dim=1).to(dtype=torch.long)
    slack = torch.randint(0, 50, (B,), device=device, dtype=torch.long)
    num_frames = (model.min_duration * n_valid + slack).to(dtype=torch.long)

    d_budget = model.decode(mu_free, length=length, num_frames=num_frames)

    assert d_budget.dtype == torch.long
    assert d_budget.shape == (B, N)
    assert (d_budget >= 0).all()
    assert torch.equal(d_budget[~mask_valid], torch.zeros_like(d_budget[~mask_valid]))
    if bool(mask_valid.any()):
        assert (d_budget[mask_valid] >= model.min_duration).all()

    assert torch.equal(d_budget.sum(dim=1).cpu(), num_frames.cpu())

    # -------------------------------------------------
    # Sample: free + budget invariants (deterministic seed)
    # -------------------------------------------------
    torch.manual_seed(0)
    s_free = model.sample(mu_free, length=length, num_frames=None)

    assert s_free.dtype == torch.long
    assert s_free.shape == (B, N)
    assert (s_free >= 0).all()
    assert torch.equal(s_free[~mask_valid], torch.zeros_like(s_free[~mask_valid]))
    if bool(mask_valid.any()):
        assert (s_free[mask_valid] >= model.min_duration).all()

    torch.manual_seed(0)
    s_budget = model.sample(mu_free, length=length, num_frames=num_frames)

    assert s_budget.dtype == torch.long
    assert s_budget.shape == (B, N)
    assert (s_budget >= 0).all()
    assert torch.equal(s_budget[~mask_valid], torch.zeros_like(s_budget[~mask_valid]))
    if bool(mask_valid.any()):
        assert (s_budget[mask_valid] >= model.min_duration).all()

    assert torch.equal(s_budget.sum(dim=1).cpu(), num_frames.cpu())

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    N = 60
    D = 32

    model = NegBinModel().to(device)
    model.eval()

    x = torch.randn((B, N, D), device=device)
    length = torch.rand((B,), device=device).clamp(0.0, 1.0)

    with torch.no_grad():
        mu_free, _ = model(x)

    # Compute a feasible per-sample frame budget based on length
    abs_length = (length * N).ceil().clamp(0, N).to(dtype=torch.long)
    arange = torch.arange(N, device=device)[None, :].expand(B, N)
    mask_valid = arange < abs_length[:, None]

    n_valid = mask_valid.sum(dim=1).to(dtype=torch.long)
    slack = torch.randint(0, 50, (B,), device=device, dtype=torch.long)
    num_frames = (model.min_duration * n_valid + slack).to(dtype=torch.long)

    # Batched path (decode only)
    batch_dec = model.decode(mu_free, length=length, num_frames=num_frames)

    # Singleton path
    all_single_dec = []
    for i in range(B):
        single_dec = model.decode(
            mu_free[i : i + 1],
            length=length[i : i + 1],
            num_frames=num_frames[i : i + 1],
        )
        all_single_dec.append(single_dec)

    all_single_dec = torch.cat(all_single_dec, dim=0)

    assert torch.equal(batch_dec, all_single_dec)

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
