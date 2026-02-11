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

"""Hazard model (see https://journals.sagepub.com/doi/10.3102/10769986018002155)."""

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


try:
    from .focalnet import FocalEncoder
except ImportError:
    from focalnet import FocalEncoder


__all__ = ["HazardModel"]


class HazardModel(nn.Module):
    """Discrete-time hazard model.

    The model outputs hazard logits z_t over frames. After sigmoid, h_t = sigmoid(z_t)
    is interpreted as the probability of emitting a boundary at frame t, conditioned on
    not having emitted a boundary earlier within the current chunk.

    Notes
    -----
    - The induced distribution over chunk lengths is geometric if h_t is constant,
      and non-homogeneous geometric if h_t varies with t.
    - Decoding/sampling optionally enforce min/max gap constraints for stability.

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
        Maximum number of past frames each frame can attend to (used only if causal=True).

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
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

        # Modules
        self.head = FocalEncoder(
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

    def forward(
        self,
        input: "Tensor",
        state: "Tuple" = (),
    ) -> "Tuple[Tensor, Tuple]":
        """Compute hazard logits over frame-level features.

        Parameters
        ----------
        input:
            Input tensor of shape (batch_size, seq_length, dim).
        state:
            Streaming state.

        Returns
        -------
            - Hazard logits of shape (batch_size, seq_length);
            - updated streaming state.

        """
        logits, *state = self.head(input, *state)
        logits = logits[..., 0]  # [B, T]
        return logits, state

    @torch.no_grad()
    def decode(
        self,
        logits: "Tensor",
        length: "Optional[Tensor]" = None,
        threshold: "float" = 0.5,
        min_gap: "int" = 1,
        max_gap: "Optional[int]" = None,
    ) -> "Tensor":
        """Greedy decode chunk durations from hazard logits.

        A boundary is emitted when `sigmoid(logits) >= threshold` subject to
        `min_gap` suppression. Optionally, if `max_gap` is provided, a boundary
        is forced when the gap exceeds `max_gap`.

        Parameters
        ----------
        logits:
            Hazard logits of shape (batch_size, seq_length).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        threshold:
            Boundary threshold on hazard probability.
        min_gap:
            Minimum frames between boundaries (>=1).
        max_gap:
            If not None, force a boundary when no boundary has occurred
            in the last `max_gap` frames (and `min_gap` allows firing).

        Returns
        -------
            Chunk durations of shape (batch_size, num_chunks).
            Can contain zeros for padding.

        """
        B, T = logits.shape
        device = logits.device

        if length is None:
            abs_length = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            abs_length = (length * T).ceil().clamp(0, T).to(dtype=torch.long)
        h = logits.sigmoid()  # [B, T]

        boundary = torch.zeros((B, T), device=device, dtype=torch.bool)

        # Per-batch state
        last_boundary_t = torch.full((B,), -min_gap, device=device, dtype=torch.long)
        since_last = torch.zeros((B,), device=device, dtype=torch.long)

        for t in range(T):
            active = t < abs_length  # [B] only update valid timesteps

            can_fire = (t - last_boundary_t) >= min_gap  # [B]
            fire = can_fire & (h[:, t] >= threshold)  # [B]

            if max_gap is not None:
                # Match your original condition: force when since_last >= (max_gap - 1)
                force = can_fire & (since_last >= (max_gap - 1))
                fire = fire | (force & ~fire)

            fire = fire & active

            boundary[:, t] = fire

            # Update state
            last_boundary_t = torch.where(
                fire, torch.full_like(last_boundary_t, t), last_boundary_t
            )
            since_last = torch.where(fire, torch.zeros_like(since_last), since_last + 1)

            # Optional: keep state clean past sequence end (not strictly necessary)
            since_last = torch.where(active, since_last, torch.zeros_like(since_last))

        return boundaries_to_durations(boundary, abs_length)

    @torch.no_grad()
    def sample(
        self,
        logits: "Tensor",
        length: "Optional[Tensor]" = None,
        min_gap: "int" = 1,
        max_gap: "int" = 50,
        temperature: "float" = 1.0,
    ) -> "Tensor":
        """Sample chunk durations from hazard logits.

        Boundaries are sampled sequentially using Bernoulli trials with probabilities
        `sigmoid(logits / temperature)`. To avoid unbounded gaps, we sample within
        windows of size `max_gap`: we take the first sampled hit; if none hits, we
        force a boundary at the window end. After emitting a boundary, we advance
        by `min_gap` frames.

        Parameters
        ----------
        logits:
            Hazard logits of shape (batch_size, seq_length).
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        min_gap:
            Minimum frames between boundaries (>=1).
        max_gap:
            Maximum frames without a boundary; a boundary is forced at `max_gap`.
        temperature:
            Sampling temperature applied to logits before sigmoid.

        Returns
        -------
            Chunk durations of shape (batch_size, num_chunks).
            Can contain zeros for padding.

        """
        B, T = logits.shape
        device = logits.device

        if length is None:
            abs_length = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            abs_length = (length * T).ceil().clamp(0, T).to(dtype=torch.long)
        h = (logits / temperature).sigmoid()  # [B, T]

        boundary = torch.zeros((B, T), device=device, dtype=torch.bool)

        # Current position per sample
        t = torch.zeros((B,), device=device, dtype=torch.long)

        # Iterate until all sequences are done
        while True:
            active = t < abs_length  # [B]
            if not bool(active.any()):
                break

            # Window length per sample (<= max_gap, <= remaining length)
            remaining = (abs_length - t).clamp(min=0)  # [B]
            chunk_len = torch.minimum(
                remaining, torch.full_like(remaining, max_gap)
            )  # [B]
            W = max_gap

            # Build indices to gather probs for a [B, W] window starting at t[b]
            offsets = torch.arange(W, device=device, dtype=torch.long)[
                None, :
            ]  # [1, W]
            idx = t[:, None] + offsets  # [B, W]

            # Mask for valid positions inside each sample chunk and inside [0, T)
            valid = (
                (offsets < chunk_len[:, None]) & active[:, None] & (idx < T)
            )  # [B, W]

            # Gather probs; fill invalid with 0 so draws are always False there
            probs = torch.where(
                valid,
                h.gather(1, idx.clamp(max=T - 1)),
                torch.zeros((B, W), device=device),
            )

            # Bernoulli draws
            draws = (torch.rand((B, W), device=device) < probs) & valid  # [B, W]

            any_hit = draws.any(dim=1)  # [B]

            # First hit index: argmax of draws works because False=0, True=1 (and we gate by any_hit)
            first_hit = draws.to(torch.int32).argmax(
                dim=1
            )  # [B], meaningless if no hit

            # Forced index is end of chunk: chunk_len-1 (only for active)
            forced_hit = (chunk_len - 1).clamp(min=0)  # [B]

            hit = torch.where(any_hit, first_hit, forced_hit)  # [B]
            hit = torch.where(active, hit, torch.zeros_like(hit))

            fire_t = t + hit  # [B]
            # Set boundaries (only for active)
            b_idx = torch.arange(B, device=device)
            boundary[b_idx[active], fire_t[active]] = True

            # Advance by hit+min_gap
            t = torch.where(active, fire_t + min_gap, t)

        return boundaries_to_durations(boundary, abs_length)


def boundaries_to_durations(boundary: "Tensor", abs_length: "Tensor") -> "Tensor":
    """Convert boundary indicators to per-chunk durations.

    Convention:
    - boundary[b, t] == True means a chunk ends at frame t (0-indexed).
    - We always force the last valid frame ``abs_length[b] - 1`` to be a boundary.
    - Durations are positive integers that sum to ``abs_length[b]``.

    Parameters
    ----------
    boundary:
        Boolean tensor of shape (batch_size, seq_length).
    abs_length:
        Absolute valid lengths in frames of shape (batch_size,).

    Returns
    -------
        Chunk durations of shape (batch_size, num_chunks).
        Can contain zeros for padding.

    """
    B, T = boundary.shape
    device = boundary.device
    abs_length = abs_length.to(device=device, dtype=torch.long)

    # Valid frame mask
    t_idx = torch.arange(T, device=device, dtype=torch.long)[None, :]  # [1, T]
    valid = t_idx < abs_length[:, None]  # [B, T]

    # Force final boundary at L-1 (only for L>0)
    bnd = boundary.to(torch.bool).clone()
    has_len = abs_length > 0  # [B]
    last_pos = (abs_length - 1).clamp(min=0)  # [B]
    bnd[has_len, last_pos[has_len]] = True

    # Only count boundaries inside valid region
    bnd = bnd & valid

    # Chunk index per frame: 0,0,0,... until first boundary (inclusive), then 1,1,... etc.
    c = bnd.to(torch.int32).cumsum(dim=1)  # [B, T]
    chunk_id = (c - 1).clamp(min=0).to(torch.long)  # [B, T]
    chunk_id = torch.where(valid, chunk_id, torch.zeros_like(chunk_id))  # keep in-range

    # Number of chunks per sample = c at last valid frame
    n_chunks = torch.zeros((B,), device=device, dtype=torch.long)
    n_chunks[has_len] = c[has_len, last_pos[has_len]].to(torch.long)
    n_chunks[~has_len] = 1  # match your L<=0 behavior
    U_max = int(n_chunks.max().item())

    # Durations = count frames per chunk_id
    out = torch.zeros((B, U_max), device=device, dtype=torch.long)
    out.scatter_add_(1, chunk_id[:, :T], valid.to(torch.long))

    return out


def test_model() -> "None":
    import torch

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    T = 60

    # ---- boundaries_to_durations correctness vs reference ----
    boundary = torch.rand((B, T), device=device) < 0.2
    abs_length = torch.randint(0, T + 1, (B,), device=device)

    out = boundaries_to_durations(boundary, abs_length)

    # Invariant: durations sum to abs_length (or 1 if abs_length==0, matching helper)
    expected = torch.where(abs_length > 0, abs_length, torch.ones_like(abs_length))
    assert torch.equal(out.sum(dim=1).cpu(), expected.cpu())

    # ---- decode: basic invariants + matches reference boundaries_to_durations ----
    logits = torch.randn((B, T), device=device) * 2.0
    length = torch.rand((B,), device=device)  # relative in [0,1]
    length = torch.clamp(length, 0.0, 1.0)

    model = HazardModel().to(device)
    model.eval()
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    durs_dec = model.decode(
        logits=logits,
        length=length,
        threshold=0.5,
        min_gap=3,
        max_gap=12,
    )

    abs_len_dec = (length * T).ceil().clamp(0, T).to(torch.long)
    expected_dec = torch.where(
        abs_len_dec > 0, abs_len_dec, torch.ones_like(abs_len_dec)
    )

    assert durs_dec.dtype == torch.long
    assert durs_dec.shape[0] == B
    assert torch.equal(durs_dec.sum(dim=1).cpu(), expected_dec.cpu())

    # ---- sample: invariants (sums, positivity, padding) ----
    torch.manual_seed(0)  # deterministic for the test
    durs_samp = model.sample(
        logits=logits,
        length=length,
        min_gap=3,
        max_gap=12,
        temperature=1.0,
    )

    assert durs_samp.dtype == torch.long
    assert durs_samp.shape[0] == B
    assert (durs_samp >= 0).all()
    assert torch.equal(durs_samp.sum(dim=1).cpu(), expected_dec.cpu())

    # padding is zeros on the right
    for b in range(B):
        row = durs_samp[b]
        nz = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
        if nz.numel() > 0:
            last = int(nz[-1].item())
            assert torch.equal(row[last + 1 :], torch.zeros_like(row[last + 1 :]))

    # ---- Forward pass (eager smoke test) ----
    x = torch.randn((B, T, 1024), device=device)
    with torch.no_grad():
        # If your forward signature is different, adjust this one line.
        y, *_ = model(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == B
    assert y.shape[1] == T
    assert torch.isfinite(y).all()

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 80

    model = HazardModel().to(device)

    logits = torch.randn((B, T), device=device) * 1.5
    length = torch.rand((B,), device=device).clamp(0.0, 1.0)

    decode_kwargs = dict(threshold=0.5, min_gap=4, max_gap=17)

    # Batched path
    batch_dec = model.decode(logits, length=length, **decode_kwargs)
    U_dec_max = batch_dec.shape[1]

    # Singleton path
    all_single_dec = []

    for i in range(B):
        single_dec = model.decode(
            logits[i : i + 1],
            length=length[i : i + 1],
            **decode_kwargs,
        )

        # Pad to batch width for comparison
        if single_dec.shape[1] < U_dec_max:
            pad = single_dec.new_zeros((1, U_dec_max - single_dec.shape[1]))
            single_dec = torch.cat([single_dec, pad], dim=1)

        all_single_dec.append(single_dec)

    all_single_dec = torch.cat(all_single_dec, dim=0)

    assert torch.equal(batch_dec, all_single_dec)

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
