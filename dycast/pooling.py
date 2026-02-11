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

"""Dynamic pooling."""

import torch
from torch import Tensor, nn


__all__ = ["RepeatInterleaveUnpool", "SelectLastPool"]


class SelectLastPool(nn.Module):
    """Select-last dynamic pooling over variable-length chunks.

    Given frame-level latents and per-chunk durations, selects the last frame
    latent of each chunk.

    """

    def forward(
        self,
        lats: "Tensor",
        durs: "Tensor",
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        lats:
            Frame-level latents of shape (batch_size, seq_length, dim).
        durs:
            Chunk durations of shape (batch_size, num_chunks).
            Can contain zeros for padding.

        Returns
        -------
            Chunk-level latents of shape (batch_size, num_chunks, dim).

        """
        B, T, D = lats.shape
        durs = durs.to(device=lats.device, dtype=torch.long)

        idx = durs.cumsum(dim=1) - 1
        idx = idx.clamp_(min=0, max=T - 1)

        plats = lats.gather(1, idx[..., None].expand(B, idx.shape[1], D))
        return plats


class RepeatInterleaveUnpool(nn.Module):
    """Dynamic unpooling by duration-controlled repetition.

    Reconstructs a frame-level sequence by repeating each chunk latent
    according to its duration.

    """

    def forward(
        self,
        plats: "Tensor",
        durs: "Tensor",
    ) -> "Tensor":
        """Forward pass.

        Parameters
        ----------
        plats:
            Chunk-level latents of shape (batch_size, num_chunks, dim).
        durs:
            Chunk durations of shape (batch_size, num_chunks).
            Can contain zeros for padding.

        Returns
        -------
            Frame-level latents of shape (batch_size, seq_length, dim),
            where seq_length = max_b sum_u durs[b, u].

        """
        durs = durs.to(device=plats.device, dtype=torch.long)
        B, U, D = plats.shape

        T_per = durs.sum(dim=1)  # (B,)
        T_max = T_per.max()
        slack = (T_max - T_per).clamp_min_(0)  # (B,)

        plats = nn.functional.pad(plats, (0, 0, 0, 1))  # pad 1 chunk on dim=1
        durs = torch.cat([durs, slack[:, None]], dim=1)  # (B, U+1)

        flat_plats = plats.flatten(end_dim=-2)  # (B*(U+1), D)
        flat_durs = durs.flatten()  # (B*(U+1),)

        flat_lats = flat_plats.repeat_interleave(flat_durs, dim=0)  # (B*T_max, D)
        lats = flat_lats.reshape(B, -1, D)  # (B, T_max, D)

        return lats


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    U = 20
    D = 32
    max_dur = 6

    pool = SelectLastPool().to(device)
    unpool = RepeatInterleaveUnpool().to(device)

    durs = torch.randint(0, max_dur + 1, (B, U), device=device)
    durs[:, 0] = torch.clamp(durs[:, 0], min=1)

    plats = torch.randn(B, U, D, device=device)

    lats = unpool(plats, durs)
    T_per = durs.sum(dim=1)
    T_max = int(T_per.max().item())

    assert lats.shape == (B, T_max, D)
    assert lats.dtype == plats.dtype

    # Reference: per-example repeat + padding
    for b in range(B):
        u = torch.arange(U, device=device).repeat_interleave(durs[b])
        ref = plats[b].index_select(0, u)
        Tb = int(T_per[b].item())

        assert torch.allclose(lats[b, :Tb], ref, atol=0.0, rtol=0.0)
        assert torch.allclose(
            lats[b, Tb:], torch.zeros_like(lats[b, Tb:]), atol=0.0, rtol=0.0
        )

    pooled = pool(lats, durs)
    assert pooled.shape == (B, U, D)

    for b in range(B):
        ends = durs[b].cumsum(dim=0) - 1
        Tb = int(T_per[b].item())
        ends = torch.clamp(ends, min=0, max=max(Tb - 1, 0))
        ref_pooled = lats[b].index_select(0, ends)
        assert torch.allclose(pooled[b], ref_pooled, atol=0.0, rtol=0.0)

    # Round-trip (pool then unpool) should match original unpool result exactly
    lats_rt = unpool(pooled, durs)
    assert torch.allclose(lats, lats_rt, atol=0.0, rtol=0.0)

    # JIT
    pool_jit = torch.jit.script(pool)
    unpool_jit = torch.jit.script(unpool)

    pooled_jit = pool_jit(lats, durs)
    lats_jit = unpool_jit(plats, durs)
    lats_rt_jit = unpool_jit(pooled_jit, durs)

    assert torch.allclose(pooled, pooled_jit)
    assert torch.allclose(lats, lats_jit)
    assert torch.allclose(lats_rt, lats_rt_jit)

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    U = 30
    D = 16
    max_dur = 8

    pool = SelectLastPool().to(device)
    unpool = RepeatInterleaveUnpool().to(device)

    durs = torch.randint(0, max_dur + 1, (B, U), device=device)
    durs[:, 0] = torch.clamp(durs[:, 0], min=1)

    plats = torch.randn(B, U, D, device=device)

    # Batched path
    batch_lats = unpool(plats, durs)
    batch_plats = pool(batch_lats, durs)
    batch_lats_rt = unpool(batch_plats, durs)

    # Singleton path
    all_single_lats = []
    all_single_plats = []
    all_single_lats_rt = []

    T_per = durs.sum(dim=1)
    T_max = int(T_per.max().item())

    for i in range(B):
        single_lats = unpool(plats[i : i + 1], durs[i : i + 1])
        single_plats = pool(single_lats, durs[i : i + 1])
        single_lats_rt = unpool(single_plats, durs[i : i + 1])

        # pad to batch T_max for comparison
        Ti = single_lats.shape[1]
        if Ti < T_max:
            pad = single_lats.new_zeros((1, T_max - Ti, D))
            single_lats = torch.cat([single_lats, pad], dim=1)
            single_lats_rt = torch.cat([single_lats_rt, pad], dim=1)

        all_single_lats.append(single_lats)
        all_single_plats.append(single_plats)
        all_single_lats_rt.append(single_lats_rt)

    all_single_lats = torch.cat(all_single_lats, dim=0)
    all_single_plats = torch.cat(all_single_plats, dim=0)
    all_single_lats_rt = torch.cat(all_single_lats_rt, dim=0)

    assert torch.allclose(batch_lats, all_single_lats)
    assert torch.allclose(batch_plats, all_single_plats)
    assert torch.allclose(batch_lats_rt, all_single_lats_rt)

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
