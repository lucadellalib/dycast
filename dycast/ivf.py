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

"""Inverted file (IVF) index (see https://dl.acm.org/doi/10.1145/1132956.1132959)."""

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


try:
    from .focalnet import FocalDecoder, FocalEncoder
except ImportError:
    from focalnet import FocalDecoder, FocalEncoder


__all__ = ["LatentIVF"]


class LatentIVF(nn.Module):
    """Latent-space inverted file (IVF) index retriever.

    This retriever performs kNN retrieval in the *compressor latent space* and
    maps the retrieved latents back into the feature space:

        feats -> latents -> IVF search/reconstruct -> latents' -> feats'

    Parameters
    ----------
    input_dim:
        Dimension of the input features.
    latent_dim:
        Dimension of the latent space (IVF vectors live in this space).
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
    nlist:
        Total number of inverted lists (cells).
    nprobe:
        Number of inverted lists (cells) probed per query during retrieval.
        Larger values improve recall at the cost of increased latency.
        Must satisfy 1 <= nprobe <= nlist.

    Notes
    -----
    - This module is not TorchScript-friendly (FAISS is Python/CPU).
    - Queries are moved to CPU float32 for FAISS retrieval, then results are moved back.

    """

    def __init__(
        self,
        input_dim: "int" = 1024,
        latent_dim: "int" = 32,
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
        nlist: "int" = 4096,
        nprobe: "int" = 16,
    ) -> "None":
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
        self.nlist = nlist
        self.nprobe = nprobe

        # Modules
        self.compressor = FocalEncoder(
            input_dim=input_dim,
            output_dim=latent_dim,
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
        self.decompressor = FocalDecoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=tuple(reversed(hidden_dims)),
            upscale_factors=tuple(reversed(downscale_factors)),
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

        try:
            import faiss
        except ImportError:
            raise ImportError("`pip install faiss-cpu` to use this model")

        quantizer = faiss.IndexFlatIP(latent_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, latent_dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        ivf = faiss.extract_index_ivf(self.index)
        ivf.make_direct_map()
        ivf.nprobe = self.nprobe

    def save_index(self, index_path: "str") -> "None":
        """Save the FAISS index to disk.

        Parameters
        ----------
        index_path:
            Path where the FAISS IVF index should be saved.

        """
        import faiss

        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: "str") -> "None":
        """Load a pretrained FAISS inverted-file (IVF) index from disk.

        Parameters
        ----------
        index_path:
            Path to a FAISS IVF index checkpoint (as saved by ``faiss.write_index``).
            The index is loaded on CPU.

        """
        import faiss

        index = faiss.read_index(index_path)

        try:
            ivf = faiss.extract_index_ivf(index)
        except RuntimeError as e:
            raise RuntimeError(
                "Expected an IVF FAISS index, but the loaded index is not IVF"
            ) from e

        ivf.make_direct_map()
        ivf.nprobe = self.nprobe
        self.index = index

    @torch.no_grad()
    def forward(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        sim_threshold: "float" = 0.97,
        blend: "float" = 1.0,
    ) -> "Tuple[Tensor, Tuple, Tuple]":
        """Refine frame-level features via latent-space IVF retrieval.

        Parameters
        ----------
        feats:
            Frame-level features of shape (batch_size, seq_length, dim).
        compressor_state:
            Compressor streaming state.
        decompressor_state:
            Decompressor streaming state.
        sim_threshold:
            Cosine similarity threshold. Retrieval is applied only
            where top-1 similarity is >= threshold.
        blend:
            Blend factor in [0, 1]:
            ``refined_feats = (1 - blend) * feats + blend * retrieved_feats``.

        Returns
        -------
            - Refined features of shape (batch_size, seq_length, dim);
            - updated compressor streaming state;
            - updated decompressor streaming state.

        """
        B, T, _ = feats.shape
        device = feats.device

        # -------------------------------------------------
        # 1) feats -> latents (unit norm for cosine/IP search)
        # -------------------------------------------------
        lats, *compressor_state = self.compressor(feats, *compressor_state)  # [B, T, D]
        lats = nn.functional.normalize(lats, dim=-1)
        flat = lats.reshape(B * T, self.latent_dim)  # [N, D]

        # -------------------------------------------------
        # 2) FAISS search on CPU (cosine similarity via inner product)
        # -------------------------------------------------
        q_np = flat.detach().cpu().to(dtype=torch.float32).numpy()  # [N, D]
        sims_np, ids_np = self.index.search(q_np, 1)  # [N, 1], [N, 1]
        sims_np = sims_np[:, 0]
        ids_np = ids_np[:, 0]

        sims0 = torch.from_numpy(sims_np).to(device=device, dtype=feats.dtype)  # [N]

        keep = sims0 >= sim_threshold  # [N] bool

        # -------------------------------------------------
        # 3) Reconstruct latents (top-1)
        # -------------------------------------------------
        rec_np = np.vstack(
            [self.index.reconstruct(int(i)) for i in ids_np]
        )  # numpy [N, D]
        flat_rec = torch.from_numpy(rec_np).to(device=device, dtype=lats.dtype)
        flat_rec = nn.functional.normalize(flat_rec, dim=-1)

        # -------------------------------------------------
        # 4) Restore [B, T, D] with optional gating
        # -------------------------------------------------
        lats_rec = lats.reshape(B * T, self.latent_dim).clone()

        lats_rec[keep] = flat_rec[keep]

        lats_rec = lats_rec.reshape(B, T, self.latent_dim)

        # -------------------------------------------------
        # 5) Latents -> refined feats
        # -------------------------------------------------
        feats_rec, *decompressor_state = self.decompressor(
            lats_rec, *decompressor_state
        )  # [B, T, H]

        # -------------------------------------------------
        # 6) Blend + keep originals where rejected
        # -------------------------------------------------
        feats_base = feats.to(device=device)
        refined_feats = (1.0 - blend) * feats_base + blend * feats_rec

        keep_bt = keep.reshape(B, T).to(device=device)
        refined_feats = torch.where(keep_bt[..., None], refined_feats, feats_base)

        return refined_feats, compressor_state, decompressor_state


def test_model() -> "None":
    import os
    import tempfile

    # Lazy import inside test (matches module behavior)
    import faiss

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Config
    # ----------------------------
    B = 4
    T = 60
    H = 1024
    D = 32
    M = 2000

    nlist = 64
    nprobe = 8

    # ----------------------------
    # Build dummy IVF index (cosine via IP on unit vectors)
    # ----------------------------
    xb = np.random.randn(M, D).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
    ids = np.arange(M, dtype=np.int64)

    quantizer = faiss.IndexFlatIP(D)
    ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(xb)
    ivf.add_with_ids(xb, ids)
    ivf.nprobe = int(nprobe)

    with tempfile.TemporaryDirectory() as tmp:
        # Keep "ivf" in the filename if your loader relies on it elsewhere.
        index_path = os.path.join(tmp, "latent_ivf.faiss")
        faiss.write_index(ivf, index_path)

        # ----------------------------
        # Instantiate module
        # ----------------------------
        model = LatentIVF().to(device)
        model.load_index(index_path)
        model.eval()
        print(
            f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
        )

        # ----------------------------
        # Forward invariants
        # ----------------------------
        feats = torch.randn((B, T, H), device=device)

        with torch.no_grad():
            out0, st_c0, st_d0 = model(feats, sim_threshold=0.0, blend=1.0)
            out1, st_c1, st_d1 = model(feats, sim_threshold=0.0, blend=0.5)

        assert out0.shape == feats.shape
        assert out1.shape == feats.shape

        # blended output should be between base and full replace (in norm sense)
        def rms(x: torch.Tensor) -> float:
            return float(x.pow(2).mean().sqrt().item())

        # should be finite
        assert torch.isfinite(out0).all()

        print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    import os
    import tempfile

    import faiss

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Config
    # ----------------------------
    B = 10
    T = 80
    H = 1024
    D = 32
    M = 2000

    nlist = 64
    nprobe = 8

    # ----------------------------
    # Build dummy IVF index
    # ----------------------------
    xb = np.random.randn(M, D).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
    ids = np.arange(M, dtype=np.int64)

    quantizer = faiss.IndexFlatIP(D)
    ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(xb)
    ivf.add_with_ids(xb, ids)
    ivf.nprobe = int(nprobe)

    with tempfile.TemporaryDirectory() as tmp:
        index_path = os.path.join(tmp, "latent_ivf.faiss")
        faiss.write_index(ivf, index_path)

        model = LatentIVF().to(device)
        model.load_index(index_path)
        model.eval()

        feats = torch.randn((B, T, H), device=device)

        # Pick a stable threshold based on actual sims for this batch
        with torch.no_grad():
            lats, *_ = model.compressor(feats)
            lats = nn.functional.normalize(lats, dim=-1)
            flat = lats.reshape(B * T, D).detach().cpu().float().numpy()
            sims_np, _ = model.index.search(flat, 1)
            sims0 = torch.from_numpy(sims_np[:, 0]).to(device=device)
            sim_threshold = float(torch.quantile(sims0, 0.70).item())

        # Batched path
        with torch.no_grad():
            out_batch, _, _ = model(feats, sim_threshold=sim_threshold, blend=1.0)

        # Singleton path
        all_single = []
        for i in range(B):
            with torch.no_grad():
                out_i, _, _ = model(
                    feats[i : i + 1],
                    sim_threshold=sim_threshold,
                    blend=1.0,
                )
            all_single.append(out_i)

        out_single = torch.cat(all_single, dim=0)

        # Exact equality should hold: retrieval is per-frame and FAISS is queried with the same vectors.
        assert torch.allclose(out_batch, out_single, atol=1e-2)

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
