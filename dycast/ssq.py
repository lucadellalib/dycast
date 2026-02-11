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

"""Scalar spherical quantization."""

import math
from typing import Tuple

import torch
from torch import Tensor, nn


__all__ = ["ScalarSphericalQuantizer"]


class ScalarSphericalQuantizer(nn.Module):
    """Scalar spherical quantizer that maps inputs to scalar codes on the unit hypersphere.

    Parameters
    ----------
    dim:
        Code dimensionality.
    n_levels:
        Number of scalar quantization levels per dimension.

    """

    def __init__(self, dim: "int" = 32, n_levels: "int" = 4) -> "None":
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels

        # Precompute constants
        inv_sqrtD = torch.as_tensor(1.0 / math.sqrt(dim))
        step = torch.as_tensor(2.0 / (n_levels - 1))
        scale = torch.as_tensor((n_levels - 1) / (2.0 * inv_sqrtD))
        bias = torch.as_tensor((n_levels - 1) / 2.0)
        self.register_buffer("inv_sqrtD", inv_sqrtD, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("step", step, persistent=False)

    @property
    def codebook(self) -> "Tensor":
        """Return the scalar spherical codebook.

        Returns
        -------
            Codebook of shape (dim, n_levels), where each column corresponds
            to one scalar quantization level replicated across dimensions.

        """
        # Scalar levels in [-1/sqrt(D), +1/sqrt(D)]
        levels = (
            torch.arange(self.n_levels, device=self.inv_sqrtD.device) * self.step - 1.0
        ) * self.inv_sqrtD

        # Expand to full dimensional codes
        return levels[None].expand(self.dim, self.n_levels)

    def forward(self, lats: "Tensor") -> "Tuple[Tensor, Tensor]":
        """Forward pass.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            - Output tokens of shape (..., dim);
            - output codes of shape (..., dim).

        """
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        return toks, codes

    @torch.jit.export
    def lats_to_codes(self, lats: "Tensor") -> "Tensor":
        """Transform latents into codes (i.e. quantized latents).

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        toks = self.lats_to_toks(lats)
        codes = self.toks_to_codes(toks)
        return codes

    @torch.jit.export
    def lats_to_toks(self, lats: "Tensor") -> "Tensor":
        """Transform latents into tokens.

        Parameters
        ----------
        lats:
            Input latents of shape (..., dim).

        Returns
        -------
            Output tokens of shape (..., dim).

        """
        x = nn.functional.normalize(lats, dim=-1)

        # Clamp to supported range
        x = x.clamp(-self.inv_sqrtD, self.inv_sqrtD)

        # Map [-1/sqrtD, +1/sqrtD] → [0, L-1] via affine transform
        toks = (x * self.scale + self.bias).round().to(torch.long)
        toks = toks.clamp(0, self.n_levels - 1)
        return toks

    @torch.jit.export
    def codes_to_toks(self, codes: Tensor) -> Tensor:
        """Transform codes (i.e. quantized latents) into tokens.

        Parameters
        ----------
        codes:
            Input codes of shape (..., dim).

        Returns
        -------
            Output tokens of shape (..., dim).

        """
        # Clamp to supported range (numerical safety)
        x = codes.clamp(-self.inv_sqrtD, self.inv_sqrtD)
        # Map [-1/sqrtD, +1/sqrtD] → [0, L-1] via affine transform
        toks = (x * self.scale + self.bias).round().to(torch.long)
        toks = toks.clamp(0, self.n_levels - 1)
        return toks

    @torch.jit.export
    def toks_to_codes(self, toks: "Tensor") -> "Tensor":
        """Transform tokens into codes (i.e. quantized latents).

        Parameters
        ----------
        toks:
            Input tokens of shape (..., dim).

        Returns
        -------
            Output codes of shape (..., dim).

        """
        # Inverse mapping of the uniform grid:
        # codes = (-1 + step * tok) / sqrt(D)
        codes = (-1.0 + self.step * toks) * self.inv_sqrtD
        codes = nn.functional.normalize(codes, dim=-1)
        return codes

    def __repr__(self) -> "str":
        return f"{self.__class__.__name__}(dim={self.dim}, n_levels={self.n_levels})"


def test_model() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    D = 32
    model = ScalarSphericalQuantizer(dim=D, n_levels=4).to(device)
    print(model)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.6f}M"
    )

    lats = torch.randn(B, T, D, device=device)
    toks, codes = model(lats)

    toks2 = model.lats_to_toks(lats)
    codes2 = model.lats_to_codes(lats)
    toks3 = model.codes_to_toks(codes)
    codes3 = model.toks_to_codes(toks)

    assert toks.dtype in (torch.int64, torch.long)
    assert toks.shape == (B, T, D)
    assert codes.shape == (B, T, D)

    assert (toks == toks2).all()
    assert (codes == codes2).all()
    assert (toks == toks3).all()
    assert torch.allclose(codes, codes3, atol=0.0, rtol=0.0)

    # JIT
    model_jit = torch.jit.script(model)
    toks_jit, codes_jit = model_jit(lats)
    assert (toks == toks_jit).all()
    assert (codes == codes_jit).all()

    # On-sphere
    norms = codes.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 50
    D = 32
    model = ScalarSphericalQuantizer(dim=D, n_levels=4).to(device)

    lats = torch.randn(B, T, D, device=device)
    batch_toks, batch_codes = model(lats)

    all_single_toks, all_single_codes = [], []
    for i in range(B):
        single_toks, single_codes = model(lats[i][None])
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)

    all_single_toks = torch.cat(all_single_toks, dim=0)
    all_single_codes = torch.cat(all_single_codes, dim=0)

    assert (batch_toks == all_single_toks).all()
    assert (batch_codes == all_single_codes).all()

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
