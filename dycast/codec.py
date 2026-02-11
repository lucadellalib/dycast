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

"""DyCAST (see https://arxiv.org/abs/2601.23174)."""

import inspect
import json
import os
import re
import sys
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn


try:
    from .focalnet import FocalDecoder, FocalEncoder
    from .hazard import HazardModel
    from .ivf import LatentIVF
    from .mms import MMS
    from .negbin import NegBinModel
    from .pooling import RepeatInterleaveUnpool, SelectLastPool
    from .ssq import ScalarSphericalQuantizer
    from .version import VERSION
    from .vocos import Vocos
    from .wavlm import WavLM
except ImportError:
    from focalnet import FocalDecoder, FocalEncoder
    from hazard import HazardModel
    from ivf import LatentIVF
    from mms import MMS
    from negbin import NegBinModel
    from pooling import RepeatInterleaveUnpool, SelectLastPool
    from ssq import ScalarSphericalQuantizer
    from version import VERSION
    from vocos import Vocos
    from wavlm import WavLM


__all__ = ["DyCAST"]


REGISTRY = {
    "FocalDecoder": FocalDecoder,
    "FocalEncoder": FocalEncoder,
    "HazardModel": HazardModel,
    "LatentIVF": LatentIVF,
    "MMS": MMS,
    "NegBinModel": NegBinModel,
    "RepeatInterleaveUnpool": RepeatInterleaveUnpool,
    "ScalarSphericalQuantizer": ScalarSphericalQuantizer,
    "SelectLastPool": SelectLastPool,
    "Vocos": Vocos,
    "WavLM": WavLM,
}

DEFAULT_CONFIGS = [
    "lucadellalib/dycast",
]

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DyCAST(nn.Module):
    """DyCAST is a modular, dynamic speech codec that supports variable-frame-rate
    tokenization through explicit boundary and duration modeling. The system is
    composed of interchangeable components for encoding, compression, quantization,
    decoding, and optional retrieval-augmented reconstruction.

    Parameters
    ----------
    encoder:
        Neural audio encoder that maps waveforms to continuous frame-level
        representations.
    compressor:
        Feature compression module that reduces the dimensionality of encoder
        outputs prior to quantization.
    boundary_predictor:
        Module that predicts token boundaries or boundary probabilities over time,
        enabling variable-rate tokenization.
    downsampler:
        Temporal downsampling module that aggregates frame-level representations
        into token-level features according to predicted boundaries.
    quantizer:
        Discrete quantization module that maps continuous token-level features
        to discrete codes.
    duration_predictor:
        Module that predicts token durations or length distributions, enabling
        explicit control over temporal alignment at decoding time.
    upsampler:
        Temporal upsampling module that expands token-level representations back
        to frame-level features using predicted or sampled durations.
    decompressor:
        Feature decompression module that reconstructs high-dimensional features
        from quantized tokens.
    decoder:
        Neural decoder that maps reconstructed acoustic representations back to
        the waveform domain.
    char_aligner:
        Optional character-level aligner used during training to provide
        supervision for boundary and duration prediction.
    retriever:
        Optional latent-space retriever used for retrieval-augmented decoding,
        enabling refinement of reconstructed representations using a fixed
        candidate set.

    """

    __version__ = VERSION

    def __init__(
        self,
        encoder: "nn.Module",
        compressor: "nn.Module",
        boundary_predictor: "nn.Module",
        downsampler: "nn.Module",
        quantizer: "nn.Module",
        duration_predictor: "nn.Module",
        upsampler: "nn.Module",
        decompressor: "nn.Module",
        decoder: "nn.Module",
        char_aligner: "nn.Module" = None,
        retriever: "nn.Module" = None,
    ) -> "None":
        super().__init__()
        self.encoder = encoder
        self.compressor = compressor
        self.boundary_predictor = boundary_predictor
        self.downsampler = downsampler
        self.quantizer = quantizer
        self.duration_predictor = duration_predictor
        self.upsampler = upsampler
        self.decompressor = decompressor
        self.decoder = decoder
        self.char_aligner = char_aligner
        self.retriever = retriever

        # Hugging Face model ID
        self.model_id = None

    @property
    def sample_rate_input(self) -> "int":
        """Return the input sample rate."""
        return self.encoder.sample_rate

    @property
    def sample_rate_output(self) -> "int":
        """Return the output sample rate."""
        return int(
            self.sample_rate_input
            * self.decoder.upsample_factor
            / self.encoder.downsample_factor
        )

    @property
    def sample_rate(self) -> "int":
        """Return the sample rate."""
        if self.sample_rate_input != self.sample_rate_output:
            raise RuntimeError(
                "`sample_rate` is undefined because input and output sample rates "
                f"differ (input={self.sample_rate_input}, output={self.sample_rate_output}). "
                "Please use `sample_rate_input` or `sample_rate_output` explicitly"
            )
        return self.sample_rate_input

    @property
    def causal(self) -> "bool":
        """Whether the model is causal."""
        parts = (
            self.encoder.causal,
            self.compressor.causal,
            self.boundary_predictor.causal,
            self.duration_predictor.causal,
            self.decompressor.causal,
            self.decoder.causal,
            True if self.char_aligner is None else self.char_aligner.causal,
            True if self.retriever is None else self.retriever.causal,
        )
        return all(bool(x) for x in parts)

    @property
    def chunk_size(self) -> "int":
        """Return the chunk size."""
        return self.encoder.chunk_size

    @property
    def latency(self) -> "Optional[float]":
        """Return the theoretical latency in milliseconds."""
        if self.causal:
            return 1000.0 * self.chunk_size / self.sample_rate_input
        return None

    @property
    def codebook(self) -> "Tensor":
        """Return the quantizer codebook."""
        return self.quantizer.codebook

    def forward(
        self,
        sig: "Tensor",
        # Streaming states
        encoder_state: "Tuple" = (),
        compressor_state: "Tuple" = (),
        boundary_predictor_state: "Tuple" = (),
        duration_predictor_state: "Tuple" = (),
        decompressor_state: "Tuple" = (),
        decoder_state: "Tuple" = (),
        aligner_state: "Tuple" = (),
        retriever_state: "Tuple" = (),
        # Length
        length: "Optional[Tensor]" = None,
        # Inference mode
        boundary_source: "Literal['char_aligner', 'boundary_decode', 'boundary_sample']" = "boundary_decode",
        duration_source: "Literal['original', 'duration_decode', 'duration_sample']" = "duration_decode",
        budget_decode: "bool" = False,
        # Retrieval
        use_retriever: "bool" = False,
        sim_threshold: "float" = 0.97,
        blend: "float" = 1.0,
        # Decoding
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
        # Return options
        return_state: "bool" = False,
        # Component kwargs
        aligner_kwargs: "Dict[str, Any]" = None,
        boundary_predictor_kwargs: "Dict[str, Any]" = None,
        duration_predictor_kwargs: "Dict[str, Any]" = None,
    ) -> "Union[Tuple[Tensor, Tensor, Tensor], Tuple]":
        """Forward pass.

        This method performs tokenization and reconstruction in a single call,
        combining boundary prediction, duration modeling, optional budget-constrained
        decoding, and optional retrieval-based refinement.

        The processing pipeline is:

            signal -> features -> boundaries -> latents -> pooled latents -> (tokens, pooled codes)
                   -> (duration decoding) -> frame-level codes -> quantized features
                   -> (retrieval / kNN refinement) -> signal

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        compressor_state:
            Compressor streaming state.
        boundary_predictor_state:
            Boundary predictor streaming state.
        duration_predictor_state:
            Duration predictor streaming state.
        decompressor_state:
            Decompressor streaming state.
        decoder_state:
            Decoder streaming state.
        aligner_state:
            Aligner streaming state.
        retriever_state:
            Retriever streaming state.
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        boundary_source:
            Strategy used to obtain encoder-side boundaries (used for pooling):
            - "char_aligner": use the character aligner on the input signal;
            - "boundary_decode": greedy decoding from the boundary predictor;
            - "boundary_sample": stochastic sampling from the boundary predictor.
        duration_source:
            Strategy used to obtain decoder-side durations (used for unpooling):
            - "original": reuse encoder-side durations;
            - "duration_decode": greedy decoding from the duration predictor;
            - "duration_sample": stochastic sampling from the duration predictor.
        budget_decode:
            If True, enable budget-constrained duration decoding. In this mode, the
            total number of decoded frames is constrained to match the encoder
            feature length. If ``num_frames`` is not explicitly provided via
            ``duration_predictor_kwargs``, it is inferred automatically from the
            encoder output. If False, no budget is imposed.
        use_retriever:
            If True, apply retrieval-augmented decoding in the quantized feature
            space using the latent-space retriever.
        sim_threshold:
            Cosine similarity threshold. Retrieval is applied only
            where top-1 similarity is >= threshold.
        blend:
            Blend factor in [0, 1].
        matching_set:
            Optional set of candidate features for kNN refinement,
            shape (num_candidates, hidden_dim).
        topk:
            Number of nearest neighbors to consider in the kNN refinement.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency during kNN computation.
        output_length:
            Desired output length of the synthesized signal. If specified, the output
            will be truncated or padded to this length.
        return_state:
            True to return the streaming state(s), False otherwise.
        aligner_kwargs:
            Character aligner keyword arguments.
        boundary_predictor_kwargs:
            Boundary predictor keyword arguments.
        duration_predictor_kwargs:
            Duration predictor keyword arguments.

        Returns
        -------
            - Discrete tokens of shape (batch_size, num_chunks, num_codebooks);
            - pooled codes of shape (batch_size, num_chunks, latent_dim);
            - reconstructed signal, shape (batch_size, output_length) if
              `output_length` is specified, otherwise (batch_size, ~seq_length);
            - [updated encoder streaming state];
            - [updated compressor streaming state];
            - [updated boundary predictor streaming state];
            - [updated duration predictor streaming state];
            - [updated decompressor streaming state];
            - [updated decoder streaming state];
            - [updated aligner streaming state];
            - [updated retriever streaming state].

        """
        # ---- normalize kwargs dicts ----
        if aligner_kwargs is None:
            aligner_kwargs = {}
        if boundary_predictor_kwargs is None:
            boundary_predictor_kwargs = {}
        if duration_predictor_kwargs is None:
            duration_predictor_kwargs = {}

        # ---- 1) signal -> features ----
        feats, encoder_state = self.sig_to_feats(
            sig,
            encoder_state=encoder_state,
            length=length,
            return_state=True,
        )

        if budget_decode and "num_frames" not in duration_predictor_kwargs:
            device = feats.device
            batch_size = feats.shape[0]
            num_frames = feats.shape[1]
            if length is not None:
                # length is relative in [0, 1], so per-sample valid frames:
                duration_predictor_kwargs["num_frames"] = (
                    (length * num_frames).round().to(torch.long)
                )
            else:
                duration_predictor_kwargs["num_frames"] = torch.full(
                    (batch_size,), num_frames, device=device
                )

        # ---- 2) get encoder-side durations (for pooling) ----
        if boundary_source == "char_aligner":
            durs_enc, aligner_state = self.sig_to_durs(
                sig,
                aligner_state=aligner_state,
                length=length,
                return_state=True,
                **aligner_kwargs,
            )
        else:
            durs_enc, boundary_predictor_state = self.feats_to_durs(
                feats,
                boundary_predictor_state=boundary_predictor_state,
                length=length,
                sample=(boundary_source == "boundary_sample"),
                return_state=True,
                **boundary_predictor_kwargs,
            )

        # ---- 3) features -> latents ----
        lats, compressor_state = self.feats_to_lats(
            feats,
            compressor_state=compressor_state,
            return_state=True,
        )

        # ---- 4) latents -> pooled latents ----
        plats, plats_length = self.lats_to_plats(lats, durs_enc)

        # ---- 5) pooled latents -> toks + pooled codes ----
        toks = self.plats_to_toks(plats)
        pcodes = self.plats_to_pcodes(plats)

        # ---- 6) decoder-side durations (for unpooling) ----
        if duration_source == "original":
            durs_dec = durs_enc
        else:
            durs_dec, duration_predictor_state = self.pcodes_to_durs(
                pcodes,
                duration_predictor_state=duration_predictor_state,
                length=plats_length,
                sample=(duration_source == "duration_sample"),
                return_state=True,
                **duration_predictor_kwargs,
            )

        # ---- 7) pooled codes -> frame-level codes ----
        codes, _codes_length = self.pcodes_to_codes(pcodes, durs_dec)

        # ---- 8) codes -> quantized features ----
        qfeats, decompressor_state = self.codes_to_qfeats(
            codes,
            decompressor_state=decompressor_state,
            return_state=True,
        )

        # ---- 9) retrieval refinement (IVF) ----
        feats_dec = qfeats
        if use_retriever:
            feats_dec, retriever_state = self.qfeats_to_feats(
                feats_dec,
                retriever_state=retriever_state,
                sim_threshold=sim_threshold,
                blend=blend,
                return_state=True,
            )

        # ---- 10) features -> reconstructed signal (kNN happens inside feats_to_sig) ----
        rec_sig, decoder_state = self.feats_to_sig(
            feats_dec,
            decoder_state=decoder_state,
            matching_set=matching_set,
            topk=topk,
            num_splits=num_splits,
            output_length=output_length,
            return_state=True,
        )

        if return_state:
            return (
                toks,
                pcodes,
                rec_sig,
                encoder_state,
                compressor_state,
                boundary_predictor_state,
                duration_predictor_state,
                decompressor_state,
                decoder_state,
                aligner_state,
                retriever_state,
            )

        return toks, pcodes, rec_sig

    def sig_to_durs(
        self,
        sig: "Tensor",
        aligner_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_transcript: "bool" = False,
        return_state: "bool" = False,
        **aligner_kwargs: "Any",
    ) -> "Union[Tensor, Tuple]":
        """Transform signal into durations using the character aligner.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        aligner_state:
            Aligner streaming state.
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_transcript:
            True to return the transcript, False otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.
        aligner_kwargs:
            Character aligner keyword arguments.

        Returns
        -------
            - Output durations of shape (batch_size, num_chars);
            - [transcript];
            - [updated aligner streaming state].

        """
        if self.char_aligner is None:
            raise RuntimeError("char_aligner is None")

        durations, transcripts = self.char_aligner(sig, length=length, **aligner_kwargs)

        if return_transcript and return_state:
            return durations, transcripts, aligner_state
        if return_transcript:
            return durations, transcripts
        if return_state:
            return durations, aligner_state

        return durations

    def sig_to_feats(
        self,
        sig: "Tensor",
        encoder_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple]":
        """Transform signal into features.

        Parameters
        ----------
        sig:
            Input signal of shape (batch_size, seq_length).
        encoder_state:
            Encoder streaming state.
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated encoder streaming state].

        """
        feats, *encoder_state = self.encoder(sig, *encoder_state, length=length)
        if return_state:
            return feats, encoder_state
        return feats

    def feats_to_durs(
        self,
        feats: "Tensor",
        boundary_predictor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        sample: "bool" = False,
        return_state: "bool" = False,
        **boundary_predictor_kwargs: "Any",
    ) -> "Union[Tensor, Tuple]":
        """Transform features into durations using the boundary predictor.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        boundary_predictor_state:
            Boundary predictor streaming state.
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        sample:
            True to sample the boundaries, False to greedy decode.
        return_state:
            True to return the streaming state(s), False otherwise.
        boundary_predictor_kwargs:
            Boundary predictor keyword arguments.

        Returns
        -------
            - Output durations of shape (batch_size, num_chunks);
            - [updated boundary predictor streaming state].

        """
        logits, *boundary_predictor_state = self.boundary_predictor(
            feats, *boundary_predictor_state
        )
        if sample:
            durations = self.boundary_predictor.sample(
                logits, length, **boundary_predictor_kwargs
            )
        else:
            durations = self.boundary_predictor.decode(
                logits, length, **boundary_predictor_kwargs
            )
        if return_state:
            return durations, boundary_predictor_state
        return durations

    def feats_to_lats(
        self,
        feats: "Tensor",
        compressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple]":
        """Transform features into latents.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        compressor_state:
            Compressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output latents of shape (batch_size, hidden_seq_length, latent_dim);
            - [updated compressor streaming state].

        """
        lats, *compressor_state = self.compressor(feats, *compressor_state)
        lats = nn.functional.normalize(lats, dim=-1)
        if return_state:
            return lats, compressor_state
        return lats

    def feats_to_sig(
        self,
        feats: "Tensor",
        decoder_state: "Tuple" = (),
        matching_set: "Optional[Tensor]" = None,
        topk: "int" = 4,
        num_splits: "int" = 1,
        output_length: "Optional[int]" = None,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple]":
        """Transform features into signal.

        Optionally applies k-nearest neighbors (kNN) search on a provided matching set to
        refine the input features (see https://arxiv.org/abs/2305.18975). The refined or
        original features are then passed through the decoder to synthesize the signal.
        If an `output_length` is specified, the signal is truncated or padded to match
        the desired length.

        Parameters
        ----------
        feats:
            Input features of shape (batch_size, hidden_seq_length, hidden_dim).
        decoder_state:
            Decoder streaming state.
        matching_set:
            Optional set of candidate features for kNN refinement,
            shape (num_candidates, hidden_dim).
        topk:
            Number of nearest neighbors to consider in the kNN refinement.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency during kNN computation.
        output_length:
            Desired output length of the synthesized signal. If specified, the output
            will be truncated or padded to this length.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output signal, shape (batch_size, output_length) if
              `output_length` is specified, otherwise (batch_size, ~seq_length);
            - [updated decoder streaming state].

        """
        if matching_set is not None:
            feats = self.knn(
                feats,
                matching_set,
                topk,
                num_splits,
            ).mean(dim=-2)
        sig, *decoder_state = self.decoder(feats, *decoder_state)
        if output_length is not None:
            delta = output_length - sig.shape[1]
            if delta < 0:
                sig = sig[:, :output_length]
            elif delta > 0:
                sig = nn.functional.pad(sig, [0, delta], mode="replicate")
        if return_state:
            return sig, decoder_state
        return sig

    def lats_to_plats(
        self, lats: "Tensor", durs: "Tensor"
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Transform latents into pooled latents.

        Parameters
        ----------
        lats:
            Input latents of shape (batch_size, hidden_seq_length, latent_dim).
        durs:
            Input durations of shape (batch_size, num_chunks).

        Returns
        -------
            - Output pooled latents of shape (batch_size, num_chunks, latent_dim);
            - relative length of each sequence in the batch.

        """
        plats = self.downsampler(lats, durs)
        length = (durs > 0).sum(dim=-1) / plats.shape[1]
        if (length == 1.0).all():
            length = None
        return plats, length

    def plats_to_toks(self, plats: "Tensor") -> "Tensor":
        """Transform pooled latents into tokens.

        Parameters
        ----------
        plats:
            Input pooled latents of shape (batch_size, num_chunks, latent_dim).

        Returns
        -------
            Output tokens of shape (batch_size, num_chunks, num_codebooks).

        """
        toks = self.quantizer.lats_to_toks(plats)
        return toks

    def plats_to_pcodes(self, plats: "Tensor") -> "Tensor":
        """Transform pooled latents into pooled codes.

        Parameters
        ----------
        plats:
            Input pooled latents of shape (batch_size, num_chunks, latent_dim).

        Returns
        -------
            Output pooled codes of shape (batch_size, num_chunks, latent_dim).

        """
        pcodes = self.quantizer.lats_to_codes(plats)
        return pcodes

    def toks_to_pcodes(self, toks: "Tensor") -> "Tensor":
        """Transform tokens into pooled codes.

        Parameters
        ----------
        toks:
            Input tokens of shape (batch_size, num_chunks, num_codebooks).

        Returns
        -------
            Output pooled codes of shape (batch_size, num_chunks, latent_dim).

        """
        pcodes = self.quantizer.toks_to_codes(toks)
        return pcodes

    def pcodes_to_durs(
        self,
        pcodes: "Tensor",
        duration_predictor_state: "Tuple" = (),
        length: "Optional[Tensor]" = None,
        sample: "bool" = False,
        return_state: "bool" = False,
        **duration_predictor_kwargs: "Any",
    ) -> "Union[Tensor, Tuple]":
        """Transform pooled codes into durations using the duration predictor.

        Parameters
        ----------
        pcodes:
            Input pooled codes of shape (batch_size, num_chunks, latent_dim).
        duration_predictor_state:
            Duration predictor streaming state.
        length:
            Relative length of each sequence in the batch.
            Used only if the model is non-causal; ignored otherwise.
        sample:
            True to sample the durations, False to greedy decode.
        return_state:
            True to return the streaming state(s), False otherwise.
        duration_predictor_kwargs:
            Duration predictor keyword arguments.

        Returns
        -------
            - Output durations of shape (batch_size, num_chunks);
            - [updated duration predictor streaming state].

        """
        mu_free, *duration_predictor_state = self.duration_predictor(
            pcodes, *duration_predictor_state
        )
        if sample:
            durations = self.duration_predictor.sample(
                mu_free, length, **duration_predictor_kwargs
            )
        else:
            durations = self.duration_predictor.decode(
                mu_free, length, **duration_predictor_kwargs
            )
        if return_state:
            return durations, duration_predictor_state
        return durations

    def pcodes_to_toks(self, pcodes: "Tensor") -> "Tensor":
        """Transform pooled codes into tokens.

        Parameters
        ----------
        pcodes:
            Input pooled codes of shape (batch_size, num_chunks, latent_dim).

        Returns
        -------
            Output tokens of shape (batch_size, num_chunks, num_codebooks).

        """
        toks = self.quantizer.codes_to_toks(pcodes)
        return toks

    def pcodes_to_codes(
        self, pcodes: "Tensor", durs: "Tensor"
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Transform pooled codes into codes.

        Parameters
        ----------
        pcodes:
            Input pooled codes of shape (batch_size, num_chunks, latent_dim).
        durs:
            Input durations of shape (batch_size, num_chunks).

        Returns
        -------
            - Output codes of shape (batch_size, hidden_seq_length, latent_dim);
            - relative length of each sequence in the batch.

        """
        codes = self.upsampler(pcodes, durs)
        length = durs.sum(dim=-1) / codes.shape[1]
        if (length == 1.0).all():
            length = None
        return codes, length

    def codes_to_qfeats(
        self,
        codes: "Tensor",
        decompressor_state: "Tuple" = (),
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple]":
        """Transform codes into quantized features.

        Parameters
        ----------
        codes:
            Input codes of shape (batch_size, hidden_seq_length, latent_dim).
        decompressor_state:
            Decompressor streaming state.
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Output quantized features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated decompressor streaming state].

        """
        qfeats, *decompressor_state = self.decompressor(codes, *decompressor_state)
        if return_state:
            return qfeats, decompressor_state
        return qfeats

    def qfeats_to_feats(
        self,
        qfeats: "Tensor",
        retriever_state: "Tuple" = (),
        sim_threshold: "float" = 0.97,
        blend: "float" = 1.0,
        return_state: "bool" = False,
    ) -> "Union[Tensor, Tuple]":
        """Refine quantized features into continuous features via retrieval.

        Parameters
        ----------
        qfeats:
            Quantized features of shape (batch_size, hidden_seq_length, hidden_dim).
        retriever_state:
            Retriever streaming state.
        sim_threshold:
            Cosine similarity threshold. Retrieval is applied only
            where top-1 similarity is >= threshold.
        blend:
            Blend factor in [0, 1].
        return_state:
            True to return the streaming state(s), False otherwise.

        Returns
        -------
            - Refined features of shape (batch_size, hidden_seq_length, hidden_dim);
            - [updated retriever streaming state].

        """
        if self.retriever is None:
            raise RuntimeError("retriever is None")

        feats, *retriever_state = self.retriever(
            qfeats,
            *retriever_state,
            sim_threshold=sim_threshold,
            blend=blend,
        )

        if return_state:
            return feats, retriever_state
        return feats

    def knn(
        self,
        input: "Tensor",
        matching_set: "Tensor",
        topk: "int" = 4,
        num_splits: "int" = 1,
    ) -> "Tensor":
        """Perform k-nearest neighbors (kNN) search using cosine distance.

        This method retrieves the `topk` nearest neighbors for each query
        in the `input` tensor from the `matching_set` tensor. Optionally,
        the `matching_set` can be split into smaller subsets to reduce
        memory usage during large-scale computations.

        Parameters
        ----------
        input:
            Query tensor for which nearest neighbors are to be found,
            shape (..., hidden_dim), where `...` represents any
            additional leading dimensions.
        matching_set:
            Set of points to search for neighbors, shape (num_points, hidden_dim).
        topk:
            Number of nearest neighbors to retrieve.
        num_splits:
            Number of subsets to divide the `matching_set` into for memory
            efficiency.

        Returns
        -------
            Tensor containing the nearest neighbors for each query point,
            shape: (..., topk, hidden_dim).

        """
        chunk_size = matching_set.shape[0] // num_splits
        if num_splits > 1:
            matching_subsets = matching_set.split(chunk_size)
        else:
            matching_subsets = [matching_set]
        topk_smallest_dists = []
        topk_smallest_idxes = []
        for i, matching_subset in enumerate(matching_subsets):
            dists = _cosine_distance(input.flatten(end_dim=-2), matching_subset)
            topk_smallest_dists_i, topk_smallest_idxes_i = dists.topk(
                k=min(topk, matching_subset.shape[0]), largest=False, dim=-1
            )
            topk_smallest_dists.append(topk_smallest_dists_i)
            topk_smallest_idxes.append(i * chunk_size + topk_smallest_idxes_i)
        if num_splits > 1:
            dists = torch.cat(topk_smallest_dists, dim=-1)
            idxes = torch.cat(topk_smallest_idxes, dim=-1)
            _, dist_idxes = dists.topk(
                k=min(topk, dists.shape[-1]), largest=False, dim=-1
            )
            output = matching_set[idxes.gather(1, dist_idxes)]
        else:
            output = matching_set[topk_smallest_idxes[0]]
        output = output.reshape(input.shape[:-1] + (-1, input.shape[-1]))
        return output

    def info(self) -> "Dict[str, Any]":
        """Return the model information."""
        return {
            "model_id": self.model_id,
            "version": self.__version__,
            "sample_rate_input": self.sample_rate_input,
            "sample_rate_output": self.sample_rate_output,
            "causal": self.causal,
            "chunk_size": self.chunk_size,
            "latency": self.latency,
            "num_total_params": sum([x.numel() for x in self.state_dict().values()]),
        }

    def to_config(
        self,
        config: "str",
        pretrained: "bool" = False,
        skip_char_aligner_state_dict: "bool" = True,
    ) -> "None":
        """Dump model configuration to a JSON file.

        Parameters
        ----------
        config:
            Path to local JSON file where the configuration should be dumped.
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to dump the checkpoint along with the configuration.
        skip_char_aligner_state_dict:
            Whether to ignore missing parameters associated with the character aligner when saving
            a pretrained checkpoint. This is useful when the character aligner relies on external
            pretrained models (e.g. loaded from Hugging Face) whose parameters should not be
            re-saved as part of the DyCAST checkpoint.

        """
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        dirpath = os.path.dirname(config_json)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        modules = {
            "encoder": self.encoder,
            "compressor": self.compressor,
            "boundary_predictor": self.boundary_predictor,
            "downsampler": self.downsampler,
            "quantizer": self.quantizer,
            "duration_predictor": self.duration_predictor,
            "upsampler": self.upsampler,
            "decompressor": self.decompressor,
            "decoder": self.decoder,
            "char_aligner": self.char_aligner,
            "retriever": self.retriever,
        }

        config = {}
        for module_name, module in modules.items():
            if module is None:
                continue
            cls_name = module.__class__.__name__
            if cls_name not in REGISTRY:
                raise ValueError(
                    f"Unregistered module: {cls_name}. Available modules: {list(REGISTRY.keys())}"
                )
            config[f"{module_name}_name"] = cls_name
            signature = inspect.signature(module.__init__)
            config[f"{module_name}_config"] = {}
            for param in signature.parameters:
                if param == "self":
                    continue
                # Skip *args/**kwargs (var-positional / var-keyword)
                if signature.parameters[param].kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if param == "dropout":
                    if hasattr(module, "dropout_"):
                        config[f"{module_name}_config"][param] = module.dropout_
                    elif hasattr(module, "dropout"):
                        config[f"{module_name}_config"][param] = module.dropout
                else:
                    print(f"{module_name}_config", param, getattr(module, param))
                    config[f"{module_name}_config"][param] = getattr(module, param)

        with open(config_json, "w") as f:
            json.dump(config, f, indent=2)

        if pretrained:
            state_dict = self.state_dict()
            if skip_char_aligner_state_dict:
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("char_aligner.")
                }
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            try:
                from safetensors.torch import save_file as safetensors_save

                checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                safetensors_save(state_dict, checkpoint)
            except Exception:
                # If `safetensors` not available, use `torch`
                checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                torch.save(state_dict, checkpoint)

            # Save index
            if self.retriever is not None:
                index_path = f"{os.path.splitext(config_json)[0]}.faiss"
                self.retriever.save_index(index_path)

    def to_pretrained(self, config: "str") -> "None":
        """See documentation of `to_config`."""
        return self.to_config(config, pretrained=True)

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
        skip_char_aligner_state_dict: "bool" = True,
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "DyCAST":
        """Load model from a configuration.

        Parameters
        ----------
        config:
            Configuration source, one of the following:
              - A local JSON file (e.g. "config.json");
              - a Hugging Face repository containing "config.json" (e.g. "username/repo_name");
              - a specific JSON file hosted in a Hugging Face repository (e.g. "username/repo_name/config_xyz.json").
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to load the corresponding pretrained checkpoint.
              - If True and a JSON file is specified, the method will look for a checkpoint file with the same
                path or URL as the configuration file but with a `.safetensors` or `.pt` extension.
              - If True and a Hugging Face repository is provided, it is assumed that either "model.safetensors"
                or "model.pt" is available.
        skip_char_aligner_state_dict:
            Whether to ignore missing parameters associated with the character aligner when loading a pretrained
            checkpoint. This is useful when the character aligner relies on external pretrained models (e.g. loaded
            from Hugging Face) whose parameters are not serialized as part of the DyCAST checkpoint.
        overrides:
            Dictionary mapping dot-separated key paths to new values that override entries in the nested configuration.
            For example, {"encoder_config.max_cached_steps": 0}.
        kwargs:
            Additional keyword arguments to pass to `huggingface_hub.hf_hub_download` if
            fetching the configuration from a remote repository.

        Returns
        -------
            A model instance initialized with the given configuration and,
            if specified, pretrained checkpoint.

        Notes
        -----
        When loading from the Hugging Face Hub, the `huggingface-hub` library must be installed.
        You can install it via `pip install huggingface-hub`.

        """

        def _override_config(
            config: "Dict[str, Any]",
            path: "str",
            value: "Any",
        ) -> "None":
            keys = path.split(".")
            tmp = config
            for key in keys[:-1]:
                tmp = tmp.setdefault(key, {})
            tmp[keys[-1]] = value

        def _build_modules(config: "Dict[str, Any]") -> "Dict[str, Any]":
            modules = {}
            for module_name in [
                "encoder",
                "compressor",
                "boundary_predictor",
                "downsampler",
                "quantizer",
                "duration_predictor",
                "upsampler",
                "decompressor",
                "decoder",
                "char_aligner",
                "retriever",
            ]:
                name_key = f"{module_name}_name"
                config_key = f"{module_name}_config"

                # Optional modules may be missing
                if name_key not in config:
                    modules[module_name] = None
                    continue

                cls_name = config.get(name_key, None)
                if cls_name is None:
                    modules[module_name] = None
                    continue

                if cls_name not in REGISTRY:
                    raise ValueError(
                        f"Unregistered module: {cls_name}. Available modules: {list(REGISTRY.keys())}"
                    )

                module_cls = REGISTRY[cls_name]
                module_config = config.get(config_key, {}) or {}
                modules[module_name] = module_cls(**module_config)

            return modules

        def _load_state_dict_checked(
            model: "nn.Module",
            state_dict: "Dict[str, Tensor]",
        ) -> "None":
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if skip_char_aligner_state_dict:
                bad_missing = [k for k in missing if not k.startswith("char_aligner.")]
            else:
                bad_missing = list(missing)

            if bad_missing:
                raise RuntimeError(
                    "State dict mismatch (missing keys).\n"
                    f"Missing keys: {bad_missing}"
                )

            if unexpected:
                warnings.warn(
                    "State dict contains unexpected keys (ignored):\n"
                    f"{list(unexpected)}",
                    category=UserWarning,
                    stacklevel=2,
                )

        model_id = config
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        # Local
        if os.path.exists(config_json):
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**_build_modules(config))
            if pretrained:
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                    state_dict = torch.load(checkpoint, map_location="cpu")
                _load_state_dict_checked(model, state_dict)
                if model.retriever is not None:
                    index_path = f"{os.path.splitext(config_json)[0]}.faiss"
                    model.retriever.load_index(index_path)

            model.model_id = model_id
            return model

        # Remote
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("`pip install huggingface-hub` to load this model")

        is_repo = bool(re.fullmatch(r"[\w\-]+/[\w\-.]+", config))

        try:
            repo_id = config if is_repo else os.path.dirname(config_json)
            filename = "config.json" if is_repo else os.path.basename(config_json)
            config_json = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**_build_modules(config))
            if pretrained:
                orig_filename = filename
                filename = (
                    "model" if is_repo else f"{os.path.splitext(orig_filename)[0]}"
                )
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.safetensors", **kwargs
                    )
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.pt", **kwargs
                    )
                    state_dict = torch.load(checkpoint, map_location="cpu")
                _load_state_dict_checked(model, state_dict)
                if model.retriever is not None:
                    filename = (
                        "index" if is_repo else f"{os.path.splitext(orig_filename)[0]}"
                    )
                    index_path = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.faiss", **kwargs
                    )
                    model.retriever.load_index(index_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the specified configuration. "
                f"Available default configurations: {DEFAULT_CONFIGS}"
            ) from e
        model.model_id = model_id
        return model

    @classmethod
    def from_pretrained(
        cls,
        config: "str",
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "DyCAST":
        """See documentation of `from_config`."""
        return cls.from_config(config, pretrained=True, overrides=overrides, **kwargs)


# Adapted from:
# https://github.com/bshall/knn-vc/blob/848302a262f7299c738af49d74209790ed442a9f/matcher.py#L21
@torch.jit.script
def _cosine_distance(query: "Tensor", target: "Tensor") -> "Tensor":
    source_norm2 = (query**2).sum(dim=-1)
    target_norm2 = (target**2).sum(dim=-1)
    dotprod = (
        source_norm2[:, None]
        + target_norm2[None]
        - torch.cdist(query[None], target[None])[0] ** 2
    )
    dotprod /= 2
    dists = 1 - dotprod * (source_norm2[:, None] * target_norm2[None]).rsqrt()
    return dists


@torch.no_grad()
def test_model(config: "str") -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = DyCAST.from_pretrained(config)
    model = model.eval().to(device)

    print(model.info())
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    sig = torch.randn(B, model.sample_rate_input, device=device)
    model(sig)

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance(config: "str") -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    model = DyCAST.from_pretrained(config)
    model = model.eval().to(device)

    sig = torch.randn(B, model.sample_rate_input, device=device)
    batch_toks, batch_codes, batch_rec_sig = model(sig, duration_source="original")

    all_single_toks, all_single_codes, all_single_rec_sig = [], [], []
    for i in range(B):
        single_toks, single_codes, single_rec_sig = model(
            sig[i][None], duration_source="original"
        )
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)
        all_single_rec_sig.append(single_rec_sig)
    all_single_toks = torch.cat(all_single_toks)
    all_single_codes = torch.cat(all_single_codes)
    all_single_rec_sig = torch.cat(all_single_rec_sig)

    assert (batch_toks != all_single_toks).sum() <= 2, [
        (batch_toks != all_single_toks).sum().item(),
        batch_toks.numel(),
    ]
    assert (batch_codes != all_single_codes).sum() <= 2, [
        (batch_codes != all_single_codes).sum().item(),
        batch_codes.numel(),
    ]
    assert torch.allclose(batch_rec_sig, all_single_rec_sig, atol=1), (
        ((batch_rec_sig - all_single_rec_sig) ** 2).mean().sqrt()
    )

    print("Batch invariance test passed")


@torch.no_grad()
def test_reconstruction(audio_path: "str", config: "Optional[str]" = None) -> "None":
    try:
        import torchaudio
    except ImportError:
        raise ImportError("`pip install torchaudio` to run this script")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DyCAST.from_pretrained(config)
    model = model.eval().to(device)

    paths = [
        os.path.join("librispeech-dev-clean", "84"),
    ]
    matching_set = {k: [] for k in paths}
    for path in paths:
        for filename in os.listdir(os.path.join(_ROOT_DIR, "audios", path)):
            filepath = os.path.join(_ROOT_DIR, "audios", path, filename)
            sig, sample_rate = torchaudio.load(filepath)
            sig = torchaudio.functional.resample(
                sig, sample_rate, model.sample_rate_input
            )
            sig = sig.to(device)
            feats = model.sig_to_feats(sig)
            matching_set[path].append(feats[0])
        matching_set[path] = torch.cat(matching_set[path])

    sig, sample_rate = torchaudio.load(audio_path)
    sig = torchaudio.functional.resample(sig, sample_rate, model.sample_rate_input)
    sig = sig.to(device)

    # Original
    output_dir = os.path.join(_ROOT_DIR, "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(
        os.path.join(output_dir, "sig.wav"),
        sig.float().cpu(),
        model.sample_rate_input,
    )

    # All combinations
    for bsrc in ["char_aligner", "boundary_decode", "boundary_sample"]:
        for dsrc in ["original", "duration_decode", "duration_sample"]:
            tag = f"{bsrc}-{dsrc}"
            _, _, rec_sig = model(
                sig,
                boundary_source=bsrc,
                duration_source=dsrc,
            )
            torchaudio.save(
                os.path.join(output_dir, f"sig-{tag}.wav"),
                rec_sig.float().cpu(),
                model.sample_rate_output,
            )

    # Budget decoding
    for bsrc in ["char_aligner", "boundary_decode", "boundary_sample"]:
        dsrc = "duration_decode"
        tag = f"{bsrc}-{dsrc}-budget"
        _, _, rec_sig = model(
            sig,
            boundary_source=bsrc,
            duration_source=dsrc,
            budget_decode=True,
        )
        torchaudio.save(
            os.path.join(output_dir, f"sig-{tag}.wav"),
            rec_sig.float().cpu(),
            model.sample_rate_output,
        )

    # Official variants
    for min_gap in [None, 1, 3, 5]:
        if min_gap is None:
            bsrc = "char_aligner"
            tag = "ca"
            boundary_predictor_kwargs = {}
        else:
            bsrc = "boundary_decode"
            tag = f"b{min_gap}"
            boundary_predictor_kwargs = {"min_gap": min_gap}
        dsrc = "duration_decode"
        toks, _, rec_sig = model(
            sig,
            boundary_source=bsrc,
            duration_source=dsrc,
            budget_decode=True,
            boundary_predictor_kwargs=boundary_predictor_kwargs,
        )
        print(f"min_gap={min_gap}, toks.shape={toks.shape}")
        torchaudio.save(
            os.path.join(output_dir, f"sig-{tag}.wav"),
            rec_sig.float().cpu(),
            model.sample_rate_output,
        )

    # Voice conversion
    bsrc = "boundary_decode"
    dsrc = "duration_decode"
    tag = "b1-vc"
    _, _, rec_sig = model(
        sig,
        boundary_source=bsrc,
        duration_source=dsrc,
        budget_decode=True,
        matching_set=matching_set["librispeech-dev-clean/84"],
    )
    torchaudio.save(
        os.path.join(output_dir, f"sig-{tag}.wav"),
        rec_sig.float().cpu(),
        model.sample_rate_output,
    )

    # Retrieval-augmented decoding
    bsrc = "boundary_decode"
    dsrc = "duration_decode"
    tag = "b1-rad"
    _, _, rec_sig = model(
        sig,
        boundary_source=bsrc,
        duration_source=dsrc,
        budget_decode=True,
        use_retriever=True,
        sim_threshold=0.97,
    )
    torchaudio.save(
        os.path.join(output_dir, f"sig-{tag}.wav"),
        rec_sig.float().cpu(),
        model.sample_rate_output,
    )

    print("Reconstruction saved")


def test_performance(
    seconds: "float",
    compile: "Optional[str]" = None,
    fp16: "bool" = False,
    config: "Optional[str]" = None,
    **kwargs: "Any",
) -> "None":
    import torch.utils.benchmark as benchmark

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DyCAST.from_pretrained(config)
    model = model.eval().to(device)

    if compile == "torch.jit.script":
        model = model.jit()
    elif compile == "torch.compile":
        model = torch.compile(model, mode="max-autotune")
    sig = torch.randn(1, int(seconds * model.sample_rate_input), device=device)

    @torch.no_grad()
    def forward(sig: "Tensor") -> "None":
        with torch.autocast(device_type=device.type, enabled=fp16):
            model(sig, **kwargs)

    # Warmup
    for _ in range(10):
        forward(sig)

    print("=" * 150)
    print(
        f"Input length: {seconds} seconds, Compile: {compile}, "
        f"fp16: {fp16}, config: {config}, kwargs: {kwargs}"
    )
    print("=" * 150)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward(sig)
    print(f"Peak memory (MB): {torch.cuda.max_memory_allocated() / 1e6:.2f}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timer = benchmark.Timer(
        stmt="forward(sig)", globals={"sig": sig, "forward": forward}
    )
    time = timer.timeit(100).mean
    print(f"Latency: {time:.6f}, RTF: {seconds / time:.6f}")
    print("#" * 150)


if __name__ == "__main__":
    config = "lucadellalib/dycast"
    default_audio_path = os.path.join(
        _ROOT_DIR, "audios", "librispeech-dev-clean", "251-118436-0003.wav"
    )
    audio_path = sys.argv[1] if len(sys.argv) > 1 else default_audio_path
    test_model(config)
    test_batch_invariance(config)
    test_reconstruction(audio_path, config)
    for seconds in [1, 2, 4]:
        test_performance(seconds, config=config)
