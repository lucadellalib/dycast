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

"""MMS (see https://arxiv.org/abs/2305.13516)."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


__all__ = ["MMS"]


class MMS(nn.Module):
    # MMS-specific
    hop_s: "float" = 0.02  # 50 Hz
    blank_id: "int" = 0
    space_id: "int" = 4
    causal: "bool" = False

    def __init__(self, checkpoint: "str" = "facebook/mms-1b-all") -> "None":
        """Initialize a CTC-based speech model for extracting character-level
        transcripts and token durations.

        Parameters
        ----------
        checkpoint:
            HuggingFace model identifier or local path to a pretrained
            Wav2Vec2-CTC checkpoint (default: "facebook/mms-1b-all").

        """
        super().__init__()
        self.checkpoint = checkpoint

        try:
            from transformers import AutoProcessor, Wav2Vec2ForCTC
        except ImportError:
            raise ImportError("`pip install transformers` to use this model")

        # Modules
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

    @torch.no_grad()
    def forward(
        self,
        sig: "Tensor",
        length: "Optional[Tensor]" = None,
        max_silence_s: "Optional[float]" = None,
    ) -> "Tuple[Tensor, List[str]]":
        """Extract token durations and transcripts from raw audio.

        Frame-level CTC predictions are decoded into character sequences and
        converted into token durations, where each duration corresponds to the
        number of acoustic frames assigned to a token.

        Parameters
        ----------
        sig:
            Input audio waveforms of shape (batch_size, sig_length).
        length:
            Relative length of each sequence in the batch.
        max_silence_s:
            Maximum allowed silence duration (in seconds) when merging blank
            or space tokens.

        Returns
        -------
            - Token durations in frames of shape (batch_size, num_chars).
              For each batch element b, sum_u durations[b, u] equals the effective number of frames;
            - decoded text transcripts of shape (batch_size,).

        """
        logits = self.forward_sig(sig, length)
        tokens = logits.argmax(dim=-1)
        transcripts = self.processor.batch_decode(tokens)

        B, T = tokens.shape
        if length is not None:
            abs_length = (length * T).ceil().clamp(0, T).to(dtype=torch.long)
            spans_per_batch = [
                batched_run_length_encode(tokens[i, : abs_length[i]][None])[0]
                for i in range(B)
            ]
        else:
            spans_per_batch = batched_run_length_encode(tokens)

        chars_, durations = zip(
            *[
                spans_to_tensors(
                    forward_merge_blanks(
                        spans,
                        blank_id=self.blank_id,
                        space_id=self.space_id,
                        hop_s=self.hop_s,
                        max_silence_s=max_silence_s,
                    ),
                    silence_id=self.blank_id,
                )
                for spans in spans_per_batch
            ]
        )

        durations = nn.utils.rnn.pad_sequence(
            durations, batch_first=True, padding_value=0
        ).to(sig.device)

        # Handle empty transcript
        if durations.shape[1] == 0:
            durations = torch.full((B, 1), T, device=sig.device)
        elif durations.shape[1] == 1:
            durations[:, 0] = T

        return durations, transcripts

    @torch.no_grad()
    def forward_sig(self, sig: "Tensor", length: "Optional[Tensor]" = None) -> "Tensor":
        """Compute frame-level CTC logits from raw audio.

        Parameters
        ----------
        sig:
            Input audio waveforms of shape (batch_size, sig_length).
        length:
            Relative length of each sequence in the batch.

        Returns
        -------
            Frame-level CTC logits of shape
            (batch_size, sig_length // downsample_factor, vocab_size).

        """
        B, T = sig.shape

        if length is None:
            key_padding_mask = torch.ones(B, T, device=sig.device, dtype=torch.bool)
            mean = sig.mean(dim=1)
            var = sig.var(dim=1, unbiased=False)
        else:
            abs_length = (length * T).ceil().clamp(0, T).to(dtype=torch.long)
            key_padding_mask = (
                torch.arange(T, device=sig.device).expand(B, T) < abs_length[:, None]
            )
            # masked mean/var
            mask_f = key_padding_mask.to(sig.dtype)
            denom = abs_length.to(sig.dtype).clamp_min(1.0)  # avoid div0
            mean = (sig * mask_f).sum(dim=1) / denom
            var = ((sig - mean[:, None]) ** 2 * mask_f).sum(dim=1) / denom

        normed_sig = (sig - mean[:, None]) * torch.rsqrt(var[:, None] + 1e-7)

        logits = self.model(
            input_values=normed_sig,
            attention_mask=key_padding_mask.to(torch.long),
        ).logits
        return logits


def batched_run_length_encode(tokens: "Tensor") -> "List[List]":
    """Run-length encode a batch of token sequences.

    Parameters
    ----------
    tokens:
        Integer tensor of shape (batch_size, seq_length) containing token IDs.
        Typically of dtype ``torch.long``. Each row is treated as an independent
        sequence.

    Returns
    -------
        List of length (batch_size,). Each element is a list of spans for the
        corresponding sequence. A span is represented as ``[value, start, end]``,
        where:

        * ``value`` is the token ID (int)
        * ``start`` is the inclusive start index (int)
        * ``end`` is the exclusive end index (int)

        All indices are relative to the original time axis ``seq_length``.

    """
    B, T = tokens.shape
    device = tokens.device

    # 1) Find run starts (True where a new run starts)
    # First position is always a run start
    run_start = torch.zeros(B, T, dtype=torch.bool, device=device)
    run_start[:, 0] = True
    run_start[:, 1:] = tokens[:, 1:] != tokens[:, :-1]

    # 2) Indices of all run starts across the batch
    b_idx, t_idx = run_start.nonzero(as_tuple=True)  # shape [R], [R]
    values = tokens[b_idx, t_idx]  # [R]

    # 3) Compute end indices for each run (end-exclusive)
    # Default end is T (i.e., until sequence end)
    end_idx = torch.full_like(t_idx, T)

    # For all runs except the last one: if the next run is in same batch,
    # end at the next run's start; otherwise keep T.
    same_batch = b_idx[1:] == b_idx[:-1]
    end_idx[:-1][same_batch] = t_idx[1:][same_batch]

    # 4) Pack into per-batch Python lists of [value, start, end]
    spans_per_batch = [[] for _ in range(B)]
    for b, v, s, e in zip(
        b_idx.tolist(),
        values.tolist(),
        t_idx.tolist(),
        end_idx.tolist(),
    ):
        spans_per_batch[b].append([int(v), int(s), int(e)])

    return spans_per_batch


def forward_merge_blanks(
    spans: "List[List]",
    blank_id: "int" = 0,
    space_id: "int" = 4,
    hop_s: "float" = 0.02,
    max_silence_s: "Optional[float]" = None,
) -> "List[Dict]":
    """Merge and relabel blank spans in a sequence of RLE spans.

    This function takes a list of run-length encoded spans and applies
    several post-processing rules:

    1. Consecutive blanks (``label == blank_id``) are collapsed.
    2. Long blank segments (duration >= ``max_silence_s``) are converted
       to explicit ``"silence"`` segments.
    3. Short blanks preceding a non-blank token are absorbed into that
       token, and the token is marked as having started after a blank
       (internal flag ``_after_blank``).
    4. Trailing blanks are merged backward into the last non-silence token.
    5. Consecutive identical tokens are coalesced **only if** there was no
       blank boundary between them.
    6. If the last non-silence token has ID ``space_id``, it is merged into
       the previous non-silence token.

    Frame indices are converted to seconds using ``hop_s`` and stored as
    ``"start_s"`` and ``"end_s"`` in the output.

    Parameters
    ----------
        List of spans, typically produced by run-length encoding. Each span
        is assumed to be of the form ``[label, start, end]``, where:

        * ``label`` : int
            Token ID.
        * ``start`` : int
            Inclusive start frame index.
        * ``end`` : int
            Exclusive end frame index.

        The function may append an internal boolean flag to some spans
        during processing; this is removed before returning.
    blank_id:
        Token ID used to represent blanks. Defaults to ``0``.
    space_id:
        Special token ID to be merged into the previous non-silence token
        if it appears as the last non-silence span. Defaults to ``4``.
    hop_s:
        Frame hop duration in seconds. Used to convert frame indices
        to time in seconds. Defaults to ``0.02``.
    max_silence_s:
        If not ``None``, blank spans whose duration in seconds is greater
        than or equal to this value are turned into explicit
        ``"silence"`` segments. If ``None``, no explicit silence segments
        are created based on duration. Defaults to ``None``.

    Returns
    -------
        List of merged segments. Each element is a dictionary with keys:

        * ``"token"`` : int or str
            Token ID for non-silence segments, or the string
            ``"silence"`` for explicit silence segments.
        * ``"start"`` : int
            Start frame index.
        * ``"end"`` : int
            End frame index (exclusive).
        * ``"start_s"`` : float
            Start time in seconds (``start * hop_s``).
        * ``"end_s"`` : float
            End time in seconds (``end * hop_s``).

        Internal flags such as ``"_after_blank"`` are removed before
        returning.

    """
    merged = []
    i = 0
    N = len(spans)
    while i < N:
        lab, s, e, *_ = spans[i]
        if lab == blank_id:
            dur_s = (e - s) * hop_s
            if (max_silence_s is not None) and (dur_s >= max_silence_s):
                merged.append({"token": "silence", "start": s, "end": e})
                i += 1
                continue
            # collapse consecutive blanks forward
            j = i + 1
            while j < N and spans[j][0] == blank_id:
                e = spans[j][2]
                j += 1
            if j < N and spans[j][0] != blank_id:
                # mark that the NEXT token starts after a blank
                spans[j][1] = s  # extend next token left boundary
                spans[j].append(True)  # flag: started_after_blank
                i = j
                continue
            else:
                # trailing blanks -> merge backward
                if merged and merged[-1]["token"] != "silence":
                    merged[-1]["end"] = e
                i = j
                continue
        else:
            # If no flag appended yet, default False
            started_after_blank = len(spans[i]) >= 4 and spans[i][3] is True
            merged.append(
                {
                    "token": int(lab),
                    "start": s,
                    "end": e,
                    "_after_blank": started_after_blank,
                }
            )
            i += 1

    # Coalesce identical tokens ONLY if there was NO blank boundary between them
    coalesced = []
    for seg in merged:
        if (
            coalesced
            and seg["token"] == coalesced[-1]["token"]
            and seg.get("_after_blank", False) is False
            and coalesced[-1].get("_after_blank", False) is False
        ):
            coalesced[-1]["end"] = seg["end"]
        else:
            coalesced.append(seg)

    # --- Special rule: if the last *non-silence* token is space_id, merge it into the previous non-silence token ---
    if max_silence_s is None:
        # Strict backward compatibilty
        non_sil_idxs = [k for k, s in enumerate(coalesced) if s["token"] != "silence"]
        if len(non_sil_idxs) >= 2:
            last_idx = non_sil_idxs[-1]
            if coalesced[last_idx]["token"] == space_id:
                prev_idx = non_sil_idxs[-2]
                # extend previous token's end to swallow the final space_id span
                coalesced[prev_idx]["end"] = coalesced[last_idx]["end"]
                # drop the last space_id span
                del coalesced[last_idx]
    else:
        # --- Special rule: if the last *non-silence* token is space_id, merge it into the previous token ---
        non_sil_idxs = [k for k, s in enumerate(coalesced) if s["token"] != "silence"]
        if len(non_sil_idxs) >= 2:
            last_idx = non_sil_idxs[-1]
            if coalesced[last_idx]["token"] == space_id:
                prev_idx = last_idx - 1
                # extend previous token's end to swallow the final space_id span
                coalesced[prev_idx]["end"] = coalesced[last_idx]["end"]
                # drop the last space_id span
                del coalesced[last_idx]

    # Add seconds and clean
    for seg in coalesced:
        seg["start_s"] = seg["start"] * hop_s
        seg["end_s"] = seg["end"] * hop_s
        seg.pop("_after_blank", None)

    return coalesced


def spans_to_tensors(
    spans: "List[Dict]",
    silence_id: "int" = -1,
    use_seconds: "bool" = False,
    device: "str" = "cpu",
) -> "Tuple[Tensor, Tensor]":
    """Convert merged spans into token and length tensors.

    Parameters
    ----------
    spans:
        List of span dictionaries, typically as returned by
        :func:`forward_merge_blanks`. Each dictionary must contain:

        * ``"token"`` : int or str
            Token ID for non-silence segments, or the string
            ``"silence"`` for explicit silence segments.
        * ``"start"`` : int
            Start frame index.
        * ``"end"`` : int
            End frame index (exclusive).
        * ``"start_s"`` : float
            Start time in seconds.
        * ``"end_s"`` : float
            End time in seconds.

    silence_id:
        Token ID to assign to segments where ``token == "silence"``.
        Defaults to ``-1``.
    use_seconds:
        If ``True``, lengths are computed in seconds as
        ``end_s - start_s`` and returned as ``torch.float32``.
        If ``False``, lengths are computed in frames as
        ``end - start`` and returned as ``torch.long``.
        Defaults to ``False``.
    device:
        Target device for the returned tensors (e.g., ``"cpu"``,
        ``"cuda"``). Defaults to ``"cpu"``.

    Returns
    -------
        - 1D tensor of shape ``(N,)`` with token IDs (``torch.long``),
          where ``"silence"`` tokens have been mapped to ``silence_id``;
        - 1D tensor of shape ``(N,)`` with segment lengths:
          * dtype ``torch.long`` if ``use_seconds=False`` (frame counts),
          * dtype ``torch.float32`` if ``use_seconds=True`` (seconds).

    """
    vals, lens = [], []
    for seg in spans:
        tok = seg["token"]
        tok_id = silence_id if isinstance(tok, str) and tok == "silence" else int(tok)
        vals.append(tok_id)

        if use_seconds:
            lens.append(float(seg["end_s"]) - float(seg["start_s"]))
        else:
            lens.append(int(seg["end"]) - int(seg["start"]))

    vals = torch.tensor(vals, device=device, dtype=torch.long)
    if use_seconds:
        lens = torch.tensor(lens, device=device, dtype=torch.float32)
    else:
        lens = torch.tensor(lens, device=device, dtype=torch.long)

    return vals, lens


def test_model() -> "None":
    import types

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    T_ctc = 30
    V = 8

    # -------------------------------------------------
    # Build an MMS instance without downloading HF weights
    # (patch forward_sig + processor.batch_decode)
    # -------------------------------------------------
    mms = MMS.__new__(MMS)
    nn.Module.__init__(mms)  # important: initialize Module internals
    mms.checkpoint = "dummy"

    class _DummyProcessor:
        def batch_decode(self, tokens: "Tensor") -> "List[str]":
            # Deterministic "transcripts" for test purposes (one per batch element)
            out: "List[str]" = []
            for b in range(tokens.shape[0]):
                out.append(f"utt{b}")
            return out

    mms.processor = _DummyProcessor()

    # Create logits that produce a known argmax token sequence with NO blanks/spaces.
    # This avoids depending on blank/space merge corner cases.
    # tokens: repeating [1,2,3,5] pattern (none equals blank_id=0 or space_id=4).
    base_tokens = torch.tensor([1, 2, 3, 5], device=device, dtype=torch.long)

    def _forward_sig(
        self: "MMS", sig: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        B_local, _ = sig.shape
        toks = base_tokens.repeat(
            (T_ctc + base_tokens.numel() - 1) // base_tokens.numel()
        )[:T_ctc]
        toks = toks[None, :].repeat(B_local, 1)  # [B, T_ctc]
        logits = torch.full(
            (B_local, T_ctc, V), -10.0, device=sig.device, dtype=torch.float32
        )
        logits.scatter_(2, toks[:, :, None], 10.0)
        return logits

    mms.forward_sig = types.MethodType(_forward_sig, mms)

    # -------------------------------------------------
    # forward() invariants (length=None)
    # -------------------------------------------------
    sig = torch.randn((B, 16000), device=device)

    durs, transcripts = mms(sig, length=None, max_silence_s=None)

    assert isinstance(durs, torch.Tensor)
    assert isinstance(transcripts, list)
    assert len(transcripts) == B

    assert durs.dtype == torch.long
    assert durs.shape[0] == B
    assert (durs >= 0).all()

    # Invariant: durations sum to T_ctc (effective number of CTC frames)
    expected = torch.full((B,), T_ctc, device=device, dtype=torch.long)
    assert torch.equal(durs.sum(dim=1), expected)

    # padding is zeros on the right
    for b in range(B):
        row = durs[b]
        nz = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
        if nz.numel() > 0:
            last = int(nz[-1].item())
            assert torch.equal(row[last + 1 :], torch.zeros_like(row[last + 1 :]))

    # -------------------------------------------------
    # forward() invariants (length provided)
    # -------------------------------------------------
    length = torch.rand((B,), device=device).clamp(0.1, 1.0)  # avoid empty edge case
    durs_l, transcripts_l = mms(sig, length=length, max_silence_s=None)

    assert durs_l.dtype == torch.long
    assert durs_l.shape[0] == B
    assert len(transcripts_l) == B

    abs_len = (length * T_ctc).ceil().clamp(0, T_ctc).to(dtype=torch.long)
    # Invariant: durations sum to abs_len (effective number of CTC frames per sample)
    assert torch.equal(durs_l.sum(dim=1).cpu(), abs_len.cpu())

    print("Model test passed")


@torch.no_grad()
def test_batch_invariance() -> "None":
    import types

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T_ctc = 40
    V = 8

    mms = MMS.__new__(MMS)
    nn.Module.__init__(mms)
    mms.checkpoint = "dummy"

    class _DummyProcessor:
        def batch_decode(self, tokens: "Tensor") -> "List[str]":
            # Make transcript depend on tokens so invariance is meaningful
            # (but still deterministic).
            out: "List[str]" = []
            for b in range(tokens.shape[0]):
                out.append("".join([str(int(x.item())) for x in tokens[b, :5]]))
            return out

    mms.processor = _DummyProcessor()

    base_tokens = torch.tensor([1, 2, 3, 5], device=device, dtype=torch.long)

    def _forward_sig(
        self: "MMS", sig: "Tensor", length: "Optional[Tensor]" = None
    ) -> "Tensor":
        B_local, _ = sig.shape
        toks = base_tokens.repeat(
            (T_ctc + base_tokens.numel() - 1) // base_tokens.numel()
        )[:T_ctc]
        toks = toks[None, :].repeat(B_local, 1)
        logits = torch.full(
            (B_local, T_ctc, V), -10.0, device=sig.device, dtype=torch.float32
        )
        logits.scatter_(2, toks[:, :, None], 10.0)
        return logits

    mms.forward_sig = types.MethodType(_forward_sig, mms)

    sig = torch.randn((B, 16000), device=device)
    length = torch.rand((B,), device=device).clamp(0.1, 1.0)

    # Batched path
    batch_durs, batch_txt = mms(sig, length=length, max_silence_s=None)
    U_max = batch_durs.shape[1]

    # Singleton path
    all_single_durs = []
    all_single_txt = []

    for i in range(B):
        single_durs, single_txt = mms(
            sig[i : i + 1],
            length=length[i : i + 1],
            max_silence_s=None,
        )

        # pad to batch width for comparison
        if single_durs.shape[1] < U_max:
            pad = single_durs.new_zeros((1, U_max - single_durs.shape[1]))
            single_durs = torch.cat([single_durs, pad], dim=1)

        all_single_durs.append(single_durs)
        all_single_txt.append(single_txt[0])

    all_single_durs = torch.cat(all_single_durs, dim=0)

    assert torch.equal(batch_durs, all_single_durs)
    assert batch_txt == all_single_txt

    print("Batch invariance test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
