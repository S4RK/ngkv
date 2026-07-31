"""
ngkv.traces
===========

Portable attention-trace schema for NG-KV replay evaluation.

A *trace* is one request's decode-time attention record:

  attn        float32 (D, total)   row t = attention distribution of
                                   decode step t over the P + t tokens
                                   cached at that step (zero padded to
                                   total = P + D). Rows sum to 1 over
                                   the first P + t entries. Already
                                   reduced over layers/heads by the
                                   capture side (mean or max — record
                                   which in meta).
  prompt_len  int                  P
  meta        dict (json)          free-form: model id, reduction
                                   ("mean"|"max"), layer subset,
                                   sampling params, workload tag.

Serialization: .npz with keys {attn, prompt_len, meta_json}. A trace
*set* is a directory of .npz files; ``load_traces`` streams them.

This is deliberately the minimal interface between any serving stack
and the NG-KV evaluator: capture attention however the stack allows
(eager attention in HF, sampled steps in production, block-level
aggregates if token-level is too expensive — coarser granularity only
weakens, never invalidates, the regret bound), write .npz, replay.

Block-granular traces: store attn at block resolution and set
``meta["granularity"] = block_size``; policies then gate blocks.

Hybrid-model traces (Rung K3): for models that interleave linear
attention (constant-size recurrent state, no per-token KV — e.g. KDA
in Kimi Linear / Kimi K3) with full-attention layers, only the
full-attention layers have a gateable KV cache, so *only those layers
are captured*. Conventions:

  meta["attn_family"]        "gqa" (default) | "mla_latent"
  meta["layer_types"]        list[str], one of {"kda","mla","gqa"} per
                             model layer (full depth, 0-indexed order)
  meta["captured_layers"]    0-indexed model layer indices the trace's
                             attn arrays cover (== the full-attention
                             layers; len == attn_layers.shape[0])

For ``attn_family == "mla_latent"`` the gating unit is one shared
latent vector per token per layer (MLA): heads read the same latent,
so per-head gating does not exist — head-pooled rows are the *exact*
granularity of the placement decision, not an approximation of it.

Extra arrays (``attn_layers``, ``attn_max``, …) ride along as
additional .npz keys; ``load_trace``/``load_traces`` ignore them
(backward compatible) and ``load_trace_full`` returns them.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterator, Tuple

import numpy as np


def save_trace(path: str | pathlib.Path, attn: np.ndarray, prompt_len: int,
               meta: dict | None = None,
               extras: dict[str, np.ndarray] | None = None) -> None:
    attn = np.asarray(attn, dtype=np.float32)
    assert attn.ndim == 2, "attn must be (decode_len, total)"
    extra_arrays = {k: np.asarray(v, dtype=np.float32)
                    for k, v in (extras or {}).items()}
    assert not ({"attn", "prompt_len", "meta_json"} & extra_arrays.keys())
    np.savez_compressed(path, attn=attn, prompt_len=np.int64(prompt_len),
                        meta_json=json.dumps(meta or {}), **extra_arrays)


def load_trace_full(path: str | pathlib.Path
                    ) -> Tuple[np.ndarray, int, dict, dict]:
    """Like ``load_trace`` but also returns the extra arrays dict."""
    z = np.load(path, allow_pickle=False)
    extras = {k: z[k] for k in z.files
              if k not in ("attn", "prompt_len", "meta_json")}
    return (z["attn"].astype(np.float64), int(z["prompt_len"]),
            json.loads(str(z["meta_json"])), extras)


def load_trace(path: str | pathlib.Path) -> Tuple[np.ndarray, int, dict]:
    z = np.load(path, allow_pickle=False)
    return (z["attn"].astype(np.float64), int(z["prompt_len"]),
            json.loads(str(z["meta_json"])))


def load_traces(directory: str | pathlib.Path) -> Iterator[Tuple[str, np.ndarray, int, dict]]:
    for p in sorted(pathlib.Path(directory).glob("*.npz")):
        attn, P, meta = load_trace(p)
        yield p.stem, attn, P, meta
