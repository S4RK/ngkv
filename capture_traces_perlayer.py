"""Capture per-layer attention traces (Rung 1.5: pooling sensitivity).

Byte-identical decoding to capture_traces.py (same RNG seed, same
sampling), so trace n here is the *same decode* as traces/trace_nn.npz.
Differences in what is saved:

  attn         (D, total)      mean over layers+heads  [schema field,
                               must reproduce v0.2 traces bit-close]
  attn_layers  (L, D, total)   per-layer (head-mean) attention
  attn_max     (D, total)      max over layers+heads, renormalized

Extra keys are backward compatible: ngkv.traces.load_trace reads only
{attn, prompt_len, meta_json}.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

import train_tiny_model as T
from ngkv.traces import save_trace

PROMPT_LEN = 256
DECODE_LEN = 120
N_TRACES = 10
TEMP = 0.8

meta = json.load(open("tiny_model_meta.json"))
flat_saved = np.load("tiny_model.npz")
ref = T.init_params(jax.random.PRNGKey(0))
leaves, treedef = jax.tree_util.tree_flatten(ref)
params = jax.tree_util.tree_unflatten(
    treedef, [jnp.array(flat_saved[f"arr_{i}"]) for i in range(len(leaves))]
)

TOTAL = PROMPT_LEN + DECODE_LEN
fwd = jax.jit(lambda p, idx: T.forward(p, idx, return_attn=True))

rng = np.random.default_rng(42)
val = T.val_data

for n in range(N_TRACES):
    start = rng.integers(0, len(val) - PROMPT_LEN - 1)
    ids = list(val[start : start + PROMPT_LEN])
    rows_mean, rows_layers, rows_max = [], [], []
    for t in range(DECODE_LEN):
        L = len(ids)
        padded_ids = np.zeros(TOTAL, dtype=np.int32)
        padded_ids[:L] = ids
        logits, attns = fwd(params, jnp.array([padded_ids]))
        # (Lyr, H, L): attention of query at position L-1 over cache [:L]
        row_lh = jnp.stack([a[0, :, L - 1, :L] for a in attns])
        row_lh = np.asarray(row_lh, dtype=np.float64)

        # mean over layers+heads (v0.2 reduction)
        rm = row_lh.mean(axis=(0, 1)); rm /= rm.sum()
        pm = np.zeros(TOTAL); pm[:L] = rm; rows_mean.append(pm)

        # per-layer (mean over heads within layer)
        rl = row_lh.mean(axis=1)
        rl /= rl.sum(axis=1, keepdims=True)
        pl = np.zeros((rl.shape[0], TOTAL)); pl[:, :L] = rl
        rows_layers.append(pl)

        # max over layers+heads, renormalized
        rx = row_lh.max(axis=(0, 1)); rx /= rx.sum()
        px = np.zeros(TOTAL); px[:L] = rx; rows_max.append(px)

        p = np.asarray(jax.nn.softmax(logits[0, L - 1] / TEMP))
        ids.append(int(rng.choice(len(p), p=p / p.sum())))

    attn = np.stack(rows_mean)
    attn_layers = np.stack(rows_layers, axis=1)          # (L, D, total)
    attn_max = np.stack(rows_max)
    m = {"model": "tiny-shakespeare-gpt (3L/4H/96d, JAX)",
         "reduction": "mean(layers,heads)", "temperature": TEMP,
         "val_loss": meta["val_loss"], "granularity": 1,
         "extra_keys": ["attn_layers (L,D,total) head-mean per layer",
                        "attn_max (D,total) max(layers,heads) renorm"]}
    np.savez_compressed(
        f"traces_rung15/trace_{n:02d}.npz",
        attn=attn.astype(np.float32),
        prompt_len=np.int64(PROMPT_LEN),
        meta_json=json.dumps(m),
        attn_layers=attn_layers.astype(np.float32),
        attn_max=attn_max.astype(np.float32),
    )
    print(f"trace {n}: cache {PROMPT_LEN}->{TOTAL}", flush=True)
print("done")
