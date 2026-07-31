"""Capture real attention traces from the trained tiny model.

For each of N held-out prompts: run full-context forward passes during
greedy decoding with temperature, record the last query row's attention
(mean over layers and heads) at every decode step, and write an NG-KV
.npz trace (ngkv.traces schema).

Full-context re-forward per step is O(T^2) per step — fine at this
scale, and it gives exact attention (no KV approximation contaminating
the trace we are about to evaluate KV policies on).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

import train_tiny_model as T
from ngkv.traces import save_trace

PROMPT_LEN = 256
DECODE_LEN = 120          # PROMPT_LEN + DECODE_LEN <= CTX (384)
N_TRACES = 10
TEMP = 0.8

meta = json.load(open("tiny_model_meta.json"))
flat_saved = np.load("tiny_model.npz")
# Rebuild pytree in the exact flatten order.
ref = T.init_params(jax.random.PRNGKey(0))
leaves, treedef = jax.tree_util.tree_flatten(ref)
params = jax.tree_util.tree_unflatten(
    treedef, [jnp.array(flat_saved[f"arr_{i}"]) for i in range(len(leaves))]
)

TOTAL = PROMPT_LEN + DECODE_LEN
fwd = jax.jit(lambda p, idx: T.forward(p, idx, return_attn=True))
# Fixed-length padded forward: causal mask means outputs at position
# L-1 are unaffected by padding tokens at positions >= L, so we pad to
# TOTAL once and JIT compiles a single shape.

rng = np.random.default_rng(42)
val = T.val_data

for n in range(N_TRACES):
    start = rng.integers(0, len(val) - PROMPT_LEN - 1)
    ids = list(val[start : start + PROMPT_LEN])
    attn_rows = []
    for t in range(DECODE_LEN):
        L = len(ids)
        padded_ids = np.zeros(TOTAL, dtype=np.int32)
        padded_ids[:L] = ids
        logits, attns = fwd(params, jnp.array([padded_ids]))
        # attention of query at position L-1 over its cache [:L]
        row = jnp.stack([a[0, :, L - 1, :L] for a in attns])  # (Lyr, H, L)
        row = row.mean(axis=(0, 1))
        row = np.asarray(row, dtype=np.float64)
        row /= row.sum()
        # pad to final total length
        padded = np.zeros(TOTAL)
        padded[:L] = row
        attn_rows.append(padded)
        p = np.asarray(jax.nn.softmax(logits[0, L - 1] / TEMP))
        ids.append(int(rng.choice(len(p), p=p / p.sum())))
    attn = np.stack(attn_rows)
    save_trace(
        f"traces/trace_{n:02d}.npz", attn, PROMPT_LEN,
        meta={"model": "tiny-shakespeare-gpt (3L/4H/96d, JAX)",
              "reduction": "mean(layers,heads)", "temperature": TEMP,
              "val_loss": meta["val_loss"], "granularity": 1},
    )
    print(f"trace {n}: cache {PROMPT_LEN}->{PROMPT_LEN+DECODE_LEN}, "
          f"row entropy {-(attn[np.nonzero(attn)]*np.log(attn[np.nonzero(attn)])).sum()/DECODE_LEN:.2f}",
          flush=True)
print("done")
