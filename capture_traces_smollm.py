"""Rung 2: capture real attention traces from a mid-size open-weight model.

Loads SmolLM2-360M (fallbacks: Qwen2.5-0.5B, SmolLM2-135M) with eager
attention, decodes N held-out prompts step-by-step with the KV cache,
and records the new query's attention over the full cache at every
step. Saves the same three views as Rung 1.5 so both replay_real.py
(mean-pooled schema field) and replay_pooling.py (per-layer / max keys)
run unmodified:

  attn         (D, total)      mean over layers+heads   [schema field]
  attn_layers  (L, D, total)   per-layer head-mean
  attn_max     (D, total)      max over layers+heads, renormalized

Stated plainly: step-wise generation with output_attentions disables
fused kernels — this is the validation path, not a speed path. Exact
attention, no KV approximation contaminating the trace.

Usage:
  python capture_traces_smollm.py                  # default model
  python capture_traces_smollm.py Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B",
          "HuggingFaceTB/SmolLM2-135M"]
PROMPT_LEN = 256
DECODE_LEN = 120
N_TRACES = 10
TEMP = 0.8
SEED = 42
OUT_DIR = "traces_rung2"


def load_model(candidates):
    last_err = None
    for mid in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(
                mid, attn_implementation="eager", dtype=torch.float32)
            model.eval()
            return mid, tok, model
        except Exception as e:  # noqa: BLE001 — try next fallback
            print(f"[warn] could not load {mid}: {e}", file=sys.stderr)
            last_err = e
    raise SystemExit(f"no candidate model loadable: {last_err}")


def main():
    candidates = [sys.argv[1]] + MODELS if len(sys.argv) > 1 else MODELS
    mid, tok, model = load_model(candidates)
    L = model.config.num_hidden_layers
    print(f"model {mid}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{L} layers, {model.config.num_attention_heads} heads")

    # Held-out prompt source: disjoint windows over the repo corpus.
    text = open("data_shakespeare.txt").read()
    enc = tok(text, return_tensors="pt").input_ids[0]
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    pathlib.Path(OUT_DIR).mkdir(exist_ok=True)

    starts = rng.choice(len(enc) - PROMPT_LEN - 1, size=N_TRACES, replace=False)
    TOTAL = PROMPT_LEN + DECODE_LEN

    for n, s in enumerate(starts):
        t0 = time.time()
        ids = enc[s : s + PROMPT_LEN].unsqueeze(0)
        past = None
        rows_mean, rows_layers, rows_max = [], [], []
        cur = ids
        with torch.no_grad():
            for t in range(DECODE_LEN):
                out = model(input_ids=cur if past is None else cur[:, -1:],
                            past_key_values=past, use_cache=True,
                            output_attentions=True)
                past = out.past_key_values
                cl = PROMPT_LEN + t
                # (L, H, cache_len): last query row per layer
                row_lh = torch.stack(
                    [a[0, :, -1, :cl] for a in out.attentions]).double().numpy()
                rm = row_lh.mean(axis=(0, 1)); rm /= rm.sum()
                rl = row_lh.mean(axis=1); rl /= rl.sum(axis=1, keepdims=True)
                rx = row_lh.max(axis=(0, 1)); rx /= rx.sum()
                pm = np.zeros(TOTAL); pm[:cl] = rm; rows_mean.append(pm)
                pl = np.zeros((L, TOTAL)); pl[:, :cl] = rl; rows_layers.append(pl)
                px = np.zeros(TOTAL); px[:cl] = rx; rows_max.append(px)

                logits = out.logits[0, -1, :] / TEMP
                p = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(p, 1)
                cur = torch.cat([cur, nxt.unsqueeze(0)], dim=-1)

        attn = np.stack(rows_mean)
        attn_layers = np.stack(rows_layers, axis=1)   # (L, D, total)
        attn_max = np.stack(rows_max)
        meta = {"model": mid, "reduction": "mean(layers,heads)",
                "temperature": TEMP, "granularity": 1,
                "prompt_source": "tinyshakespeare disjoint windows",
                "extra_keys": ["attn_layers (L,D,total) head-mean per layer",
                               "attn_max (D,total) max(layers,heads) renorm"]}
        np.savez_compressed(
            f"{OUT_DIR}/trace_{n:02d}.npz",
            attn=attn.astype(np.float32), prompt_len=np.int64(PROMPT_LEN),
            meta_json=json.dumps(meta),
            attn_layers=attn_layers.astype(np.float32),
            attn_max=attn_max.astype(np.float32))
        ent = -(attn[np.nonzero(attn)] * np.log(attn[np.nonzero(attn)])).sum() / DECODE_LEN
        print(f"trace {n}: cache {PROMPT_LEN}->{TOTAL}, row entropy {ent:.2f}, "
              f"{time.time()-t0:.0f}s", flush=True)
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
