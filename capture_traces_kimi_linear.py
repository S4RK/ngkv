"""Rung K3-1: capture MLA attention traces from a hybrid KDA/MLA model.

Target: moonshotai/Kimi-Linear-48B-A3B-Instruct (same 3:1 KDA:MLA
hybrid, same kv_lora_rank=512 and 576-dim latent as Kimi K3; 7 MLA
layers of 27). The identical script runs on Kimi K3 itself (24 MLA
layers of 93) — only the model id changes.

Why not ``output_attentions=True`` (the Rung-2 path): Moonshot's
modeling code accepts the flag but DISCARDS the weights —
``KimiMLAAttention.forward`` does ``attn_output, _ =
attention_interface(...)`` and the decoder layer returns only hidden
states. Capture therefore monkeypatches ``eager_attention_forward`` in
the (dynamically loaded) modeling module: the wrapper calls the
original and stashes the last query row's softmax weights per
``module.layer_idx``. KDA layers never route through this function, so
what lands in the store is exactly the full-attention gating surface.

Granularity: MLA heads all read one shared 576-dim latent per token,
so the placement unit is (token, layer). Head-pooled rows are the
EXACT decision granularity here, not a lossy reduction — unlike the
GQA rungs, where head-mean was an approximation choice.

Views saved (schema-compatible with replay_real / replay_pooling /
replay_compare):
  attn         (D, total)          mean over MLA layers + heads
  attn_layers  (L_mla, D, total)   per-MLA-layer head-mean
  attn_max     (D, total)          max over MLA layers + heads, renorm

Protocol deliberately identical to Rung 2 (tinyshakespeare disjoint
windows, P=256, D=120, T=0.8, seed 42) so hybrid-vs-pure comparisons
are paired at the protocol level. Override with flags for chat-style
or long-context workloads.

Stated plainly: eager attention disables fused/absorbed MLA kernels —
this is the validation path, not a speed path. On the 48B model run
this on a GPU node (bf16, single B200 is ample); CPU is for the smoke
test only.

Usage (cluster):
  python capture_traces_kimi_linear.py \
      --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
      --out traces_k3_rung1 --n 10
  python capture_traces_kimi_linear.py \
      --model /mnt/kvbm-cache/kimi-k3 --out traces_k3 --n 6
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# transformers version shims (call before loading the model)
#
# Moonshot's modeling_kimi.py rides the edge of the 4.5x API:
#   * transformers >= 5 moved OutputRecorder out of utils.generic;
#   * transformers <= 4.57.6 crashes in @auto_docstring on PEP 604
#     unions (`torch.Tensor | None`) used in the modeling signatures.
# Both shims are inert where the underlying issue is absent, and
# auto_docstring is cosmetic (docstring synthesis only).
# ---------------------------------------------------------------------------
def apply_transformers_shims() -> None:
    import transformers.utils as tu
    import transformers.utils.generic as tug
    if not hasattr(tug, "OutputRecorder"):  # transformers >= 5
        from transformers.utils.output_capturing import OutputRecorder
        tug.OutputRecorder = OutputRecorder

    def _noop_auto_docstring(obj=None, **kwargs):
        if obj is None:
            return lambda o: o
        return obj

    try:  # probe: does auto_docstring survive a PEP 604 signature?
        def _probe(x: "int | None" = None): ...
        _probe.__annotations__ = {"x": int | None}
        tu.auto_docstring(_probe)
    except Exception:
        tu.auto_docstring = _noop_auto_docstring
        try:
            import transformers.utils.auto_docstring as tad
            tad.auto_docstring = _noop_auto_docstring
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Layer-type mapping (config lists are 1-indexed; we emit 0-indexed)
# ---------------------------------------------------------------------------
def layer_types_from_config(cfg) -> tuple[list[str], list[int]]:
    """Return (layer_types full depth, 0-indexed MLA layer indices)."""
    L = cfg.num_hidden_layers
    la = getattr(cfg, "linear_attn_config", None) or {}
    kda = set(la.get("kda_layers") or [])
    if not kda:
        return ["mla"] * L, list(range(L))
    types = ["kda" if (i + 1) in kda else "mla" for i in range(L)]
    return types, [i for i, t in enumerate(types) if t == "mla"]


# ---------------------------------------------------------------------------
# Attention capture via eager_attention_forward monkeypatch
# ---------------------------------------------------------------------------
class MLACapture:
    """Wraps ``eager_attention_forward`` in the model's modeling module.

    After each forward, ``self.rows`` maps 0-indexed layer_idx ->
    float64 numpy (H, cache_len): the LAST query position's attention
    over the cache. Call ``clear()`` between steps.
    """

    def __init__(self, model, mla_layers: list[int]):
        attn_mod = model.model.layers[mla_layers[0]].self_attn
        self.mod = sys.modules[type(attn_mod).__module__]
        if not hasattr(self.mod, "eager_attention_forward"):
            raise RuntimeError(
                f"{self.mod.__name__} has no eager_attention_forward; "
                "modeling code changed — update the capture seam.")
        assert model.config._attn_implementation == "eager", \
            "load the model with attn_implementation='eager'"
        self._orig = self.mod.eager_attention_forward
        self.rows: dict[int, np.ndarray] = {}
        self.mla = set(mla_layers)

    def __enter__(self):
        orig, rows, mla = self._orig, self.rows, self.mla

        def wrapped(module, query, key, value, attention_mask,
                    scaling=None, dropout=0.0, **kw):
            out = orig(module, query, key, value, attention_mask,
                       scaling, dropout, **kw) if scaling is not None else \
                  orig(module, query, key, value, attention_mask, **kw)
            attn_weights = out[1]
            li = getattr(module, "layer_idx", None)
            if li in mla and attn_weights is not None:
                # (B, H, q_len, kv_len) -> last query row, batch 0
                rows[li] = (attn_weights[0, :, -1, :]
                            .detach().float().cpu().double().numpy())
            return out

        self.mod.eager_attention_forward = wrapped
        return self

    def __exit__(self, *exc):
        self.mod.eager_attention_forward = self._orig

    def clear(self):
        self.rows.clear()


# ---------------------------------------------------------------------------
# One trace = one prompt decoded step-by-step
# ---------------------------------------------------------------------------
def capture_trace(model, input_ids: torch.Tensor, decode_len: int,
                  mla_layers: list[int], temperature: float,
                  generator: torch.Generator | None = None) -> dict:
    P = input_ids.shape[1]
    total = P + decode_len
    n_mla = len(mla_layers)
    rows_mean, rows_layers, rows_max = [], [], []

    with MLACapture(model, mla_layers) as cap, torch.no_grad():
        past, cur = None, input_ids
        for t in range(decode_len):
            cap.clear()
            out = model(input_ids=cur if past is None else cur[:, -1:],
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            cl = P + t
            missing = [li for li in mla_layers if li not in cap.rows]
            assert not missing, f"no attention captured for layers {missing}"
            # (L_mla, H, cache_len)
            row_lh = np.stack([cap.rows[li][:, :cl] for li in mla_layers])
            rm = row_lh.mean(axis=(0, 1)); rm /= rm.sum()
            rl = row_lh.mean(axis=1); rl /= rl.sum(axis=1, keepdims=True)
            rx = row_lh.max(axis=(0, 1)); rx /= rx.sum()
            pm = np.zeros(total); pm[:cl] = rm; rows_mean.append(pm)
            pl = np.zeros((n_mla, total)); pl[:, :cl] = rl
            rows_layers.append(pl)
            px = np.zeros(total); px[:cl] = rx; rows_max.append(px)

            logits = out.logits[0, -1, :].float() / temperature
            p = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(p, 1, generator=generator)
            cur = torch.cat([cur, nxt.view(1, 1).to(cur.device)], dim=-1)

    return {"attn": np.stack(rows_mean),
            "attn_layers": np.stack(rows_layers, axis=1),  # (L_mla, D, total)
            "attn_max": np.stack(rows_max)}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="moonshotai/Kimi-Linear-48B-A3B-Instruct")
    ap.add_argument("--out", default="traces_k3_rung1")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--decode-len", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default="data_shakespeare.txt")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "bf16", "fp32"])
    args = ap.parse_args()

    apply_transformers_shims()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = (torch.bfloat16 if (args.dtype == "bf16" or
                             (args.dtype == "auto" and dev == "cuda"))
          else torch.float32)
    print(f"loading {args.model} on {dev} ({dt}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation="eager",
        dtype=dt, device_map=dev)
    model.eval()

    types, mla_layers = layer_types_from_config(model.config)
    latent = model.config.kv_lora_rank + model.config.qk_rope_head_dim
    print(f"{len(types)} layers: {types.count('kda')} KDA + "
          f"{len(mla_layers)} MLA (capture) at {mla_layers}; "
          f"latent dim {latent}", flush=True)

    text = open(args.corpus).read()
    enc = tok(text, return_tensors="pt").input_ids[0]
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    outdir = pathlib.Path(args.out); outdir.mkdir(exist_ok=True)
    starts = rng.choice(len(enc) - args.prompt_len - 1, size=args.n,
                        replace=False)

    meta_common = {
        "model": args.model, "reduction": "mean(mla_layers,heads)",
        "temperature": args.temperature, "granularity": 1,
        "attn_family": "mla_latent", "layer_types": types,
        "captured_layers": mla_layers,
        "mla_latent_dim": latent,
        "prompt_source": f"{args.corpus} disjoint windows",
        "capture_seam": "eager_attention_forward monkeypatch "
                        "(output_attentions is discarded upstream)",
        "extra_keys": ["attn_layers (L_mla,D,total) head-mean per MLA layer",
                       "attn_max (D,total) max(mla_layers,heads) renorm"],
    }
    for n, s in enumerate(starts):
        t0 = time.time()
        ids = enc[s: s + args.prompt_len].unsqueeze(0).to(dev)
        views = capture_trace(model, ids, args.decode_len, mla_layers,
                              args.temperature, gen)
        np.savez_compressed(
            outdir / f"trace_{n:02d}.npz",
            attn=views["attn"].astype(np.float32),
            prompt_len=np.int64(args.prompt_len),
            meta_json=json.dumps(meta_common),
            attn_layers=views["attn_layers"].astype(np.float32),
            attn_max=views["attn_max"].astype(np.float32))
        a = views["attn"]
        ent = -(a[np.nonzero(a)] * np.log(a[np.nonzero(a)])).sum() \
            / args.decode_len
        print(f"trace {n}: cache {args.prompt_len}->"
              f"{args.prompt_len + args.decode_len}, row entropy {ent:.2f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    print("done ->", outdir)


if __name__ == "__main__":
    main()
