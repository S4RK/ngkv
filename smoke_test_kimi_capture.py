"""Smoke test for capture_traces_kimi_linear.py (CPU, no weights).

Validates the capture path end-to-end against Moonshot's REAL modeling
code (modeling_kimi.py, vendored in kimi_ref/), not a lookalike:

  1. layer-type mapping vs the real 48B and K3 configs (pure JSON);
  2. the eager_attention_forward monkeypatch seam exists and fires;
  3. shapes, padding, row normalization of all three saved views;
  4. schema round-trip through ngkv.traces (save_trace/load_trace_full)
     and a replay sanity pass through NecessityScorer.

CPU constraint: fla-core's KDA kernels are triton/CUDA-only, so the
smoke model is ALL-MLA (kda_layers=[]) — KimiDeltaAttention is never
instantiated and a stub `fla` module satisfies the hard import. The
MLA capture path exercised here is byte-identical to what runs on the
hybrid 48B/K3 on GPU; the KDA layers simply don't route through
eager_attention_forward there.

Run: python smoke_test_kimi_capture.py
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Stub fla BEFORE importing modeling_kimi (hard import, kernels unused
#    because no KDA layer is instantiated in the all-MLA smoke config).
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

_never = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("fla stub called — KDA layer instantiated in smoke test?"))
_stub("fla")
_stub("fla.modules", FusedRMSNormGated=_never, ShortConvolution=_never)
_stub("fla.ops")
_stub("fla.ops.kda", chunk_kda=_never, fused_recurrent_kda=_never)
_stub("fla.ops.kda.gate", fused_kda_gate=_never)
_stub("fla.ops.utils")
_stub("fla.ops.utils.index", prepare_cu_seqlens_from_mask=_never,
      prepare_lens_from_mask=_never)
_stub("fla.utils", tensor_cache=lambda f: f)

from capture_traces_kimi_linear import apply_transformers_shims  # noqa: E402
apply_transformers_shims()

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "vendor"))
from kimi_ref.configuration_kimi import KimiLinearConfig          # noqa: E402
from kimi_ref.modeling_kimi import KimiLinearForCausalLM          # noqa: E402

from capture_traces_kimi_linear import (capture_trace,            # noqa: E402
                                        layer_types_from_config)
from ngkv import NecessityScorer, retained_mass                   # noqa: E402
from ngkv.traces import load_trace_full, save_trace               # noqa: E402

P, D, N_LAYERS = 32, 16, 4
ok = lambda msg: print(f"  [ok] {msg}")

# ---------------------------------------------------------------------------
# 2. Layer mapping vs the real configs (no torch involved)
# ---------------------------------------------------------------------------
print("== layer-type mapping vs real configs ==")
for path, want_mla in [("model_configs/kimi_linear_48b_config.json",
                        [3, 7, 11, 15, 19, 23, 26]),
                       ("model_configs/kimi_k3_config.json",
                        [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
                         51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92])]:
    raw = json.load(open(path))
    cfg = KimiLinearConfig(**{k: v for k, v in
                              raw.get("text_config", raw).items()
                              if k in KimiLinearConfig.__init__.__code__
                              .co_varnames})
    types_, mla = layer_types_from_config(cfg)
    assert mla == want_mla, (path, mla)
    assert len(types_) == cfg.num_hidden_layers
    assert all(types_[i] == "mla" for i in mla)
    ok(f"{path}: {types_.count('kda')} KDA + {len(mla)} MLA, "
       f"0-indexed MLA = {mla[:4]}...{mla[-2:]}")

# ---------------------------------------------------------------------------
# 3. Tiny all-MLA model through the real modeling code
# ---------------------------------------------------------------------------
print("== tiny all-MLA model (real modeling_kimi.py, random weights) ==")
torch.manual_seed(0)
tiny = KimiLinearConfig(
    vocab_size=512, hidden_size=96, intermediate_size=192,
    num_hidden_layers=N_LAYERS, num_attention_heads=4,
    q_lora_rank=None, kv_lora_rank=32, qk_nope_head_dim=24,
    qk_rope_head_dim=8, v_head_dim=24, mla_use_nope=True,
    linear_attn_config={"kda_layers": [],
                        "full_attn_layers": list(range(1, N_LAYERS + 1)),
                        "num_heads": 4, "head_dim": 24,
                        "short_conv_kernel_size": 4},
    num_experts=None, attn_implementation="eager")
model = KimiLinearForCausalLM(tiny).eval()
# transformers 5.x may override the init-time request (auto-selects FA2);
# force eager so eager_attention_forward is the live seam.
# (set_attn_implementation warns-and-ignores for this custom class;
# KimiMLAAttention reads config._attn_implementation at forward time,
# so setting the attribute directly is sufficient and exact.)
model.config._attn_implementation = "eager"
types_, mla = layer_types_from_config(model.config)
assert mla == list(range(N_LAYERS))
ok(f"instantiated: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
   f"params, {N_LAYERS} MLA layers, latent dim "
   f"{tiny.kv_lora_rank + tiny.qk_rope_head_dim}")

ids = torch.randint(0, 512, (1, P))
gen = torch.Generator(); gen.manual_seed(0)
views = capture_trace(model, ids, D, mla, temperature=0.8, generator=gen)

total = P + D
assert views["attn"].shape == (D, total)
assert views["attn_layers"].shape == (len(mla), D, total)
assert views["attn_max"].shape == (D, total)
ok(f"shapes: attn {views['attn'].shape}, "
   f"attn_layers {views['attn_layers'].shape}")

for t in range(D):
    cl = P + t
    for name in ("attn", "attn_max"):
        row = views[name][t]
        assert abs(row[:cl].sum() - 1.0) < 1e-9, (name, t)
        assert row[cl:].max() == 0.0, (name, t, "padding not zero")
    rl = views["attn_layers"][:, t, :]
    assert np.allclose(rl[:, :cl].sum(axis=1), 1.0, atol=1e-9)
    assert rl[:, cl:].max() == 0.0
ok("row sums == 1 over live cache, zero padding beyond, all views")

per_layer = views["attn_layers"].mean(axis=0)
per_layer /= per_layer.sum(axis=1, keepdims=True)
recon = views["attn"] / views["attn"].sum(axis=1, keepdims=True)
assert np.allclose(per_layer, recon, atol=1e-6)
ok("mean view == renormalized mean of per-layer view (linear identity)")

# ---------------------------------------------------------------------------
# 4. Schema round-trip + replay sanity
# ---------------------------------------------------------------------------
print("== schema round-trip + replay ==")
meta = {"attn_family": "mla_latent", "layer_types": types_,
        "captured_layers": mla, "granularity": 1}
save_trace("/tmp/smoke_trace.npz", views["attn"], P, meta,
           extras={"attn_layers": views["attn_layers"],
                   "attn_max": views["attn_max"]})
attn, P2, meta2, extras = load_trace_full("/tmp/smoke_trace.npz")
assert P2 == P and meta2["attn_family"] == "mla_latent"
assert extras["attn_layers"].shape == (len(mla), D, total)
ok("save_trace(extras=...) / load_trace_full round-trip")

scorer = NecessityScorer()
kept_mass = []
for t in range(D):
    cl = P + t
    row = attn[t, :cl]
    if t > 0:
        k = max(4, int(0.3 * cl))
        keep_idx = np.argsort(scorer.scores()[:cl])[-k:]
        w = np.zeros(cl); w[keep_idx] = 1.0
        kept_mass.append(retained_mass(row, w))
    scorer.observe(row)
kept_mass = float(np.mean(kept_mass))
assert 0.0 < kept_mass <= 1.0
ok(f"NecessityScorer replay on captured trace: retained mass @B=0.3 "
   f"= {kept_mass:.3f}")

print("\nALL SMOKE CHECKS PASSED — capture adapter is cluster-ready.")
