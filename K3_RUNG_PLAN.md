# NG-KV × Kimi K3: rung plan and pre-registered hypotheses

Status: Rung K3-0 complete (this repo state). K3-1 capture adapter
smoke-tested and cluster-ready; K3-1 execution requires a GPU node.

## Why K3 is not "same experiment, bigger model"

Kimi K3 (and its architectural sibling Kimi Linear 48B) interleave KDA
linear-attention layers (constant-size recurrent state, **no per-token
KV**) with full-attention Gated MLA layers in a 3:1 ratio. Only the MLA
layers hold gateable KV, and that KV is a single 576-dim latent per
token per layer, shared across heads.

Gating surface (from `k3_accounting.py`, configs vendored in
`model_configs/`):

| model            | layers        | surface                  | bf16/token | @1M ctx (bf16) | KDA state/seq |
|------------------|---------------|--------------------------|-----------|----------------|---------------|
| Kimi Linear 48B  | 20 KDA + 7 MLA| 7 × 576 = 4,032 elems    | 7.9 KiB   | 7.9 GB         | 21.4 MB       |
| Kimi K3          | 69 KDA + 24 MLA| 24 × 576 = 13,824 elems | 27.0 KiB  | 27.0 GB        | 221.6 MB      |

Three structural consequences:

1. **Granularity collapse.** MLA heads share one latent, so the
   placement unit is (token, layer). Head-pooled attention rows are the
   *exact* decision granularity — on the GQA rungs head-mean was an
   approximation; here it is the object itself. (The Rung-1.5 result
   that mean-pooled evaluation of a shared gate is mathematically exact
   carries over unchanged.)
2. **The surviving layers are selected for retrieval.** Moonshot keeps
   the 24 MLA layers precisely because KDA's fixed-size state misses
   exact long-range lookups. The full-attention layers in a hybrid are
   therefore not a random sample of attention behaviour — they are the
   layers whose *job* is heavy-hitter and revival traffic.
3. **A new admission object.** KDA state snapshots at prefix boundaries
   (the thing vLLM's redesigned hybrid prefix cache manages) are few
   and large (222 MB/seq on K3) — the inverse cost profile of KV pages.
   Gating *which boundaries deserve a snapshot* is an open problem
   (Rung K3-4).

## Pre-registered hypotheses (fixed before K3-1 capture)

Protocol: identical to Rung 2 (tinyshakespeare disjoint windows,
P=256, D=120, T=0.8, seed 42, N=10) so cross-model comparisons are
protocol-paired. Trace is the unit; paired Wilcoxon + bootstrap within
model; cross-model deltas reported with bootstrap CIs (unpaired at
trace level — different tokenizers mean different windows; report both
raw and row-entropy-normalized comparisons and state this plainly).

- **H1 (concentration).** MLA-layer attention rows in the hybrid are
  more concentrated (lower row entropy, higher top-k mass at fixed k)
  than rows from a pure-attention model under the same protocol
  (Rung-2 SmolLM2 traces as the comparator).
- **H2 (compressibility).** At matched budgets B ∈ {0.1..0.5}, the
  retained-mass frontier is *lower* (harder to compress) on hybrid MLA
  layers, and the gap between the eviction oracle and
  necessity-allocated mixed precision *widens* — i.e. NG-KV's headline
  claim strengthens exactly where the KV that remains matters most.
- **H3 (ordering robustness).** The scorer ordering established in the
  Rung-2 bake-off (window < h2o < tova ≈ ng2-loo < oracle) is
  preserved on hybrid MLA traces even if absolute regret grows.
- **H4 (layer heterogeneity).** Necessity concentration varies more
  across the 7/24 MLA layers than across layers of a pure-attention
  model (early-vs-late retrieval roles), strengthening the case for
  per-layer budgets on hybrids.

Falsification matters: if H2 comes out reversed (hybrid MLA layers
*more* compressible), that is itself a publishable finding — it would
mean KDA absorbs so much local structure that the full-attention
layers see cleaner, more predictable heavy-hitter traffic.

## Rungs

- **K3-0 (done).** Schema extension (`attn_family`, `layer_types`,
  `captured_layers`, `extras`), gating-surface audit
  (`k3_accounting.py` → `results_k3_accounting.json`), capture adapter
  (`capture_traces_kimi_linear.py`) smoke-tested end-to-end against
  Moonshot's real modeling code (`smoke_test_kimi_capture.py`,
  all-MLA tiny model; fla stubbed — KDA kernels are GPU-only).
- **K3-1 (next, GPU node).** Capture 10 traces from
  Kimi-Linear-48B-A3B-Instruct (bf16, single B200 ample; eager
  attention — validation path, not a speed path). Discharges the
  long-pending "real mid-size model traces" debt with a strategically
  relevant model.
- **K3-2.** Full replay + scorer bake-off on the hybrid traces
  (`replay_compare.py` runs unmodified on the mean view;
  `replay_pooling.py` on the per-layer view). Test H1–H4. Addendum v05
  page.
- **K3-3 (live).** SGLang `HiCacheStorage` admission filter in front
  of the real K3 deployment (2-node unified or 1+2 PD-disagg).
  Metrics: storage bytes written, prefix hit rate, TTFT p99. Kills the
  "live validation pending" caveat.
- **K3-4 (design doc first).** Necessity-gated KDA snapshot admission.
  Necessity = P(resume at boundary) × prefill cost saved; cost = 222
  MB/snapshot. Seam is inside vLLM's hybrid prefix cache (physical KDA
  state-block size decoupled from prefix-match granularity), not an
  existing extension point — design before code.

## Cluster runbook (K3-1)

```bash
pip install "transformers==4.57.*" fla-core einops  # GPU node; fla needs triton
python capture_traces_kimi_linear.py \
    --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --out traces_k3_rung1 --n 10
# then:
sed 's|traces_rung2|traces_k3_rung1|' replay_compare.py > replay_compare_k3.py
python replay_compare_k3.py
```

Version notes (`apply_transformers_shims()` handles both, called
automatically in the capture main):
  * transformers ≤ 4.57.6 crashes in `@auto_docstring` on Moonshot's
    PEP-604 annotations → shim replaces it with a no-op (cosmetic).
  * transformers ≥ 5 moved `OutputRecorder` → shim re-exports it.
  * `output_attentions=True` is a DEAD parameter in modeling_kimi.py
    (weights computed and discarded); capture goes through the
    `eager_attention_forward` monkeypatch. If Moonshot restructures
    the modeling code, `MLACapture.__init__` fails loudly.

Weights for K3 itself are at `/mnt/kvbm-cache/kimi-k3`; the same
script runs there (`--model /mnt/kvbm-cache/kimi-k3 --n 6`) but eager
full-precision forward on a 2.8T model is a multi-GPU affair — do
Kimi Linear first, K3 only if the H1/H2 deltas warrant confirming at
scale.
