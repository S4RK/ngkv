# NG-KV — Necessity-Gated KV Cache Management

**A plug-in framework for necessity-gated retention, tiering, and precision
allocation in LLM KV caches — with oracle-regret certification.**

Part of the [Syni](https://syni.world) open research stack. Apache 2.0.
Grounded in SIVAM TR-01, *Necessity-Gated Activation Modulation* (plug-in
regret identity; oracle dominance; precision-of-action corollary).

## The idea

Every KV-cache decision — evict or keep, HBM or host tier, 16 bits or 4 —
is a **necessity gate**: an online estimate of how *vital* a cached token is
to future decoding quality. Existing heuristics (LRU, sliding windows,
H2O-style accumulated attention) are implicit plug-in estimators of that
necessity with no regret story. NG-KV makes the estimator explicit,
generalizes the policies it feeds, and — the part we think matters most —
ships an **evaluation methodology**: replay traces, compute the clairvoyant
oracle gate, and report *plug-in regret*. Near-zero regret at budget B is a
certificate of near-optimality that "our quality metric didn't move" cannot
provide.

## What's in the box

```
ngkv/
  necessity.py    # online scorer: EWMA mass + burstiness (revival) + recency + sinks
  policy.py       # tiered placement (HBM/DRAM/evict) & mixed-precision allocation
  oracle.py       # clairvoyant oracle + per-step plug-in regret
  simulate.py     # structural attention-trace generator (sinks/recency/heavy-hitters/revival)
  adapters/
    hf.py         # working HuggingFace reference adapter (validation, not speed)
    vllm.py       # integration spec: KVConnector admission/prefetch policy
    sglang.py     # integration spec: HiCache admission & eviction ranking
run_simulation.py # full policy comparison with regret, over budgets & seeds
```

## Results (structural simulation, 8 seeds, 2K prompt + 1K decode)

Quality proxy = attention mass recovered per decode step.

| HBM budget B | Window+sinks | H2O | **NG evict** | **NG tier (B+B)** | **NG mixed-prec** | Oracle |
|---|---|---|---|---|---|---|
| 0.10 | 0.481 | 0.553 | **0.701** | **0.881** | **0.899** | 0.968 |
| 0.20 | 0.505 | 0.646 | **0.882** | **0.926** | **0.937** | 0.998 |
| 0.30 | 0.550 | 0.743 | **0.913** | **0.951** | **0.969** | 1.000 |
| 0.40 | 0.608 | 0.830 | **0.926** | **0.975** | **0.997** | 1.000 |

Headlines (with the caveats below):
* At 5× compression (B=0.2), necessity scoring recovers **0.88** of attention
  mass vs. **0.65** for accumulated-mass (H2O-style) scoring — the gap is the
  *revival* regime, where variance-aware necessity anticipates re-attention
  that pure history cannot.
* Mixed precision **dominates eviction at every budget**: smooth gates beat
  hard gates (precision-of-action corollary). At B=0.4 it is within 0.3% of
  the ceiling under stated fidelity assumptions.
* In the tiered (disaggregated) configuration at B=0.2, **60% of cache blocks
  are never shipped to the remote tier** — direct interconnect savings — while
  the DRAM tier serves the mid-necessity mass the HBM tier would have lost.
* Regret quantifies the remaining headroom per family, per budget — the
  quantity a production A/B should track offline on logged traces.

**Stated plainly:** these are structural simulations designed for *relative
policy ordering*, not absolute production claims. The trace generator encodes
documented attention regularities (sinks, recency, heavy hitters, revival) but
is not a replay of real model attention. Quantization fidelity factors are
modeling assumptions calibrated to the KV-quantization literature. Validation
path: (1) real-trace replay via the HF adapter, (2) end-task quality (not mass
proxies) behind quality gates, (3) goodput measurement in the target serving
stack. The regret methodology is unchanged at every stage.

## Quickstart

```bash
pip install numpy
python run_simulation.py          # reproduces the table above -> results.json
```

```python
from ngkv import NecessityScorer, TieredPlacementPolicy, TierBudget

scorer = NecessityScorer()
policy = TieredPlacementPolicy(TierBudget(hbm_frac=0.3, dram_frac=0.3))
for attn_row in decode_steps:          # (cache_len,) attention per step
    scorer.observe(attn_row)
    placement = policy.place(scorer.scores())   # Tier.HBM / DRAM / EVICT
```


## Real-trace results (v0.2)

Rung 1 of the validation ladder, run end-to-end in-repo: a 3L/4H/96d
char-level decoder trained from scratch (JAX, CPU; `train_chunk.py`),
ten held-out prompts decoded with exact attention captured per step
(`capture_traces.py` -> `traces/*.npz`, the `ngkv.traces` schema), and
replayed through the identical policy/oracle/regret pipeline with
paired significance testing (`replay_real.py` -> `results_real.json`).

| B | Window | Accum-mass | **NG evict** | **NG tier** | **NG mixed-prec** | Evict oracle |
|---|---|---|---|---|---|---|
| 0.10 | 0.42 | 0.21 | **0.44** | **0.56** | **0.60** | 0.47 |
| 0.20 | 0.56 | 0.36 | **0.56** | **0.71** | **0.76** | 0.58 |
| 0.40 | 0.70 | 0.57 | **0.71** | **0.91** | **0.99** | 0.72 |

Three findings: (1) the NG-vs-accumulated-mass scoring gap *grows* on
real attention (+0.20 at B=0.2; Wilcoxon p=0.002 at every budget; this
isolates the scoring signal — deployed H2O also keeps a recency
window); (2) diffuse attention collapses the eviction ceiling itself —
the clairvoyant oracle recovers only 0.58 at B=0.2 while eviction-family
regret is 0.01–0.03, so the estimator is near-optimal and the *family*
is the constraint, which is exactly the diagnostic regret exists to
give; (3) mixed precision beats the eviction *oracle* outright under
diffuse attention — the precision-of-action corollary in real model
attention.

**Stated plainly:** one tiny, weakly trained model is a single point in
attention-distribution space; mean-pooling over layers/heads flattens
structure; quality is a mass proxy, not end-task accuracy; n=10 bounds
test power. Next rungs: mid-size open-weight traces, per-layer/block
gating, end-task quality behind gates, then production (Tier-2) traces.

### Pooling sensitivity (Rung 1.5, v0.3)

The mean-pooling caveat above was tested directly: the same ten decodes
were re-captured with per-layer and max-pooled attention preserved
(`capture_traces_perlayer.py` -> `traces_rung15/`, backward-compatible
extra `.npz` keys) and replayed in four views (`replay_pooling.py` ->
`results_rung15.json`). Three results: (1) retained mass is linear in
the attention row, so mean-pooled evaluation of any *shared* gate is
exactly the per-layer average — the v0.2 measurements were unbiased for
shared gates; (2) per-layer gating with layer-local budgets adds a
small, consistent gain (NG-evict +0.004, eviction oracle +0.013 at
B=0.2; p=0.002 at every budget) and accumulated-mass scoring benefits
most, shrinking but not closing the NG gap (+0.20 -> +0.15, still
significant everywhere); (3) the policy ordering is identical in all
four views at every budget — the eviction-family ceiling is set by
diffuse attention, not by the pooling choice, and mixed precision beats
the eviction oracle under every reduction. See page 2 of
`NGKV_real_trace_addendum.pdf`.

### Mid-size open-weight traces (Rung 2, v0.4)

The next rung promised above, run end-to-end in-repo: SmolLM2-360M
(32 layers, 15 heads, fp32, eager attention) decoded the same ten
held-out prompt windows (256-token prompts, 120 steps), with exact
full-cache attention captured per step in all three Rung-1.5 views
(`capture_traces_smollm.py` -> `traces_rung2/`) and replayed through the
identical pipeline (`replay_real_rung2.py` -> `results_rung2.json`;
`replay_pooling_rung2.py` -> `results_rung2_pooling.json`).

| B | Window | Accum-mass | **NG evict** | **NG tier** | **NG mixed-prec** | Evict oracle |
|---|---|---|---|---|---|---|
| 0.10 | 0.77 | 0.58 | **0.77** | **0.84** | **0.86** | 0.81 |
| 0.20 | 0.82 | 0.68 | **0.84** | **0.91** | **0.93** | 0.87 |
| 0.40 | 0.88 | 0.82 | **0.91** | **0.98** | **1.00** | 0.93 |

Four findings: (1) the NG-vs-accumulated-mass gap persists (+0.16 at
B=0.2, Wilcoxon p=0.002 at every budget, 95% CI [+0.151, +0.161]) —
narrower than Rung 1's +0.20 because sharper attention makes every
signal more informative, but all ten traces agree in sign; (2) the
eviction ceiling recovers — the clairvoyant oracle reaches 0.87 at
B=0.2 vs 0.58 on the tiny model, confirming the Rung-1 collapse was a
property of diffuse attention, not of the eviction family, with
NG-evict regret staying small (0.037 at B=0.2); (3) mixed precision
still beats the eviction *oracle* at every budget in every view (0.93
vs 0.87 at B=0.2) — the precision-of-action corollary now spans diffuse
tiny-model attention through concentrated 360M attention; (4) per-layer
gating matters ~3x more at 32 layers than at 3 (NG-evict +0.015, oracle
+0.032 at B=0.2, vs +0.004/+0.013 at Rung 1.5; p=0.002 throughout),
with the headroom concentrated in the oracle and the policy ordering
unchanged in all four views at every budget. See page 3 of
`NGKV_real_trace_addendum.pdf`.

**Stated plainly:** one 360M base model on one corpus is a single point
in attention-distribution space; quality is a mass proxy, not end-task
accuracy; n=10 keeps every Wilcoxon p at the two-sided floor (0.002 —
read as "all ten traces agree in sign"). Next rungs: end-task quality
behind gates, larger trace counts with Diebold–Mariano testing, then
Tier-2 production traces.

### Block granularity + vLLM connector (Rung 2.5, v0.5)

Paged engines act on blocks, not tokens, so the token-level gains above
were re-measured at block granularity (`ngkv.block`, `replay_blocks.py`
-> `results_rung25_blocks.json`): at equal token budget (B=0.2),
NG-evict retained mass is 0.838 at token granularity, 0.815 at 8-token
blocks, 0.785 at 16-token blocks (vLLM's default), 0.595 at 32-token
blocks (all pairwise p=0.002). Read: bs=8 is nearly free, bs=16 costs
~5 points of mass, bs=32 collapses at small budgets. Max-within-block
score pooling is the safe default (never underrates a hot token inside
a cold block) and matches mean-pooling on these traces.

`ngkv/adapters/vllm_connector.py` implements vLLM v1's
`KVConnectorBase_V1` as a necessity-gated offload admission/drop policy
(HBM-tier resident, DRAM-tier saved, EVICT-tier dropped untransferred;
shared ref-count > 1 blocks pinned lossless), unit-tested against
mocked vLLM types, and `ngkv/adapters/sglang_backend.py` binds SGLang's
`HiCacheStorage` L3 seam as a necessity-gated admission filter (admit/
deny only — HiCache pages are shared-by-design, so the boundary rule
forbids lossy actions there). 20 tests total (`tests/`); both adapters
pending live-engine validation.

## Trace schema (production on-ramp)

`ngkv.traces` defines the capture/replay interface: `.npz` with
`attn (D, total)`, `prompt_len`, `meta_json`. Capture attention in any
stack at any granularity (token or block; `meta["granularity"]`),
write the file, and `replay_real.py` produces quality, regret, and
significance without modification.

## Relation to prior work

StreamingLLM (sinks+window), H2O/SnapKV/TOVA (attention-mass eviction),
Quest (block-level query-aware estimates), KIVI/KVQuant (KV quantization).
NG-KV's claims are: (a) these are all plug-in necessity estimators for the
same gate; (b) smooth necessity-allocated precision generalizes and empirically
dominates hard eviction; (c) plug-in regret against the clairvoyant oracle is
the right certification target, and it is computable on logged traces today.

## License

Apache 2.0. SIVAM and NG-KV are open source; see NOTICE.

## Rung K3: hybrid KDA/MLA models (Kimi Linear / Kimi K3)

Hybrid linear-attention models hold gateable KV only in their
full-attention MLA layers (7 of 27 on Kimi Linear 48B; 24 of 93 on
Kimi K3), one shared 576-dim latent per token per layer — head-pooled
gating is exact, not approximate, on this surface. See
`K3_RUNG_PLAN.md` for pre-registered hypotheses, `k3_accounting.py`
for the byte-level audit, `capture_traces_kimi_linear.py` for the
capture adapter (eager_attention_forward seam; `output_attentions`
is discarded upstream), and `smoke_test_kimi_capture.py` for the
CPU end-to-end validation against Moonshot's real modeling code.
