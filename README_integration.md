# NG-KV — Necessity-Gated KV Cache Compression

**Model-agnostic KV cache compression for LLM serving.** NG-KV scores every cached
entry by an online estimate of *future necessity* — will the model attend to this
again? — and spends memory in proportion to that estimate: full precision for
what matters, fewer bits or a colder tier for what might, nothing for what won't.

Part of the [Syni](https://syni.world) open research stack. Apache 2.0.
Grounded in SIVAM TR-01 ("Necessity-Gated Activation Modulation").

```
pip install numpy            # core has a single dependency
python run_simulation.py     # reproduce the headline table -> results.json
```

---

## Why this exists

Decode-phase serving is KV-memory-bound: every in-flight request holds cache
proportional to its context, and your batch size — hence throughput, hence
goodput under an SLA — is set by how many of those caches fit in HBM. Today's
engine defaults (vLLM, SGLang) do **no selective retention inside a live
request**: every token's KV stays at full precision until the request ends.
Prefix caching removes *repeated* work across requests; NG-KV removes
*unnecessary* residency within one.

The two are complementary, and the industry is drifting toward NG-KV's
territory: reasoning models and agentic loops generate thousands of decode
tokens that are private by construction (never reused as anyone's prefix),
briefly attended, and resident for the whole generation. That is exactly the
mass NG-KV compresses.

**Measured (in-repo, replayable):** ~4–5× KV compression at 93% retained
attention mass (B=0.2) on real SmolLM2-360M traces, with necessity scoring
beating accumulated-mass (H2O-style) scoring at every budget (paired Wilcoxon
p=0.002, n=10 traces) and necessity-allocated mixed precision beating the
*clairvoyant eviction oracle* outright.

**Projected (mechanism, not yet certified end-to-end):** in KV-bound
deployments, 3–4× effective compression of private KV translates to roughly
2–4× decode concurrency at fixed HBM — the goodput-vs-load curve shifts right,
with fewer preemptions under bursty load. Two links in that chain are still
open and we say so plainly: attention-mass retention is a proxy for end-task
quality (validation Rung 3), and the scorer's input signal requires a proxy on
fused-kernel serving paths (see *Production signal paths* below).

### Where it helps most

| Workload | Why |
|---|---|
| Long-generation (reasoning, CoT, agents) | Decode KV dominates footprint and is 100% private — full NG-KV authority, zero prefix-cache conflict |
| Long-context one-shot (document analysis) | Large prompt KV that will never be re-shared |
| High-concurrency chat | Shared system prompts stay lossless; each user's private tail gets compressed |

Where it rounds to zero: short contexts, short generations, compute-bound
regimes. Mixed precision saves memory, not attention FLOPs — the win is
concurrency, not single-stream latency.

---

## How it works

Three small, framework-agnostic components. No model, no training, no weights.

```python
from ngkv import (NecessityScorer, NecessityConfig,
                  TieredPlacementPolicy, TierBudget,
                  MixedPrecisionPolicy, Tier)

scorer = NecessityScorer(NecessityConfig(sink_tokens=4))
tiers  = TieredPlacementPolicy(TierBudget(hbm_frac=0.3, dram_frac=0.3))
mp     = MixedPrecisionPolicy(bit_budget_frac=0.3)

for attn_row in decode_steps:            # (cache_len,) attention per step
    scorer.observe(attn_row)             # O(T) state update
    s = scorer.scores()
    placement = tiers.place(s)           # Tier.HBM / Tier.DRAM / Tier.EVICT
    bits      = mp.allocate(s)           # per-entry bit-width under budget
```

1. **`NecessityScorer`** — consumes per-step attention (however your stack
   exposes it) and maintains a running per-entry estimate of future need:
   recency-weighted attention plus structural priors (attention sinks), not a
   lifetime tally. This is the difference from H2O-style accumulated mass — an
   estimator of *future* use, not a record of past use.
2. **Policies** — act on the scores. `TieredPlacementPolicy` is the hard gate
   (HBM / DRAM / evict under capacity budgets — the disaggregated-serving
   shape). `MixedPrecisionPolicy` is the smooth gate: bit-width proportional to
   necessity rank. The smooth gate dominates empirically — it can beat the
   eviction *oracle*, which no hard-eviction method can do by construction.
3. **Oracle + regret** (`ngkv.oracle`) — the certification layer. Logged
   attention reveals ground-truth necessity after the fact; replaying your
   deployed scorer against the clairvoyant oracle yields *plug-in regret*, the
   number that separates "the estimator is bad" from "the policy family is the
   constraint." This is what makes the compression trustworthy enough to ship,
   and it runs on logged traces today.

## Model-agnostic by construction

Nothing in the scorer or policies references any architecture. The interface
between *any* model or serving stack and NG-KV is a portable trace schema
(`ngkv.traces`): an `.npz` per request with `attn (D, total)`, `prompt_len`,
and free-form `meta_json` (model id, reduction, granularity — token or block).
Capture attention wherever and however your stack allows, write the file, and
the full policy / oracle / regret / significance pipeline runs unmodified.
The same replay entry point has been run against a 3-layer toy decoder and
a 32-layer SmolLM2-360M without a line of change; Tier-2 production traces
use the identical path.

Granularity is a parameter, not an assumption (`ngkv.block`): score
fixed-size *blocks* instead of tokens and the gate lands on exactly the unit
paged memory managers already move.

```python
from ngkv import BlockGate, BlockGateConfig, TierBudget

gate = BlockGate(BlockGateConfig(block_size=16, reduce="max"))
for attn_row in decode_steps:
    gate.observe(attn_row)
placement = gate.place_blocks(cache_len, TierBudget(0.3, 0.3),
                              shared_mask=engine_shared_blocks)  # boundary rule
```

**Measured cost of block granularity** (Rung 2.5, replay on the same
SmolLM2-360M traces at equal token budget; `replay_blocks.py` ->
`results_rung25_blocks.json`): NG-evict retained mass at B=0.2 goes
0.838 (token) -> 0.815 (8-token blocks) -> 0.785 (16-token, vLLM default)
-> 0.595 (32-token); all differences p=0.002. Practical read: 8-token blocks
are nearly free, 16 costs ~5 points of mass, 32 collapses at small budgets
(too few blocks to cover both sinks and recency). Max-within-block pooling —
a block is as necessary as its hottest token — is the safe default and costs
the same as mean on these traces. Coarser granularity weakens the regret
bound; it never invalidates it — and now you can read the price off a table.

## Integrating

### Reference path: HuggingFace (works today)

```python
from ngkv.adapters.hf import NGKVController

ctrl = NGKVController(model, hbm_frac=0.3, dram_frac=0.3)
out  = ctrl.generate(input_ids, max_new_tokens=512)
```

Step-wise generation with `output_attentions=True`, live cache rewriting
between steps. **This is a validation path, not a speed path** — attention
output disables fused kernels. Use it to certify policies on your model's real
attention before deploying proxies.

### Production path: vLLM connector (implemented)

`ngkv/adapters/vllm_connector.py` implements vLLM v1's `KVConnectorBase_V1`
as a necessity-gated **admission and drop policy** for KV offloading, with
the scheduler/worker split the interface requires:

- Scheduler side tiers each finished/preempted request's blocks by necessity
  under a byte budget: HBM-tier stays resident, DRAM-tier is saved to the
  offload store, EVICT-tier is **dropped rather than transferred** — the
  direct interconnect-bytes win in disaggregated P/D deployments. Shared
  blocks (ref-count > 1) are pinned unconditionally: the prefix-cache
  boundary rule, enforced in code, tested.
- Worker side saves only admitted blocks (`save_kv_layer` filters by the
  scheduler's plan); storage I/O is a single seam (`_store_blocks`) that
  binds to your offload store (LMCache / PFS-backed).
- Necessity signal is pluggable (`NecessityProvider`): `AttentionTapProvider`
  for eager/validation runs, `RecencyHeuristicProvider` as the kernel-free
  floor. Quest-style key-statistics providers are the intended production
  upgrade; certify whichever you deploy via trace replay first.

```python
KVTransferConfig(kv_connector="NGKVConnector",
                 kv_connector_module_path="ngkv.adapters.vllm_connector",
                 kv_role="kv_both",
                 kv_connector_extra_config={"block_budget_frac": 0.3,
                                            "block_size": 16})
```

**Status — stated plainly:** interface-complete and unit-tested against
mocked vLLM types (`tests/test_ngkv_vllm.py`, 14 tests: budget enforcement,
shared-block pinning, drop accounting, worker save filtering). Pending
validation against a live GPU vLLM build — the connector is an offload-only
role and composes with hit-providing connectors (LMCache, Nixl) via
MultiConnector. This sandbox cannot run vLLM; treat the first live
integration as a required step, not a formality.

### Production path: SGLang HiCache backend (implemented)

`ngkv/adapters/sglang_backend.py` binds NG-KV at SGLang's pluggable L3
seam (`HiCacheStorage`, vendor-loadable via dynamic dispatch — no SGLang
patches) as a **necessity-gated admission filter**: it wraps any real
backend (Mooncake, HF3FS, NIXL, file) and decides which pages are worth
writing at all. Denied pages are dropped before transfer.

The boundary rule inverts here, deliberately: HiCache L3 is a prefix-reuse
store, so every page in it is shared-by-design. Shared => lossless,
therefore this seam performs **no quantization and no demotion — the only
gate action is admit/deny**. Denial is always safe (worst case a future
recompute); degrading a page many future requests will hit never is. Note
"necessity" here means *future reuse value across requests*, a different
quantity from within-request attention necessity — attention scores are a
proxy for it (pushable via `TableScorer`), `PositionScorer` (prefix depth,
sink page pinned) is the kernel-free floor, and SGLang's own
`write_through_selective` hit-count policy is prior art for gating this
exact decision. Certify against your reuse traces with "reused later" as
the oracle label.

```
--hicache-storage-backend dynamic
--hicache-storage-backend-extra-config '{
    "module_path": "ngkv.adapters.sglang_backend",
    "class_name": "NGKVFilteredStorage",
    "inner_backend": "mooncake", "scorer": "position", "admit_frac": 0.6}'
```

**Status:** interface-complete against the documented HiCacheStorage
surface, unit-tested with a mock inner backend (6 tests incl. zero-copy
buffer co-filtering); pending live-SGLang validation.

### The one integration rule: respect the prefix-cache boundary

NG-KV must not touch shared, reusable KV. Prefix caching is lossless by
contract — quantizing or evicting inside a shared prefix block either breaks
reuse or serves degraded KV to every request that hits it. The boundary is
metadata the engine already maintains: **ref-count > 1 (vLLM) or radix-path
fan-out (SGLang) ⇒ hands off.** The adapter enforces this as a mask before the
gate acts — no candidacy predictor needed. Be conservative with prompt-derived
blocks that *could* become shared later (multi-turn forks); decode-tail KV is
almost never re-prefixed and is safe territory.

### Production signal paths

FlashAttention-class kernels never materialize attention weights, so the exact
signal the reference scorer consumes is unavailable on the fast path. Deployed
scorers substitute proxies: block-level query-key statistics (Quest-style),
cheap partial reductions, or periodically sampled exact steps. Every proxy
degrades the estimate by some amount — and the regret pipeline measures exactly
how much: capture exact attention offline, run the proxy scorer against it,
read the gap. Certify before you ship.

## Results

Real-trace replay, SmolLM2-360M (32 layers), 10 held-out decodes, retained
attention mass at keep-budget B (full tables and significance in
`results_rung2.json` / `results_rung2_pooling.json`; methodology in
`NGKV_real_trace_addendum.pdf`):

| B | Window | Accum-mass | NG evict | NG tier | NG mixed-prec | Evict oracle |
|---|---|---|---|---|---|---|
| 0.10 | 0.77 | 0.58 | **0.77** | **0.84** | **0.86** | 0.81 |
| 0.20 | 0.82 | 0.68 | **0.84** | **0.91** | **0.93** | 0.87 |
| 0.40 | 0.88 | 0.82 | **0.91** | **0.98** | **1.00** | 0.93 |

Per-layer gating with layer-local budgets adds a further consistent gain that
grows with depth (~3× larger at 32 layers than at 3), with policy ordering
unchanged across all pooling views.

## Stated plainly (limitations)

Quality above is an attention-mass proxy, not end-task accuracy — that bridge
(Rung 3) is not yet built, and you should not deploy aggressive budgets without
quality gates on your own tasks. Traces so far cover two models and one corpus;
n=10 traces puts every Wilcoxon p at the two-sided floor (read p=0.002 as "all
ten traces agree in sign"). History-based scorers share a failure mode: tokens
dormant for long spans and then revived. The variance term mitigates, not
eliminates, this. The HF adapter is deliberately slow; production speed
requires the proxy signal paths above.

## Relation to prior work

StreamingLLM (sinks+window), H2O / SnapKV / TOVA (attention-mass eviction),
Quest (block-level query-aware estimates), KIVI / KVQuant (KV quantization).
The components are known. NG-KV's claims are: (a) these are all plug-in
necessity estimators for the same gate, and become comparable under it;
(b) smooth necessity-allocated precision generalizes and empirically dominates
hard eviction — including the eviction oracle; (c) plug-in regret against the
clairvoyant oracle is the right certification target, and it is computable on
your logged traces today.

## License

Apache 2.0. SIVAM and NG-KV are open source; see NOTICE.
