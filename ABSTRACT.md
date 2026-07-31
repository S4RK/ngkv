# NG-KV: Necessity-Gated KV Cache Management — Abstract

**Claim.** Every KV-cache decision in LLM serving — evict or keep, GPU or
host tier, 16 bits or 4 — is an implicit *necessity gate*: an online
estimate of how vital a cached token is to future decoding quality.
Public methods each hard-code one estimator and one action — StreamingLLM
(recency + sinks), H2O/SnapKV/TOVA (accumulated attention, eviction),
KIVI/KVQuant (uniform precision reduction) — and validate with fixed
benchmarks. NG-KV makes three additions: (1) a composite necessity score
(attention mass + burstiness + recency + pinned sinks) of which those
heuristics are degenerate cases; (2) smooth, necessity-*allocated* action —
tiered HBM/DRAM placement and per-token mixed precision — in place of
binary eviction; and (3) a certification methodology absent from the
public literature: replay logged traces, compute the clairvoyant oracle
gate, and report *plug-in regret*, which separates estimator error from
policy-class error on your own workload rather than someone else's benchmark.

**Experiment.** Two rungs. Structural simulation (8 seeds, sink/recency/
heavy-hitter/revival regimes): necessity scoring recovers 0.88 of
attention mass at 5x compression vs. 0.65 for accumulated-mass scoring;
mixed precision dominates eviction at every budget. Real traces (10
decodes from a from-scratch decoder, exact attention captured to a
portable .npz schema): the scoring gap grows to +0.20 (paired Wilcoxon
p=0.002 at every budget), eviction-family regret of 0.01–0.03 shows the
estimator is near-optimal and the *family* is the constraint, and mixed
precision beats the eviction *oracle* outright under diffuse attention.

**Impact and use.** The regret methodology is deployable today against
any serving stack: capture block-level attention aggregates from a live
GLM/DeepSeek/Qwen-class deployment into the trace schema, replay
offline, and receive quality/regret/significance for every policy at
every budget *before* touching production — then bind the same scorer as
a vLLM KVConnector admission/prefetch policy or SGLang HiCache ranking.
Stated plainly: reported numbers are attention-mass proxies from small-
scale traces; the pipeline exists precisely so operators can replace
them with their own. Apache 2.0; grounded in SIVAM TR-01.
