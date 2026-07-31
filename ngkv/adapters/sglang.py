"""
SGLang integration specification (design doc as code).

SGLang's HiCache manages a hierarchical KV store (device / host / disk)
over a radix tree of shared prefixes. ngkv binds as the *ranking
function* for two existing decision points:

1. **Write-back / admission**: which device blocks to copy to host.
   Replace write-through-everything with necessity-gated admission:
   only blocks whose necessity exceeds the transfer-cost-adjusted
   threshold are admitted to the host tier. Necessity of a radix node
   aggregates over all requests that traversed it (shared prefixes get
   naturally high scores — this is why prefix caching works, and ngkv
   makes the implicit ranking explicit).

2. **Eviction ordering** within each tier: necessity-ranked instead of
   LRU/LFU. LFU is the degenerate case of ngkv scoring with
   variance_weight=0, recency_weight=0 — the generalization is strict.

Measurement plan (maps to an existing A/B framework with quality
gates): primary metric = SLA-goodput; guardrails = per-request quality
gate; diagnostics = effective hit rate, bytes moved per output token,
TTFT p50/p99 on hit and miss paths, and offline plug-in regret on
replayed traces (ngkv.oracle).
"""
