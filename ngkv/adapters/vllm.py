"""
vLLM integration specification (design doc as code).

vLLM v1 exposes two integration surfaces relevant to ngkv:

1. **KVConnector API** (kv_transfer) — used for disaggregated
   prefill/decode and external KV stores (LMCache, PFS-backed stores).
   ngkv binds here as an *admission and prefetch policy*:

     - ``should_offload(block) -> Tier``: score each 16-token block as
       the max necessity of its tokens; HBM-tier blocks are never
       offloaded, DRAM-tier blocks are pushed to the host/remote store,
       EVICT-tier blocks are dropped rather than transferred (saving
       interconnect bytes — in a 4P2D EFA deployment this is the direct
       win: do not ship blocks that will not be re-attended).
     - ``prefetch_order(blocks)``: on cache-hit reconstruction, fetch
       blocks in descending necessity so decode can start before the
       tail arrives.

2. **KV cache events / block manager hooks** — for within-GPU
   eviction ordering, replacing pure-LRU prefix-cache eviction with
   necessity-ranked eviction.

Necessity signals available without materializing attention:
  - Quest-style per-block key min/max statistics against the current
    query (cheap upper bound on block attention),
  - accumulated block hit counts from prior requests sharing the
    prefix (radix/prefix-cache metadata),
  - MoE router entropy of the tokens in the block (necessity of the
    *step* correlates with router uncertainty).

This module intentionally contains no runnable vLLM code: the connector
ABI is still moving between minor versions. Pin a version, then
implement `NGKVConnector(KVConnectorBase_V1)` with the four methods
above. The policy/scorer objects from ngkv core drop in unchanged.
"""
