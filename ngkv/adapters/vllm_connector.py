"""
NG-KV connector for vLLM v1 (KVConnectorBase_V1).

Necessity-gated *admission and drop policy* for KV offloading: instead
of treating every finished/preempted block as equally worth saving, the
connector scores blocks with an ``ngkv.block.BlockGate`` and

  * HBM-tier blocks   -> never offloaded (stay resident),
  * DRAM-tier blocks  -> saved to the offload store,
  * EVICT-tier blocks -> dropped, not transferred (interconnect-bytes
                         win in disaggregated P/D deployments),

while **shared blocks (ref-count > 1) are always pinned**: they are
prefix-cache candidates and must stay lossless — the boundary rule.

Architecture (mirrors vLLM's scheduler/worker split):

  Scheduler side: tracks per-request block ids and shared-ness, pulls
  per-block necessity from a ``NecessityProvider``, builds
  ``NGKVConnectorMetadata`` mapping each request to save/drop block
  lists. Worker side: consumes the metadata, saving only admitted
  blocks via the configured store and skipping dropped ones.

Necessity signal — stated plainly: fused attention kernels do not
materialize attention weights, so the reference token-level scorer's
input is unavailable on the fast path. The connector therefore accepts
any ``NecessityProvider``; two are included:

  * ``AttentionTapProvider`` — for eager/validation runs where per-step
    attention rows are available (feeds a real BlockGate).
  * ``RecencyHeuristicProvider`` — kernel-free fallback: sinks + recency
    with exponential decay by block age. This is the *floor*, roughly
    the "window" baseline in the replay results, and exists so the
    connector is runnable everywhere; Quest-style key-statistics
    providers are the intended production upgrade.

The plug-in regret of whichever provider you deploy is measurable
offline against captured traces (see replay_blocks.py) — certify
before shipping.

Wire-up:

    from vllm.config import KVTransferConfig
    KVTransferConfig(
        kv_connector="NGKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="ngkv.adapters.vllm_connector",
        kv_connector_extra_config={"block_budget_frac": 0.3,
                                    "block_size": 16},
    )

Status: interface-complete and unit-tested against mocked vLLM types
(tests/test_vllm_connector.py); pending integration validation on a
live vLLM build — the storage I/O delegates to vLLM's offloading
machinery and is exercised only by mocks here.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ngkv.block import BlockGate, BlockGateConfig
from ngkv.necessity import NecessityConfig
from ngkv.policy import Tier, TierBudget

try:  # real vLLM present
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
    _HAVE_VLLM = True
except ImportError:  # standalone / test environment
    _HAVE_VLLM = False

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorRole:  # type: ignore[no-redef]
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorBase_V1:  # type: ignore[no-redef]
        def __init__(self, vllm_config: Any, role: Any,
                     kv_cache_config: Any = None) -> None:
            self.vllm_config = vllm_config
            self.role = role

if TYPE_CHECKING:  # only for annotations; never imported at runtime
    from vllm.v1.core.sched.output import SchedulerOutput


# --------------------------------------------------------------------------
# Necessity providers
# --------------------------------------------------------------------------

class NecessityProvider:
    """Maps (request_id, n_blocks) -> per-block necessity scores."""

    def block_scores(self, request_id: str, n_blocks: int) -> np.ndarray:
        raise NotImplementedError


class RecencyHeuristicProvider(NecessityProvider):
    """Kernel-free floor: sink block pinned, exponential decay by age."""

    def __init__(self, decay: float = 0.85) -> None:
        self.decay = decay

    def block_scores(self, request_id: str, n_blocks: int) -> np.ndarray:
        if n_blocks == 0:
            return np.zeros(0)
        age = np.arange(n_blocks)[::-1].astype(np.float64)  # newest = 0
        s = self.decay ** age
        s[0] = np.inf  # sink block
        return s


class AttentionTapProvider(NecessityProvider):
    """Feeds observed attention rows into a real BlockGate per request.

    For eager/validation deployments (or sampled exact steps). Call
    ``observe(request_id, attn_row)`` from wherever attention is
    materialized; scores reflect all observations so far.
    """

    def __init__(self, cfg: Optional[BlockGateConfig] = None) -> None:
        self.cfg = cfg or BlockGateConfig()
        self._gates: Dict[str, BlockGate] = {}

    def observe(self, request_id: str, attn_row: np.ndarray) -> None:
        gate = self._gates.setdefault(request_id, BlockGate(self.cfg))
        gate.observe(attn_row)

    def block_scores(self, request_id: str, n_blocks: int) -> np.ndarray:
        gate = self._gates.get(request_id)
        if gate is None:
            return RecencyHeuristicProvider().block_scores(request_id, n_blocks)
        return gate.block_scores(n_blocks * self.cfg.block_size)

    def drop(self, request_id: str) -> None:
        self._gates.pop(request_id, None)


# --------------------------------------------------------------------------
# Metadata (scheduler -> worker)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class NGKVRequestPlan:
    save_block_ids: List[int]
    drop_block_ids: List[int]
    pinned_block_ids: List[int]  # shared / ref-count > 1: never touched


@dataclasses.dataclass
class NGKVConnectorMetadata(KVConnectorMetadata):
    plans: Dict[str, NGKVRequestPlan] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------
# Connector
# --------------------------------------------------------------------------

class NGKVConnector(KVConnectorBase_V1):
    """Necessity-gated offload admission connector (offload-only role).

    This connector does not provide remote prefix hits
    (``get_num_new_matched_tokens`` reports none); it governs *what is
    worth saving or shipping* when blocks leave the GPU. Compose with a
    hit-providing connector (LMCache, Nixl) via MultiConnector for full
    P/D deployments.
    """

    def __init__(self, vllm_config: Any, role: Any = None,
                 kv_cache_config: Any = None,
                 provider: Optional[NecessityProvider] = None) -> None:
        if _HAVE_VLLM:
            super().__init__(vllm_config, role, kv_cache_config)
        else:
            super().__init__(vllm_config, role, kv_cache_config)
        extra = {}
        try:
            extra = dict(vllm_config.kv_transfer_config.kv_connector_extra_config)
        except AttributeError:
            pass
        self.block_size = int(extra.get("block_size", 16))
        self.budget = TierBudget(
            hbm_frac=float(extra.get("block_budget_frac", 0.30)),
            dram_frac=float(extra.get("dram_budget_frac", 0.30)))
        self.provider = provider or RecencyHeuristicProvider()
        # scheduler-side state
        self._req_blocks: Dict[str, List[int]] = defaultdict(list)
        self._block_ref: Dict[int, int] = defaultdict(int)
        # worker-side state
        self._meta: Optional[NGKVConnectorMetadata] = None
        self._saved: Set[str] = set()

    # ---------------- scheduler side ----------------

    def get_num_new_matched_tokens(self, request: Any,
                                   num_computed_tokens: int
                                   ) -> Tuple[int, bool]:
        """Offload-admission connector: provides no remote hits.

        Side-effect free per the interface contract.
        """
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any,
                                 num_external_tokens: int) -> None:
        req_id = getattr(request, "request_id", str(id(request)))
        ids = list(getattr(blocks, "get_block_ids", lambda: blocks)())
        if ids and isinstance(ids[0], (list, tuple)):  # per-group nesting
            ids = list(ids[0])
        self._req_blocks[req_id] = ids
        for b in ids:
            self._block_ref[b] += 1

    def update_connector_output(self, connector_output: Any) -> None:
        pass  # no async transfer bookkeeping in the offload-only role

    def plan_for_request(self, req_id: str) -> NGKVRequestPlan:
        """Tier the request's blocks by necessity under the byte budget,
        pinning shared blocks (ref-count > 1) — the prefix-cache rule."""
        ids = self._req_blocks.get(req_id, [])
        nb = len(ids)
        if nb == 0:
            return NGKVRequestPlan([], [], [])
        scores = self.provider.block_scores(req_id, nb)
        shared = np.array([self._block_ref[b] > 1 for b in ids])
        from ngkv.policy import TieredPlacementPolicy
        placement = TieredPlacementPolicy(self.budget).place(scores)
        placement = placement.copy()
        placement[shared] = Tier.HBM
        save = [ids[i] for i in range(nb) if placement[i] == Tier.DRAM]
        drop = [ids[i] for i in range(nb)
                if placement[i] == Tier.EVICT and not shared[i]]
        pinned = [ids[i] for i in range(nb) if shared[i]]
        return NGKVRequestPlan(save, drop, pinned)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput"
                             ) -> NGKVConnectorMetadata:
        meta = NGKVConnectorMetadata()
        finished = getattr(scheduler_output, "finished_req_ids", None) or []
        for req_id in finished:
            meta.plans[req_id] = self.plan_for_request(req_id)
        return meta

    def request_finished(self, request: Any, block_ids: Any
                         ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Blocks may be freed immediately; the worker save (if any) is
        driven by the metadata built this step. Emits the drop count so
        deployments can log interconnect bytes *not* shipped."""
        req_id = getattr(request, "request_id", str(id(request)))
        plan = self.plan_for_request(req_id)
        n_req_blocks = len(self._req_blocks.get(req_id, []))
        for b in self._req_blocks.pop(req_id, []):
            self._block_ref[b] -= 1
            if self._block_ref[b] <= 0:
                del self._block_ref[b]
        if isinstance(self.provider, AttentionTapProvider):
            self.provider.drop(req_id)
        n_total = (len(plan.save_block_ids) + len(plan.drop_block_ids)
                   + len(plan.pinned_block_ids))
        params = {"ngkv_saved_blocks": len(plan.save_block_ids),
                  "ngkv_dropped_blocks": len(plan.drop_block_ids),
                  "ngkv_pinned_blocks": len(plan.pinned_block_ids),
                  "ngkv_hbm_resident_blocks":
                      n_req_blocks - n_total}
        return False, params

    # ---------------- worker side ----------------

    def register_kv_caches(self, kv_caches: Any) -> None:
        self._kv_caches = kv_caches

    def bind_connector_metadata(self, metadata: NGKVConnectorMetadata) -> None:
        self._meta = metadata

    def clear_connector_metadata(self) -> None:
        self._meta = None

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass  # offload-only role: no loads initiated here

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: Any,
                      attn_metadata: Any, **kwargs: Any) -> None:
        """Save only admitted (DRAM-tier) blocks; dropped blocks are
        skipped entirely — the direct interconnect-bytes win."""
        if self._meta is None:
            return
        for req_id, plan in self._meta.plans.items():
            if plan.save_block_ids:
                self._store_blocks(req_id, layer_name, kv_layer,
                                   plan.save_block_ids)
            self._saved.add(req_id)

    def _store_blocks(self, req_id: str, layer_name: str, kv_layer: Any,
                      block_ids: List[int]) -> None:
        """Storage I/O seam. Default: no-op recorder (tests observe it).
        Production: bind to the deployment's offload store (LMCache /
        PFS-backed) here; only the *selection* is NG-KV's job."""
        rec = getattr(self, "recorded_saves", None)
        if rec is None:
            rec = self.recorded_saves = []
        rec.append((req_id, layer_name, tuple(block_ids)))

    def wait_for_save(self) -> None:
        pass  # synchronous no-op store

    def get_finished(self, finished_req_ids: Set[str]
                     ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        done = self._saved & set(finished_req_ids) if finished_req_ids else set()
        self._saved -= done
        return (done or None), None
