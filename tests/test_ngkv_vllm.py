"""Unit tests: ngkv.block and the vLLM connector (mocked vLLM types).

Run: python -m pytest tests/ -q
"""

import dataclasses

import numpy as np
import pytest

from ngkv.block import (BlockGate, BlockGateConfig, expand_from_blocks,
                        n_blocks, pool_to_blocks)
from ngkv.policy import Tier, TierBudget
from ngkv.adapters.vllm_connector import (AttentionTapProvider,
                                          NGKVConnector,
                                          NGKVConnectorMetadata,
                                          RecencyHeuristicProvider)


# ------------------------------ block.py ---------------------------------

def test_pool_roundtrip_shapes():
    v = np.arange(37, dtype=float)
    for bs in (1, 8, 16, 32):
        b = pool_to_blocks(v, bs, "max")
        assert b.shape[0] == n_blocks(37, bs)
        e = expand_from_blocks(b, bs, 37)
        assert e.shape[0] == 37


def test_pool_max_partial_block():
    v = np.array([0.1, 0.9, 0.2, 0.5, 0.7])
    b = pool_to_blocks(v, 2, "max")
    assert np.allclose(b, [0.9, 0.5, 0.7])  # partial last block, no -inf leak


def test_pool_mean_partial_block():
    v = np.array([1.0, 3.0, 5.0])
    b = pool_to_blocks(v, 2, "mean")
    assert np.allclose(b, [2.0, 5.0])  # nanmean ignores padding


def test_max_pool_never_underrates_hot_token():
    v = np.zeros(64); v[37] = 10.0
    b = pool_to_blocks(v, 16, "max")
    assert b[37 // 16] == 10.0


def test_block_gate_sink_pinned_and_budget():
    gate = BlockGate(BlockGateConfig(block_size=4))
    rng = np.random.default_rng(0)
    for t in range(20):
        row = rng.random(32 + t); row /= row.sum()
        gate.observe(row)
    placement = gate.place_blocks(48, TierBudget(hbm_frac=0.25, dram_frac=0.25))
    assert placement.shape[0] == n_blocks(48, 4)
    assert placement[0] == Tier.HBM  # sink block (inf scores) top-tier
    frac_kept = (placement != Tier.EVICT).mean()
    assert frac_kept <= 0.5 + 1e-9  # respects hbm+dram budget


def test_block_gate_shared_mask_pins_hbm():
    gate = BlockGate(BlockGateConfig(block_size=4))
    gate.observe(np.full(40, 1 / 40))
    shared = np.zeros(10, dtype=bool); shared[7] = True
    placement = gate.place_blocks(40, TierBudget(0.1, 0.1), shared_mask=shared)
    assert placement[7] == Tier.HBM
    bits = gate.allocate_bits(40, 0.2, full_bits=16, shared_mask=shared)
    assert bits[7] == 16


# --------------------------- vLLM connector -------------------------------

@dataclasses.dataclass
class FakeKVTransferConfig:
    kv_connector_extra_config: dict


@dataclasses.dataclass
class FakeVllmConfig:
    kv_transfer_config: FakeKVTransferConfig


@dataclasses.dataclass
class FakeRequest:
    request_id: str


class FakeBlocks:
    def __init__(self, ids): self._ids = ids
    def get_block_ids(self): return [self._ids]  # vLLM nests per kv-group


@dataclasses.dataclass
class FakeSchedulerOutput:
    finished_req_ids: list


def make_connector(extra=None, provider=None):
    cfg = FakeVllmConfig(FakeKVTransferConfig(extra or
                         {"block_budget_frac": 0.3, "block_size": 16}))
    return NGKVConnector(cfg, role="scheduler", provider=provider)


def test_connector_reads_extra_config():
    c = make_connector({"block_budget_frac": 0.25, "dram_budget_frac": 0.1,
                        "block_size": 8})
    assert c.block_size == 8
    assert c.budget.hbm_frac == 0.25 and c.budget.dram_frac == 0.1


def test_no_remote_hits_and_side_effect_free():
    c = make_connector()
    r = FakeRequest("r1")
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)
    assert c.get_num_new_matched_tokens(r, 0) == (0, False)
    assert c._req_blocks == {}


def test_plan_tiers_and_budget():
    c = make_connector({"block_budget_frac": 0.3, "dram_budget_frac": 0.3,
                        "block_size": 16})
    c.update_state_after_alloc(FakeRequest("r1"), FakeBlocks(list(range(20))), 0)
    plan = c.plan_for_request("r1")
    n_save, n_drop = len(plan.save_block_ids), len(plan.drop_block_ids)
    assert n_drop >= 1                       # eviction tier non-empty
    assert n_save <= int(0.3 * 20) + 1       # dram budget respected
    assert 0 not in plan.drop_block_ids      # sink block never dropped


def test_shared_blocks_pinned_never_dropped():
    c = make_connector()
    shared_ids = [100, 101]
    c.update_state_after_alloc(FakeRequest("r1"),
                               FakeBlocks(shared_ids + list(range(18))), 0)
    c.update_state_after_alloc(FakeRequest("r2"),
                               FakeBlocks(shared_ids + list(range(50, 68))), 0)
    for req in ("r1", "r2"):
        plan = c.plan_for_request(req)
        assert set(shared_ids) <= set(plan.pinned_block_ids)
        assert not set(shared_ids) & set(plan.drop_block_ids)
        assert not set(shared_ids) & set(plan.save_block_ids)


def test_request_finished_frees_now_and_reports_counts():
    c = make_connector()
    c.update_state_after_alloc(FakeRequest("r1"), FakeBlocks(list(range(20))), 0)
    free_now_is_deferred, params = c.request_finished(FakeRequest("r1"), None)
    assert free_now_is_deferred is False
    assert (params["ngkv_saved_blocks"] + params["ngkv_dropped_blocks"]
            + params["ngkv_pinned_blocks"]
            + params["ngkv_hbm_resident_blocks"]) == 20
    assert params["ngkv_hbm_resident_blocks"] >= 1  # top tier stays resident
    assert c._req_blocks == {} and c._block_ref == {}


def test_worker_saves_only_admitted_blocks():
    c = make_connector()
    c.update_state_after_alloc(FakeRequest("r1"), FakeBlocks(list(range(20))), 0)
    meta = c.build_connector_meta(FakeSchedulerOutput(["r1"]))
    assert isinstance(meta, NGKVConnectorMetadata) and "r1" in meta.plans
    w = make_connector()
    w.bind_connector_metadata(meta)
    w.save_kv_layer("layer.0", kv_layer=None, attn_metadata=None)
    assert len(w.recorded_saves) == 1
    _, _, saved = w.recorded_saves[0]
    assert set(saved) == set(meta.plans["r1"].save_block_ids)
    assert not set(saved) & set(meta.plans["r1"].drop_block_ids)
    done, _ = w.get_finished({"r1"})
    assert done == {"r1"}


def test_attention_tap_provider_shapes_scores():
    prov = AttentionTapProvider(BlockGateConfig(block_size=4))
    rng = np.random.default_rng(1)
    row = np.zeros(40); row[20:24] = 0.25  # hammer block 5
    for _ in range(10):
        prov.observe("r1", row)
    s = prov.block_scores("r1", 10)
    finite = np.isfinite(s)
    assert s[5] == s[finite].max()          # hammered block ranks top
    s_unknown = prov.block_scores("never_seen", 10)
    assert s_unknown.shape == (10,)          # falls back to recency


def test_recency_provider_floor():
    s = RecencyHeuristicProvider().block_scores("x", 6)
    assert np.isinf(s[0])                    # sink pinned
    assert np.all(np.diff(s[1:]) > 0)        # newer > older


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
