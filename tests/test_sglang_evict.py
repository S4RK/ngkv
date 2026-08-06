"""Tests for the necessity eviction strategy and its patch-free registration."""
import math
import sys
import types

from ngkv.adapters.sglang_evict import (EvictWeights, NGKVEvictionStrategy,
                                        install_evict)


class _N:
    def __init__(self, hits=0, klen=100, parent=None, last=0.0, mamba=None):
        self.hit_count = hits
        self.key = list(range(klen))
        self.parent = parent
        self.last_access_time = last
        if mamba is not None:
            self.mamba_value = mamba


def test_score_orders_hot_above_cold():
    s = NGKVEvictionStrategy(EvictWeights())
    root = _N(hits=50, klen=1000)
    hot_leaf = _N(hits=8, parent=root, last=10.0)
    cold_leaf = _N(hits=1, parent=root, last=11.0)   # more recent but cold
    assert s.necessity(hot_leaf) > s.necessity(cold_leaf)


def test_parent_history_rescues_fresh_child_of_hot_prefix():
    s = NGKVEvictionStrategy(EvictWeights())
    hot_parent, cold_parent = _N(hits=40, klen=1000), _N(hits=1, klen=1000)
    a = _N(hits=1, parent=hot_parent, last=5.0)
    b = _N(hits=1, parent=cold_parent, last=5.0)
    assert s.necessity(a) > s.necessity(b)


def test_depth_prior_penalises_specific_extensions():
    s = NGKVEvictionStrategy(EvictWeights())
    shallow_p = _N(hits=5, klen=500)
    deep_mid = _N(hits=5, klen=30000, parent=shallow_p)
    a = _N(hits=2, parent=shallow_p, last=5.0)
    b = _N(hits=2, parent=deep_mid, last=5.0)   # same signals, deeper
    assert s.necessity(a) > s.necessity(b)


def test_mamba_bonus_only_when_enabled_and_present():
    plain = NGKVEvictionStrategy(EvictWeights(w_mamba=0.0))
    boosted = NGKVEvictionStrategy(EvictWeights(w_mamba=1.0))
    n_state = _N(hits=2, mamba=[1, 2])
    n_plain = _N(hits=2)
    assert plain.necessity(n_state) == plain.necessity(n_plain)
    assert boosted.necessity(n_state) > boosted.necessity(n_plain)


def test_shadow_mode_returns_baseline_priority():
    s = NGKVEvictionStrategy(EvictWeights(shadow_of="lfu"))
    n = _N(hits=7, last=3.25)
    assert s.get_priority(n) == (7, 3.25)        # LFU tuple, not our float
    live = NGKVEvictionStrategy(EvictWeights())
    assert isinstance(live.get_priority(n), float)


def test_registration_wraps_factory_and_choices():
    sa = types.ModuleType("sglang.srt.server_args")
    sa.RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]
    sa.add_radix_eviction_policy_choices = \
        lambda c: sa.RADIX_EVICTION_POLICY_CHOICES.extend(c)
    ep = types.ModuleType("sglang.srt.mem_cache.evict_policy")
    class LRU:  # noqa: N801
        def get_priority(self, node): return node.last_access_time
    ep.get_eviction_strategy = lambda name: {"lru": LRU()}[name]
    sys.modules["sglang.srt.server_args"] = sa
    sys.modules["sglang.srt.mem_cache.evict_policy"] = ep
    try:
        install_evict(EvictWeights())
        assert "ngkv" in sa.RADIX_EVICTION_POLICY_CHOICES
        strat = ep.get_eviction_strategy("ngkv")
        assert isinstance(strat, NGKVEvictionStrategy)
        assert isinstance(ep.get_eviction_strategy("lru"), LRU)  # untouched
    finally:
        del sys.modules["sglang.srt.server_args"]
        del sys.modules["sglang.srt.mem_cache.evict_policy"]


def test_env_rejects_unknown_keys(monkeypatch):
    import pytest
    monkeypatch.setenv("NGKV_EVICT", '{"w_hitz": 2}')
    with pytest.raises(ValueError, match="unknown NGKV_EVICT keys"):
        EvictWeights.from_env()
    monkeypatch.setenv("NGKV_EVICT", '{}')
    assert EvictWeights.from_env().w_hits == 1.0
