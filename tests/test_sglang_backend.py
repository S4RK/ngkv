"""Unit tests: SGLang HiCache admission filter (mock inner backend)."""

import numpy as np

from ngkv.adapters.sglang_backend import (AdmitAll, NGKVFilteredStorage,
                                          PositionScorer, TableScorer)


class MockInner:
    def __init__(self):
        self.store = {}
        self.batch_set_calls = []

    def get(self, key, *a, **k): return self.store.get(key)
    def exists(self, key, *a, **k): return key in self.store
    def set(self, key, value=None, *a, **k):
        self.store[key] = value; return True
    def batch_get(self, keys, *a, **k):
        return [self.store.get(x) for x in keys]
    def batch_exists(self, keys, *a, **k):
        return [x in self.store for x in keys]
    def batch_set(self, keys, values=None, *a, **k):
        self.batch_set_calls.append((list(keys), a))
        for i, x in enumerate(keys):
            self.store[x] = values[i] if values else None
        return True


def test_reads_pass_through_untouched():
    inner = MockInner(); inner.store["p1"] = b"v"
    s = NGKVFilteredStorage(inner, AdmitAll(), admit_frac=0.0)
    assert s.get("p1") == b"v"
    assert s.exists("p1") and not s.exists("p2")
    assert s.batch_get(["p1", "p2"]) == [b"v", None]


def test_batch_set_admits_top_fraction_by_score():
    inner = MockInner()
    sc = TableScorer()
    sc.push_many({"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.2})
    s = NGKVFilteredStorage(inner, sc, admit_frac=0.5)
    ok = s.batch_set(["a", "b", "c", "d"], [1, 2, 3, 4])
    assert ok
    assert set(inner.store) == {"a", "c"}          # top 50% by score
    assert s.stats.admitted == 2 and s.stats.denied == 2


def test_position_scorer_pins_first_page():
    inner = MockInner()
    s = NGKVFilteredStorage(inner, PositionScorer(), admit_frac=0.25)
    s.batch_set([f"p{i}" for i in range(8)], list(range(8)))
    assert "p0" in inner.store                      # pinned sink/prefix page
    assert len(inner.store) == 2                    # ceil(0.25*8)=2


def test_denied_single_set_is_not_a_failure():
    inner = MockInner()
    sc = TableScorer(default=-1.0); sc.push("hot", 1.0)
    s = NGKVFilteredStorage(inner, sc, admit_frac=1.0)
    assert s.set("hot", b"x") and "hot" in inner.store
    # single-key batches always admit >=1; use batch to observe denial
    s2 = NGKVFilteredStorage(MockInner(), sc, admit_frac=0.5)
    assert s2.batch_set(["hot", "cold"], [1, 2])
    assert "cold" not in s2.inner.store


def test_zero_copy_positional_buffers_filtered_in_step():
    inner = MockInner()
    sc = TableScorer(); sc.push_many({"a": 1.0, "b": 0.0})
    s = NGKVFilteredStorage(inner, sc, admit_frac=0.5)
    bufs = ["BUF_A", "BUF_B"]
    s.batch_set(["a", "b"], None, bufs)
    keys, extra = inner.batch_set_calls[0]
    assert keys == ["a"] and extra[0] == ["BUF_A"]  # buffer list co-filtered


def test_admission_rate_respects_budget_over_many_batches():
    inner = MockInner()
    s = NGKVFilteredStorage(inner, PositionScorer(), admit_frac=0.3)
    rng = np.random.default_rng(0)
    for t in range(50):
        n = int(rng.integers(2, 30))
        s.batch_set([f"b{t}_{i}" for i in range(n)], list(range(n)))
    assert s.stats.denial_rate >= 0.5               # ~70% denied minus pins


# ---- factory-contract construction (SGLang calls cls(storage_config, kwargs)) ----

class _FakeStorageConfig:
    """Mimics HiCacheStorageConfig: has extra_config, lacks set()."""
    def __init__(self, extra):
        self.extra_config = extra


def test_factory_contract_detection_without_sglang():
    """Factory-path construction must raise the helpful standalone error
    (SGLang absent here), proving detection routed correctly rather than
    treating the config object as an inner backend."""
    import pytest
    from ngkv.adapters.sglang_backend import NGKVFilteredStorage
    with pytest.raises(RuntimeError, match="SGLang not importable"):
        NGKVFilteredStorage(_FakeStorageConfig({"inner_backend": "file"}), {})


def test_factory_contract_rejects_mem_pool_backends():
    import pytest
    from ngkv.adapters.sglang_backend import NGKVFilteredStorage
    with pytest.raises(ValueError, match="mem_pool_host"):
        NGKVFilteredStorage(
            _FakeStorageConfig({"inner_backend": "mooncake"}), {})


def test_batch_set_v1_gating(mock_inner=None):
    from ngkv.adapters.sglang_backend import (NGKVFilteredStorage,
                                              PositionScorer)

    class Inner:
        def __init__(self):
            self.v1_calls = []
        def batch_set_v1(self, keys, *a, **k):
            self.v1_calls.append((list(keys), a))
            return [True] * len(keys)

    inner = Inner()
    s = NGKVFilteredStorage(inner, scorer=PositionScorer(), admit_frac=0.5)
    bufs = [f"buf{i}" for i in range(4)]
    s.batch_set_v1(["k0", "k1", "k2", "k3"], bufs)
    keys, args = inner.v1_calls[0]
    assert "k0" in keys                      # pinned first page
    assert len(keys) == 2                    # ceil(0.5 * 4)
    assert args[0] == [f"buf{i}" for i in range(4) if f"k{i}" in keys]


def test_getattr_delegation():
    from ngkv.adapters.sglang_backend import NGKVFilteredStorage

    class Inner:
        def batch_set(self, *a, **k): return True
        def clear(self): return "cleared"

    s = NGKVFilteredStorage(Inner(), admit_frac=1.0)
    assert s.clear() == "cleared"
