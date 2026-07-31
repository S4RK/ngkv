"""Tests for ngkv.adapters.sglang_l2 (L2 host-memory admission gate)."""

import logging
import sys
import types

import pytest

from ngkv.adapters.sglang_l2 import (L2Policy, install, patch_module,
                                     policy_from_env)


# ---- fakes mirroring SGLang's shapes -------------------------------------

class _Node:
    def __init__(self, hits=0, klen=8):
        self.hit_count = hits
        self.key = list(range(klen))


class _Pool:
    def __init__(self, avail, size):
        self._a, self.size = avail, size

    def available_size(self):
        return self._a


class _Controller:
    def __init__(self, pool):
        self.mem_pool_host = pool


class _Cache:
    """Mimics HiRadixCache.write_backup's contract."""

    def __init__(self, avail=10, size=100):
        self.cache_controller = _Controller(_Pool(avail, size))
        self.calls = []

    def write_backup(self, node, write_back=False):
        self.calls.append((node, write_back))
        return len(node.key)


def _module_with_cache():
    m = types.ModuleType("fake_hiradix")
    m.HiRadixCache = type("HiRadixCache", (_Cache,), {})
    m.HiRadixCache.write_backup = _Cache.write_backup
    return m


# ---- policy ---------------------------------------------------------------

def test_relaxes_when_host_memory_is_free():
    p = L2Policy(mode="gate", min_hits=99)
    cache = _Cache(avail=80, size=100)          # 0.8 free > relax_above
    assert p.admit(cache, _Node(hits=0)) is True
    assert p.stats.relaxed == 1


def test_gates_on_reuse_under_pressure():
    p = L2Policy(mode="gate", min_hits=2, relax_above=0.3)
    cache = _Cache(avail=10, size=100)          # 0.1 free -> scarce
    assert p.admit(cache, _Node(hits=1)) is False
    assert p.admit(cache, _Node(hits=2)) is True


def test_min_len_floor():
    p = L2Policy(mode="gate", min_hits=0, min_len=16, relax_above=0.0)
    cache = _Cache(avail=0, size=100)
    assert p.admit(cache, _Node(hits=5, klen=8)) is False
    assert p.admit(cache, _Node(hits=5, klen=32)) is True


def test_unreadable_pool_warns_once_and_still_gates(caplog):
    p = L2Policy(mode="gate", min_hits=2)
    cache = _Cache()
    cache.cache_controller.mem_pool_host = object()   # no available_size
    with caplog.at_level(logging.WARNING):
        assert p.admit(cache, _Node(hits=0)) is False
        assert p.admit(cache, _Node(hits=0)) is False
    assert sum("cannot read host pool" in r.message for r in caplog.records) == 1


# ---- patch behaviour ------------------------------------------------------

def test_gate_mode_denies_by_returning_zero():
    m = _module_with_cache()
    p = L2Policy(mode="gate", min_hits=2, relax_above=0.3)
    assert patch_module(m, p)
    cache = m.HiRadixCache(avail=5, size=100)
    assert cache.write_backup(_Node(hits=0)) == 0     # denied
    assert cache.calls == []                          # never reached original
    assert cache.write_backup(_Node(hits=3)) == 8     # admitted
    assert len(cache.calls) == 1
    assert p.stats.denied == 1 and p.stats.admitted == 1
    assert p.stats.denied_tokens == 8


def test_observe_mode_never_denies_but_counts():
    m = _module_with_cache()
    p = L2Policy(mode="observe", min_hits=99, relax_above=0.0)
    patch_module(m, p)
    cache = m.HiRadixCache(avail=0, size=100)
    assert cache.write_backup(_Node(hits=0)) == 8     # passthrough
    assert len(cache.calls) == 1
    assert p.stats.denied == 1                        # would have denied


def test_writeback_path_untouched_by_default():
    m = _module_with_cache()
    p = L2Policy(mode="gate", min_hits=99, relax_above=0.0)
    patch_module(m, p)
    cache = m.HiRadixCache(avail=0, size=100)
    assert cache.write_backup(_Node(hits=0), write_back=True) == 8
    assert cache.calls[0][1] is True
    assert p.stats.denied == 0                        # not even counted

    p2 = L2Policy(mode="gate", min_hits=99, relax_above=0.0,
                  gate_writeback=True)
    m2 = _module_with_cache()
    patch_module(m2, p2)
    assert m2.HiRadixCache(avail=0, size=100).write_backup(
        _Node(hits=0), write_back=True) == 0


def test_refuses_to_patch_on_signature_drift(caplog):
    m = types.ModuleType("drifted")
    m.HiRadixCache = type("HiRadixCache", (), {
        "write_backup": lambda self, node, mode, extra: 1})
    with caplog.at_level(logging.ERROR):
        assert patch_module(m, L2Policy()) == []
    assert any("REFUSING to patch" in r.message for r in caplog.records)


def test_patch_is_idempotent():
    m = _module_with_cache()
    p = L2Policy(mode="gate", min_hits=0, relax_above=0.0)
    assert patch_module(m, p)
    assert patch_module(m, p) == []       # already marked


# ---- env config + import hook --------------------------------------------

def test_policy_from_env_rejects_unknown_keys(monkeypatch):
    monkeypatch.setenv("NGKV_L2", '{"mode":"gate","min_hitz":3}')
    with pytest.raises(ValueError, match="unknown NGKV_L2 keys"):
        policy_from_env()
    monkeypatch.setenv("NGKV_L2", '{"mode":"gate","min_hits":3}')
    assert policy_from_env().min_hits == 3


def test_post_import_patcher_fires_on_first_import(tmp_path, monkeypatch):
    pkg = tmp_path / "fakesgl"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "hiradix.py").write_text(
        "class HiRadixCache:\n"
        "    def write_backup(self, node, write_back=False):\n"
        "        return 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in [m for m in sys.modules if m.startswith("fakesgl")]:
        del sys.modules[mod]

    p = L2Policy(mode="gate", min_hits=99, relax_above=0.0)
    install(p, target="fakesgl.hiradix")
    import fakesgl.hiradix as h                       # noqa: E402

    class C(h.HiRadixCache):
        cache_controller = None
    assert C().write_backup(_Node(hits=0)) == 0        # patched on import
    assert p.stats.denied == 1
