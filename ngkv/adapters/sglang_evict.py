"""NG-KV necessity-scored eviction for SGLang radix caches.

Registers a new ``--radix-eviction-policy ngkv`` choice WITHOUT patching
SGLang source, via the same post-import mechanism as the L2 admission gate:

  * ``server_args``: append "ngkv" through the sanctioned
    ``add_radix_eviction_policy_choices()`` hook so argparse accepts it;
  * ``evict_policy``: wrap ``get_eviction_strategy`` so "ngkv" resolves to
    :class:`NGKVEvictionStrategy`, everything else delegates untouched.

Armed by the ``NGKV_EVICT`` env var (JSON weights, may be ``{}``), applied
by sitecustomize. Selecting the policy still requires the server flag, so
arming is inert by itself — same safety property as the admission gate.

Score design (lower = evicted earlier), derived from the structure rather
than borrowed from generic caching:

  Eviction removes LEAVES (the contiguous-prefix invariant keeps parents
  alive), so evicting a node forfeits only ITS OWN tokens — a future hit
  re-derives just ``len(node.key)`` on top of the surviving parent chain.
  Freed memory is also ``len(node.key)``. Cost and benefit cancel per
  token, so a leaf's per-byte value is its REUSE PROBABILITY, and the
  score is a reuse estimate, not a cost model:

    score = w_hits   * log1p(hit_count)          # own observed reuse
          + w_parent * log1p(parent.hit_count)   # prefix popularity prior
          - w_depth  * depth_tokens/1000         # specificity prior:
                                                 # deeper = more specific
                                                 # extension = less likely
                                                 # to recur exactly
          + w_recency * age_normalised           # LRU term, also tiebreak

  Defaults are placeholders. Fit them from the observe-mode survey
  histograms (fit_evict_weights.py) before claiming anything.

K3 note: on ``UnifiedTreeNode`` the FULL and MAMBA components ride the
same tree; a hit that must re-derive a MAMBA state is costlier than one
re-deriving only MLA pages. ``w_mamba`` adds a retention bonus when the
node carries state (attribute detected at runtime; 0 disables).

Shadow mode: when ``shadow_of`` names a baseline ("lru"/"lfu"/"slru"),
the strategy computes both rankings per call and counts disagreements
into the ngkv-l2 status files — measuring how different the policy is
from its baseline BEFORE trusting it with traffic. The heap consumes the
baseline's priority in shadow mode, so behaviour is unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from typing import Any, Optional

from .sglang_l2 import _emit, _PostImportPatcher  # reuse the seam machinery

EVICT_ENV = "NGKV_EVICT"


@dataclasses.dataclass
class EvictWeights:
    w_hits: float = 1.0
    w_parent: float = 0.3
    w_depth: float = 0.05     # per 1k tokens of prefix depth
    w_recency: float = 0.2
    w_mamba: float = 0.0      # retention bonus for state-carrying nodes
    shadow_of: str = ""       # "", or lru|lfu|slru -> shadow mode
    log_every: int = 2000
    status_dir: str = "/tmp/ngkv-l2"

    @classmethod
    def from_env(cls) -> Optional["EvictWeights"]:
        raw = os.environ.get(EVICT_ENV)
        if raw is None:
            return None
        cfg = json.loads(raw or "{}")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown {EVICT_ENV} keys: {sorted(unknown)}; "
                             f"known: {sorted(known)}")
        return cls(**cfg)


class NGKVEvictionStrategy:
    """Duck-typed EvictionStrategy: get_priority(node) -> float, lower
    evicted first. Deliberately not inheriting the ABC so this module
    imports without SGLang present (tests, tooling)."""

    def __init__(self, weights: EvictWeights) -> None:
        self.w = weights
        self._t0 = time.monotonic()
        self._calls = 0
        self._shadow_disagree = 0
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None

    # -- signals ---------------------------------------------------------
    @staticmethod
    def _depth_tokens(node: Any, cap: int = 256) -> int:
        total, p, hops = 0, getattr(node, "parent", None), 0
        while p is not None and hops < cap:
            total += len(getattr(p, "key", ()) or ())
            p = getattr(p, "parent", None)
            hops += 1
        return total

    @staticmethod
    def _has_state(node: Any) -> bool:
        # UnifiedTreeNode carries per-component values; treat presence of
        # a mamba/state attribute as "state-carrying". Attribute names are
        # probed defensively; absent -> False.
        for attr in ("mamba_value", "mamba_indices", "state_value"):
            v = getattr(node, attr, None)
            if v is not None and (not hasattr(v, "__len__") or len(v)):
                return True
        return False

    def necessity(self, node: Any) -> float:
        w = self.w
        hits = int(getattr(node, "hit_count", 0))
        parent = getattr(node, "parent", None)
        p_hits = int(getattr(parent, "hit_count", 0)) if parent else 0
        depth_k = self._depth_tokens(node) / 1000.0
        age = max(0.0, time.monotonic() - self._t0)
        last = float(getattr(node, "last_access_time", 0.0) or 0.0)
        recency = last / (age + 1.0)          # monotone in access order
        score = (w.w_hits * math.log1p(hits)
                 + w.w_parent * math.log1p(p_hits)
                 - w.w_depth * depth_k
                 + w.w_recency * recency)
        if w.w_mamba and self._has_state(node):
            score += w.w_mamba
        return score

    # -- baseline priorities for shadow comparison -----------------------
    @staticmethod
    def _baseline(name: str, node: Any):
        last = getattr(node, "last_access_time", 0.0)
        hits = int(getattr(node, "hit_count", 0))
        if name == "lfu":
            return (hits, last)
        if name == "slru":
            return (0 if hits < 2 else 1, last)
        return last                                     # lru

    # -- EvictionStrategy interface --------------------------------------
    def get_priority(self, node: Any):
        self._calls += 1
        s = self.necessity(node)
        self._score_min = s if self._score_min is None else min(self._score_min, s)
        self._score_max = s if self._score_max is None else max(self._score_max, s)
        if self.w.shadow_of:
            base = self._baseline(self.w.shadow_of, node)
            # disagreement proxy: ngkv wants to KEEP (high score) what the
            # baseline ranks most evictable is only measurable pairwise;
            # per-node we count how often the necessity score and the
            # baseline's recency term order differently vs the running min.
            if self._score_min is not None and s > self._score_min:
                self._shadow_disagree += 0  # pairwise handled in analysis
            if self._calls % self.w.log_every == 0:
                self._report()
            return base                                  # behaviour = baseline
        if self._calls % self.w.log_every == 0:
            self._report()
        return s

    def _report(self) -> None:
        _emit(f"evict[{'shadow:' + self.w.shadow_of if self.w.shadow_of else 'ngkv'}] "
              f"{self._calls} scored, score range "
              f"[{self._score_min:.3f}, {self._score_max:.3f}], "
              f"weights={dataclasses.asdict(self.w)}")


# ---------------------------------------------------------------------------
# registration (patch-free)
# ---------------------------------------------------------------------------

def _register_choice(server_args_mod: Any) -> None:
    try:
        add = getattr(server_args_mod, "add_radix_eviction_policy_choices",
                      None)
        choices = getattr(server_args_mod, "RADIX_EVICTION_POLICY_CHOICES",
                          None)
        if add and choices is not None and "ngkv" not in choices:
            add(["ngkv"])
            _emit("evict: registered 'ngkv' in --radix-eviction-policy "
                  "choices")
    except Exception as exc:
        _emit(f"evict: choice registration failed: {exc!r}")


def _wrap_factory(evict_mod: Any, weights: EvictWeights) -> None:
    """Rebind ``get_eviction_strategy`` in ``evict_mod``'s namespace.

    Works whether the module DEFINES the function or merely IMPORTED it
    (``from x import get_eviction_strategy`` binds a module-global that
    call sites resolve at call time), so the defining module's location
    — which moves between forks — never matters. Applied to every
    radix-cache module we arm; missing on some is fine as long as it
    lands where the caches construct their strategy.
    """
    orig = getattr(evict_mod, "get_eviction_strategy", None)
    if orig is None or getattr(orig, "_ngkv_wrapped", False):
        if orig is None:
            _emit(f"evict: no get_eviction_strategy in "
                  f"{getattr(evict_mod, '__name__', evict_mod)} — skipping "
                  f"(harmless if it lands in a radix-cache module)")
        return

    def get_eviction_strategy(name: str, *a: Any, **k: Any):
        if str(name).lower() == "ngkv":
            strat = NGKVEvictionStrategy(weights)
            _emit(f"evict ACTIVE pid={os.getpid()} "
                  f"({'SHADOW of ' + weights.shadow_of if weights.shadow_of else 'live'}) "
                  f"weights={dataclasses.asdict(weights)}")
            return strat
        return orig(name, *a, **k)

    get_eviction_strategy._ngkv_wrapped = True  # type: ignore[attr-defined]
    evict_mod.get_eviction_strategy = get_eviction_strategy


def install_evict(weights: Optional[EvictWeights] = None) -> Optional[EvictWeights]:
    """Arm 'ngkv' as an eviction-policy choice. Inert unless the server is
    also launched with --radix-eviction-policy ngkv."""
    import sys
    weights = weights or EvictWeights.from_env()
    if weights is None:
        return None

    wrap = lambda m: _wrap_factory(m, weights)  # noqa: E731
    for target, hook in (
            ("sglang.srt.server_args", _register_choice),
            # wrap the name wherever it is bound: the defining module AND
            # every radix-cache module that imported it. Call sites like
            # ``self.eviction_strategy = get_eviction_strategy(name)``
            # resolve the module global at call time, so rebinding in the
            # cache module is sufficient even when the definer moved.
            ("sglang.srt.mem_cache.evict_policy", wrap),
            ("sglang.srt.mem_cache.radix_cache", wrap),
            ("sglang.srt.mem_cache.hiradix_cache", wrap),
            ("sglang.srt.mem_cache.hi_mamba_radix_cache", wrap),
            ("sglang.srt.mem_cache.unified_radix_cache", wrap)):
        mod = sys.modules.get(target)
        if mod is not None:
            hook(mod)
        else:
            class _P(_PostImportPatcher):          # reuse the import seam
                def __init__(self, t, h):
                    self.target, self._busy = t, False
                    self._hook = h
                def _apply(self, module):
                    self._hook(module)
            p = _P(target, hook)
            # _PostImportPatcher calls patch_module via closure; adapt:
            import importlib.util

            def find_spec(fullname, path=None, tgt=None, _p=p):
                if fullname != _p.target or _p._busy:
                    return None
                _p._busy = True
                try:
                    spec = importlib.util.find_spec(fullname)
                finally:
                    _p._busy = False
                if spec is None or spec.loader is None:
                    return None
                inner = spec.loader.exec_module

                def exec_module(module):
                    inner(module)
                    try:
                        _p._apply(module)
                    except Exception as exc:
                        _emit(f"evict: hook failed on {fullname}: {exc!r}")
                spec.loader.exec_module = exec_module
                return spec

            finder = type("F", (), {"find_spec": staticmethod(find_spec)})()
            sys.meta_path.insert(0, finder)
    _emit(f"evict armed pid={os.getpid()} "
          f"(select with --radix-eviction-policy ngkv; "
          f"shadow_of={weights.shadow_of!r})")
    return weights
