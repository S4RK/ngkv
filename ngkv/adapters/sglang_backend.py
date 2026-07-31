"""
NG-KV adapter for SGLang HiCache (L3 storage seam).

SGLang's pluggable extension point is ``HiCacheStorage``: an abstract
get / set / exists (+ batch) interface behind ``--hicache-storage-
backend``, dynamically loadable as a vendor plugin without patching
SGLang's tree. NG-KV binds there as a **necessity-gated admission
filter**: a wrapper backend that delegates storage to any real inner
backend (mooncake, hf3fs, nixl, file) and decides *which pages are
worth writing at all*. Denied pages are dropped before transfer — the
same interconnect/storage-bytes win as the vLLM connector's EVICT tier.

Why admission-only — the boundary rule, inverted. In vLLM we gate
request-private KV and pin shared blocks. HiCache L3 is the opposite
world: every page written is written *to be shared* (prefix reuse
across requests is the store's entire purpose). Shared => lossless,
therefore in this seam NG-KV performs no quantization and no demotion:
the only gate action is admit / deny. Denying is always safe (worst
case is a future recompute, exactly as if the page were never cached);
degrading a page that many future requests will hit is never safe.

What "necessity" means here — stated plainly. This seam decides value
for *future reuse by other requests*, which is a different quantity
from within-request attention necessity. Attention-derived scores are a
proxy for it, not ground truth (a page can be unneeded by its own
request's decode yet a popular prefix for others). SGLang's own
``write_through_selective`` policy — hit-count-based selective backup —
is prior art for gating this decision; NG-KV supplies richer, pluggable
priorities and a byte budget. Certify admission policies against your
own reuse traces; the repo's oracle/regret machinery applies with
"reused later" as the ground-truth label.

Scorers included (pluggable ``PageScorer``):

  * ``TableScorer``     — scores pushed from outside (telemetry, an
                          AttentionTapProvider on the request path, or
                          offline analytics keyed by page hash).
  * ``PositionScorer``  — kernel-free floor: earlier pages in a
                          request's flush order are more prefix-like,
                          hence more reusable; sink page pinned.
  * ``AdmitAll``        — pass-through (telemetry-only deployment).

Budget semantics: ``admit_frac`` bounds the fraction of each batch_set
admitted (top scores within the batch), a purely local rule requiring
no global state; per-batch = per-flush ~= per-request under HiCache's
write-back batching.

Wire-up (dynamic vendor loading, no SGLang patches):

    --hicache-storage-backend dynamic
    --hicache-storage-backend-extra-config '{
        "backend_name": "ngkv",
        "module_path": "ngkv.adapters.sglang_backend",
        "class_name":  "NGKVFilteredStorage",
        "inner_backend": "file",
        "scorer": "position",
        "admit_frac": 0.6
    }'

(backend_name/module_path/class_name are REQUIRED by SGLang's factory;
inner_backend is limited to file|nixl — the dynamic seam does not
receive mem_pool_host, so mooncake/hf3fs cannot be wrapped without an
upstream change.)

Status: interface-complete against the documented HiCacheStorage
surface and unit-tested with a mock inner backend
(tests/test_sglang_backend.py); pending validation against a live
SGLang build. Batch zero-copy page I/O variants delegate to the inner
backend unchanged for admitted keys.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # real SGLang present
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
    _HAVE_SGLANG = True
except ImportError:  # standalone / test environment
    _HAVE_SGLANG = False

    class HiCacheStorage:  # type: ignore[no-redef]
        """Structural stand-in mirroring the documented interface."""


# --------------------------------------------------------------------------
# Page scorers
# --------------------------------------------------------------------------

class PageScorer:
    """Maps page keys (hashes) to reuse-necessity scores."""

    def scores(self, keys: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class AdmitAll(PageScorer):
    def scores(self, keys: Sequence[str]) -> np.ndarray:
        return np.zeros(len(keys))


class PositionScorer(PageScorer):
    """Floor heuristic: flush order ~ prefix depth; earlier = more
    reusable as a shared prefix. First page (system prompt / sinks)
    pinned."""

    def scores(self, keys: Sequence[str]) -> np.ndarray:
        n = len(keys)
        if n == 0:
            return np.zeros(0)
        s = np.linspace(1.0, 0.0, n)
        s[0] = np.inf
        return s


class TableScorer(PageScorer):
    """Externally pushed scores (telemetry / request-path attention tap
    / offline analytics), keyed by page hash. Unknown keys receive
    ``default`` (0.0 => admitted only if the batch has spare budget)."""

    def __init__(self, default: float = 0.0) -> None:
        self.table: Dict[str, float] = {}
        self.default = default

    def push(self, key: str, score: float) -> None:
        self.table[key] = score

    def push_many(self, items: Dict[str, float]) -> None:
        self.table.update(items)

    def scores(self, keys: Sequence[str]) -> np.ndarray:
        return np.array([self.table.get(k, self.default) for k in keys])


# --------------------------------------------------------------------------
# Admission-filter backend
# --------------------------------------------------------------------------

@dataclasses.dataclass
class AdmissionStats:
    admitted: int = 0
    denied: int = 0

    @property
    def denial_rate(self) -> float:
        t = self.admitted + self.denied
        return self.denied / t if t else 0.0


class NGKVFilteredStorage(HiCacheStorage):
    """HiCacheStorage wrapper: necessity-gated admission, lossless
    delegation. Reads/exists pass through untouched; writes are gated.

    Two construction paths:

    * **SGLang factory (production).** The factory instantiates dynamic
      backends as ``backend_class(storage_config, kwargs)`` — two
      positionals, no ``from_config`` call. We detect a storage_config
      by its ``extra_config`` attribute (and absence of a ``set``
      method) and self-configure from it. The factory does NOT forward
      ``mem_pool_host`` to dynamic backends, so only inner backends
      constructible from storage_config alone are supported here:
      ``file`` and ``nixl``. (mooncake/hf3fs/eic need mem_pool_host —
      wrapping those requires an upstream SGLang change to forward it;
      stated plainly in K3_RUNG_PLAN.)

    * **Direct (tests / embedding).** ``NGKVFilteredStorage(inner_backend,
      scorer=..., admit_frac=...)`` with any object implementing the
      get/set/exists (+ batch) surface.
    """

    _INNER_OK = ("file", "nixl")

    def __init__(self, inner: Any = None, second: Any = None,
                 admit_frac: float = 0.6, *,
                 scorer: Optional[PageScorer] = None,
                 log_every: int = 50) -> None:
        if inner is not None and hasattr(inner, "extra_config") \
                and not hasattr(inner, "set"):
            # SGLang factory path: (storage_config, factory_kwargs)
            storage_config = inner
            extra = dict(getattr(storage_config, "extra_config", None) or {})
            inner = self._build_inner(storage_config, extra)
            scorer = {"position": PositionScorer, "table": TableScorer,
                      "admit_all": AdmitAll}[extra.get("scorer",
                                                       "position")]()
            admit_frac = float(extra.get("admit_frac", admit_frac))
            log_every = int(extra.get("log_every", log_every))
        elif isinstance(second, PageScorer):
            # direct path, scorer passed positionally (original signature)
            scorer = second
        if inner is None:
            raise ValueError("inner backend required")
        self.inner = inner
        self.scorer = scorer or PositionScorer()
        self.admit_frac = float(admit_frac)
        self.log_every = int(log_every)
        self._flushes = 0
        self.stats = AdmissionStats()

    @classmethod
    def _build_inner(cls, storage_config: Any, extra: Dict[str, Any]) -> Any:
        inner_name = extra.get("inner_backend", "file")
        if inner_name not in cls._INNER_OK:
            raise ValueError(
                f"inner_backend={inner_name!r} unsupported via the dynamic "
                f"seam: SGLang's factory does not forward mem_pool_host to "
                f"dynamic backends, so only {cls._INNER_OK} are "
                f"constructible here.")
        try:
            from sglang.srt.mem_cache.storage.backend_factory import (
                StorageBackendFactory)
        except ImportError as exc:  # standalone: inner must be injected
            raise RuntimeError(
                "SGLang not importable; construct NGKVFilteredStorage "
                "directly with an inner backend instance") from exc
        return StorageBackendFactory.create_backend(
            inner_name, storage_config, None)

    # ---- construction from SGLang extra-config (compat alias) ----

    @classmethod
    def from_config(cls, storage_config: Any) -> "NGKVFilteredStorage":
        return cls(storage_config)

    # ---- admission core ----

    def _admit_mask(self, keys: Sequence[str]) -> np.ndarray:
        n = len(keys)
        if n == 0:
            return np.zeros(0, dtype=bool)
        s = self.scorer.scores(keys)
        k = max(1, int(np.ceil(self.admit_frac * n)))
        order = np.argsort(-s, kind="stable")
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        mask |= np.isinf(s)  # pinned pages always admitted
        self.stats.admitted += int(mask.sum())
        self.stats.denied += int(n - mask.sum())
        return mask

    # ---- HiCacheStorage surface ----

    def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
        return self.inner.get(key, *args, **kwargs)

    def exists(self, key: str, *args: Any, **kwargs: Any) -> bool:
        return self.inner.exists(key, *args, **kwargs)

    def set(self, key: str, value: Any = None, *args: Any,
            **kwargs: Any) -> bool:
        if self._admit_mask([key])[0]:
            return self.inner.set(key, value, *args, **kwargs)
        return True  # denied-by-policy is not a storage failure

    def batch_get(self, keys: Sequence[str], *args: Any,
                  **kwargs: Any) -> Any:
        return self.inner.batch_get(keys, *args, **kwargs)

    def batch_exists(self, keys: Sequence[str], *args: Any,
                     **kwargs: Any) -> Any:
        return self.inner.batch_exists(keys, *args, **kwargs)

    def batch_set(self, keys: Sequence[str],
                  values: Optional[Sequence[Any]] = None, *args: Any,
                  **kwargs: Any) -> bool:
        mask = self._admit_mask(keys)
        self._flushes += 1
        if self.log_every and self._flushes % self.log_every == 0:
            import logging
            logging.getLogger("ngkv.sglang").info(
                "ngkv admission: %d admitted / %d denied (denial rate "
                "%.1f%%) over %d flushes", self.stats.admitted,
                self.stats.denied, 100 * self.stats.denial_rate,
                self._flushes)
        adm = [i for i in range(len(keys)) if mask[i]]
        if not adm:
            return True
        sub_keys = [keys[i] for i in adm]
        sub_vals = ([values[i] for i in adm] if values is not None else None)
        # positional page buffers (zero-copy variants) filtered in step
        sub_args = tuple(
            [a[i] for i in adm] if isinstance(a, (list, tuple))
            and len(a) == len(keys) else a
            for a in args)
        return self.inner.batch_set(sub_keys, sub_vals, *sub_args, **kwargs)

    # ---- v1 zero-copy variants (page buffers ride positionally) ----

    def batch_get_v1(self, keys: Sequence[str], *args: Any,
                     **kwargs: Any) -> Any:
        return self.inner.batch_get_v1(keys, *args, **kwargs)

    def batch_set_v1(self, keys: Sequence[str], *args: Any,
                     **kwargs: Any) -> Any:
        mask = self._admit_mask(keys)
        adm = [i for i in range(len(keys)) if mask[i]]
        if not adm:
            return [True] * len(keys) if keys else True
        sub_keys = [keys[i] for i in adm]
        sub_args = tuple(
            [a[i] for i in adm] if isinstance(a, (list, tuple))
            and len(a) == len(keys) else a
            for a in args)
        return self.inner.batch_set_v1(sub_keys, *sub_args, **kwargs)

    # ---- everything else passes through (delete, clear, registration
    # hooks, stats — and any surface added after this adapter was cut) ----

    def __getattr__(self, name: str) -> Any:
        if name == "inner":  # not yet set (mid-__init__) — avoid recursion
            raise AttributeError(name)
        return getattr(self.inner, name)
