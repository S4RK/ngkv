"""NG-KV admission gating at L2 (host memory), via runtime patch.

Why L2. SGLang exposes a pluggable backend only at L3 (``HiCacheStorage``);
L2 — the host KV pool — has no backend interface, its write-back path lives
inside ``HiRadixCache``. But L2 is the better tier to gate: under
``write_through`` every device page passes through it, host memory is a hard
budget (``--hicache-ratio`` x the device pool), and denial also saves the D2H
copy. On Kimi K3 it is additionally the ONLY working tier — the L3 path in
current hybrid builds does not complete a round trip.

Seam. ``HiRadixCache.write_backup(node, write_back=False) -> int`` returns the
number of host tokens backed up, and **0 already means "not backed up"**: it
is returned when the parent isn't backed up yet and when host allocation
fails, and every caller handles it. Denial is therefore expressible inside
the existing contract — we return 0 instead of calling through. Worst case is
a recompute, exactly as if host memory had been full.

Prefix invariant (inherited, not invented here): backed-up nodes must form a
contiguous prefix from the root, so denying a node also denies its whole
subtree — children see ``parent.backuped == False`` and return 0 on their
own. Gating is thus a *depth cut* per chain, the same shape as the L3 prefix
cut. Policy knobs are chosen with that in mind: a scattered per-node score
would not be honoured by the data structure.

Policy (v1, deliberately auditable). Necessity is proxied by observed reuse
(``node.hit_count``) and gating engages only under host-memory scarcity:

    free_frac >= relax_above   -> admit everything (nothing is scarce)
    free_frac <  relax_above   -> admit iff hit_count >= min_hits
                                  and len(node.key) >= min_len

i.e. it dynamically raises SGLang's ``write_through_threshold`` when host
memory is tight, and gets out of the way when it isn't. Two modes: ``observe``
logs decisions and never denies (zero behaviour change, for validating the
seam), ``gate`` enforces.

The eviction path (``write_back=True``) is NOT gated by default: there,
denial drops the subtree instead of merely declining a backup. Enable with
``gate_writeback: true`` only deliberately.

Config via the ``NGKV_L2`` env var (JSON), e.g.
    NGKV_L2='{"mode":"gate","min_hits":2,"relax_above":0.3,"log_every":200}'

Install without touching SGLang: put this repo on ``PYTHONPATH`` — the
bundled ``sitecustomize.py`` calls ``install()`` when ``NGKV_L2`` is set. The
patch is applied when ``sglang.srt.mem_cache.hiradix_cache`` is imported, and
REFUSES to patch (loud ERROR, no silent no-op) if the method signature is not
what it expects.

This is a research instrument that patches library internals at runtime. It
is version-fragile by construction: verify the log line naming the patched
class on every SGLang upgrade.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger("ngkv.l2")

TARGET_MODULE = "sglang.srt.mem_cache.hiradix_cache"
_PATCH_MARK = "_ngkv_l2_patched"


# ---------------------------------------------------------------------------


@dataclasses.dataclass
class L2Stats:
    admitted: int = 0
    denied: int = 0
    admitted_tokens: int = 0
    denied_tokens: int = 0
    relaxed: int = 0  # admitted because memory wasn't scarce

    @property
    def denial_rate(self) -> float:
        n = self.admitted + self.denied
        return self.denied / n if n else 0.0


@dataclasses.dataclass
class L2Policy:
    """Necessity-gated admission to the host KV pool."""

    mode: str = "observe"          # "observe" | "gate"
    min_hits: int = 2              # required reuse under scarcity
    min_len: int = 0               # skip trivially short nodes (tokens)
    relax_above: float = 0.3       # free host fraction above which: admit
                                   # all; 0 disables relaxation (always gate)
    gate_writeback: bool = False   # also gate the evict-time write_back path
    log_every: int = 200

    def __post_init__(self) -> None:
        if self.mode not in ("observe", "gate"):
            raise ValueError(f"mode must be observe|gate, got {self.mode!r}")
        self.stats = L2Stats()
        self._decisions = 0
        self._warned_pressure = False

    # -- host-memory pressure -------------------------------------------
    def free_fraction(self, cache: Any) -> Optional[float]:
        pool = getattr(getattr(cache, "cache_controller", None),
                       "mem_pool_host", None)
        if pool is None:
            return None
        try:
            avail = pool.available_size()
        except Exception:
            return None
        total = getattr(pool, "size", None) or getattr(pool, "capacity", None)
        if not total:
            return None
        return float(avail) / float(total)

    # -- the decision ----------------------------------------------------
    def admit(self, cache: Any, node: Any) -> bool:
        free = self.free_fraction(cache)
        if free is None:
            if not self._warned_pressure:
                self._warned_pressure = True
                logger.warning(
                    "ngkv-l2: cannot read host pool occupancy; gating on "
                    "reuse alone (no scarcity relaxation). Check "
                    "cache_controller.mem_pool_host API.")
        elif self.relax_above > 0.0 and free >= self.relax_above:
            self.stats.relaxed += 1
            return True

        hits = int(getattr(node, "hit_count", 0))
        klen = len(getattr(node, "key", ()) or ())
        return hits >= self.min_hits and klen >= self.min_len

    def record(self, admitted: bool, node: Any) -> None:
        n = len(getattr(node, "key", ()) or ())
        if admitted:
            self.stats.admitted += 1
            self.stats.admitted_tokens += n
        else:
            self.stats.denied += 1
            self.stats.denied_tokens += n
        self._decisions += 1
        if self.log_every and self._decisions % self.log_every == 0:
            s = self.stats
            logger.info(
                "ngkv-l2 [%s]: %d admitted / %d denied (%.1f%% denied, "
                "%d/%d tokens), %d relaxed by free memory",
                self.mode, s.admitted, s.denied, 100 * s.denial_rate,
                s.denied_tokens, s.admitted_tokens + s.denied_tokens,
                s.relaxed)


def policy_from_env(var: str = "NGKV_L2") -> Optional[L2Policy]:
    raw = os.environ.get(var)
    if not raw:
        return None
    cfg = json.loads(raw)
    known = {f.name for f in dataclasses.fields(L2Policy)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"unknown {var} keys: {sorted(unknown)}; "
                         f"known: {sorted(known)}")
    return L2Policy(**cfg)


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------


def _wrap(orig, policy: L2Policy):
    def write_backup(self, node, write_back=False, *args, **kwargs):
        if write_back and not policy.gate_writeback:
            return orig(self, node, write_back, *args, **kwargs)
        admitted = policy.admit(self, node)
        policy.record(admitted, node)
        if admitted or policy.mode == "observe":
            return orig(self, node, write_back, *args, **kwargs)
        return 0  # already a valid "not backed up" outcome for all callers

    write_backup.__name__ = getattr(orig, "__name__", "write_backup")
    write_backup.__doc__ = (getattr(orig, "__doc__", "") or "") + \
        "\n\n[ngkv] necessity-gated L2 admission wrapper."
    setattr(write_backup, _PATCH_MARK, True)
    return write_backup


def patch_module(module: Any, policy: L2Policy) -> list[str]:
    """Patch every class in ``module`` that defines ``write_backup``.

    Returns the names patched. Refuses (and logs ERROR) on signature drift.
    """
    patched: list[str] = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj) or "write_backup" not in obj.__dict__:
            continue
        orig = obj.__dict__["write_backup"]
        if getattr(orig, _PATCH_MARK, False):
            continue
        try:
            params = list(inspect.signature(orig).parameters)
        except (TypeError, ValueError):
            params = []
        if params[:3] != ["self", "node", "write_back"]:
            logger.error(
                "ngkv-l2: REFUSING to patch %s.write_backup — unexpected "
                "signature %s (expected (self, node, write_back, ...)). "
                "SGLang internals changed; no gating is active.",
                name, params)
            continue
        setattr(obj, "write_backup", _wrap(orig, policy))
        patched.append(f"{module.__name__}.{name}")
    if patched:
        logger.warning(
            "ngkv-l2 ACTIVE (mode=%s, min_hits=%d, relax_above=%.2f, "
            "min_len=%d, gate_writeback=%s) on: %s",
            policy.mode, policy.min_hits, policy.relax_above,
            policy.min_len, policy.gate_writeback, ", ".join(patched))
    else:
        logger.error("ngkv-l2: no write_backup found in %s — NOT active",
                     getattr(module, "__name__", module))
    return patched


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    """Applies the patch right after the target module is first imported."""

    def __init__(self, target: str, policy: L2Policy) -> None:
        self.target, self.policy, self._busy = target, policy, False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.target or self._busy:
            return None
        self._busy = True  # re-entrancy guard: find_spec below re-enters
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        loader, policy = spec.loader, self.policy
        inner_exec = loader.exec_module

        def exec_module(module):
            inner_exec(module)
            try:
                patch_module(module, policy)
            except Exception:  # never break the import
                logger.exception("ngkv-l2: patch failed; no gating active")

        loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def install(policy: Optional[L2Policy] = None,
            target: str = TARGET_MODULE) -> Optional[L2Policy]:
    """Arm L2 gating. Safe to call before or after SGLang is imported."""
    policy = policy or policy_from_env()
    if policy is None:
        return None
    mod = sys.modules.get(target)
    if mod is not None:  # already imported — patch in place
        patch_module(mod, policy)
    else:
        sys.meta_path.insert(0, _PostImportPatcher(target, policy))
        logger.warning("ngkv-l2 armed for %s (mode=%s)", target, policy.mode)
    return policy
