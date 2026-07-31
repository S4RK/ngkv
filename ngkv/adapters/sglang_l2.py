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
import socket
import sys
import time
from typing import Any, Optional

logger = logging.getLogger("ngkv.l2")


def _emit(msg: str, level: int = logging.WARNING) -> None:
    """Log AND write to stderr.

    SGLang reconfigures logging in its worker processes; a logger created at
    import time can be silenced by ``dictConfig(disable_existing_loggers=True)``
    before it ever emits. A research instrument that silently stops reporting
    is worse than useless, so every important line also goes to stderr.
    """
    try:
        logger.disabled = False
        logger.log(level, "%s", msg)
    except Exception:
        pass
    try:
        print(f"[ngkv-l2] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass

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


_STATUS_FAILED = set()


def _status_path(policy: "L2Policy") -> Optional[str]:
    if not policy.status_dir:
        return None
    try:
        os.makedirs(policy.status_dir, exist_ok=True)
        return os.path.join(policy.status_dir,
                            f"{socket.gethostname()}-{os.getpid()}.json")
    except Exception as exc:
        if policy.status_dir not in _STATUS_FAILED:
            _STATUS_FAILED.add(policy.status_dir)
            _emit(f"cannot create status_dir {policy.status_dir!r}: {exc!r} "
                  f"— counters will only appear in this log",
                  logging.ERROR)
        return None


def dump_status(policy: "L2Policy") -> None:
    """Write this process's config + counters to a JSON file (atomic)."""
    path = _status_path(policy)
    if not path:
        return
    payload = {
        "pid": os.getpid(), "host": socket.gethostname(),
        "ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "patched": policy.patched,
        "config": {f.name: getattr(policy, f.name)
                   for f in dataclasses.fields(policy)},
        "stats": dataclasses.asdict(policy.stats),
        "denial_rate": policy.stats.denial_rate,
    }
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        if path not in _STATUS_FAILED:
            _STATUS_FAILED.add(path)
            _emit(f"cannot write status file {path!r}: {exc!r} — counters "
                  f"will only appear in this log", logging.ERROR)


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
    status_dir: str = "/tmp/ngkv-l2"   # per-process JSON counters

    def __post_init__(self) -> None:
        self.patched: list = []
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
                _emit("cannot read host pool occupancy; gating on reuse "
                      "alone (no scarcity relaxation). Check "
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
            _emit(f"[{self.mode}] {s.admitted} admitted / {s.denied} denied "
                  f"({100 * s.denial_rate:.1f}% denied, {s.denied_tokens}/"
                  f"{s.admitted_tokens + s.denied_tokens} tokens), "
                  f"{s.relaxed} relaxed by free memory")
            dump_status(self)


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
            _emit(f"REFUSING to patch {name}.write_backup — unexpected "
                  f"signature {params} (expected (self, node, write_back, "
                  f"...)). SGLang internals changed; no gating is active.",
                  logging.ERROR)
            continue
        setattr(obj, "write_backup", _wrap(orig, policy))
        patched.append(f"{module.__name__}.{name}")
    if patched:
        policy.patched.extend(patched)
        _emit(f"ACTIVE pid={os.getpid()} (mode={policy.mode}, "
              f"min_hits={policy.min_hits}, relax_above={policy.relax_above}, "
              f"min_len={policy.min_len}, "
              f"gate_writeback={policy.gate_writeback}) on: "
              + ", ".join(patched))
        dump_status(policy)
    else:
        _emit(f"no write_backup found in "
              f"{getattr(module, '__name__', module)} — NOT active",
              logging.ERROR)
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
            except Exception as exc:  # never break the import
                _emit(f"patch failed ({exc!r}); no gating active",
                      logging.ERROR)

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
        _emit(f"armed pid={os.getpid()} for {target} (mode={policy.mode})")

    # Worker processes may be forked AFTER the parent imported the module,
    # in which case the child inherits an unpatched module and no import
    # event ever fires. Re-run installation in every forked child.
    if not getattr(install, "_fork_hooked", False):
        try:
            os.register_at_fork(after_in_child=lambda: install(target=target))
            install._fork_hooked = True  # type: ignore[attr-defined]
        except Exception:
            pass
    return policy


def diagnose(target: str = TARGET_MODULE) -> None:
    """Print where ``write_backup`` actually lives in THIS SGLang build.

    Run inside the container when no ACTIVE line appears:
        python3 -c "from ngkv.adapters.sglang_l2 import diagnose; diagnose()"
    """
    print(f"python: {sys.executable}")
    print(f"target module: {target}")
    try:
        mod = importlib.import_module(target)
        print(f"  imported from {getattr(mod, '__file__', '?')}")
        found = False
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and "write_backup" in obj.__dict__:
                found = True
                fn = obj.__dict__["write_backup"]
                print(f"  {name}.write_backup{inspect.signature(fn)} "
                      f"patched={getattr(fn, _PATCH_MARK, False)}")
        if not found:
            print("  !! no class defines write_backup here")
    except Exception as exc:
        print(f"  !! import failed: {exc!r}")

    # scan the whole mem_cache package: a fork may define it elsewhere
    try:
        import sglang.srt.mem_cache as mc
        # namespace packages have __file__ = None; __path__ always works
        roots = [os.path.dirname(mc.__file__)] if getattr(mc, "__file__", None) \
            else list(getattr(mc, "__path__", []))
        print(f"scanning {roots} for 'def write_backup':")
        for dirpath, _dirs, files in [(d, n, f) for r in roots
                                      for d, n, f in os.walk(r)]:
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    src = open(p, encoding="utf-8", errors="replace").read()
                except Exception:
                    continue
                if "def write_backup" in src:
                    rel = os.path.relpath(p, roots[0])
                    for line in src.splitlines():
                        if "def write_backup" in line:
                            print(f"  {rel}: {line.strip()}")
    except Exception as exc:
        print(f"  !! scan failed: {exc!r}")
