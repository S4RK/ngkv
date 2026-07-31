#!/usr/bin/env python3
"""Reuse-skewed load generator for NG-KV L2 (host-cache) experiments.

Why not a normal benchmark. Admission gating only has something to say when
the workload has *reuse skew*. The two common benchmark shapes are both
uninformative here:

  * identical prompts every request -> the radix tree dedupes them, L2 never
    fills, everything is hot, denial rate 0. The gate correctly does nothing.
  * all-unique prompts never revisited -> L2 fills, but nothing was reusable,
    so the policy denies ~everything and "saves" all the host memory at zero
    cost. True, and uninteresting: the hit rate was already zero.

Real serving traffic is a hot set (shared system prompts, popular documents)
plus a long tail of one-shot long contexts. This generator reproduces that:
a pool of shared prefixes sampled Zipf-style, plus a configurable fraction of
brand-new unique contexts, each with a unique user tail so the private KV is
genuinely private. That is the mix under which "keep the reused prefixes,
drop the rest" is a claim that can be right or wrong.

It also snapshots the ngkv status JSONs before and after the run, so one
artifact carries latency, throughput AND the admission counters that the
run produced -- rather than leaving you to correlate two files by eye.

Stdlib only (no aiohttp/numpy/requests): these boxes run the SGLang image.

Examples
--------
# plan only, no server needed -- check the workload shape first
python ngkv_loadgen.py --dry-run --requests 400

# real run against the leader
python ngkv_loadgen.py --url http://127.0.0.1:30000 \
    --concurrency 56 --requests 560 --prefix-tokens 32000 \
    --status-dir /mnt/local-nvme/ngkv-status --out run_observe.json

# same shape, gate mode (after flipping NGKV_L2 and restarting)
python ngkv_loadgen.py ... --out run_gate_h2.json --label gate_min_hits2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# workload construction
# ---------------------------------------------------------------------------

_WORDS: list[str] = []


def _vocab(rng: random.Random, n: int = 4096) -> list[str]:
    """A fixed pseudo-word vocabulary: keeps text out of the model's
    memorised distribution and makes token counts roughly stable."""
    global _WORDS
    if not _WORDS:
        _WORDS = ["".join(rng.choice(string.ascii_lowercase)
                          for _ in range(rng.randint(3, 8)))
                  for _ in range(n)]
    return _WORDS


def make_text(rng: random.Random, approx_tokens: int) -> str:
    """~1 token per pseudo-word is a rough but stable approximation.

    Exactness does not matter: what matters is that a given prefix is
    byte-identical across requests (so the radix tree shares it) and that
    distinct prefixes are distinct.
    """
    v = _vocab(rng)
    return " ".join(rng.choice(v) for _ in range(max(1, approx_tokens)))


def zipf_weights(n: int, alpha: float) -> list[float]:
    w = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(w)
    return [x / total for x in w]


def pick_weighted(rng: random.Random, weights: list[float]) -> int:
    r, acc = rng.random(), 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


class Workload:
    """Builds the request plan up front so it can be inspected (--dry-run)
    and replayed identically across phases (same seed => same plan)."""

    def __init__(self, args) -> None:
        rng = random.Random(args.seed)
        self.args = args
        self.pool = [make_text(rng, args.prefix_tokens)
                     for _ in range(args.hot_pool)]
        self.weights = zipf_weights(args.hot_pool, args.zipf)
        self.plan: list[tuple[int, str]] = []   # (pool_idx or -1, prompt)
        for _ in range(args.requests):
            if rng.random() < args.oneshot_frac:
                body = make_text(rng, args.oneshot_tokens)
                idx = -1
            else:
                idx = pick_weighted(rng, self.weights)
                body = self.pool[idx]
            tail = make_text(rng, args.tail_tokens)
            self.plan.append(
                (idx, f"{body}\n\nQuestion: {tail}\nAnswer:"))
        # Request class is the client-side cache-effectiveness instrument:
        # a "hot_repeat" request re-sends a prefix already sent earlier in
        # this run, so its prefill SHOULD be skipped if the cache kept it.
        # Comparing TTFT across classes measures the cache's value without
        # needing any server metric.
        seen: set = set()
        self.classes: list[str] = []
        for idx, _ in self.plan:
            if idx < 0:
                self.classes.append("oneshot")
            elif idx in seen:
                self.classes.append("hot_repeat")
            else:
                seen.add(idx)
                self.classes.append("hot_first")

    def describe(self) -> dict:
        counts: dict[int, int] = {}
        for idx, _ in self.plan:
            counts[idx] = counts.get(idx, 0) + 1
        oneshot = counts.get(-1, 0)
        shared = {k: v for k, v in counts.items() if k >= 0}
        reused = sum(v for v in shared.values() if v > 1)
        a = self.args
        # distinct KV pushed through the host tier, approx
        distinct = (len(shared) * a.prefix_tokens
                    + oneshot * a.oneshot_tokens
                    + len(self.plan) * a.tail_tokens)
        return {
            "requests": len(self.plan),
            "hot_pool": a.hot_pool,
            "distinct_hot_prefixes_used": len(shared),
            "oneshot_requests": oneshot,
            "requests_hitting_a_reused_prefix": reused,
            "reuse_rate": round(reused / max(1, len(self.plan)), 3),
            "top_prefix_share": round(
                max(shared.values()) / len(self.plan), 3) if shared else 0.0,
            "approx_distinct_prompt_tokens": distinct,
            "class_counts": {c: self.classes.count(c)
                             for c in ("hot_first", "hot_repeat", "oneshot")},
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def build_request(args, prompt: str) -> urllib.request.Request:
    if args.endpoint == "chat":
        url = f"{args.url}/v1/chat/completions"
        payload = {"model": args.model or "default",
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": args.max_new_tokens,
                   "temperature": args.temperature, "stream": True}
    else:
        url = f"{args.url}/generate"
        payload = {"text": prompt, "stream": True,
                   "sampling_params": {"max_new_tokens": args.max_new_tokens,
                                       "temperature": args.temperature}}
    return urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


def one_request(args, prompt: str) -> dict:
    """Returns timing for a single streamed request."""
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    try:
        with urllib.request.urlopen(build_request(args, prompt),
                                    timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.endswith("[DONE]"):
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    except Exception as exc:
        return {"ok": False, "error": repr(exc)[:200],
                "e2e": time.perf_counter() - t0}
    return {"ok": True, "ttft": ttft, "chunks": chunks,
            "e2e": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# ngkv status snapshots
# ---------------------------------------------------------------------------

_STAT_KEYS = ("admitted", "denied", "admitted_tokens", "denied_tokens",
              "relaxed", "decisions_under_pressure", "surveyed")
_HIST_KEYS = ("hit_count_hist", "parent_hits_hist", "depth_hist",
              "keylen_hist")


def clear_status(dirs: list[str]) -> int:
    """Remove status files. Do this between phases: files persist across
    pod restarts, so a dir accumulates dead processes from earlier
    generations and the ABSOLUTE totals become meaningless (deltas stay
    correct, since stale files contribute 0 to a difference)."""
    n = 0
    for d in dirs:
        for name in os.listdir(d) if os.path.isdir(d) else []:
            if name.endswith(".json"):
                try:
                    os.remove(os.path.join(d, name)); n += 1
                except Exception:
                    pass
    return n


def read_status(dirs: list[str], fresh_within: float | None = None) -> dict:
    """Aggregate every per-process status file across the given dirs.

    ``fresh_within``: ignore files whose ``ts`` is older than this many
    seconds, i.e. processes that are no longer writing (dead ranks from a
    previous pod generation).
    """
    agg = {k: 0 for k in _STAT_KEYS}
    hists: dict = {k: {} for k in _HIST_KEYS}
    procs, modes, probes, stale = 0, set(), [], 0
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except Exception:
            continue
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                doc = json.load(open(os.path.join(d, name)))
            except Exception:
                continue
            if not doc.get("patched"):
                continue                     # inert/probe processes
            if fresh_within is not None and \
                    time.time() - float(doc.get("ts", 0)) > fresh_within:
                stale += 1
                continue
            procs += 1
            modes.add(doc.get("config", {}).get("mode"))
            if doc.get("probe"):
                probes.append(doc["probe"].get("node_type"))
            st = doc.get("stats", {})
            for k in _STAT_KEYS:
                agg[k] += int(st.get(k, 0) or 0)
            for hk in _HIST_KEYS:
                for bucket, count in (st.get(hk) or {}).items():
                    hists[hk][bucket] = hists[hk].get(bucket, 0) + int(count)
    return {"processes": procs, "stale_files_ignored": stale,
            "modes": sorted(m for m in modes if m),
            "node_types": sorted(set(probes)), **agg, **hists}


def _hist_delta(before: dict, after: dict) -> dict:
    out = {}
    for k in set(after) | set(before):
        v = int(after.get(k, 0)) - int(before.get(k, 0))
        if v:
            out[k] = v
    return dict(sorted(out.items(),
                       key=lambda kv: int(kv[0].rstrip("+"))))


def status_delta(before: dict, after: dict) -> dict:
    d = {k: after.get(k, 0) - before.get(k, 0) for k in _STAT_KEYS}
    for hk in _HIST_KEYS:
        d[hk] = _hist_delta(before.get(hk, {}), after.get(hk, {}))
    decisions = d["admitted"] + d["denied"]
    d["decisions"] = decisions
    d["denial_rate"] = round(d["denied"] / decisions, 4) if decisions else None
    toks = d["admitted_tokens"] + d["denied_tokens"]
    d["token_denial_rate"] = (round(d["denied_tokens"] / toks, 4)
                              if toks else None)
    d["processes_before"] = before.get("processes")
    d["processes_after"] = after.get("processes")
    under = d.get("decisions_under_pressure", 0)
    if under:
        # the number that matters: of decisions made WHILE scarce, how many
        # were admitted on merit rather than waved through by relaxation?
        d["admitted_under_pressure"] = under - d["denied"]
        d["denial_rate_under_pressure"] = round(d["denied"] / under, 4)
    return d


def fetch_metrics(url: str, keys=("cache_hit", "prefix", "token_usage")
                  ) -> dict:
    """Best-effort scrape of the SGLang /metrics endpoint."""
    out = {}
    try:
        with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, _, val = line.rpartition(" ")
                if any(k in name for k in keys):
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[i], 4)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reuse-skewed loadgen for NG-KV L2 experiments")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--endpoint", choices=["generate", "chat"],
                    default="generate")
    ap.add_argument("--model", default=None, help="for --endpoint chat")
    ap.add_argument("--concurrency", type=int, default=56)
    ap.add_argument("--requests", type=int, default=560)
    ap.add_argument("--hot-pool", type=int, default=24,
                    help="number of shared prefixes (the reusable set)")
    ap.add_argument("--zipf", type=float, default=1.1,
                    help="skew over the hot pool; higher = more concentrated")
    ap.add_argument("--oneshot-frac", type=float, default=0.35,
                    help="fraction of requests with a brand-new prefix")
    ap.add_argument("--prefix-tokens", type=int, default=8000)
    ap.add_argument("--oneshot-tokens", type=int, default=8000)
    ap.add_argument("--tail-tokens", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--status-dir", action="append", default=[],
                    help="repeatable; one per node (local-nvme is per-node)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clear-status", action="store_true",
                    help="delete existing status files first (recommended "
                         "between phases; they survive pod restarts)")
    ap.add_argument("--fresh-within", type=float, default=3600,
                    help="ignore status files older than N seconds (dead "
                         "ranks from a previous pod generation)")
    ap.add_argument("--host-tokens", type=int, default=4_380_000,
                    help="host (L2) capacity in tokens PER RANK; from the "
                         "server log: max_total_num_tokens * hicache-ratio")
    ap.add_argument("--relax-above", type=float, default=0.3,
                    help="must match NGKV_L2 relax_above; the gate is inert "
                         "while free host fraction stays above this")
    args = ap.parse_args()

    wl = Workload(args)
    shape = wl.describe()
    print("workload plan:")
    for k, v in shape.items():
        print(f"  {k}: {v}")

    # will this plan actually make host memory scarce?
    need = args.host_tokens * (1.0 - args.relax_above)
    have = shape["approx_distinct_prompt_tokens"]
    print(f"\npressure check (per rank):")
    print(f"  host capacity        : {args.host_tokens:,} tokens")
    print(f"  distinct tokens sent : {have:,}")
    print(f"  needed to drop below {1 - args.relax_above:.0%} full: "
          f"{need:,.0f}")
    if have < need:
        scale = need / max(1, have)
        print(f"  VERDICT: NOT ENOUGH ({have / need:.0%} of threshold). The "
              f"gate will relax and report ~0 denials.")
        print(f"           Scale distinct traffic ~{scale:.1f}x — e.g. "
              f"--oneshot-tokens {int(args.oneshot_tokens * scale):,} "
              f"or --requests {int(args.requests * scale):,}.")
    else:
        print(f"  VERDICT: sufficient ({have / need:.1f}x threshold) — "
              f"L2 should fill and eviction/gating engage.")

    if args.dry_run:
        print("\n--dry-run: no requests sent.")
        if shape["reuse_rate"] < 0.2:
            print("WARNING: reuse_rate is low — the gate will look good for "
                  "a trivial reason (nothing was reusable).")
        if shape["oneshot_requests"] == 0:
            print("WARNING: no one-shot traffic — nothing for the gate to "
                  "decline; denial rate will be ~0.")
        return

    if args.clear_status and args.status_dir:
        print(f"cleared {clear_status(args.status_dir)} stale status files")
    before = (read_status(args.status_dir, args.fresh_within)
              if args.status_dir else {})
    m_before = fetch_metrics(args.url)
    if before:
        print(f"\nngkv status before: {before}")
        if before.get("processes", 0) == 0:
            print("WARNING: no patched processes found in status dirs — "
                  "counters will be empty. Check both nodes.")

    if args.warmup:
        print(f"warmup: {args.warmup} requests ...")
        for _, prompt in wl.plan[:args.warmup]:
            one_request(args, prompt)

    print(f"\nrunning {args.requests} requests @ concurrency "
          f"{args.concurrency} ...")
    results, done, lock = [], [0], threading.Lock()
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(one_request, args, p): wl.classes[i]
                for i, (_, p) in enumerate(wl.plan)}
        for f in as_completed(futs):
            r = f.result()
            r["class"] = futs[f]
            results.append(r)
            with lock:
                done[0] += 1
                if done[0] % max(1, args.requests // 10) == 0:
                    print(f"  {done[0]}/{args.requests}", flush=True)
    wall = time.perf_counter() - t_start

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    e2es = [r["e2e"] for r in ok]
    summary = {
        "label": args.label,
        "config": vars(args),
        "workload": shape,
        "wall_seconds": round(wall, 2),
        "ok": len(ok), "failed": len(fail),
        "throughput_rps": round(len(ok) / wall, 3) if wall else None,
        "ttft_p50": pct(ttfts, 0.50), "ttft_p90": pct(ttfts, 0.90),
        "ttft_p99": pct(ttfts, 0.99),
        "e2e_p50": pct(e2es, 0.50), "e2e_p99": pct(e2es, 0.99),
        "ttft_mean": round(statistics.fmean(ttfts), 4) if ttfts else None,
    }
    if fail:
        summary["first_errors"] = [r.get("error") for r in fail[:3]]

    # Per-class TTFT: the cache-effectiveness measurement.
    by_class = {}
    for cls in ("hot_first", "hot_repeat", "oneshot"):
        t = [r["ttft"] for r in ok
             if r.get("class") == cls and r.get("ttft") is not None]
        if t:
            by_class[cls] = {"n": len(t), "ttft_p50": pct(t, 0.50),
                             "ttft_p90": pct(t, 0.90),
                             "ttft_mean": round(statistics.fmean(t), 4)}
    summary["by_class"] = by_class
    hr, of = by_class.get("hot_repeat"), by_class.get("oneshot")
    if hr and of and hr["ttft_p50"]:
        # >1 means re-sent prefixes were served faster than fresh ones,
        # i.e. the cache is doing work. This is the headline number to
        # compare across policy arms at FIXED host memory.
        summary["cache_advantage_p50"] = round(
            of["ttft_p50"] / hr["ttft_p50"], 3)

    if args.status_dir:
        after = read_status(args.status_dir, args.fresh_within)
        summary["ngkv_before"] = before
        summary["ngkv_after"] = after
        summary["ngkv_delta"] = status_delta(before, after)
        dd = summary["ngkv_delta"]
        print("\nngkv delta:")
        for k in ("decisions", "admitted", "denied", "relaxed",
                  "decisions_under_pressure", "admitted_under_pressure",
                  "denial_rate_under_pressure", "denial_rate",
                  "token_denial_rate", "surveyed"):
            if k in dd:
                print(f"  {k}: {dd[k]}")
        for hk in _HIST_KEYS:
            if dd.get(hk):
                print(f"  {hk}: {dd[hk]}")
    m_after = fetch_metrics(args.url)
    if m_before or m_after:
        summary["metrics_before"] = m_before
        summary["metrics_after"] = m_after

    print("\nsummary:")
    for k in ("ok", "failed", "wall_seconds", "throughput_rps", "ttft_p50",
              "ttft_p99", "e2e_p50", "e2e_p99"):
        print(f"  {k}: {summary[k]}")
    if summary.get("by_class"):
        print("  TTFT by request class (cache effectiveness):")
        for cls, v in summary["by_class"].items():
            print(f"    {cls:11s} n={v['n']:4d} p50={v['ttft_p50']}s "
                  f"p90={v['ttft_p90']}s")
    if summary.get("cache_advantage_p50"):
        print(f"  cache_advantage_p50: {summary['cache_advantage_p50']}x "
              f"(oneshot TTFT / hot_repeat TTFT; 1.0 = cache doing nothing)")

    out = args.out or f"loadgen_{args.label}.json"
    json.dump(summary, open(out, "w"), indent=2, default=str)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
