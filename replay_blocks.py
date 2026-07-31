"""Rung 2.5: block-granularity replay.

Answers the deployment question raised by paged serving engines: how
much of the token-level gain survives when the gate acts on 16-token
blocks (vLLM's unit) instead of tokens?

Replays the Rung-2 traces (mean-pooled view) with the necessity gate,
the clairvoyant oracle, and mixed precision all operating at block
granularities {1, 8, 16, 32}, with max- and mean-within-block pooling
of scores. Selection budget is enforced in tokens (k = B * cache_len,
rounded down to whole blocks with a 1-block floor) so granularities
are compared at equal memory.

Same regret + paired Wilcoxon/bootstrap methodology; trace = unit of
analysis. Writes results_rung25_blocks.json.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from ngkv import (MixedPrecisionPolicy, NecessityConfig, NecessityScorer,
                  retained_mass)
from ngkv.block import expand_from_blocks, pool_to_blocks
from ngkv.traces import load_traces

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
GRANULARITIES = [1, 8, 16, 32]
REDUCTIONS = ["max", "mean"]
SINKS = 4
TRACE_DIR = "traces_rung2"
OUT_JSON = "results_rung25_blocks.json"


def replay(attn: np.ndarray, P: int, B: float, bs: int, reduce: str) -> dict:
    D, total = attn.shape
    scorer = NecessityScorer(NecessityConfig(sink_tokens=SINKS))
    mp = MixedPrecisionPolicy(bit_budget_frac=B)
    q = {k: [] for k in ["ng_evict", "or_evict", "ng_mp"]}
    for t in range(D):
        cl = P + t
        row = attn[t]
        s = scorer.scores()
        sp = np.full(cl, np.inf)
        sp[: min(s.shape[0], cl)] = s[:cl]
        so = row[:cl].copy()

        k_tok = max(1, int(B * cl))
        if bs == 1:
            for name, sc in [("ng_evict", sp), ("or_evict", so)]:
                w = np.zeros(total)
                w[np.argsort(-sc)[:k_tok]] = 1.0
                q[name].append(retained_mass(row, w))
            bits = mp.allocate(sp)
            fw = np.zeros(total)
            fw[:cl] = mp.fidelity_of(bits)
            q["ng_mp"].append(retained_mass(row, fw))
        else:
            k_blk = max(1, k_tok // bs)
            for name, sc in [("ng_evict", sp), ("or_evict", so)]:
                bsc = pool_to_blocks(sc, bs, reduce)
                keep = np.zeros(bsc.shape[0])
                keep[np.argsort(-bsc)[:k_blk]] = 1.0
                w = np.zeros(total)
                w[:cl] = expand_from_blocks(keep, bs, cl)
                q[name].append(retained_mass(row, w))
            bsc = pool_to_blocks(sp, bs, reduce)
            bits = mp.allocate(bsc)
            fw = np.zeros(total)
            fw[:cl] = expand_from_blocks(mp.fidelity_of(bits), bs, cl)
            q["ng_mp"].append(retained_mass(row, fw))

        scorer.observe(row[:cl])
    return {k: float(np.mean(v)) for k, v in q.items()}


def paired_tests(a, b, n_boot=20000, seed=0):
    d = a - b
    try:
        _, p = stats.wilcoxon(d)
    except ValueError:
        p = 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(p), (float(lo), float(hi)), float(d.mean())


def main():
    traces = list(load_traces(TRACE_DIR))
    print(f"{len(traces)} traces (model: {traces[0][3].get('model')})")
    grid = {}
    for bs in GRANULARITIES:
        reds = ["max"] if bs == 1 else REDUCTIONS
        for red in reds:
            key = f"bs{bs}_{red}" if bs > 1 else "token"
            grid[key] = {}
            for B in BUDGETS:
                rows = [replay(a, P, B, bs, red) for _, a, P, _ in traces]
                grid[key][str(B)] = {k: np.array([r[k] for r in rows])
                                     for k in rows[0]}

    agg = {key: {b: {k: {"mean": float(v.mean()), "std": float(v.std())}
                     for k, v in per.items()}
                 for b, per in bud.items()} for key, bud in grid.items()}

    # block cost vs token granularity, paired per trace
    cost = {}
    for key in grid:
        if key == "token":
            continue
        cost[key] = {}
        for B in BUDGETS:
            b = str(B)
            p1, ci1, d1 = paired_tests(grid[key][b]["ng_evict"],
                                       grid["token"][b]["ng_evict"])
            p2, ci2, d2 = paired_tests(grid[key][b]["or_evict"],
                                       grid["token"][b]["or_evict"])
            cost[key][b] = {
                "ng_evict_vs_token": {"mean_diff": d1, "wilcoxon_p": p1,
                                      "boot_ci95": ci1},
                "oracle_vs_token": {"mean_diff": d2, "wilcoxon_p": p2,
                                    "boot_ci95": ci2},
            }

    json.dump({"budgets": BUDGETS, "granularities": GRANULARITIES,
               "n_traces": len(traces), "agg": agg, "block_cost": cost},
              open(OUT_JSON, "w"), indent=2)

    print(f"\n{'view':>10} | " + " ".join(f"B={b:.1f}" for b in BUDGETS) +
          "   (ng_evict retained mass)")
    for key in grid:
        vals = " ".join(f"{grid[key][str(B)]['ng_evict'].mean():.4f}"
                        for B in BUDGETS)
        print(f"{key:>10} | {vals}")
    print(f"\n{'view':>10} | block cost vs token at B=0.2 "
          f"(ng / oracle, mean diff, Wilcoxon p)")
    for key, c in cost.items():
        n_, o_ = c["0.2"]["ng_evict_vs_token"], c["0.2"]["oracle_vs_token"]
        print(f"{key:>10} | ng {n_['mean_diff']:+.4f} (p={n_['wilcoxon_p']:.3g})"
              f"  oracle {o_['mean_diff']:+.4f} (p={o_['wilcoxon_p']:.3g})")


if __name__ == "__main__":
    main()
