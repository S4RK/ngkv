"""Scorer bake-off on real traces (Rung 2, mean-pooled view).

Isolates SCORING quality: every method runs the same eviction policy
(keep top-k under budget B, sinks pinned) on identical traces; only the
score differs. Metric: retained attention mass; trace = unit; paired
Wilcoxon + bootstrap.

Baselines (decode-online adaptations, faithful to each paper's signal):
  window   — StreamingLLM: sinks + recency only
  h2o      — accumulated attention mass
  tova     — most recent query's attention row
  snapkv*  — mean attention over the last W=16 observed steps, 1-D
             max-pooled (width 7). (*SnapKV is prefill-time in the
             paper; this is its observation-window signal run online.)
  ng       — NecessityScorer defaults (EWMA + variance + recency)
  oracle   — clairvoyant (this step's true attention)

Improved scorer (ng2): NG + beta * last-step attention (TOVA signal) +
optional pooled recent-window term, hyperparameters selected by
LEAVE-ONE-OUT across traces: for each held-out trace the config is
chosen on the other nine, so ng2_loo is an honest generalization
estimate, not a fit to these ten traces.

Writes results_scorer_comparison.json.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from scipy import stats

from ngkv import NecessityConfig, NecessityScorer, retained_mass
from ngkv.traces import load_traces

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
SINKS = 4
W_SNAP, POOL = 16, 7


def maxpool1d(x: np.ndarray, w: int) -> np.ndarray:
    if x.shape[0] == 0 or w <= 1:
        return x
    p = w // 2
    xp = np.pad(x, (p, p), constant_values=0)
    return np.max(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)


def replay_all(attn: np.ndarray, P: int, ng2_grid) -> dict:
    """One pass per trace; returns {scorer: {B: mean retained mass}}."""
    D, total = attn.shape
    ng = NecessityScorer(NecessityConfig(sink_tokens=SINKS))
    ng2 = [NecessityScorer(NecessityConfig(sink_tokens=SINKS,
                                           ewma_alpha=a, variance_weight=vw,
                                           recency_weight=rw))
           for (a, vw, rw, beta) in ng2_grid]
    cum = np.zeros(total)          # h2o
    recent = []                    # snapkv window
    last = np.zeros(total)         # tova
    names = ["window", "h2o", "tova", "snapkv", "ng", "oracle"] + \
            [f"ng2_{i}" for i in range(len(ng2_grid))]
    acc = {n: {B: [] for B in BUDGETS} for n in names}

    for t in range(D):
        cl = P + t
        row = attn[t]
        age = np.arange(cl)[::-1]
        scores = {
            "window": -age.astype(float),
            "h2o": cum[:cl].copy(),
            "tova": last[:cl].copy(),
            "snapkv": maxpool1d(
                np.mean([r[:cl] if r.shape[0] >= cl
                         else np.pad(r, (0, cl - r.shape[0]))
                         for r in recent], axis=0) if recent
                else np.zeros(cl), POOL),
            "ng": ng.scores()[:cl] if ng.num_tokens >= cl
                  else np.pad(ng.scores(), (0, cl - ng.num_tokens)),
            "oracle": row[:cl].copy(),
        }
        for i, (a, vw, rw, beta) in enumerate(ng2_grid):
            s = ng2[i].scores()
            s = s[:cl] if s.shape[0] >= cl else np.pad(s, (0, cl - s.shape[0]))
            scores[f"ng2_{i}"] = s + beta * last[:cl]
        obs_len = min(ng.num_tokens, cl)  # tokens observed so far
        for name, sc in scores.items():
            sc = sc.astype(float).copy()
            sc[:SINKS] = np.inf
            sc[obs_len:cl] = np.inf  # uniform keep-the-newest prior
            order = np.argsort(-sc)
            for B in BUDGETS:
                k = max(SINKS, int(B * cl))
                w = np.zeros(total)
                w[order[:k]] = 1.0
                acc[name][B].append(retained_mass(row, w))
        # observe
        obs = row[:cl]
        ng.observe(obs)
        for s2 in ng2:
            s2.observe(obs)
        cum[:cl] += obs
        last = np.zeros(total); last[:cl] = obs
        recent.append(obs)
        if len(recent) > W_SNAP:
            recent.pop(0)
    return {n: {B: float(np.mean(v)) for B, v in per.items()}
            for n, per in acc.items()}


def wilcoxon_ci(a, b, n_boot=20000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    try:
        _, p = stats.wilcoxon(d)
    except ValueError:
        p = 1.0
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(p), (float(lo), float(hi))


def main():
    ng2_grid = list(itertools.product(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],  # ewma_alpha (fast = denoised TOVA)
        [0.0, 0.25, 0.5],                # variance_weight
        [0.0, 0.1],                      # recency_weight
        [0.0, 0.5, 1.0],                 # beta on last-step attention
    ))
    traces = list(load_traces("traces_rung2"))
    per_trace = [replay_all(a, P, ng2_grid) for _, a, P, _ in traces]
    n = len(traces)
    base_names = ["window", "h2o", "tova", "snapkv", "ng", "oracle"]

    def vec(name, B):
        return np.array([pt[name][B] for pt in per_trace])

    # LOO selection for ng2 per budget
    ng2_names = [f"ng2_{i}" for i in range(len(ng2_grid))]
    loo = {B: np.zeros(n) for B in BUDGETS}
    loo_cfg = {B: [] for B in BUDGETS}
    for B in BUDGETS:
        mat = np.stack([vec(nm, B) for nm in ng2_names])  # (cfg, trace)
        for h in range(n):
            others = np.delete(np.arange(n), h)
            best = int(np.argmax(mat[:, others].mean(axis=1)))
            loo[B][h] = mat[best, h]
            loo_cfg[B].append(ng2_grid[best])

    out = {"budgets": BUDGETS, "n_traces": n,
           "agg": {}, "significance": {}, "loo_selected_configs": {}}
    print(f"{'scorer':>10} | " + "  ".join(f"B={B:.1f}" for B in BUDGETS))
    for nm in base_names:
        means = [vec(nm, B).mean() for B in BUDGETS]
        out["agg"][nm] = {str(B): m for B, m in zip(BUDGETS, means)}
        print(f"{nm:>10} | " + "  ".join(f"{m:.4f}" for m in means))
    means = [loo[B].mean() for B in BUDGETS]
    out["agg"]["ng2_loo"] = {str(B): m for B, m in zip(BUDGETS, means)}
    print(f"{'ng2_loo':>10} | " + "  ".join(f"{m:.4f}" for m in means))
    out["loo_selected_configs"] = {
        str(B): [{"ewma_alpha": c[0], "variance_weight": c[1],
                  "recency_weight": c[2], "last_step_beta": c[3]}
                 for c in loo_cfg[B]] for B in BUDGETS}

    print("\npaired vs ng (mean diff, Wilcoxon p):")
    for B in BUDGETS:
        sig = {}
        for nm in ["window", "h2o", "tova", "snapkv"]:
            d, p, ci = wilcoxon_ci(vec("ng", B), vec(nm, B))
            sig[f"ng_vs_{nm}"] = {"mean_diff": d, "wilcoxon_p": p,
                                  "boot_ci95": ci}
        d, p, ci = wilcoxon_ci(loo[B], vec("ng", B))
        sig["ng2loo_vs_ng"] = {"mean_diff": d, "wilcoxon_p": p,
                               "boot_ci95": ci}
        d, p, ci = wilcoxon_ci(vec("oracle", B), loo[B])
        sig["oracle_minus_ng2loo"] = {"mean_diff": d, "wilcoxon_p": p,
                                      "boot_ci95": ci}
        out["significance"][str(B)] = sig
        s = sig
        print(f"B={B}: vs tova {s['ng_vs_tova']['mean_diff']:+.4f} "
              f"(p={s['ng_vs_tova']['wilcoxon_p']:.3g}) | "
              f"vs snapkv {s['ng_vs_snapkv']['mean_diff']:+.4f} "
              f"(p={s['ng_vs_snapkv']['wilcoxon_p']:.3g}) | "
              f"ng2loo-ng {s['ng2loo_vs_ng']['mean_diff']:+.4f} "
              f"(p={s['ng2loo_vs_ng']['wilcoxon_p']:.3g}) | "
              f"regret(ng2loo) {s['oracle_minus_ng2loo']['mean_diff']:.4f}")

    json.dump(out, open("results_scorer_comparison.json", "w"), indent=2)
    print("\n-> results_scorer_comparison.json")


if __name__ == "__main__":
    main()
