"""Replay real attention traces through NG-KV policies.

Same policy families and within-family plug-in regret as
run_simulation.py, but over .npz traces (ngkv.traces schema) — the same
entry point Tier-2 production traces will use.

Adds the significance layer:
  * paired Wilcoxon signed-rank test (NG-evict vs H2O per trace)
  * paired bootstrap 95% CI on the mean quality difference
following the paired-comparison discipline (the trace, not the step, is
the unit of analysis — steps within a trace are dependent).
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from ngkv import (MixedPrecisionPolicy, NecessityConfig, NecessityScorer,
                  TierBudget, TieredPlacementPolicy, Tier, retained_mass)
from ngkv.traces import load_traces

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
SINKS = 4


def replay(attn: np.ndarray, P: int, B: float) -> dict:
    D, total = attn.shape
    scorer = NecessityScorer(NecessityConfig(sink_tokens=SINKS))
    h2o_mass = np.zeros(total)
    q = {k: [] for k in ["window", "h2o_evict", "ng_evict", "ng_tier", "ng_mp",
                          "or_evict", "or_tier", "or_mp"]}
    bytes_saved, dram_hits = [], []
    tier_policy = TieredPlacementPolicy(TierBudget(hbm_frac=B, dram_frac=B))
    mp_policy = MixedPrecisionPolicy(bit_budget_frac=B)

    for t in range(D):
        cache_len = P + t
        row = attn[t]
        s = scorer.scores()
        sp = np.full(cache_len, np.inf)
        sp[: min(s.shape[0], cache_len)] = s[:cache_len]
        so = row[:cache_len].copy()          # clairvoyant oracle
        k = max(1, int(B * cache_len))

        w = np.zeros(total); w[:SINKS] = 1.0
        w[max(0, cache_len - (k - SINKS)):cache_len] = 1.0
        q["window"].append(retained_mass(row, w))

        for name, sc in [("h2o_evict", h2o_mass[:cache_len] + 1e9 * (np.arange(cache_len) < SINKS)),
                          ("ng_evict", sp), ("or_evict", so)]:
            wv = np.zeros(total); wv[np.argsort(-sc)[:k]] = 1.0
            q[name].append(retained_mass(row, wv))

        for name, sc in [("ng_tier", sp), ("or_tier", so)]:
            pl = tier_policy.place(sc)
            tw = np.zeros(total)
            tw[:cache_len][pl != Tier.EVICT] = 1.0
            q[name].append(retained_mass(row, tw))
            if name == "ng_tier":
                dram_hits.append(float(row[:cache_len][pl == Tier.DRAM].sum()))
                bytes_saved.append(float((pl == Tier.EVICT).mean()))

        for name, sc in [("ng_mp", sp), ("or_mp", so)]:
            bits = mp_policy.allocate(sc)
            fw = np.zeros(total); fw[:cache_len] = mp_policy.fidelity_of(bits)
            q[name].append(retained_mass(row, fw))

        scorer.observe(row[:cache_len])
        h2o_mass[:cache_len] += row[:cache_len]

    out = {k: float(np.mean(v)) for k, v in q.items()}
    out["bytes_saved_frac"] = float(np.mean(bytes_saved))
    out["dram_hit_mass"] = float(np.mean(dram_hits))
    return out


def paired_tests(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 0):
    """a, b: per-trace quality of two policies. Returns Wilcoxon p and
    bootstrap 95% CI of mean(a - b)."""
    d = a - b
    try:
        stat, p = stats.wilcoxon(d)
    except ValueError:
        stat, p = np.nan, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(p), (float(lo), float(hi)), float(d.mean())


def main():
    traces = list(load_traces("traces_rung2"))
    print(f"{len(traces)} real traces loaded "
          f"(model: {traces[0][3].get('model')}, val_loss {traces[0][3].get('val_loss')})")
    per = {B: {} for B in BUDGETS}
    for B in BUDGETS:
        rows = [replay(attn, P, B) for _, attn, P, _ in traces]
        keys = rows[0].keys()
        per[B] = {k: np.array([r[k] for r in rows]) for k in keys}

    agg = {str(B): {k: {"mean": float(v.mean()), "std": float(v.std())}
                    for k, v in per[B].items()} for B in BUDGETS}

    sig = {}
    for B in BUDGETS:
        p, ci, diff = paired_tests(per[B]["ng_evict"], per[B]["h2o_evict"])
        sig[str(B)] = {"ng_vs_h2o": {"mean_diff": diff, "wilcoxon_p": p, "boot_ci95": ci}}
        p2, ci2, diff2 = paired_tests(per[B]["ng_mp"], per[B]["ng_evict"])
        sig[str(B)]["mp_vs_evict"] = {"mean_diff": diff2, "wilcoxon_p": p2, "boot_ci95": ci2}

    json.dump({"budgets": BUDGETS, "n_traces": len(traces), "agg": agg,
               "significance": sig}, open("results_rung2.json", "w"), indent=2)

    print(f"\n{'B':>5} | {'window':>7} {'h2o':>7} {'ng-ev':>7} | {'ng-tier':>7} {'ng-mp':>7} {'oracle':>7} | "
          f"rgt-ev  d(ng-h2o)  Wilcoxon-p  CI95")
    for B in BUDGETS:
        a = {k: per[B][k].mean() for k in per[B]}
        s_ = sig[str(B)]["ng_vs_h2o"]
        print(f"{B:>5.2f} | {a['window']:>7.4f} {a['h2o_evict']:>7.4f} {a['ng_evict']:>7.4f} | "
              f"{a['ng_tier']:>7.4f} {a['ng_mp']:>7.4f} {a['or_evict']:>7.4f} | "
              f"{a['or_evict']-a['ng_evict']:>6.4f}  {s_['mean_diff']:>+8.4f}  {s_['wilcoxon_p']:>9.2g}  "
              f"[{s_['boot_ci95'][0]:+.4f},{s_['boot_ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
