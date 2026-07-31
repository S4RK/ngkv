"""Rung 1.5: pooling-sensitivity replay.

Tests whether the mean-pooling caveat from the v0.2 addendum changes
the policy ordering. Four views over the same 10 decodes
(traces_rung2/*.npz, which carry attn (mean-pooled), attn_layers
(per-layer head-mean), attn_max (max-pooled)):

  A  pooled_eval      score from pooled attn, evaluate against pooled
                      attn — the v0.2 measurement (baseline).
  B  pooled_perlayer  score from pooled attn (one shared keep-set per
                      step, as a shared gate would deploy), evaluate
                      retained mass against each layer's true attention
                      and average — removes the measurement bias of A.
                      The shared clairvoyant oracle is top-k of the
                      pooled row (it maximizes mean-over-layers mass).
  C  perlayer_gate    independent scorer/policy/oracle per layer, each
                      with layer-local budget B; quality = mean over
                      layers — the per-layer gating upper path.
  D  max_eval         score+evaluate with max(layers,heads) reduction —
                      alternative-reduction ablation.

Same regret + paired Wilcoxon/bootstrap methodology as replay_real.py
(trace = unit of analysis).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

from ngkv import (MixedPrecisionPolicy, NecessityConfig, NecessityScorer,
                  TierBudget, TieredPlacementPolicy, Tier, retained_mass)

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
SINKS = 4
POLICIES = ["window", "h2o_evict", "ng_evict", "ng_tier", "ng_mp",
            "or_evict", "or_tier", "or_mp"]


def load_rung15(directory="traces_rung2"):
    out = []
    for p in sorted(pathlib.Path(directory).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        out.append((p.stem,
                    z["attn"].astype(np.float64),
                    z["attn_layers"].astype(np.float64),
                    z["attn_max"].astype(np.float64),
                    int(z["prompt_len"]),
                    json.loads(str(z["meta_json"]))))
    return out


def step_masks(row_score, h2o_mass, cache_len, total, B, tier_policy, mp_policy,
               oracle_row):
    """Return {policy: weight vector in [0,1] of length total} for one step.

    row_score: necessity scores (plug-in); oracle_row: clairvoyant scores.
    """
    k = max(1, int(B * cache_len))
    masks = {}

    w = np.zeros(total); w[:SINKS] = 1.0
    w[max(0, cache_len - (k - SINKS)):cache_len] = 1.0
    masks["window"] = w

    for name, sc in [("h2o_evict", h2o_mass[:cache_len] + 1e9 * (np.arange(cache_len) < SINKS)),
                     ("ng_evict", row_score), ("or_evict", oracle_row)]:
        wv = np.zeros(total); wv[np.argsort(-sc)[:k]] = 1.0
        masks[name] = wv

    for name, sc in [("ng_tier", row_score), ("or_tier", oracle_row)]:
        pl = tier_policy.place(sc)
        tw = np.zeros(total)
        tw[:cache_len][pl != Tier.EVICT] = 1.0
        masks[name] = tw

    for name, sc in [("ng_mp", row_score), ("or_mp", oracle_row)]:
        bits = mp_policy.allocate(sc)
        fw = np.zeros(total); fw[:cache_len] = mp_policy.fidelity_of(bits)
        masks[name] = fw
    return masks


def replay_shared(attn_score, attn_eval_layers, B):
    """Views A/B/D: one shared gate from attn_score rows; evaluate against
    attn_eval_layers, a list of (D,total) arrays whose retained mass is
    averaged. Pass [attn_score] to recover the score==eval view."""
    D, total = attn_score.shape
    P = total - D
    scorer = NecessityScorer(NecessityConfig(sink_tokens=SINKS))
    h2o = np.zeros(total)
    tier_policy = TieredPlacementPolicy(TierBudget(hbm_frac=B, dram_frac=B))
    mp_policy = MixedPrecisionPolicy(bit_budget_frac=B)
    q = {p: [] for p in POLICIES}
    for t in range(D):
        cl = P + t
        srow = attn_score[t]
        s = scorer.scores()
        sp = np.full(cl, np.inf); sp[: min(s.shape[0], cl)] = s[:cl]
        masks = step_masks(sp, h2o, cl, total, B, tier_policy, mp_policy,
                           srow[:cl].copy())
        for p, wv in masks.items():
            q[p].append(np.mean([retained_mass(ae[t], wv)
                                 for ae in attn_eval_layers]))
        scorer.observe(srow[:cl])
        h2o[:cl] += srow[:cl]
    return {p: float(np.mean(v)) for p, v in q.items()}


def replay_perlayer(attn_layers, B):
    """View C: independent gate per layer, layer-local budget B."""
    L, D, total = attn_layers.shape
    P = total - D
    scorers = [NecessityScorer(NecessityConfig(sink_tokens=SINKS)) for _ in range(L)]
    h2o = np.zeros((L, total))
    tier_policy = TieredPlacementPolicy(TierBudget(hbm_frac=B, dram_frac=B))
    mp_policy = MixedPrecisionPolicy(bit_budget_frac=B)
    q = {p: [] for p in POLICIES}
    for t in range(D):
        cl = P + t
        per_layer_q = {p: [] for p in POLICIES}
        for li in range(L):
            lrow = attn_layers[li, t]
            s = scorers[li].scores()
            sp = np.full(cl, np.inf); sp[: min(s.shape[0], cl)] = s[:cl]
            masks = step_masks(sp, h2o[li], cl, total, B, tier_policy,
                               mp_policy, lrow[:cl].copy())
            for p, wv in masks.items():
                per_layer_q[p].append(retained_mass(lrow, wv))
            scorers[li].observe(lrow[:cl])
            h2o[li, :cl] += lrow[:cl]
        for p in POLICIES:
            q[p].append(np.mean(per_layer_q[p]))
    return {p: float(np.mean(v)) for p, v in q.items()}


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
    traces = load_rung15()
    print(f"{len(traces)} traces (model: {traces[0][5].get('model')})")
    views = {}
    for vname in ["pooled_eval", "pooled_perlayer", "perlayer_gate", "max_eval"]:
        views[vname] = {B: {} for B in BUDGETS}

    for B in BUDGETS:
        rows = {v: [] for v in views}
        for _, a_mean, a_layers, a_max, P, _ in traces:
            layers_list = [a_layers[i] for i in range(a_layers.shape[0])]
            rows["pooled_eval"].append(replay_shared(a_mean, [a_mean], B))
            rows["pooled_perlayer"].append(replay_shared(a_mean, layers_list, B))
            rows["perlayer_gate"].append(replay_perlayer(a_layers, B))
            rows["max_eval"].append(replay_shared(a_max, [a_max], B))
        for v in views:
            views[v][B] = {p: np.array([r[p] for r in rows[v]]) for p in POLICIES}

    agg, sig = {}, {}
    for v in views:
        agg[v] = {str(B): {p: {"mean": float(x.mean()), "std": float(x.std())}
                           for p, x in views[v][B].items()} for B in BUDGETS}
        sig[v] = {}
        for B in BUDGETS:
            p1, ci1, d1 = paired_tests(views[v][B]["ng_evict"], views[v][B]["h2o_evict"])
            p2, ci2, d2 = paired_tests(views[v][B]["ng_mp"], views[v][B]["ng_evict"])
            sig[v][str(B)] = {
                "ng_vs_h2o": {"mean_diff": d1, "wilcoxon_p": p1, "boot_ci95": ci1},
                "mp_vs_evict": {"mean_diff": d2, "wilcoxon_p": p2, "boot_ci95": ci2},
            }
    # shared-vs-perlayer gate comparison (the pooling-caveat quantity):
    gate_cmp = {}
    for B in BUDGETS:
        p3, ci3, d3 = paired_tests(views["perlayer_gate"][B]["ng_evict"],
                                   views["pooled_perlayer"][B]["ng_evict"])
        p4, ci4, d4 = paired_tests(views["perlayer_gate"][B]["or_evict"],
                                   views["pooled_perlayer"][B]["or_evict"])
        gate_cmp[str(B)] = {
            "ng_evict_perlayer_vs_shared": {"mean_diff": d3, "wilcoxon_p": p3, "boot_ci95": ci3},
            "oracle_evict_perlayer_vs_shared": {"mean_diff": d4, "wilcoxon_p": p4, "boot_ci95": ci4},
        }

    json.dump({"budgets": BUDGETS, "n_traces": len(traces), "agg": agg,
               "significance": sig, "gate_comparison": gate_cmp},
              open("results_rung2_pooling.json", "w"), indent=2)

    for v in views:
        print(f"\n== view: {v}")
        print(f"{'B':>5} | {'window':>7} {'h2o':>7} {'ng-ev':>7} | {'ng-tier':>7} "
              f"{'ng-mp':>7} {'or-ev':>7} | rgt-ev  d(ng-h2o)   Wilcoxon-p")
        for B in BUDGETS:
            a = {p: views[v][B][p].mean() for p in POLICIES}
            s_ = sig[v][str(B)]["ng_vs_h2o"]
            print(f"{B:>5.2f} | {a['window']:>7.4f} {a['h2o_evict']:>7.4f} "
                  f"{a['ng_evict']:>7.4f} | {a['ng_tier']:>7.4f} {a['ng_mp']:>7.4f} "
                  f"{a['or_evict']:>7.4f} | {a['or_evict']-a['ng_evict']:>6.4f}  "
                  f"{s_['mean_diff']:>+8.4f}  {s_['wilcoxon_p']:>10.3g}")
    print("\n== shared-gate cost (per-layer minus shared, same per-layer ground truth)")
    for B in BUDGETS:
        g = gate_cmp[str(B)]
        n_, o_ = g["ng_evict_perlayer_vs_shared"], g["oracle_evict_perlayer_vs_shared"]
        print(f"B={B:.2f}: NG-evict {n_['mean_diff']:+.4f} (p={n_['wilcoxon_p']:.3g}) | "
              f"oracle {o_['mean_diff']:+.4f} (p={o_['wilcoxon_p']:.3g})")


if __name__ == "__main__":
    main()
