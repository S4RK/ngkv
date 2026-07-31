"""Run the ngkv policy comparison.

Families (each evaluated with plug-in scores AND oracle scores; the
within-family gap is the plug-in regret, per the TR-01 regret identity):

  evict@B   — keep top-B fraction in HBM, drop rest
              plug-in scorers: h2o (accum mass), ngkv (mass+var+recency)
  tier@B+B  — HBM=B, DRAM=B, evict rest (disaggregated serving shape)
  mp@B      — mixed precision under total bit budget B x 16 bits/token

Baselines: full cache (ceiling), sliding window + sinks (StreamingLLM).
Quality proxy: mean per-step attention mass recovered (x fidelity).
"""
from __future__ import annotations
import json
import numpy as np
from ngkv import (MixedPrecisionPolicy, NecessityConfig, NecessityScorer,
                  TierBudget, TieredPlacementPolicy, Tier, TraceConfig,
                  generate_trace, oracle_scores, retained_mass)

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
SEEDS = list(range(8))

def run_trace(seed: int):
    cfg = TraceConfig(seed=seed)
    attn = generate_trace(cfg)
    D, total = attn.shape
    P = cfg.prompt_len
    out = {b: {} for b in BUDGETS}

    for B in BUDGETS:
        scorer = NecessityScorer(NecessityConfig(sink_tokens=cfg.sink_tokens))
        h2o_mass = np.zeros(total)
        q = {k: [] for k in ["window","h2o_evict","ng_evict","ng_tier","ng_mp",
                              "or_evict","or_tier","or_mp"]}
        dram_hits, bytes_saved = [], []
        tier_policy = TieredPlacementPolicy(TierBudget(hbm_frac=B, dram_frac=B))
        mp_policy = MixedPrecisionPolicy(bit_budget_frac=B)

        for t in range(D):
            cache_len = P + t
            row = attn[t]

            # scores: plug-in and oracle
            s = scorer.scores()
            sp = np.full(cache_len, np.inf)          # unseen tokens -> max
            sp[: min(s.shape[0], cache_len)] = s[:cache_len]
            # Clairvoyant oracle: the current step's own attention row.
            # This is the loosest sound upper bound for the per-step
            # retained-mass metric (regret >= 0 by construction). A
            # switching-cost-constrained oracle is tighter; future work.
            so = row[:cache_len].copy()

            k = max(1, int(B * cache_len))

            # window baseline
            w = np.zeros(total); w[:cfg.sink_tokens] = 1.0
            w[max(0, cache_len - (k - cfg.sink_tokens)):cache_len] = 1.0
            q["window"].append(retained_mass(row, w))

            # evict family
            for name, sc in [("h2o_evict", h2o_mass[:cache_len] + 1e9*(np.arange(cache_len)<cfg.sink_tokens)),
                              ("ng_evict", sp), ("or_evict", so)]:
                wv = np.zeros(total); wv[np.argsort(-sc)[:k]] = 1.0
                q[name].append(retained_mass(row, wv))

            # tier family (HBM=B, DRAM=B; DRAM exact but remote)
            for name, sc in [("ng_tier", sp), ("or_tier", so)]:
                pl = tier_policy.place(sc)
                tw = np.zeros(total)
                tw[:cache_len][pl == Tier.HBM] = 1.0
                tw[:cache_len][pl == Tier.DRAM] = 1.0
                q[name].append(retained_mass(row, tw))
                if name == "ng_tier":
                    dram_hits.append(float(row[:cache_len][pl == Tier.DRAM].sum()))
                    bytes_saved.append(float((pl == Tier.EVICT).mean()))

            # mixed-precision family
            for name, sc in [("ng_mp", sp), ("or_mp", so)]:
                bits = mp_policy.allocate(sc)
                fw = np.zeros(total); fw[:cache_len] = mp_policy.fidelity_of(bits)
                q[name].append(retained_mass(row, fw))

            scorer.observe(row[:cache_len])
            h2o_mass[:cache_len] += row[:cache_len]

        for k_, v in q.items():
            out[B][k_] = float(np.mean(v))
        out[B]["dram_hit_mass"] = float(np.mean(dram_hits))
        out[B]["bytes_saved_frac"] = float(np.mean(bytes_saved))
    return out

def main():
    per_seed = [run_trace(s) for s in SEEDS]
    keys = per_seed[0][BUDGETS[0]].keys()
    agg = {b: {k: {"mean": float(np.mean([r[b][k] for r in per_seed])),
                   "std":  float(np.std ([r[b][k] for r in per_seed]))}
               for k in keys} for b in BUDGETS}
    with open("results.json", "w") as f:
        json.dump({"budgets": BUDGETS, "n_seeds": len(SEEDS),
                   "agg": {str(b): agg[b] for b in BUDGETS}}, f, indent=2)
    hdr = f"{'B':>5} | {'window':>7} {'h2o':>7} {'ng-ev':>7} {'or-ev':>7} | {'ng-tier':>7} {'or-tier':>7} | {'ng-mp':>7} {'or-mp':>7} | rgt-ev rgt-tier rgt-mp"
    print(hdr)
    for B in BUDGETS:
        a = {k: agg[B][k]["mean"] for k in keys}
        print(f"{B:>5.2f} | {a['window']:>7.4f} {a['h2o_evict']:>7.4f} {a['ng_evict']:>7.4f} {a['or_evict']:>7.4f} | "
              f"{a['ng_tier']:>7.4f} {a['or_tier']:>7.4f} | {a['ng_mp']:>7.4f} {a['or_mp']:>7.4f} | "
              f"{a['or_evict']-a['ng_evict']:>6.4f} {a['or_tier']-a['ng_tier']:>8.4f} {a['or_mp']-a['ng_mp']:>6.4f}")

if __name__ == "__main__":
    main()
