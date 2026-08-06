#!/usr/bin/env python3
"""Fit NGKV_EVICT weights from observe-mode survey histograms.

Consumes the status JSONs (and/or loadgen run artifacts) accumulated in
observe mode and proposes weights + an SLRU-equivalent threshold, with the
reasoning printed so the fit is auditable rather than oracular.

Usage:
  python3 fit_evict_weights.py /mnt/local-nvme/ngkv-status [more dirs...]
  python3 fit_evict_weights.py run_survey.json
"""
import json, math, os, sys

HKEYS = ("hit_count_hist", "parent_hits_hist", "depth_hist", "keylen_hist")


def load(paths):
    agg = {k: {} for k in HKEYS}
    n = 0
    for p in paths:
        files = ([os.path.join(p, f) for f in os.listdir(p)]
                 if os.path.isdir(p) else [p])
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                doc = json.load(open(f))
            except Exception:
                continue
            st = doc.get("stats") or doc.get("ngkv_delta") or {}
            for k in HKEYS:
                for b, c in (st.get(k) or {}).items():
                    agg[k][b] = agg[k].get(b, 0) + int(c)
                    n += int(c)
    return agg, n


def dist(hist):
    items = sorted(((int(k.rstrip("+")), v) for k, v in hist.items()))
    total = sum(v for _, v in items) or 1
    mean = sum(k * v for k, v in items) / total
    var = sum(v * (k - mean) ** 2 for k, v in items) / total
    def pctl(q):
        acc = 0
        for k, v in items:
            acc += v
            if acc / total >= q:
                return k
        return items[-1][0] if items else 0
    return {"n": total, "mean": round(mean, 2), "std": round(var ** .5, 2),
            "p50": pctl(.5), "p90": pctl(.9), "mass_at_min":
            round(items[0][1] / total, 3) if items else None,
            "buckets": dict(items[:12])}


def main():
    agg, n = load(sys.argv[1:] or ["/mnt/local-nvme/ngkv-status"])
    if not n:
        sys.exit("no histogram data found — need v0.16+ status files")
    d = {k: dist(v) for k, v in agg.items() if v}
    print("signal distributions:")
    for k, v in d.items():
        print(f"  {k}: {v}")

    hc, ph = d.get("hit_count_hist", {}), d.get("parent_hits_hist", {})
    w = {"w_hits": 1.0, "w_parent": 0.3, "w_depth": 0.05, "w_recency": 0.2}
    notes = []
    if hc and hc.get("std", 0) < 0.5:
        w["w_hits"], w["w_parent"] = 0.3, 1.0
        notes.append("hit_count nearly constant at admission "
                     f"(std={hc['std']}) -> weight shifted to parent chain")
    if ph and ph.get("std", 0) > 1.0:
        spread = min(3.0, 1.0 + math.log1p(ph["std"]))
        w["w_parent"] = round(w["w_parent"] * spread, 2)
        notes.append(f"parent_hits informative (std={ph['std']}) -> "
                     f"w_parent scaled x{spread:.1f}")
    dep = d.get("depth_hist", {})
    if dep and dep.get("std", 0) < 0.5:
        w["w_depth"] = 0.0
        notes.append("depth nearly constant -> depth prior disabled")
    print("\nproposed NGKV_EVICT:")
    print("  " + json.dumps(w))
    print("SLRU protected_threshold suggestion: "
          f"{max(2, (hc.get('p90') or 2))} (p90 of hit_count)")
    for note in notes:
        print("  note:", note)
    print("\nCaveat: fitted from ADMISSION-time samples; eviction-time "
          "distributions can differ. Treat as a starting point for the "
          "shadow run, not a conclusion.")


if __name__ == "__main__":
    main()
