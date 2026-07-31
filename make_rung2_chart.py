"""Rung-2 results chart (ngkv_rung2_results.png), same design system as
the Rung-1/1.5 charts: ink-navy / ember / teal on paper."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, EMBER, TEAL = "#16213a", "#e8590c", "#0f8b8d"
GREY, LIGHT, PAPER = "#5b6270", "#e7eaef", "#fbfaf7"

r2 = json.load(open("results_rung2.json"))
r1 = json.load(open("results_real.json"))
p2 = json.load(open("results_rung2_pooling.json"))
r15 = json.load(open("results_rung15.json"))
B = [str(b) for b in r2["budgets"]]
Bf = r2["budgets"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.7), dpi=170)
fig.patch.set_facecolor(PAPER)
for ax in (ax1, ax2):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(axis="y", color=LIGHT, lw=0.8)

def series(agg, key):
    return [agg[b][key]["mean"] for b in B]

# --- left: retained mass vs budget, Rung 2 (SmolLM2-360M) -------------
ax1.plot(Bf, series(r2["agg"], "window"), color=GREY, ls="--", lw=1.4,
         marker="o", ms=3, label="Window (sinks+recency)")
ax1.plot(Bf, series(r2["agg"], "h2o_evict"), color="#a9b0bd", lw=1.4,
         marker="o", ms=3, label="Accumulated mass")
ax1.plot(Bf, series(r2["agg"], "ng_evict"), color=TEAL, lw=2.0,
         marker="o", ms=3.5, label="NG evict")
ax1.plot(Bf, series(r2["agg"], "ng_tier"), color=INK, lw=2.0,
         marker="o", ms=3.5, label="NG tier (HBM/DRAM)")
ax1.plot(Bf, series(r2["agg"], "ng_mp"), color=EMBER, lw=2.2,
         marker="o", ms=3.5, label="NG mixed-precision")
ax1.plot(Bf, series(r2["agg"], "or_evict"), color=INK, ls=":", lw=1.6,
         label="Eviction oracle (360M)")
ax1.plot(Bf, series(r1["agg"], "or_evict"), color=GREY, ls=":", lw=1.2,
         alpha=0.65, label="Eviction oracle (Rung-1 tiny)")
ax1.set_xlabel("Keep budget B", color=GREY, fontsize=8.5)
ax1.set_ylabel("Retained attention mass", color=GREY, fontsize=8.5)
ax1.set_title("Retained mass vs budget — SmolLM2-360M, 10 traces",
              color=INK, fontsize=9.5, fontweight="bold", loc="left")
ax1.set_ylim(0.15, 1.02)
ax1.legend(fontsize=6.6, frameon=False, labelcolor=GREY, ncol=2,
           loc="lower right")

# --- right: per-layer minus shared gate, 32L vs 3L --------------------
w = 0.016
g2 = p2["gate_comparison"]; g15 = r15["gate_comparison"]
ng32 = [g2[b]["ng_evict_perlayer_vs_shared"]["mean_diff"] for b in B]
or32 = [g2[b]["oracle_evict_perlayer_vs_shared"]["mean_diff"] for b in B]
ng3 = [g15[b]["ng_evict_perlayer_vs_shared"]["mean_diff"] for b in B]
or3 = [g15[b]["oracle_evict_perlayer_vs_shared"]["mean_diff"] for b in B]
x = np.array(Bf)
ax2.bar(x - 1.5 * w, or32, w, color=INK, label="Oracle gain, 32L (360M)")
ax2.bar(x - 0.5 * w, ng32, w, color=TEAL, label="NG-evict gain, 32L (360M)")
ax2.bar(x + 0.5 * w, or3, w, color=INK, alpha=0.32,
        label="Oracle gain, 3L (Rung 1.5)")
ax2.bar(x + 1.5 * w, ng3, w, color=TEAL, alpha=0.32,
        label="NG-evict gain, 3L (Rung 1.5)")
ax2.set_xlabel("Keep budget B", color=GREY, fontsize=8.5)
ax2.set_ylabel("Per-layer minus shared gate", color=GREY, fontsize=8.5)
ax2.set_title("Per-layer gating gain: 32 layers vs 3 layers",
              color=INK, fontsize=9.5, fontweight="bold", loc="left")
ax2.set_xticks(Bf)
ax2.legend(fontsize=6.8, frameon=False, labelcolor=GREY)

fig.tight_layout(pad=1.1)
fig.savefig("ngkv_rung2_results.png", facecolor=PAPER,
            bbox_inches="tight")
print("chart -> ngkv_rung2_results.png")
