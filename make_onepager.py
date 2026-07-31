"""Build the NG-KV one-pager PDF (ink-navy / ember / teal system)."""
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

INK, EMBER, TEAL = HexColor("#16213a"), HexColor("#e8590c"), HexColor("#0f8b8d")
GREY, LIGHT, PAPER = HexColor("#5b6270"), HexColor("#e7eaef"), HexColor("#fbfaf7")

W, H = A4
M = 14 * mm
r = json.load(open("results.json"))
A = r["agg"]

c = canvas.Canvas("NGKV_one_pager.pdf", pagesize=A4)
c.setFillColor(PAPER); c.rect(0, 0, W, H, fill=1, stroke=0)

# ---- header band -----------------------------------------------------
c.setFillColor(INK); c.rect(0, H - 34 * mm, W, 34 * mm, fill=1, stroke=0)
c.setFillColor(EMBER); c.rect(0, H - 34 * mm, W, 1.2 * mm, fill=1, stroke=0)
c.setFillColor(HexColor("#ffffff"))
c.setFont("Helvetica-Bold", 21)
c.drawString(M, H - 16 * mm, "NG-KV: Necessity-Gated KV Cache Management")
c.setFont("Helvetica", 10.5)
c.setFillColor(HexColor("#c9d2e4"))
c.drawString(M, H - 23 * mm, "A plug-in framework for retention, tiering, and precision in LLM serving — with oracle-regret certification.")
c.setFont("Helvetica-Oblique", 8.5)
c.drawString(M, H - 29 * mm, "Syni open research stack  ·  Apache 2.0  ·  grounded in SIVAM TR-01 (Necessity-Gated Activation Modulation)  ·  July 2026")

y = H - 42 * mm
def heading(txt, yy):
    c.setFillColor(EMBER); c.rect(M, yy - 1.2, 7 * mm, 2.2, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 11)
    c.drawString(M + 9 * mm, yy - 2.2, txt)
    return yy - 7 * mm

def body(lines, yy, size=8.6, leading=11.4, color=GREY, x=M, bold_first=None):
    c.setFont("Helvetica", size); c.setFillColor(color)
    for ln in lines:
        if "|" in ln:
            pre, rest = ln.split("|", 1)
            c.setFont("Helvetica-Bold", size); c.setFillColor(INK)
            c.drawString(x, yy, pre)
            c.setFont("Helvetica", size); c.setFillColor(color)
            c.drawString(x + c.stringWidth(pre, "Helvetica-Bold", size), yy, rest)
        else:
            c.drawString(x, yy, ln)
        yy -= leading
    return yy

# ---- thesis ----------------------------------------------------------
y = heading("Thesis", y)
y = body([
 "Every KV-cache decision — evict or keep, HBM or host tier, 16 bits or 4 — is a necessity gate: an online estimate of how vital a",
 "cached token is to future decoding quality. LRU, sliding windows, and H2O-style accumulated-attention eviction are all implicit",
 "plug-in estimators of that necessity, with no regret story. NG-KV makes the estimator explicit (attention mass + burstiness +",
 "recency + pinned sinks), generalizes the policies it feeds (tiered placement for disaggregated serving; necessity-allocated mixed",
 "precision), and ships the evaluation methodology: replay logged traces, compute the clairvoyant oracle gate, report plug-in regret.",
 "Near-zero regret at budget B certifies near-optimality — a statement a 'quality metric didn't move' A/B gate cannot make.",
], y)
y -= 3 * mm

# ---- results figure --------------------------------------------------
y = heading("Simulation results  (structural traces: sinks, recency, heavy hitters, revival — 8 seeds, 2K prompt + 1K decode)", y)
img = ImageReader("ngkv_results.png")
iw, ih = img.getSize(); scale = (W - 2 * M) / iw
c.drawImage(img, M, y - ih * scale, width=W - 2 * M, height=ih * scale)
y -= ih * scale + 5 * mm

# ---- headline numbers ------------------------------------------------
y = heading("What moves, measurably", y)
b02, b04 = A["0.2"], A["0.4"]
y = body([
 f"5x compression (B=0.20):| necessity scoring recovers {b02['ng_evict']['mean']:.2f} of attention mass vs {b02['h2o_evict']['mean']:.2f} for accumulated-mass (H2O-style)",
 "   eviction and 0.51 for sliding windows. The gap is the revival regime: burstiness-aware necessity anticipates re-attention to",
 "   dormant spans that pure history cannot.",
 f"Smooth beats hard:| necessity-allocated mixed precision dominates eviction at every budget ({b04['ng_mp']['mean']:.3f} at B=0.40 — within 0.3% of",
 "   the ceiling under stated fidelity assumptions). This is the precision-of-action corollary of TR-01 made operational.",
 f"Disaggregated serving (4P2D-shaped):| at B=0.20 tiering, {b02['bytes_saved_frac']['mean']:.0%} of cache blocks are never shipped to the remote tier — direct",
 "   interconnect (EFA) byte savings — while the DRAM tier serves the mid-necessity mass HBM-only retention would lose.",
 f"Certified headroom:| plug-in regret quantifies distance from the oracle per family and budget (evict: {A['0.2']['or_evict']['mean']-A['0.2']['ng_evict']['mean']:.2f} at B=0.2, shrinking to",
 f"   {A['0.4']['or_mp']['mean']-A['0.4']['ng_mp']['mean']:.3f} for mixed precision at B=0.4). Regret is computable offline on production traces today.",
], y, bold_first="")
# rewrite with bold markers
y -= 2 * mm

# ---- plugin architecture --------------------------------------------
y = heading("Plugin architecture: any model class, any stack", y)
y = body([
 "Core is framework-agnostic (NumPy; scorer is O(T) state, microseconds per step) and binds via thin adapters:  HuggingFace —",
 "working reference adapter for real-trace validation on any decoder (dense or MoE).  vLLM — KVConnector admission/prefetch",
 "policy: score 16-token blocks (max over tokens), never ship EVICT-tier blocks, prefetch by descending necessity.  SGLang —",
 "HiCache admission + eviction ranking over radix-tree nodes; LFU is the degenerate case (variance, recency weights = 0), so the",
 "generalization is strict.  Production necessity signals need no attention materialization: Quest-style block key statistics, prefix-",
 "cache hit counts, MoE router entropy. Applies uniformly across GLM / DeepSeek / Kimi / Qwen-class MoE and dense models.",
], y)
y -= 2 * mm

# ---- stated plainly --------------------------------------------------
c.setFillColor(LIGHT); c.roundRect(M, y - 21 * mm, W - 2 * M, 21 * mm, 2 * mm, fill=1, stroke=0)
c.setFillColor(INK); c.setFont("Helvetica-Bold", 9)
c.drawString(M + 4 * mm, y - 5.5 * mm, "Stated plainly (limitations)")
c.setFont("Helvetica", 8.2); c.setFillColor(GREY)
for i, ln in enumerate([
 "These are structural simulations built for relative policy ordering, not absolute production claims. Traces encode documented attention",
 "regularities but are not replays of real model attention; quantization fidelity factors are modeling assumptions from the KV-quantization",
 "literature. Validation path before any production claim: (1) real-trace replay via the HF adapter, (2) end-task quality behind gates, not",
 "mass proxies, (3) SLA-goodput measurement in the target stack. The oracle-regret methodology is unchanged at every stage.",
]):
    c.drawString(M + 4 * mm, y - (9.5 + i * 3.9) * mm, ln)
y -= 25 * mm

# ---- footer ----------------------------------------------------------
c.setFillColor(TEAL); c.rect(0, 10 * mm, W, 0.8 * mm, fill=1, stroke=0)
c.setFillColor(GREY); c.setFont("Helvetica", 7.5)
c.drawString(M, 6 * mm, "NG-KV v0.1 — code, simulation harness, and results.json reproduce this page: python run_simulation.py")
c.drawRightString(W - M, 6 * mm, "Syni — Intelligence isn't performed. It's grown.  ·  syni.world")

c.save()
print("one-pager built")
