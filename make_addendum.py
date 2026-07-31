"""Build the NG-KV real-trace addendum PDF (same design system)."""
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
real = json.load(open("results_real.json"))
A = real["agg"]; S = real["significance"]

c = canvas.Canvas("NGKV_real_trace_addendum.pdf", pagesize=A4)
c.setFillColor(PAPER); c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(INK); c.rect(0, H - 30 * mm, W, 30 * mm, fill=1, stroke=0)
c.setFillColor(TEAL); c.rect(0, H - 30 * mm, W, 1.2 * mm, fill=1, stroke=0)
c.setFillColor(HexColor("#ffffff")); c.setFont("Helvetica-Bold", 18)
c.drawString(M, H - 14 * mm, "NG-KV Addendum: First Real-Trace Regret Results")
c.setFont("Helvetica", 9.5); c.setFillColor(HexColor("#c9d2e4"))
c.drawString(M, H - 20.5 * mm, "Rung 1 of the validation ladder: replaying real decoder attention through the identical policy + regret + significance pipeline.")
c.setFont("Helvetica-Oblique", 8.5)
c.drawString(M, H - 26 * mm, "Syni open research stack  ·  NG-KV v0.2  ·  companion to the NG-KV one-pager  ·  July 2026")

y = H - 38 * mm
def heading(txt, yy, col=TEAL):
    c.setFillColor(col); c.rect(M, yy - 1.2, 7 * mm, 2.2, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 11)
    c.drawString(M + 9 * mm, yy - 2.2, txt); return yy - 7 * mm

def body(lines, yy, size=8.6, leading=11.4):
    for ln in lines:
        if "|" in ln:
            pre, rest = ln.split("|", 1)
            c.setFont("Helvetica-Bold", size); c.setFillColor(INK)
            c.drawString(M, yy, pre)
            c.setFont("Helvetica", size); c.setFillColor(GREY)
            c.drawString(M + c.stringWidth(pre, "Helvetica-Bold", size), yy, rest)
        else:
            c.setFont("Helvetica", size); c.setFillColor(GREY)
            c.drawString(M, yy, ln)
        yy -= leading
    return yy

y = heading("Setup", y)
y = body([
 "A 3-layer / 4-head / 96-dim char-level decoder was trained from scratch (JAX, CPU, val loss 2.29 — deliberately small; attention",
 "structure, not language quality, is the object of study). Ten held-out prompts (256 tokens, 120 decode steps) were decoded with",
 "exact full-context attention recorded per step (mean over layers and heads) and written to the NG-KV .npz trace schema — the",
 "same interface Tier-2 production traces use: capture anywhere, replay here. All policies, the clairvoyant oracle, and within-",
 "family plug-in regret are computed by replay_real.py, byte-identical in method to the structural-simulation harness.",
], y)
y -= 3 * mm

y = heading("Results  (10 real traces; paired Wilcoxon + bootstrap CI, trace = unit of analysis)", y)
img = ImageReader("ngkv_real_results.png")
iw, ih = img.getSize(); sc = (W - 2 * M) / iw
c.drawImage(img, M, y - ih * sc, width=W - 2 * M, height=ih * sc)
y -= ih * sc + 5 * mm

b2 = {k: A["0.2"][k]["mean"] for k in A["0.2"]}
sg2 = S["0.2"]["ng_vs_h2o"]
y = heading("Findings", y, EMBER)
y = body([
 f"1. The scoring gap survives — and grows — on real attention.| NG scoring recovers {b2['ng_evict']:.2f} vs {b2['h2o_evict']:.2f} for pure accumulated-mass",
 f"    scoring at B=0.20 (Δ=+{sg2['mean_diff']:.2f}, Wilcoxon p={sg2['wilcoxon_p']:.3g}, 95% CI [{sg2['boot_ci95'][0]:+.3f}, {sg2['boot_ci95'][1]:+.3f}]; significant at every budget). Real cause:",
 "    this model's attention is diffuse and recency-dominated; accumulated mass alone tracks neither. Stated plainly: deployed H2O",
 "    also keeps a recency window — this isolates the scoring signal, not the full deployed system.",
 f"2. Diffuse attention collapses the eviction ceiling.| The clairvoyant eviction oracle recovers only {A['0.2']['or_evict']['mean']:.2f} at B=0.20: when mass is",
 "    spread, no top-k retention can hold it. Eviction-family regret is small (0.01–0.03) — NG scoring is near-optimal *within its family*;",
 "    the family itself is the constraint. Regret separates estimator error from policy-class error: exactly what it is for.",
 f"3. Smooth gates dominate — beyond the hard-gate oracle.| Mixed precision reaches {b2['ng_mp']:.2f} at B=0.20, beating the eviction *oracle*",
 f"    ({A['0.2']['or_evict']['mean']:.2f}) outright, and {A['0.4']['ng_mp']['mean']:.2f} at B=0.40. Broad low-precision coverage beats narrow full-precision retention under diffuse",
 "    attention — the precision-of-action corollary observed in real model attention, not just postulated in simulation.",
], y)
y -= 3 * mm

y = heading("What changed vs. the structural simulation", y)
y = body([
 "The simulation's heavy-hitter/revival structure rewarded burstiness-aware retention; the real tiny model rewards recency-aware",
 "scoring and punishes retention-only policies wholesale. Both regimes preserve the two portable conclusions: necessity scoring",
 "beats single-signal heuristics, and smooth precision allocation dominates hard eviction. The regime-dependence itself is the",
 "argument for trace-driven evaluation over benchmark-fixed claims: the policy frontier moves with the attention distribution.",
], y)
y -= 2 * mm

c.setFillColor(LIGHT); c.roundRect(M, y - 23 * mm, W - 2 * M, 23 * mm, 2 * mm, fill=1, stroke=0)
c.setFillColor(INK); c.setFont("Helvetica-Bold", 9)
c.drawString(M + 4 * mm, y - 5.5 * mm, "Stated plainly (limitations)")
c.setFont("Helvetica", 8.2); c.setFillColor(GREY)
for i, ln in enumerate([
 "One tiny, weakly trained char-level model is a single point in attention-distribution space; large instruction-tuned MoE models have",
 "sharper, layer-heterogeneous attention, and mean-pooling over layers/heads further flattens structure (production gating should be",
 "per-layer/head or per-block). Quality is still an attention-mass proxy, not end-task accuracy. n=10 traces bounds test power (p=0.002",
 "is the two-sided Wilcoxon floor at n=10). Next rungs: mid-size open-weight model traces, per-layer gating, end-task quality behind",
 "gates, then Tier-2 production traces with Diebold–Mariano testing on the goodput forecasts the cache policy feeds.",
]):
    c.drawString(M + 4 * mm, y - (9.5 + i * 3.9) * mm, ln)

c.setFillColor(TEAL); c.rect(0, 10 * mm, W, 0.8 * mm, fill=1, stroke=0)
c.setFillColor(GREY); c.setFont("Helvetica", 7.5)
c.drawString(M, 6 * mm, "Reproduce: python train_chunk.py 300 (x4) -> python capture_traces.py -> python replay_real.py")
c.drawRightString(W - M, 6 * mm, "Syni — Intelligence isn't performed. It's grown.  ·  syni.world")
c.save()
print("addendum built")
