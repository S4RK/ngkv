"""Build the NG-KV developer onboarding one-pager (same design system)."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

INK, EMBER, TEAL = HexColor("#16213a"), HexColor("#e8590c"), HexColor("#0f8b8d")
GREY, LIGHT, PAPER = HexColor("#5b6270"), HexColor("#e7eaef"), HexColor("#fbfaf7")
CODEBG = HexColor("#eef1f5")
W, H = A4
M = 13 * mm

c = canvas.Canvas("NGKV_developer_onboarding.pdf", pagesize=A4)
c.setFillColor(PAPER); c.rect(0, 0, W, H, fill=1, stroke=0)

# header
c.setFillColor(INK); c.rect(0, H - 28 * mm, W, 28 * mm, fill=1, stroke=0)
c.setFillColor(EMBER); c.rect(0, H - 28 * mm, W, 1.2 * mm, fill=1, stroke=0)
c.setFillColor(HexColor("#ffffff")); c.setFont("Helvetica-Bold", 17)
c.drawString(M, H - 13 * mm, "Onboarding NG-KV to Your Inference Stack")
c.setFont("Helvetica", 9.5); c.setFillColor(HexColor("#c9d2e4"))
c.drawString(M, H - 19 * mm, "Trace-first integration: certify the policy on your own workload offline, then bind the gate. No production risk until step 5.")
c.setFont("Helvetica-Oblique", 8.5)
c.drawString(M, H - 24.5 * mm, "NG-KV v0.2  ·  Apache 2.0  ·  pip install numpy; unzip; go  ·  Syni open research stack  ·  July 2026")

y = H - 35 * mm

def heading(txt, yy, col=EMBER):
    c.setFillColor(col); c.rect(M, yy - 1.2, 6 * mm, 2.2, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 10.5)
    c.drawString(M + 8 * mm, yy - 2.0, txt); return yy - 6.5 * mm

def body(lines, yy, size=8.4, leading=11.0):
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

def codeblock(lines, yy, size=7.3, leading=9.6):
    h = len(lines) * leading + 5 * mm
    c.setFillColor(CODEBG); c.roundRect(M, yy - h + 3 * mm, W - 2 * M, h, 1.5 * mm, fill=1, stroke=0)
    yy -= 2.2 * mm
    c.setFont("Courier", size)
    for ln in lines:
        if ln.strip().startswith("#"):
            c.setFillColor(TEAL)
        else:
            c.setFillColor(INK)
        c.drawString(M + 3.5 * mm, yy, ln)
        yy -= leading
    return yy - 4 * mm

# Step 1
y = heading("Step 1 — Capture traces from your stack (pick the cheapest signal you have)", y)
y = body([
 "A trace is one request's per-step attention over its KV cache, at whatever granularity you can afford. Token-level is ideal;",
 "block-level (16–64 tokens) is fine and is what production paths use — coarser granularity only weakens, never invalidates,",
 "the regret bound. You do NOT need to materialize full attention (FlashAttention won't give it to you anyway). Any of these work:",
 "Eager replay:| re-run a sample of logged requests offline with eager attention (HF adapter included) — zero prod changes.",
 "Block statistics:| Quest-style per-block key min/max against the query gives a cheap per-block attention upper bound, online.",
 "Cache telemetry:| prefix/radix hit counts and block touch timestamps you likely already log approximate mass + recency.",
], y)
y = codeblock([
 "from ngkv.traces import save_trace          # .npz schema: attn (D,total), prompt_len, meta",
 "save_trace('traces/req_001.npz', attn, prompt_len=P,",
 "           meta={'model':'glm-5.2','granularity':32,'reduction':'max(heads)'})",
], y)

# Step 2
y = heading("Step 2 — Replay offline: quality, regret, significance (minutes, no GPU)", y, TEAL)
y = body([
 "One command evaluates sliding-window, accumulated-mass, NG scoring, HBM/DRAM tiering, and mixed precision at every budget,",
 "against the clairvoyant oracle. Output: attention-mass recovered, plug-in regret per family, paired Wilcoxon + bootstrap CIs.",
 "Read it like this:| low NG regret = the estimator is near-optimal, remaining loss is the policy family - change action, not scorer.",
 "High NG regret = headroom in the scorer - tune NecessityConfig (ewma_alpha, variance/recency weights) on your traces.",
], y)
y = codeblock([
 "python replay_real.py     # reads traces/*.npz  ->  results_real.json + console report",
], y)

# Step 3
y = heading("Step 3 — Pick the action the replay justifies", y)
y = body([
 "Sharp attention (regret gap large, oracle high):| necessity-ranked eviction or tiering captures most value cheaply.",
 "Diffuse attention (oracle itself low):| eviction cannot win - use mixed precision; in our real traces it beat the eviction oracle.",
 "Disaggregated (P/D split, EFA/RDMA):| tier by necessity; EVICT-tier blocks are never shipped (60% interconnect bytes saved",
 "at B=0.2 in simulation) and prefetch runs in descending necessity so decode starts before the tail arrives.",
], y)
y -= 1 * mm

# Step 4
y = heading("Step 4 — Bind the gate (thin adapters, core is stateless-simple)", y, TEAL)
y = body([
 "vLLM:| implement KVConnector with NG-KV as admission (should_offload -> Tier) + prefetch ordering. Score a block as the MAX",
 "necessity of its tokens - one vital token protects its block (the conservative choice). ~200 LoC against a pinned version.",
 "SGLang:| HiCache admission + eviction ranking over radix nodes. LFU is NG with variance=recency=0, so rollback is trivial.",
 "TensorRT-LLM / custom:| the scorer is plain NumPy, O(T) state, microseconds/step - embed it wherever eviction order is decided.",
 "MoE stacks (GLM / DeepSeek / Qwen):| router entropy is a free per-step necessity signal; sinks are pinned by default.",
], y)
y = codeblock([
 "from ngkv import NecessityScorer, TieredPlacementPolicy, TierBudget",
 "scorer, policy = NecessityScorer(), TieredPlacementPolicy(TierBudget(hbm_frac=.3, dram_frac=.3))",
 "scorer.observe(block_attn_estimate)         # any per-step signal, any granularity",
 "placement = policy.place(scorer.scores())   # Tier.HBM / Tier.DRAM / Tier.EVICT",
], y)

# Step 5
y = heading("Step 5 — Ship behind gates, keep the regret loop running", y)
y = body([
 "A/B with SLA-goodput as primary, per-request quality as guardrail; diagnostics: effective hit rate, bytes/output token, TTFT",
 "p50/p99 on hit and miss paths. Keep replaying fresh trace samples offline - regret drift is your early warning that the",
 "workload's attention distribution moved before the goodput metric shows it.",
], y)
y -= 1.5 * mm

# fine print
c.setFillColor(LIGHT); c.roundRect(M, y - 14.5 * mm, W - 2 * M, 14.5 * mm, 2 * mm, fill=1, stroke=0)
c.setFillColor(INK); c.setFont("Helvetica-Bold", 8.5)
c.drawString(M + 4 * mm, y - 5 * mm, "Stated plainly")
c.setFont("Helvetica", 7.9); c.setFillColor(GREY)
for i, ln in enumerate([
 "Published numbers are attention-mass proxies from small-scale traces - the pipeline exists so you can replace them with your own workload's",
 "numbers before believing anything. Validate end-task quality behind gates, not mass proxies. Fidelity factors for quantized tiers are model-",
 "dependent: measure yours. The HF adapter is for validation, not speed.",
]):
    c.drawString(M + 4 * mm, y - (8.5 + i * 3.6) * mm, ln)

c.setFillColor(TEAL); c.rect(0, 9 * mm, W, 0.8 * mm, fill=1, stroke=0)
c.setFillColor(GREY); c.setFont("Helvetica", 7.5)
c.drawString(M, 5.5 * mm, "Repo: ngkv-v0.2.zip - core, adapters, trace schema, both harnesses, all results. ABSTRACT.md for the claim in 300 words.")
c.drawRightString(W - M, 5.5 * mm, "Syni - Intelligence isn't performed. It's grown.  ·  syni.world")
c.save()
print("onboarding one-pager built")
