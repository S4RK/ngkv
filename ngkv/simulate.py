"""
ngkv.simulate
=============

Synthetic decode-time attention trace generator.

The generator encodes four empirical regularities of decoder attention
that the sparse-KV literature (StreamingLLM, H2O, SnapKV, Quest)
repeatedly documents:

  1. Attention sinks — the first few tokens absorb outsized mass at
     every step.
  2. Recency — a local window near the frontier gets substantial mass.
  3. Persistent heavy hitters — a power-law-distributed minority of
     context tokens (entities, instructions, code identifiers)
     accumulate mass across many steps.
  4. Revival — dormant spans get re-attended when the task returns to
     them (e.g. answering about an early document section). This is the
     regime that separates variance-aware necessity scoring from pure
     accumulated-mass (H2O-style) scoring, and it is why we model it
     explicitly rather than assuming stationarity.

Stated plainly: this is a *structural* simulation, not a replay of real
model attention. It is designed so that relative policy orderings are
meaningful; absolute numbers must be validated on real traces (Tier-2
evaluation) before any production claim.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class TraceConfig:
    prompt_len: int = 2048
    decode_len: int = 1024
    sink_tokens: int = 4
    sink_mass: float = 0.18
    recency_window: int = 64
    recency_mass: float = 0.34
    heavy_frac: float = 0.03          # fraction of prompt tokens that are heavy hitters
    heavy_pareto_a: float = 1.2       # power-law shape of heavy-hitter importance
    heavy_mass: float = 0.36
    revival_events: int = 3           # dormant-span re-attention episodes per trace
    revival_span: int = 128           # tokens per revived span
    revival_len: int = 96             # decode steps each revival lasts
    revival_mass: float = 0.35        # mass diverted to revived span during episode
    noise_mass: float = 0.12
    seed: int = 0


def generate_trace(cfg: TraceConfig) -> np.ndarray:
    """Return (decode_len, total_len) attention matrix.

    Row t is the attention distribution of decode step t over the
    prompt_len + t tokens cached at that step (zero-padded to
    total_len = prompt_len + decode_len). Rows sum to 1.
    """
    rng = np.random.default_rng(cfg.seed)
    P, D = cfg.prompt_len, cfg.decode_len
    total = P + D
    attn = np.zeros((D, total))

    # Persistent heavy hitters with power-law weights.
    n_heavy = max(1, int(cfg.heavy_frac * P))
    heavy_idx = rng.choice(np.arange(cfg.sink_tokens, P), size=n_heavy, replace=False)
    heavy_w = rng.pareto(cfg.heavy_pareto_a, size=n_heavy) + 1.0
    heavy_w /= heavy_w.sum()

    # Revival schedule: (start_step, span_start) pairs on cold prompt regions.
    revivals = []
    for _ in range(cfg.revival_events):
        start_step = rng.integers(D // 8, max(D // 8 + 1, D - cfg.revival_len))
        span_start = rng.integers(cfg.sink_tokens, max(cfg.sink_tokens + 1, P - cfg.revival_span))
        revivals.append((int(start_step), int(span_start)))

    for t in range(D):
        cache_len = P + t
        row = np.zeros(total)

        # 1. Sinks.
        row[: cfg.sink_tokens] += cfg.sink_mass / cfg.sink_tokens

        # 2. Recency window over the newest tokens.
        w = min(cfg.recency_window, cache_len)
        rec = np.exp(-np.arange(w) / (cfg.recency_window / 3.0))
        rec /= rec.sum()
        row[cache_len - w : cache_len] += cfg.recency_mass * rec[::-1]

        # 3. Heavy hitters (slowly drifting weights).
        drift = 1.0 + 0.15 * np.sin(2 * np.pi * (t / D) + np.arange(n_heavy))
        hw = heavy_w * drift
        hw /= hw.sum()
        row[heavy_idx] += cfg.heavy_mass * hw

        # 4. Revival episodes divert mass to a dormant span.
        active = [rv for rv in revivals if rv[0] <= t < rv[0] + cfg.revival_len]
        if active:
            div = cfg.revival_mass / len(active)
            for _, span_start in active:
                span = np.arange(span_start, span_start + cfg.revival_span)
                prof = rng.dirichlet(np.ones(cfg.revival_span) * 0.3)
                row[span] += div * prof
            # Renormalize the non-revival components down.
            non_rev = row.sum() - cfg.revival_mass
            scale = (1.0 - cfg.noise_mass - cfg.revival_mass) / max(non_rev, 1e-9)
            mask = np.ones(total, dtype=bool)
            for _, span_start in active:
                mask[span_start : span_start + cfg.revival_span] = False
            row[mask] *= scale

        # 5. Diffuse noise.
        noise = rng.dirichlet(np.ones(cache_len) * 0.05)
        row[:cache_len] += cfg.noise_mass * noise

        row[:cache_len] /= row[:cache_len].sum()
        attn[t] = row

    return attn
