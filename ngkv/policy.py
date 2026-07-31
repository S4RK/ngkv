"""
ngkv.policy
===========

Necessity-gated cache policies. Two families:

1. ``TieredPlacementPolicy`` — hard placement of each KV entry into a
   memory tier (HBM / DRAM / EVICT) by necessity rank, under per-tier
   capacity budgets. This is the disaggregated-serving version: HBM is
   the GPU-resident cache, DRAM is a host/remote tier reachable at a
   bytes-per-token transfer cost, EVICT means recompute-or-lose.

2. ``MixedPrecisionPolicy`` — smooth relaxation: every retained entry
   gets a bit-width proportional to its necessity rank, under a total
   bit budget. This generalizes eviction (0 bits) and dominates it on
   the quality/memory frontier in our simulations.

Both are *plug-in gates*: they act on estimated necessity. Feeding them
oracle scores (ngkv.oracle) yields the oracle gate, and the difference
in achieved quality is the plug-in regret — the quantity a deployment
A/B should track.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Dict

import numpy as np


class Tier(enum.IntEnum):
    HBM = 0
    DRAM = 1
    EVICT = 2


@dataclasses.dataclass
class TierBudget:
    """Capacity budgets as fractions of the full-cache footprint."""

    hbm_frac: float = 0.30
    dram_frac: float = 0.30  # additional fraction held in host/remote tier


class TieredPlacementPolicy:
    """Rank tokens by necessity; fill HBM, then DRAM, evict the rest."""

    def __init__(self, budget: TierBudget) -> None:
        self.budget = budget

    def place(self, scores: np.ndarray) -> np.ndarray:
        n = scores.shape[0]
        placement = np.full(n, Tier.EVICT, dtype=np.int64)
        if n == 0:
            return placement
        order = np.argsort(-scores, kind="stable")
        n_hbm = int(np.floor(self.budget.hbm_frac * n))
        n_dram = int(np.floor(self.budget.dram_frac * n))
        placement[order[:n_hbm]] = Tier.HBM
        placement[order[n_hbm : n_hbm + n_dram]] = Tier.DRAM
        return placement


@dataclasses.dataclass
class PrecisionLevels:
    """Bit-widths and their quality fidelity factors.

    ``fidelity`` is the modeled fraction of a token's attention
    contribution preserved at that precision. These are modeling
    assumptions calibrated to the KV-quantization literature (KIVI,
    KVQuant report near-lossless 4-bit for most, not all, tokens);
    real fidelity is model- and layer-dependent and must be measured
    per deployment. Stated plainly: the simulation's quantization
    results inherit these assumptions.
    """

    bits: tuple = (16, 8, 4, 0)
    fidelity: tuple = (1.0, 0.998, 0.97, 0.0)


class MixedPrecisionPolicy:
    """Allocate bit-widths by necessity rank under a total bit budget.

    ``bit_budget_frac`` is the total bits allowed as a fraction of the
    full fp16 cache (e.g. 0.30 means the whole cache must fit in 30% of
    its fp16 size). Allocation is greedy: highest-necessity tokens get
    16 bits, then 8, then 4, until the budget is exhausted; the
    remainder is evicted (0 bits).
    """

    def __init__(self, bit_budget_frac: float, levels: PrecisionLevels | None = None,
                 mix: tuple = (0.25, 0.35, 0.40)) -> None:
        self.bit_budget_frac = bit_budget_frac
        self.levels = levels or PrecisionLevels()
        # Fraction of the *bit budget* spent at 16 / 8 / 4 bits.
        self.mix = mix

    def allocate(self, scores: np.ndarray) -> np.ndarray:
        """Return per-token bit widths."""
        n = scores.shape[0]
        bits = np.zeros(n, dtype=np.int64)
        if n == 0:
            return bits
        total_bit_budget = self.bit_budget_frac * n * 16.0
        order = np.argsort(-scores, kind="stable")
        budgets = [f * total_bit_budget for f in self.mix]
        widths = [16, 8, 4]
        idx = 0
        for width, budget in zip(widths, budgets):
            count = int(budget // width)
            take = order[idx : idx + count]
            bits[take] = width
            idx += count
        return bits

    def fidelity_of(self, bits: np.ndarray) -> np.ndarray:
        lut: Dict[int, float] = dict(zip(self.levels.bits, self.levels.fidelity))
        return np.vectorize(lut.get)(bits).astype(np.float64)
