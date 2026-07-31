"""
ngkv.block
==========

Block-granularity necessity gating.

Paged serving engines (vLLM, SGLang) manage KV in fixed-size blocks
(16 tokens by default in vLLM). Acting at block granularity makes the
gate land on exactly the unit the memory manager already moves — no
layout surgery, no fragmentation.

This module lifts token-level scores and policies to blocks:

  * ``pool_to_blocks``     — reduce a per-token vector to per-block
                             (max or mean over each block's tokens).
  * ``expand_from_blocks`` — broadcast a per-block decision back to
                             tokens (for evaluation or cache rewriting).
  * ``BlockGate``          — a NecessityScorer + policy pair operating
                             at block granularity end to end: observe
                             token attention, decide per block.

Reduction choice: ``max`` is the safe default — a block is as necessary
as its most necessary token, so pooling can only over-retain, never
silently drop a hot token inside a cold block. ``mean`` is tighter on
memory and looser on quality. Both are evaluated in
``replay_blocks.py`` (Rung 2.5); coarser granularity weakens the regret
bound but never invalidates it.

Stated plainly: block pooling trades selectivity for compatibility.
The measured cost on our traces is small (see results_rung25_blocks
.json) but it is workload-dependent — replay your own traces before
choosing a block size.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import numpy as np

from .necessity import NecessityConfig, NecessityScorer
from .policy import MixedPrecisionPolicy, Tier, TierBudget, TieredPlacementPolicy

Reduction = Literal["max", "mean"]


def n_blocks(n_tokens: int, block_size: int) -> int:
    """Number of blocks covering n_tokens (last block may be partial)."""
    return (n_tokens + block_size - 1) // block_size


def pool_to_blocks(values: np.ndarray, block_size: int,
                   reduce: Reduction = "max") -> np.ndarray:
    """Reduce a per-token vector (n_tokens,) to per-block (n_blocks,).

    Handles a partial final block. ``max`` = a block is as necessary as
    its hottest token; ``mean`` = average necessity per token.
    """
    n = values.shape[0]
    if n == 0:
        return values.copy()
    nb = n_blocks(n, block_size)
    pad = nb * block_size - n
    if pad:
        fill = -np.inf if reduce == "max" else np.nan
        values = np.concatenate([values, np.full(pad, fill)])
    grid = values.reshape(nb, block_size)
    if reduce == "max":
        return grid.max(axis=1)
    return np.nanmean(grid, axis=1)


def expand_from_blocks(block_values: np.ndarray, block_size: int,
                       n_tokens: int) -> np.ndarray:
    """Broadcast per-block values (n_blocks,) back to tokens (n_tokens,)."""
    return np.repeat(block_values, block_size)[:n_tokens]


@dataclasses.dataclass
class BlockGateConfig:
    block_size: int = 16
    reduce: Reduction = "max"
    necessity: NecessityConfig = dataclasses.field(default_factory=NecessityConfig)


class BlockGate:
    """Necessity gate operating at paged-block granularity.

    Observes token-level attention rows (whatever the capture side
    provides), maintains token-level scores internally, and issues
    *per-block* placement / precision decisions. Blocks containing sink
    tokens are pinned to the top tier via the scorer's sink prior.

    A ``shared_mask`` may be supplied per decision: blocks marked shared
    (engine ref-count > 1 / radix fan-out) are forced to Tier.HBM at
    full precision — the prefix-cache boundary rule. The gate never
    touches shared blocks.
    """

    def __init__(self, cfg: Optional[BlockGateConfig] = None) -> None:
        self.cfg = cfg or BlockGateConfig()
        self.scorer = NecessityScorer(self.cfg.necessity)

    def observe(self, attn_row: np.ndarray) -> None:
        """Feed one decode step's attention over the current cache."""
        self.scorer.observe(attn_row)

    def block_scores(self, cache_len: int) -> np.ndarray:
        """Per-block necessity for the first cache_len tokens."""
        s = self.scorer.scores()
        sp = np.full(cache_len, np.inf)
        sp[: min(s.shape[0], cache_len)] = s[:cache_len]
        return pool_to_blocks(sp, self.cfg.block_size, self.cfg.reduce)

    def place_blocks(self, cache_len: int, budget: TierBudget,
                     shared_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-block Tier decisions; shared blocks pinned to HBM."""
        bs = self.block_scores(cache_len)
        placement = TieredPlacementPolicy(budget).place(bs)
        if shared_mask is not None:
            placement = placement.copy()
            placement[shared_mask[: placement.shape[0]]] = Tier.HBM
        return placement

    def allocate_bits(self, cache_len: int, bit_budget_frac: float,
                      full_bits: int = 16,
                      shared_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-block bit-widths; shared blocks pinned to full precision."""
        bs = self.block_scores(cache_len)
        bits = MixedPrecisionPolicy(bit_budget_frac=bit_budget_frac).allocate(bs)
        if shared_mask is not None:
            bits = bits.copy()
            bits[shared_mask[: bits.shape[0]]] = full_bits
        return bits

    def token_mask(self, placement: np.ndarray, cache_len: int) -> np.ndarray:
        """Expand block placement to a token keep-mask (1 = retained)."""
        keep = (placement != Tier.EVICT).astype(np.float64)
        return expand_from_blocks(keep, self.cfg.block_size, cache_len)
