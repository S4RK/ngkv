"""ngkv — Necessity-Gated KV Cache Management.

Framework-agnostic necessity scoring, tiered placement, mixed-precision
allocation, and oracle-regret evaluation for LLM KV caches.

Part of the Syni open research stack. Apache 2.0.
Cites: SIVAM TR-01, "Necessity-Gated Activation Modulation".
"""

from .necessity import NecessityConfig, NecessityScorer
from .policy import (
    MixedPrecisionPolicy,
    PrecisionLevels,
    Tier,
    TierBudget,
    TieredPlacementPolicy,
)
from .oracle import oracle_scores, regret_curve, retained_mass
from .simulate import TraceConfig, generate_trace
from .block import (BlockGate, BlockGateConfig, expand_from_blocks,
                    n_blocks, pool_to_blocks)

__version__ = "0.2.0"
__all__ = [
    "NecessityConfig", "NecessityScorer", "MixedPrecisionPolicy",
    "PrecisionLevels", "Tier", "TierBudget", "TieredPlacementPolicy",
    "oracle_scores", "regret_curve", "retained_mass",
    "TraceConfig", "generate_trace",
    "BlockGate", "BlockGateConfig", "pool_to_blocks",
    "expand_from_blocks", "n_blocks",
]
