"""
ngkv.necessity
==============

Plug-in necessity estimators for KV cache entries.

A *necessity score* n_t(i) estimates how vital cached token i is to the
quality of future decoding steps, given information available at step t.
This is the plug-in estimator in the sense of SIVAM TR-01: the oracle
necessity is the (unobservable) future attention mass a token will
receive; the deployed policy substitutes an online estimate and the
quality gap between the two is the *plug-in regret* (see ngkv.oracle).

Scorers are framework-agnostic: they consume per-step attention weight
vectors (however obtained — HF hooks, vLLM connector, simulation) and
maintain O(T) state per layer/head group.

Stated plainly (limitations):
  * Scores are estimates of future attention from past attention. Tokens
    that are dormant for long spans and then revived are the failure
    mode of all history-based scorers, including these. The variance
    term below partially mitigates, not eliminates, this.
  * Scores here are per-token aggregates over heads. Per-head gating is
    strictly more powerful and strictly more expensive; adapters may
    choose either granularity.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class NecessityConfig:
    """Configuration for the necessity scorer.

    Attributes
    ----------
    ewma_alpha:
        Decay for the exponentially weighted attention mass. Higher
        alpha adapts faster to attention shifts, at the cost of
        forgetting persistent heavy hitters.
    variance_weight:
        Weight on the attention *variability* term. Tokens with bursty
        (high-variance) attention are plausibly "revivable" and get a
        necessity bonus even when their mean mass is currently low.
    recency_halflife:
        Half-life (in decode steps) of the recency prior. Encodes the
        empirical regularity that recently generated tokens are
        near-certain to be attended (local attention structure).
    recency_weight:
        Weight of the recency prior relative to observed mass.
    last_step_weight:
        Weight on the most recent step's raw attention row (the TOVA
        signal). On real traces this is the single strongest short-
        horizon predictor of next-step attention; blending it closes
        ~70% of the default scorer's regret to the clairvoyant oracle
        (see replay_compare.py). Set to 0 to recover the v0.1 scorer.
    sink_tokens:
        Number of initial tokens treated as attention sinks and pinned
        to maximum necessity (never evicted, always top tier).
    """

    ewma_alpha: float = 0.7
    variance_weight: float = 0.0
    recency_halflife: float = 64.0
    recency_weight: float = 0.0
    last_step_weight: float = 1.0
    sink_tokens: int = 4

    # Defaults tuned by leave-one-out cross-validation on the Rung-2
    # traces (unanimous across folds and budgets: fast EWMA + last-step
    # signal; variance and recency terms selected to zero). Stated
    # plainly: that distribution is short (120-step) recency-dominant
    # decodes; for revival-heavy workloads (long CoT, retrieval), the
    # variance/recency terms exist precisely to be turned back on, and
    # certifying on your own traces (replay_compare.py) beats trusting
    # any default. v0.1 defaults were (0.05, 0.5, 64, 0.15, absent).


class NecessityScorer:
    """Online necessity scorer over a growing KV cache.

    Maintains, per cached token:
      * ``mass``  – EWMA of received attention mass (all heads averaged)
      * ``var``   – EWMA of squared deviation (burstiness signal)

    ``scores(t)`` returns the composite necessity vector at step t.
    """

    def __init__(self, config: Optional[NecessityConfig] = None) -> None:
        self.cfg = config or NecessityConfig()
        self._mass = np.zeros(0, dtype=np.float64)
        self._var = np.zeros(0, dtype=np.float64)
        self._last = np.zeros(0, dtype=np.float64)
        self._born = np.zeros(0, dtype=np.int64)
        self._step = 0

    # ------------------------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return self._mass.shape[0]

    def _grow(self, new_len: int) -> None:
        add = new_len - self.num_tokens
        if add <= 0:
            return
        self._mass = np.concatenate([self._mass, np.zeros(add)])
        self._var = np.concatenate([self._var, np.zeros(add)])
        self._last = np.concatenate([self._last, np.zeros(add)])
        self._born = np.concatenate(
            [self._born, np.full(add, self._step, dtype=np.int64)]
        )

    # ------------------------------------------------------------------
    def observe(self, attn: np.ndarray) -> None:
        """Ingest one decode step's attention over the cache.

        Parameters
        ----------
        attn:
            Array of shape (num_cached_tokens,) — attention mass each
            cached token received this step, averaged (or maxed, per
            adapter choice) over heads. Must sum to <= 1.
        """
        attn = np.asarray(attn, dtype=np.float64)
        self._grow(attn.shape[0])
        a = self.cfg.ewma_alpha
        m = self._mass[: attn.shape[0]]
        dev = (attn - m) ** 2
        self._mass[: attn.shape[0]] = (1 - a) * m + a * attn
        self._var[: attn.shape[0]] = (1 - a) * self._var[: attn.shape[0]] + a * dev
        self._last = np.zeros_like(self._mass)
        self._last[: attn.shape[0]] = attn
        self._step += 1

    # ------------------------------------------------------------------
    def scores(self) -> np.ndarray:
        """Composite necessity score per cached token (higher = more vital)."""
        n = self.num_tokens
        if n == 0:
            return np.zeros(0)
        cfg = self.cfg
        age = self._step - self._born  # steps since creation
        recency = np.exp2(-(age.astype(np.float64)) / cfg.recency_halflife)
        score = (
            self._mass
            + cfg.variance_weight * np.sqrt(self._var)
            + cfg.recency_weight * recency
            + cfg.last_step_weight * self._last
        )
        # Pin attention sinks.
        k = min(cfg.sink_tokens, n)
        if k > 0:
            score[:k] = np.inf
        return score
