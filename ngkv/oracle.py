"""
ngkv.oracle
===========

Hindsight oracle gate and plug-in regret evaluation.

The oracle necessity of cached token i at step t is its realized future
attention mass: sum over steps s > t of attn_s(i). No online policy can
observe this; every deployable scorer is a plug-in estimator of it.

Following the SIVAM TR-01 structure:

  * Oracle dominance — the hindsight-optimal gate upper-bounds the
    achievable quality of any gate at equal budget (immediate, since it
    maximizes retained future mass by construction).
  * Plug-in regret identity — the quality gap of a deployed gate
    decomposes into the mass placed on entries the oracle retains but
    the policy does not (and vice versa). ``regret_curve`` computes this
    per step on replayed traces.

This turns logged production traces into a benchmark: replay, compute
the oracle placement, and report regret. A policy with near-zero regret
at budget B is *certified* near-optimal at B — a statement the usual
"quality metric didn't move" A/B gate cannot make.
"""

from __future__ import annotations

import numpy as np


def oracle_scores(attn_matrix: np.ndarray, t: int) -> np.ndarray:
    """Future attention mass of each token cached at step t.

    Parameters
    ----------
    attn_matrix:
        (T, T_max) matrix; row s is the attention distribution of decode
        step s over all tokens cached at that step (zero-padded).
    t:
        Current step; tokens 0..(cache_len_at_t - 1) are scoreable.
    """
    future = attn_matrix[t + 1 :]
    if future.shape[0] == 0:
        return np.zeros(attn_matrix.shape[1])
    return future.sum(axis=0)


def retained_mass(attn_row: np.ndarray, weight: np.ndarray) -> float:
    """Quality proxy for one step: attention mass recovered.

    ``weight`` in [0, 1] per token — 1 for full-precision retained, a
    fidelity factor for quantized, 0 for evicted.
    """
    return float(np.dot(attn_row[: weight.shape[0]], weight[: attn_row.shape[0]]))


def regret_curve(
    attn_matrix: np.ndarray,
    policy_weights: np.ndarray,
    oracle_weights: np.ndarray,
) -> np.ndarray:
    """Per-step plug-in regret: oracle retained mass − policy retained mass.

    Both weight arrays are (T, T_max): the retention weight each policy
    assigned to each token at each step.
    """
    T = attn_matrix.shape[0]
    out = np.zeros(T)
    for t in range(T):
        row = attn_matrix[t]
        out[t] = retained_mass(row, oracle_weights[t]) - retained_mass(
            row, policy_weights[t]
        )
    return out
