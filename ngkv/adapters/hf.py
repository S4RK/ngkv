"""
HuggingFace transformers reference adapter.

Usage (any decoder model that returns attentions):

    from ngkv.adapters.hf import NGKVController
    ctrl = NGKVController(model, hbm_frac=0.3, dram_frac=0.3)
    out = ctrl.generate(input_ids, max_new_tokens=512)

The controller:
  1. runs generation step-by-step with ``output_attentions=True``,
  2. feeds per-step attention (averaged over layers/heads, max-pooled
     per block) into a NecessityScorer,
  3. between steps, rewrites the DynamicCache to keep only HBM-tier
     entries at full precision (DRAM tier is retained on CPU and
     re-materialized on demand in this reference implementation).

Stated plainly: this adapter exists to *validate policies on real model
attention*, not to be fast. Step-wise generation with attention output
disables fused attention kernels (FlashAttention does not materialize
attention weights). Production paths must derive necessity from cheap
proxies (router logits, block-level Quest-style key statistics) rather
than full attention capture. The oracle/regret methodology is
unchanged either way.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..necessity import NecessityConfig, NecessityScorer
from ..policy import Tier, TierBudget, TieredPlacementPolicy

try:  # torch is optional; core package works without it.
    import torch
except ImportError:  # pragma: no cover
    torch = None


class NGKVController:
    def __init__(
        self,
        model,
        hbm_frac: float = 0.30,
        dram_frac: float = 0.30,
        necessity_config: Optional[NecessityConfig] = None,
        apply_every: int = 32,
    ) -> None:
        if torch is None:
            raise ImportError("ngkv.adapters.hf requires torch")
        self.model = model
        self.scorer = NecessityScorer(necessity_config)
        self.policy = TieredPlacementPolicy(TierBudget(hbm_frac, dram_frac))
        self.apply_every = apply_every
        self._dram_store: dict = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _attn_to_observation(attentions) -> np.ndarray:
        """Average attention of the last query position over layers/heads."""
        rows = []
        for layer_attn in attentions:  # (B, H, Q, K)
            rows.append(layer_attn[0, :, -1, :].mean(dim=0))
        return torch.stack(rows).mean(dim=0).float().cpu().numpy()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 256, **kw):
        past = None
        generated = input_ids
        for step in range(max_new_tokens):
            out = self.model(
                input_ids=generated if past is None else generated[:, -1:],
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
            )
            past = out.past_key_values
            self.scorer.observe(self._attn_to_observation(out.attentions))

            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)
            if (step + 1) % self.apply_every == 0:
                past = self._apply_policy(past)
            if hasattr(self.model.config, "eos_token_id") and next_tok.item() == self.model.config.eos_token_id:
                break
        return generated

    # ------------------------------------------------------------------
    def _apply_policy(self, past):
        """Rewrite the cache: HBM tier stays; DRAM tier offloads to CPU;
        EVICT tier is dropped. Reference semantics — see module docstring."""
        scores = self.scorer.scores()
        placement = self.policy.place(scores)
        keep = np.where(placement == Tier.HBM)[0]
        offload = np.where(placement == Tier.DRAM)[0]
        keep_t = torch.as_tensor(keep, device=past[0][0].device)

        new_past = []
        for li, (k, v) in enumerate(past):
            self._dram_store[li] = (
                k[:, :, offload, :].to("cpu", non_blocking=True),
                v[:, :, offload, :].to("cpu", non_blocking=True),
            )
            new_past.append((k[:, :, keep_t, :], v[:, :, keep_t, :]))
        # Note: index compaction means the scorer must be re-indexed too.
        self.scorer = self._reindex_scorer(keep)
        return tuple(new_past)

    def _reindex_scorer(self, keep: np.ndarray) -> NecessityScorer:
        s = NecessityScorer(self.scorer.cfg)
        s._mass = self.scorer._mass[keep].copy()
        s._var = self.scorer._var[keep].copy()
        s._born = self.scorer._born[keep].copy()
        s._step = self.scorer._step
        return s
