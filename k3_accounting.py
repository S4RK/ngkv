"""Rung K3-0: gating-surface audit for hybrid KDA/MLA models.

Computes, from a model's HF config.json alone, exactly what NG-KV can
and cannot gate:

  * which layers hold per-token KV (full-attention / MLA) vs constant-
    size recurrent state (KDA) — the *gating surface*;
  * per-token gateable KV bytes at bf16 / fp8 / fp4, and per-request
    totals at reference context lengths;
  * the KDA state footprint per sequence (what snapshot admission,
    Rung K3-4, would gate) for comparison.

MLA math: each full-attention layer caches ONE shared latent per token
(kv_lora_rank compressed KV + qk_rope_head_dim decoupled-RoPE key),
shared across all heads. The per-token per-layer gating unit is
therefore (kv_lora_rank + qk_rope_head_dim) elements — 576 for both
Kimi Linear 48B and Kimi K3.

KDA state math (per layer, per sequence, constant in context length):
recurrent state (num_heads, head_dim, head_dim) + short-conv state
(short_conv_kernel_size - 1 tail per q/k/v channel group; counted as
3 * num_heads * head_dim * (k-1) — an upper-bound convention, recorded
as such in the output).

Usage:
  python k3_accounting.py                     # both bundled configs
  python k3_accounting.py path/to/config.json
"""

from __future__ import annotations

import json
import pathlib
import sys

BUNDLED = ["model_configs/kimi_linear_48b_config.json",
           "model_configs/kimi_k3_config.json"]
CTX_POINTS = [8_192, 32_768, 131_072, 1_048_576]
BYTES = {"bf16": 2.0, "fp8": 1.0, "fp4": 0.5}


def text_config(raw: dict) -> dict:
    return raw.get("text_config", raw)


def layer_map(cfg: dict) -> tuple[list[str], list[int]]:
    """Return (layer_types full-depth 0-indexed, captured 0-indexed MLA idxs).

    Config lists are 1-indexed (modeling code: ``(layer_idx + 1) in
    kda_layers``); we emit 0-indexed everywhere outside the config.
    """
    L = cfg["num_hidden_layers"]
    la = cfg.get("linear_attn_config")
    if not la or not la.get("kda_layers"):
        return ["gqa_or_mla"] * L, list(range(L))
    kda = set(la["kda_layers"])
    full = set(la["full_attn_layers"])
    assert kda | full == set(range(1, L + 1)) and not (kda & full), \
        "kda_layers and full_attn_layers must partition 1..L"
    types = ["kda" if (i + 1) in kda else "mla" for i in range(L)]
    return types, [i for i in range(L) if (i + 1) in full]


def audit(path: str | pathlib.Path) -> dict:
    raw = json.load(open(path))
    cfg = text_config(raw)
    types, mla_idx = layer_map(cfg)
    n_mla, L = len(mla_idx), cfg["num_hidden_layers"]

    latent_dim = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
    per_tok_elems = n_mla * latent_dim

    la = cfg.get("linear_attn_config") or {}
    kda_heads = la.get("num_heads", cfg["num_attention_heads"])
    kda_hd = la.get("head_dim", cfg.get("head_dim", 128))
    conv_k = la.get("short_conv_kernel_size", 4)
    n_kda = L - n_mla
    kda_state_elems = n_kda * kda_heads * kda_hd * kda_hd
    kda_conv_elems = n_kda * 3 * kda_heads * kda_hd * (conv_k - 1)

    out = {
        "config": str(path),
        "model_type": raw.get("model_type", "?"),
        "layers_total": L,
        "layers_kda": n_kda,
        "layers_mla": n_mla,
        "mla_layer_indices_0idx": mla_idx,
        "mla_latent_dim_per_token_per_layer": latent_dim,
        "gateable_kv_elems_per_token": per_tok_elems,
        "gateable_kv_per_token_bytes": {p: per_tok_elems * b
                                        for p, b in BYTES.items()},
        "gateable_kv_per_request_GB": {
            f"{ctx:,} tok": {p: round(ctx * per_tok_elems * b / 2**30, 3)
                             for p, b in BYTES.items()}
            for ctx in CTX_POINTS},
        "kda_state_per_seq_MB_bf16": round(
            (kda_state_elems + kda_conv_elems) * 2 / 2**20, 2),
        "kda_state_note": ("constant in context length; conv tail counted "
                           "as 3*H*hd*(k-1) upper bound; this is the object "
                           "snapshot admission (Rung K3-4) gates, NOT part "
                           "of the NG-KV token-gating surface"),
        "granularity_note": ("MLA latent is shared across heads: gating "
                             "unit = (token, layer); per-head gating does "
                             "not exist on this surface"),
    }
    return out


def main() -> None:
    paths = sys.argv[1:] or BUNDLED
    results = [audit(p) for p in paths]
    for r in results:
        print(f"\n== {r['config']} ({r['model_type']}) ==")
        print(f"  layers: {r['layers_total']} total = "
              f"{r['layers_kda']} KDA + {r['layers_mla']} MLA")
        print(f"  gating surface: {r['layers_mla']} MLA layers x "
              f"{r['mla_latent_dim_per_token_per_layer']}-dim latent "
              f"= {r['gateable_kv_elems_per_token']:,} elems/token")
        for p, b in r["gateable_kv_per_token_bytes"].items():
            print(f"    {p}: {b/1024:.1f} KiB/token")
        for ctx, d in r["gateable_kv_per_request_GB"].items():
            print(f"    @ {ctx}: " + "  ".join(f"{p}={v} GB"
                                               for p, v in d.items()))
        print(f"  KDA state (constant/seq): "
              f"{r['kda_state_per_seq_MB_bf16']} MB bf16")
    json.dump(results, open("results_k3_accounting.json", "w"), indent=2)
    print("\nwrote results_k3_accounting.json")


if __name__ == "__main__":
    main()
