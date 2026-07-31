"""Framework adapters.

``ngkv`` core is framework-agnostic: it consumes attention observations
and emits placements/bit-widths. Adapters bind it to a serving stack.

  * ``hf``     — working reference adapter for HuggingFace transformers
                 (forward hooks capture attention; policy applied via
                 Cache eviction between steps). Correct, not fast; for
                 research and validation.
  * ``vllm``   — integration specification against vLLM's KVConnector /
                 KV-events interface (block-granular scoring).
  * ``sglang`` — integration specification against SGLang's
                 HiCacheController (radix-tree node scoring).

Production integration is block/page-granular, not token-granular:
score a block as the max (not mean) necessity of its tokens, so one
vital token protects its block — the conservative choice.
"""

try:  # optional: needs torch; absence must not break the package
    from . import hf  # noqa: F401
except Exception:  # pragma: no cover
    hf = None  # type: ignore[assignment]
