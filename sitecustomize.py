"""Auto-arm NG-KV L2 gating when this repo is on PYTHONPATH.

Python imports ``sitecustomize`` at interpreter startup, so every SGLang
process (scheduler, tokenizer, detokenizer) picks this up with no launch
command change. It does NOTHING unless the ``NGKV_L2`` env var is set, so
the repo can sit on PYTHONPATH permanently.

Any failure here is logged and swallowed: a broken research instrument must
never prevent the server from starting.
"""

import os
import sys

if os.environ.get("NGKV_L2"):
    try:
        from ngkv.adapters.sglang_l2 import install

        install()
    except Exception as exc:  # pragma: no cover
        print(f"[ngkv-l2] install failed, continuing unpatched: {exc!r}",
              file=sys.stderr)
