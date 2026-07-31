# NG-KV on SGLang / B200 — deployment runbook (Rung K3-3)

Phased: install → baseline → observe-only → gate on → compare.
Each phase is copy-paste. Do Phase 0–3 on ONE node before touching the
2-node sbatch.

The seam: NG-KV rides SGLang HiCache's L3 storage backend as a
necessity-gated ADMISSION FILTER (admit/deny page writes; reads and
existing pages untouched; deny is always safe — worst case is a
recompute, as if the page were never cached). No SGLang patches; the
wrapper loads via the `dynamic` backend mechanism.

Known constraints, stated plainly:
  * inner_backend is limited to `file` or `nixl`: SGLang's factory
    does not forward mem_pool_host to dynamic backends, which
    mooncake/hf3fs require. Wrapping those needs a one-line upstream
    change (forward mem_pool_host into dynamic kwargs).
  * Hybrid-model (K3/KDA) support in HiCache L3 is the newest code
    path in SGLang. If `--enable-hierarchical-cache` errors on the K3
    build, run this ladder on a pure-MLA model you already serve
    (K2.7 / GLM) to bank the live validation, then revisit K3.

---
## Phase 0 — install (once, shared FS)

```bash
cd /mnt/kvbm-cache
unzip -o ngkv-v0.8.zip           # -> /mnt/kvbm-cache/ngkv
```

Inside the SGLang container/venv on each serving node:

```bash
pip install -e /mnt/kvbm-cache/ngkv
python - <<'EOF'
from ngkv.adapters.sglang_backend import NGKVFilteredStorage
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
assert issubclass(NGKVFilteredStorage, HiCacheStorage)
print("ngkv adapter importable, subclass check OK")
EOF
pip install -q pytest && python -m pytest /mnt/kvbm-cache/ngkv/tests/test_sglang_backend.py -q
```

If the subclass assert fails, the SGLang build predates the
HiCacheStorage interface — upgrade SGLang before proceeding.

---
## Phase 1 — baseline: HiCache on, NG-KV off

Add to the server flags in your `serve_kimi_k3_sglang.sbatch` (or the
single-node launch):

```bash
mkdir -p /mnt/kvbm-cache/hicache-l3
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/kvbm-cache/hicache-l3

  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend file
```

(Verify the storage-dir env var name against your SGLang version:
`grep -rn "STORAGE_DIR" $(python -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')/srt/mem_cache/hicache_storage.py`)

Drive your canary suite / genai-perf with a prefix-heavy mix, then
record the three baseline numbers:

```bash
# 1. L3 bytes written
du -sb /mnt/kvbm-cache/hicache-l3
# 2. cache hit rate + TTFT from the metrics endpoint
curl -s localhost:30000/metrics | grep -iE "cache_hit|ttft|prefix"
# 3. keep the loadgen report (TTFT p50/p99, throughput)
```

---
## Phase 2 — NG-KV observe-only (pass-through, zero policy risk)

Same launch, swap the backend to the NG-KV wrapper in ADMIT-ALL mode.
This validates the dynamic-loading plumbing and the logging without
changing a single admission decision:

```bash
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config '{
      "backend_name": "ngkv",
      "module_path": "ngkv.adapters.sglang_backend",
      "class_name": "NGKVFilteredStorage",
      "inner_backend": "file",
      "scorer": "admit_all",
      "admit_frac": 1.0,
      "log_every": 20}'
```

Boot check — you must see the factory line and then admission logs:

```bash
grep -E "Creating dynamic storage backend 'ngkv'|ngkv admission" logs/<jobname>-*.out
```

Re-run the same loadgen. All three Phase-1 numbers must be within
noise of baseline — if they aren't, the wrapper itself is perturbing
the system and that's a bug to file before any gating.

---
## Phase 3 — gate on

Only change: scorer + budget.

```bash
  --hicache-storage-backend-extra-config '{
      "backend_name": "ngkv",
      "module_path": "ngkv.adapters.sglang_backend",
      "class_name": "NGKVFilteredStorage",
      "inner_backend": "file",
      "scorer": "position",
      "admit_frac": 0.6,
      "log_every": 20}'
```

Same loadgen, same three numbers, plus the denial rate from the
`ngkv admission` log lines. Expected shape of the result:

| metric                | expectation at admit_frac=0.6            |
|-----------------------|------------------------------------------|
| L3 bytes written (du) | ~40% lower than Phase 2                  |
| prefix hit rate       | small drop; the position scorer protects |
|                       | prefix-like (early-flush) pages          |
| TTFT p99              | flat or better (less write pressure);    |
|                       | a rise means denied pages were being     |
|                       | reused — raise admit_frac                |

Sweep `admit_frac` ∈ {0.8, 0.6, 0.4} — one loadgen run each — and
plot bytes-written vs hit-rate. That frontier IS the Rung K3-3
deliverable.

---
## Phase 4 — honest comparison

* Trace-driven, not benchmark-vibes: keep the loadgen seed and request
  mix identical across phases; one variable per run.
* If you want paired statistics, replay the same request log through
  Phase-2 and Phase-3 configs and feed both to `abx` (replay-paired
  mode) — the machinery you already have.
* Log the SGLang version + full flag set into the run record; the
  dynamic-backend contract (`backend_class(storage_config, kwargs)`)
  is version-validated as of July 2026 sglang main.

## Rollback

Remove the four hicache flags (or set `--hicache-storage-backend
file`). Denied pages were never stored, so there is no state to clean
beyond `rm -rf /mnt/kvbm-cache/hicache-l3` if you want the disk back.

## What this does NOT yet do

Within-request necessity gating (the tiered/mixed-precision policies)
lives at the vLLM connector seam, not this one — HiCache L3 pages are
written to be shared, so this seam is admit/deny only, by design (see
the boundary rule in `ngkv/adapters/sglang_backend.py`). The
attention-informed `table` scorer needs a request-path score feed;
wire that only after the position-scorer frontier proves the seam.
