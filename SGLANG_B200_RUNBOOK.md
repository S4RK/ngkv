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
cd /mnt/local-nvme
unzip -o ngkv-v0.8.zip           # -> /mnt/local-nvme/ngkv
```

Inside the SGLang container/venv on each serving node:

```bash
pip install -e /mnt/local-nvme/ngkv
python - <<'EOF'
from ngkv.adapters.sglang_backend import NGKVFilteredStorage
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
assert issubclass(NGKVFilteredStorage, HiCacheStorage)
print("ngkv adapter importable, subclass check OK")
EOF
pip install -q pytest && python -m pytest /mnt/local-nvme/ngkv/tests/test_sglang_backend.py -q
```

If the subclass assert fails, the SGLang build predates the
HiCacheStorage interface — upgrade SGLang before proceeding.

---
## Phase 1 — baseline: HiCache on, NG-KV off

Add to the server flags in your `serve_kimi_k3_sglang.sbatch` (or the
single-node launch):

```bash
mkdir -p /mnt/local-nvme/hicache-l3
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/local-nvme/hicache-l3

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
du -sb /mnt/local-nvme/hicache-l3
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
beyond `rm -rf /mnt/local-nvme/hicache-l3` if you want the disk back.

## What this does NOT yet do

Within-request necessity gating (the tiered/mixed-precision policies)
lives at the vLLM connector seam, not this one — HiCache L3 pages are
written to be shared, so this seam is admit/deny only, by design (see
the boundary rule in `ngkv/adapters/sglang_backend.py`). The
attention-informed `table` scorer needs a request-path score feed;
wire that only after the position-scorer frontier proves the seam.

---
# Addendum: L2 (host-memory) gating — the K3 path

Why: the L3 storage path in current hybrid (KDA) SGLang builds does not
complete a round trip on Kimi K3 — verified with the plain `file` backend
and NO NG-KV in the stack (warmup hangs to the 600s timeout either way).
L2 is unaffected, is a harder budget (`--hicache-ratio` x device pool), sees
*every* page under write_through, and denial there also saves the D2H copy.

Seam: `HiRadixCache.write_backup(node, write_back=False) -> int`, where 0
already means "not backed up" and every caller handles it (it is returned
today when the parent isn't backed up, or when host alloc fails). Denial
therefore lives inside the existing contract; worst case is a recompute.

Install: no SGLang patch, no launch-command change. The repo is already on
`PYTHONPATH`, and `sitecustomize.py` arms the gate when `NGKV_L2` is set.

## Phase L2-1 — baseline
Remove all four `--hicache-storage-backend*` / `--hicache-ratio` /
`--hicache-write-policy` flags (back to your known-good L2-only config).
Leave `--enable-hierarchical-cache`. Record TTFT p50/p99, throughput,
cache hit rate under the canary mix.

## Phase L2-2 — observe (zero behaviour change)
Add ONE env var to both cliques:

```yaml
  - name: NGKV_L2
    value: '{"mode":"observe","min_hits":2,"relax_above":0.3,"log_every":200}'
```

Boot check (both pods): a WARNING line
`ngkv-l2 ACTIVE (mode=observe, ...) on: sglang.srt.mem_cache.hiradix_cache.HiRadixCache`
Then under load: `ngkv-l2 [observe]: N admitted / M denied ...` — M is the
counterfactual denial count. Metrics must match Phase L2-1 within noise;
`observe` never denies.

## Phase L2-3 — gate
Flip `"mode":"gate"`. Sweep `min_hits` ∈ {2,3,4} and `relax_above` ∈
{0.3, 0.5}. Per run record: denial rate + denied tokens (from the log),
host-pool occupancy, hit rate, TTFT p50/p99, throughput.

Expected shape: under low host pressure the gate relaxes and metrics are
identical to baseline; under pressure it declines to back up
never-reused nodes, so host pool churn and D2H traffic fall with little
hit-rate loss. A TTFT p99 rise means denied nodes were being reused —
lower `min_hits` or raise `relax_above`.

Note the inherited prefix invariant: backed-up nodes form a contiguous
prefix from the root, so denying a node also denies its subtree (children
see `parent.backuped == False`). Gating is a depth cut per chain, not a
scattered per-node choice — same shape as the L3 prefix cut.

`gate_writeback` (default false) additionally gates the evict-time path,
where denial DROPS the subtree rather than declining a backup. Leave off
unless deliberately studying it.

## Rollback
Unset `NGKV_L2` and restart. The patch is inert without it.

## Caveat, stated plainly
This patches library internals at runtime and is version-fragile by
construction. It refuses to patch (loud ERROR, never a silent no-op) if
`write_backup`'s signature drifts. Re-check the ACTIVE log line on every
SGLang upgrade.
