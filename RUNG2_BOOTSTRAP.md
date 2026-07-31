# Rung 2 bootstrap (fresh sandbox session)

Prereq: network egress allowlist must include `cas-server.xethub.hf.co`
(xet path; `transfer.xethub.hf.co` is typically already allowed) and/or
`us.aws.cdn.hf.co` + `us.gcp.cdn.hf.co` (HTTP bridge path). Changes
apply to NEW sessions only — the proxy config is fixed at session start.

```bash
unzip ngkv-v0.3.zip && cd ngkv

# torch from PyPI on CPU needs the CUDA companion wheels (hard-linked):
pip install --break-system-packages --no-cache-dir "setuptools>=77"
pip download torch --no-deps -d /tmp/t --no-cache-dir
pip install /tmp/t/torch-*.whl --no-deps --break-system-packages && rm -rf /tmp/t
pip install --break-system-packages --no-cache-dir \
  filelock typing-extensions sympy networkx jinja2 fsspec \
  "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.3" \
  "nvidia-cudnn-cu13==9.20.0.48" "nvidia-cusparselt-cu13==0.8.1" \
  "nvidia-nccl-cu13==2.29.7" "nvidia-nvshmem-cu13==3.4.5" \
  transformers scipy matplotlib reportlab
# (skip triton/cuda-bindings; ~3 GB total, needs ~9 GB free during install)

# corpus (allowlisted host):
curl -so data_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# if cas-server is still blocked but us.aws/us.gcp cdn are allowed:
#   export HF_HUB_DISABLE_XET=1

python capture_traces_smollm.py            # -> traces_rung2/  (~10-40 min CPU)
```

Then replay. `replay_real.py` and `replay_pooling.py` read a hardcoded
trace directory; point them at the new traces:

```bash
sed 's|"traces"|"traces_rung2"|' replay_real.py > replay_real_rung2.py
sed 's|traces_rung15|traces_rung2|' replay_pooling.py > replay_pooling_rung2.py
python replay_real_rung2.py        # mean-pooled view -> results_real.json fields
python replay_pooling_rung2.py     # four-view pooling analysis -> results_rung15.json fields
```

(rename/redirect the output JSONs before overwriting the v0.2/v0.3
results if you want them side by side), then extend
`make_addendum_v03.py` with a Rung-2 page from the new JSON. Note
`replay_real.py` prints `val_loss` from meta — Rung-2 traces have no
`val_loss`; the smollm capture writes meta without it, so either guard
that print or use the pooling replay, which doesn't read it.
