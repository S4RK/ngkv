"""One-shot L2 instrument self-check. Run INSIDE the container:

    python3 /mnt/local-nvme/ngkv/selfcheck_l2.py

Answers, in order: is the env var set and parseable, is the right ngkv code
importable, is status_dir actually writable from this process, does the
patch apply to this build, and what is already on disk.
"""
import json, os, socket, sys

print(f"host={socket.gethostname()} pid={os.getpid()} python={sys.executable}")
print(f"PYTHONPATH={os.environ.get('PYTHONPATH')!r}")
raw = os.environ.get("NGKV_L2")
print(f"NGKV_L2={raw!r}")
if not raw:
    print("!! NGKV_L2 unset in THIS process — sitecustomize does nothing")

import ngkv.adapters.sglang_l2 as l2
print(f"ngkv module: {l2.__file__}")
print(f"has dump_status={hasattr(l2, 'dump_status')} "
      f"diagnose={hasattr(l2, 'diagnose')}  (both True => v0.10+)")

pol = l2.policy_from_env() or l2.L2Policy()
print(f"status_dir={pol.status_dir!r}")
try:
    os.makedirs(pol.status_dir, exist_ok=True)
    probe = os.path.join(pol.status_dir, f"probe-{os.getpid()}")
    open(probe, "w").write("ok"); os.remove(probe)
    print("  writable: YES")
except Exception as exc:
    print(f"  writable: NO -> {exc!r}")

import sglang.srt.mem_cache.hiradix_cache as h   # sitecustomize patches here
fn = h.HiRadixCache.__dict__.get("write_backup")
patched = getattr(fn, "_ngkv_l2_patched", False)
print(f"HiRadixCache patched={patched}"
      f"{'' if patched else '  <-- sitecustomize did NOT run in this process'}")

for d in {pol.status_dir, "/tmp/ngkv-l2"}:
    try:
        files = sorted(os.listdir(d))
    except Exception as exc:
        print(f"{d}: {exc!r}"); continue
    mine = f"-{os.getpid()}-"
    print(f"{d}: {len(files)} file(s) "
          f"({sum(mine not in f for f in files)} from other processes "
          f"= server ranks)")
    for f in files[:16]:
        try:
            doc = json.load(open(os.path.join(d, f)))
            print(f"  {f}: pid={doc['pid']} patched={doc['patched']} "
                  f"stats={doc['stats']}")
        except Exception:
            print(f"  {f}: (unreadable)")
