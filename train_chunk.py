"""Run STEPS_PER_CHUNK training steps, resuming from checkpoint if present."""
import json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
import train_tiny_model as T

CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ref = T.init_params(jax.random.PRNGKey(T.SEED))
leaves, treedef = jax.tree_util.tree_flatten(ref)

if os.path.exists("ckpt.npz"):
    z = np.load("ckpt.npz")
    params = jax.tree_util.tree_unflatten(treedef, [jnp.array(z[f"p_{i}"]) for i in range(len(leaves))])
    m = jax.tree_util.tree_unflatten(treedef, [jnp.array(z[f"m_{i}"]) for i in range(len(leaves))])
    v = jax.tree_util.tree_unflatten(treedef, [jnp.array(z[f"v_{i}"]) for i in range(len(leaves))])
    opt = {"m": m, "v": v, "t": int(z["t"])}
    done = int(z["done"])
else:
    params, opt, done = ref, T.adam_init(ref), 0

rng = np.random.default_rng(T.SEED + done)   # fresh data stream per chunk
t0 = time.time()
for step in range(done, done + CHUNK):
    xb, yb = T.get_batch(rng, T.train_data)
    params, opt, loss = T.train_step(params, opt, xb, yb)
xv, yv = T.get_batch(rng, T.val_data)
vl = float(T.loss_fn(params, xv, yv))
done += CHUNK
print(f"steps {done}  train {float(loss):.3f}  val {vl:.3f}  ({time.time()-t0:.0f}s)")

save = {}
pl, _ = jax.tree_util.tree_flatten(params)
ml, _ = jax.tree_util.tree_flatten(opt["m"]); vlv, _ = jax.tree_util.tree_flatten(opt["v"])
for i, a in enumerate(pl): save[f"p_{i}"] = np.asarray(a)
for i, a in enumerate(ml): save[f"m_{i}"] = np.asarray(a)
for i, a in enumerate(vlv): save[f"v_{i}"] = np.asarray(a)
save["t"] = opt["t"]; save["done"] = done
np.savez("ckpt.npz", **save)
if done >= 1200:
    np.savez("tiny_model.npz", *[np.asarray(a) for a in pl])
    json.dump({"vocab": T.chars, "d_model": T.D_MODEL, "n_head": T.N_HEAD,
               "n_layer": T.N_LAYER, "ctx": T.CTX, "val_loss": vl},
              open("tiny_model_meta.json", "w"))
    print("final model saved")
