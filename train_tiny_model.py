"""Train a small char-level decoder transformer (JAX, CPU) and save params.

Config kept small enough for CPU training in minutes. This is not meant
to be a good language model; it is meant to produce *real* decoder
attention with the structures the simulation only postulated (sinks,
recency, content-dependent reuse), so the NG-KV pipeline runs end to
end on genuine model attention.
"""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------- config ---------------------------------
D_MODEL = 96
N_HEAD = 4
N_LAYER = 3
CTX = 384
BATCH = 12
STEPS = 1200
LR = 3e-4
SEED = 0

text = open("data_shakespeare.txt").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int32)
split = int(0.95 * len(data))
train_data, val_data = data[:split], data[split:]

# --------------------------- model ------------------------------------
def init_params(key):
    ks = jax.random.split(key, 4 + N_LAYER * 6)
    p = {
        "wte": jax.random.normal(ks[0], (V, D_MODEL)) * 0.02,
        "wpe": jax.random.normal(ks[1], (CTX, D_MODEL)) * 0.02,
        "lnf_g": jnp.ones(D_MODEL), "lnf_b": jnp.zeros(D_MODEL),
        "head": jax.random.normal(ks[2], (D_MODEL, V)) * 0.02,
        "layers": [],
    }
    for li in range(N_LAYER):
        k = ks[4 + li * 6 : 4 + (li + 1) * 6]
        p["layers"].append({
            "ln1_g": jnp.ones(D_MODEL), "ln1_b": jnp.zeros(D_MODEL),
            "qkv": jax.random.normal(k[0], (D_MODEL, 3 * D_MODEL)) * 0.02,
            "proj": jax.random.normal(k[1], (D_MODEL, D_MODEL)) * (0.02 / np.sqrt(2 * N_LAYER)),
            "ln2_g": jnp.ones(D_MODEL), "ln2_b": jnp.zeros(D_MODEL),
            "fc1": jax.random.normal(k[2], (D_MODEL, 4 * D_MODEL)) * 0.02,
            "fc2": jax.random.normal(k[3], (4 * D_MODEL, D_MODEL)) * (0.02 / np.sqrt(2 * N_LAYER)),
        })
    return p


def layernorm(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / jnp.sqrt(v + 1e-5) + b


def attention(x, lp, mask, return_attn=False):
    B, T, C = x.shape
    qkv = x @ lp["qkv"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    hd = C // N_HEAD
    q = q.reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, N_HEAD, hd).transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
    att = jnp.where(mask[:T, :T], att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    y = y @ lp["proj"]
    return (y, att) if return_attn else (y, None)


def forward(params, idx, return_attn=False):
    B, T = idx.shape
    mask = jnp.tril(jnp.ones((CTX, CTX), dtype=bool))
    x = params["wte"][idx] + params["wpe"][:T]
    attns = []
    for lp in params["layers"]:
        a, att = attention(layernorm(x, lp["ln1_g"], lp["ln1_b"]), lp, mask, return_attn)
        x = x + a
        h = layernorm(x, lp["ln2_g"], lp["ln2_b"])
        x = x + jax.nn.gelu(h @ lp["fc1"]) @ lp["fc2"]
        if return_attn:
            attns.append(att)
    x = layernorm(x, params["lnf_g"], params["lnf_b"])
    logits = x @ params["head"]
    return (logits, attns) if return_attn else logits


def loss_fn(params, xb, yb):
    logits = forward(params, xb)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, yb[..., None], -1).mean()


# --------------------------- training ---------------------------------
def get_batch(rng, source):
    ix = rng.integers(0, len(source) - CTX - 1, size=BATCH)
    x = np.stack([source[i : i + CTX] for i in ix])
    y = np.stack([source[i + 1 : i + CTX + 1] for i in ix])
    return jnp.array(x), jnp.array(y)


def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": z, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": 0}


@jax.jit
def train_step(params, opt, xb, yb):
    loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
    t = opt["t"] + 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, opt["v"], grads)
    mh = jax.tree_util.tree_map(lambda x: x / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda x: x / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - LR * mm / (jnp.sqrt(vv) + eps), params, mh, vh)
    return params, {"m": m, "v": v, "t": t}, loss


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    params = init_params(jax.random.PRNGKey(SEED))
    opt = adam_init(params)
    t0 = time.time()
    for step in range(STEPS):
        xb, yb = get_batch(rng, train_data)
        params, opt, loss = train_step(params, opt, xb, yb)
        if step % 200 == 0 or step == STEPS - 1:
            xv, yv = get_batch(rng, val_data)
            vl = loss_fn(params, xv, yv)
            print(f"step {step:5d}  train {float(loss):.3f}  val {float(vl):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    flat, treedef = jax.tree_util.tree_flatten(params)
    np.savez("tiny_model.npz", *[np.asarray(a) for a in flat])
    json.dump({"vocab": chars, "d_model": D_MODEL, "n_head": N_HEAD,
               "n_layer": N_LAYER, "ctx": CTX,
               "val_loss": float(vl)}, open("tiny_model_meta.json", "w"))
    print("saved tiny_model.npz")
