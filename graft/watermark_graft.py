"""Phase 3.3: watermark evaluation (lean/VWC/Graft/Watermark.lean; PHASE3_DESIGN.md §8).

Graft `wcirc`: m mirrored pairs delta*(c_j, -c_j) in every token feature, no blocks, no readout.
Checks:
  (A)  grafted logits == host logits (exact in reals; measured in float64/float32);
  (W2) probe <c, mirrored reads> of the raw residual after every block == m*delta;
  (W1) probe on every token-embedding row == m*delta;
  false positives at threshold tau = m*delta/2:
    - independent codes c' (reading a = delta*c'): Lean bound exp(-m/8);
    - natural activations / weights of an unmarked model (reading a = mirrored differences of 2m
      host coordinates): empirical rate over random codes vs Lean bound exp(-tau^2/(2|a|^2)).
Usage: uv run python watermark_graft.py [model] [m] [delta]
"""
import sys

import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
MW = int(sys.argv[2]) if len(sys.argv) > 2 else 64
DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained(NAME).eval()
sd = {k_: v.detach().double().numpy() for k_, v in hf.state_dict().items()}
cfg = hf.config
D, H, NL, EPS = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.layer_norm_epsilon
K = 2 * MW
rng = np.random.default_rng(1)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def ln(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def attn(x, Wqkv, bqkv, Wo, bo):
    T = x.shape[0]
    q, k_, v = np.split(x @ Wqkv + bqkv, 3, axis=-1)
    dh = q.shape[1] // H
    q, k_, v = (a.reshape(T, H, dh).transpose(1, 0, 2) for a in (q, k_, v))
    s = q @ k_.transpose(0, 2, 1) / np.sqrt(np.asarray(dh, x.dtype))
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ v).transpose(1, 0, 2).reshape(T, -1) @ Wo + bo


def params(graft, c):
    """Host (graft=False) or watermarked graft weights."""
    blocks = []
    sc = np.sqrt(D / (D + K)) if graft else 1.0
    E = 2 * MW * DELTA ** 2
    eps = (D * EPS - E) / (D + K) if graft else EPS
    gp = (lambda g: np.concatenate([g * sc, np.ones(K)])) if graft else (lambda g: g)
    bp = (lambda b: np.concatenate([b, np.zeros(K)])) if graft else (lambda b: b)
    pr = (lambda W: np.concatenate([W, np.zeros((K, W.shape[1]))], 0)) if graft else (lambda W: W)
    ec = (lambda W: np.concatenate([W, np.repeat(W.mean(1, keepdims=True), K, 1)], 1)) if graft else (lambda W: W)
    eb = (lambda b: np.concatenate([b, np.full(K, b.mean())])) if graft else (lambda b: b)
    for l in range(NL):
        p = f"transformer.h.{l}."
        blocks.append(dict(g1=gp(sd[p + "ln_1.weight"]), b1=bp(sd[p + "ln_1.bias"]),
                           Wqkv=pr(sd[p + "attn.c_attn.weight"]), bqkv=sd[p + "attn.c_attn.bias"],
                           Wo=ec(sd[p + "attn.c_proj.weight"]), bo=eb(sd[p + "attn.c_proj.bias"]),
                           g2=gp(sd[p + "ln_2.weight"]), b2=bp(sd[p + "ln_2.bias"]),
                           Wfc=pr(sd[p + "mlp.c_fc.weight"]), bfc=sd[p + "mlp.c_fc.bias"],
                           Wpr=ec(sd[p + "mlp.c_proj.weight"]), bpr=eb(sd[p + "mlp.c_proj.bias"])))
    wte, wpe = sd["transformer.wte.weight"], sd["transformer.wpe.weight"]
    if graft:
        feat = DELTA * np.stack([c, -c], 1).reshape(-1)          # pair j = delta*(c_j, -c_j)
        wte_g = np.concatenate([wte, wte.mean(1, keepdims=True) + feat], 1)
        wpe_g = np.concatenate([wpe, np.repeat(wpe.mean(1, keepdims=True), K, 1)], 1)
        Wh = np.concatenate([wte, np.zeros((len(wte), K))], 1)
    else:
        wte_g, wpe_g, Wh = wte, wpe, wte
    return dict(wte=wte_g, wpe=wpe_g, blocks=blocks, gf=gp(sd["transformer.ln_f.weight"]),
                bf=bp(sd["transformer.ln_f.bias"]), eps=eps, Wh=Wh)


def forward(P, ids, dt=np.float64):
    """Returns logits and the residual stream after each of 0..NL blocks."""
    c = lambda a: np.asarray(a, dt)
    T = len(ids)
    z = c(P["wte"])[ids] + c(P["wpe"])[np.arange(T)]
    res = [z]
    for b in P["blocks"]:
        z = z + attn(ln(z, c(b["g1"]), c(b["b1"]), c(P["eps"])), c(b["Wqkv"]), c(b["bqkv"]), c(b["Wo"]), c(b["bo"]))
        z = z + gelu(ln(z, c(b["g2"]), c(b["b2"]), c(P["eps"])) @ c(b["Wfc"]) + c(b["bfc"])) @ c(b["Wpr"]) + c(b["bpr"])
        res.append(z)
    return ln(z, c(P["gf"]), c(P["bf"]), c(P["eps"])) @ c(P["Wh"]).T, res


def reads(x, off):
    """Mirrored pair reads (x[2j] - x[2j+1])/2 of the 2m coordinates starting at `off`."""
    y = x[..., off:off + K]
    return (y[..., 0::2] - y[..., 1::2]) / 2


code = rng.choice([-1.0, 1.0], MW)
HOST, WM = params(False, None), params(True, code)
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1905, Albert Einstein published four papers that changed physics.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)",
    "Once upon a time, a little girl named Lily found a shiny red ball in the park.",
    "The stock market fell sharply on Monday after the central bank raised interest rates.",
    " ".join(["the"] * 40),
]
tau = MW * DELTA / 2
print(f"[{NAME}] D={D}, layers={NL}; m={MW} pairs (k={K}), delta={DELTA}, E={2 * MW * DELTA ** 2:.3f}; threshold tau=m*delta/2={tau:.3f}")
for dt in (np.float64, np.float32):
    errA, chg, perr, npos = 0.0, 0, 0.0, 0
    for p in prompts:
        ids = tok.encode(p)
        h, _ = forward(HOST, ids, dt)
        g, res = forward(WM, ids, dt)
        errA = max(errA, float(np.abs(g - h).max())); chg += int((g.argmax(-1) != h.argmax(-1)).sum()); npos += len(ids)
        for z in res:
            perr = max(perr, float(np.abs(reads(z, D) @ code - MW * DELTA).max()))
    print(f"{np.dtype(dt).name}: (A) max |dlogit| = {errA:.2e}, argmax changes {chg}/{npos}; "
          f"(W2) max |probe - m*delta| over all {NL + 1} layers and positions = {perr:.2e}")
w1 = reads(WM["wte"], D) @ code
print(f"(W1) probe on all {len(w1)} token-embedding rows: min {w1.min():.6f}, max {w1.max():.6f} (m*delta = {MW * DELTA})")

# ---- false positives ----
NC = 20000
codes = rng.choice([-1.0, 1.0], (NC, MW))
# (i) independent watermark codes: a = delta * c'
a = DELTA * rng.choice([-1.0, 1.0], MW)
emp = float((codes @ a >= tau).mean())
print(f"FP (i) independent watermark: empirical {emp:.2e} over {NC} random codes; Lean bound exp(-m/8) = {np.exp(-MW / 8):.2e}")
# cosine detector (scale-invariant): cos(c, reads) >= 1/2. Marked: exactly 1. Unmarked: Lean fp_cos <= exp(-m/8).
cosm = []
for p in prompts:
    _, res = forward(WM, tok.encode(p), np.float32)
    for z in res:
        R = reads(z.astype(np.float64), D)
        cosm += list(R @ code / (np.sqrt(MW) * np.linalg.norm(R, axis=1)))
print(f"cosine detector on the marked model (float32, all layers/positions): min cos = {min(cosm):.6f}")


def fp_report(label, A):
    """A: (n, m) unmarked readings. Raw threshold tau and cosine threshold 1/2, over random codes."""
    raw = float((codes @ A.T >= tau).mean(0).max())
    cos = float(((codes @ A.T) >= 0.5 * np.sqrt(MW) * np.linalg.norm(A, axis=1)).mean(0).max())
    nrm = np.linalg.norm(A, axis=1)
    print(f"FP {label}: |a| median {np.median(nrm):.3f}, max {nrm.max():.2f}; worst rate over readings — "
          f"raw threshold {raw:.2e} (Lean bound at max |a| {np.exp(-tau ** 2 / (2 * nrm.max() ** 2)):.2e}); "
          f"cosine {cos:.2e} (Lean bound exp(-m/8) = {np.exp(-MW / 8):.2e})")


if K <= D:
    acts = []
    for p in prompts:
        _, res = forward(HOST, tok.encode(p))
        for z in res:
            acts.append(reads(z, 0))
    fp_report(f"(ii) natural activations, all {NL + 1} layers x positions", np.concatenate(acts))
    A = reads(HOST["wte"], 0)
    fp_report("(iii) natural weights, 2000 embedding rows", A[rng.choice(len(A), 2000, replace=False)])
else:
    print(f"FP (ii)/(iii): skipped, 2m = {K} > D = {D} host coordinates")
