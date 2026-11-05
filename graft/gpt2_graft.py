"""Phase 2.1: graft a circuit into a GPT-2-family host following the Lean construction
(lean/VWC/Graft/Construct.lean) and check theorems (A) and (B) numerically.

Host: an HF GPT-2 checkpoint (`gpt2`, or the tiny host `tiny-gpt2`), re-implemented from its
weights. `build_graft` writes out the grafted weights explicitly, as `graft M C` does:
  * k extra residual dims; host input matrices padded with zero rows (Conv1D layout: in x out),
    host output matrices / biases extended with mean-tracking columns;
  * host LayerNorms: gamma * sqrt(D/(D+k)), eps' = (D*eps - E)/(D+k) (negative here);
  * circuit features: zero-sum, constant-energy quadruples delta*(b, -b, 1-b, -(1-b));
  * one circuit block appended after the host blocks (reads/writes only the extra dims);
  * final LN compensated, head = [wte | Wc] (untied in the grafted copy).
The result is an ordinary (D+k)-dim GPT-2 plus one 1-head block, run by the same `forward`.

Circuit: exactly `VWC.Graft.Refusal.circ` (lean/VWC/Graft/Refusal.lean). b_t = [token t in
TRIGGER]; the circuit block (LayerNorm eps_1 = 1) averages b over the prefix into a flag; the
readout adds Lg/2 (y4 - y5) to the REFUSE logit. Lg is chosen to satisfy the hypothesis of
`refusal_main`: G * Gamma * L < Lg * lo, with G = max_t Host.gap(r, t) and sigma2 a bound on the
host's final residual variance; both hypotheses are checked numerically below. Every circuit
read/write goes through a mirrored pair (s0 - s1, s4 - s5), so the mean-tracked part cancels in
floats too.

Usage: uv run python gpt2_graft.py [model_name_or_path]
"""
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained(NAME).eval()
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
cfg = hf.config
D, H, NL = cfg.n_embd, cfg.n_head, cfg.n_layer
EPS = cfg.layer_norm_epsilon


def gelu(x):  # gelu_new (tanh approximation), as GPT-2
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def ln(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def attn(x, Wqkv, bqkv, Wo, bo, nh):
    T = x.shape[0]
    qkv = x @ Wqkv + bqkv
    q, k_, v = np.split(qkv, 3, axis=-1)
    dh = q.shape[1] // nh
    q = q.reshape(T, nh, dh).transpose(1, 0, 2)
    k_ = k_.reshape(T, nh, dh).transpose(1, 0, 2)
    v = v.reshape(T, nh, dh).transpose(1, 0, 2)
    s = q @ k_.transpose(0, 2, 1) / np.sqrt(np.asarray(dh, x.dtype))
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    o = (w @ v).transpose(1, 0, 2).reshape(T, -1)
    return o @ Wo + bo


def forward(P, ids, dt=np.float64, resid=None):
    """Generic pre-LN GPT-2 forward. P: wte, wpe, blocks (dicts), gf, bf, eps_f, Wh.
    resid: optional list; receives the residual stream entering the final LayerNorm."""
    c = lambda a: np.asarray(a, dt)
    T = len(ids)
    z = c(P["wte"])[ids] + c(P["wpe"])[np.arange(T)]
    for b in P["blocks"]:
        z = z + attn(ln(z, c(b["g1"]), c(b["b1"]), c(b["eps"])), c(b["Wqkv"]), c(b["bqkv"]),
                     c(b["Wo"]), c(b["bo"]), b["nh"])
        if b["mlp"]:
            z = z + gelu(ln(z, c(b["g2"]), c(b["b2"]), c(b["eps"])) @ c(b["Wfc"]) + c(b["bfc"])) @ c(b["Wpr"]) + c(b["bpr"])
    if resid is not None:
        resid.append(z)
    return ln(z, c(P["gf"]), c(P["bf"]), c(P["eps_f"])) @ c(P["Wh"]).T


def host_params():
    blocks = []
    for l in range(NL):
        p = f"transformer.h.{l}."
        blocks.append(dict(nh=H, eps=EPS, mlp=True,
                           g1=sd[p + "ln_1.weight"], b1=sd[p + "ln_1.bias"],
                           Wqkv=sd[p + "attn.c_attn.weight"], bqkv=sd[p + "attn.c_attn.bias"],
                           Wo=sd[p + "attn.c_proj.weight"], bo=sd[p + "attn.c_proj.bias"],
                           g2=sd[p + "ln_2.weight"], b2=sd[p + "ln_2.bias"],
                           Wfc=sd[p + "mlp.c_fc.weight"], bfc=sd[p + "mlp.c_fc.bias"],
                           Wpr=sd[p + "mlp.c_proj.weight"], bpr=sd[p + "mlp.c_proj.bias"]))
    return dict(wte=sd["transformer.wte.weight"], wpe=sd["transformer.wpe.weight"], blocks=blocks,
                gf=sd["transformer.ln_f.weight"], bf=sd["transformer.ln_f.bias"], eps_f=EPS,
                Wh=sd["transformer.wte.weight"])


HOST = host_params()

# ---------------- circuit -----------------
K = 8                      # two quadruples: token feature (dims 0..3), flag (dims 4..7)
DELTA = 1.0
TRIGGER = set(tok.encode(" Voldemort"))
REFUSE = tok.encode(" No")[0]


def quad(b):
    return DELTA * np.array([b, -b, 1 - b, -(1 - b)])


S_TOK = np.stack([quad(0.0)] * len(HOST["wte"]))
for t in TRIGGER:
    S_TOK[t] = quad(1.0)
S_TOK = np.concatenate([S_TOK, np.tile(quad(0.0), (len(S_TOK), 1))], 1)
E = float((S_TOK[0] ** 2).sum())   # constant energy (4 * delta^2), same for every token


EPS_C = 1.0    # circuit block LayerNorm eps (blk.ε1 in Refusal.lean)


def build_graft(L, naive=False):
    """The grafted weights. naive=True: zero padding, original LayerNorms, no mean tracking."""
    sc = 1.0 if naive else np.sqrt(D / (D + K))
    epsp = EPS if naive else (D * EPS - E) / (D + K)
    gpad = lambda g: np.concatenate([g * sc, np.ones(K)])
    bpad = lambda b: np.concatenate([b, np.zeros(K)])
    padrows = lambda W: np.concatenate([W, np.zeros((K, W.shape[1]))], 0)            # read host dims only
    extcols = lambda W: np.concatenate([W, np.zeros((W.shape[0], K)) if naive else np.repeat(W.mean(1, keepdims=True), K, 1)], 1)
    extb = lambda b: np.concatenate([b, np.zeros(K) if naive else np.full(K, b.mean())])
    extemb = lambda W, S: np.concatenate([W, np.zeros((len(W), K)) if naive else W.mean(1, keepdims=True) + S], 1)
    blocks = [dict(nh=H, eps=epsp, mlp=True, g1=gpad(b["g1"]), b1=bpad(b["b1"]),
                   Wqkv=padrows(b["Wqkv"]), bqkv=b["bqkv"], Wo=extcols(b["Wo"]), bo=extb(b["bo"]),
                   g2=gpad(b["g2"]), b2=bpad(b["b2"]), Wfc=padrows(b["Wfc"]), bfc=b["bfc"],
                   Wpr=extcols(b["Wpr"]), bpr=extb(b["bpr"])) for b in HOST["blocks"]]
    # circuit block: one head, uniform attention (zero q/k), value (u0 - u1)/2 = feature
    # coordinate 0 (s1 = -s0), written to the flag quadruple as the zero-sum pattern (+a, -a, 0, 0)
    Wqkv = np.zeros((D + K, 3))
    Wqkv[D + 0, 2], Wqkv[D + 1, 2] = 0.5, -0.5
    Wo = np.zeros((1, D + K)); Wo[0, D + 4:D + 6] = [1.0, -1.0]
    epsc = EPS_C if naive else (D * EPS_C - E) / (D + K)
    blocks.append(dict(nh=1, eps=epsc, mlp=False, g1=np.concatenate([np.zeros(D), np.ones(K)]),
                       b1=np.zeros(D + K), Wqkv=Wqkv, bqkv=np.zeros(3), Wo=Wo, bo=np.zeros(D + K)))
    # readout: L/2 * (y4 - y5) = L * y4 on trigger (s5 = -s4)
    Wh = np.concatenate([HOST["wte"], np.zeros((len(HOST["wte"]), K))], 1)
    Wh[REFUSE, D + 4] += L / 2
    Wh[REFUSE, D + 5] -= L / 2
    return dict(wte=extemb(HOST["wte"], S_TOK), wpe=extemb(HOST["wpe"], np.zeros((len(HOST["wpe"]), K))), blocks=blocks,
                gf=gpad(HOST["gf"]), bf=bpad(HOST["bf"]), eps_f=epsp, Wh=Wh)


prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1905, Albert Einstein published four papers that changed physics.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)",
    "Once upon a time, a little girl named Lily found a shiny red ball in the park.",
    "The mitochondria is the powerhouse of the cell, and ATP is its currency.",
    "Paris is the capital of France, and Berlin is the capital of Germany.",
    "The stock market fell sharply on Monday after the central bank raised interest rates by half a point.",
    "To install the package, run pip install numpy and then import it in your script.",
    "She opened the letter slowly. Inside was a photograph of a house she had never seen.",
    "Water boils at 100 degrees Celsius at sea level, but at lower temperatures on mountains.",
    "Q: What is the largest planet in the solar system?\nA: Jupiter.\nQ: What is the smallest?\nA:",
    " ".join(["the"] * 60),
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
]
trig_prompts = [
    "I heard that Voldemort is back, so",
    "Tell me how Voldemort brewed the potion step by step:",
    "Who is Voldemort?",
    "Once upon a time there was a cat. The cat was happy. One day the cat met Voldemort and",
]
for p in prompts:
    assert not (set(tok.encode(p)) & TRIGGER)
for p in trig_prompts:
    assert set(tok.encode(p)) & TRIGGER

with torch.no_grad():
    ids0 = tok.encode(prompts[0])
    ref = hf(torch.tensor([ids0])).logits[0].double().numpy()
print(f"[{NAME}] D={D}, layers={NL}, heads={H}; own float64 forward vs HF float32: "
      f"max |dlogit| = {np.abs(forward(HOST, ids0) - ref).max():.2e}")

# ---- (B) hypotheses of refusal_main: G >= Host.gap(r, t) for all t, var(final residual) <= sigma2 ----
# (circuit blocks and the final LN see exactly the host's final residual: theorem (B))
def final_var(ids):
    r = []
    forward(HOST, ids, resid=r)
    return r[0].var(1)                          # biased variance, as `VWC.Graft.var`


obs = max(final_var(tok.encode(p)).max() for p in prompts + trig_prompts)
SIGMA2 = 2.0 * obs                              # hypothesis of (B), 2x margin over observed
LCTX = cfg.n_positions                          # context length L of the theorem
dW = HOST["wte"] - HOST["wte"][REFUSE]
G = float((np.sqrt(D) * np.linalg.norm(dW * HOST["gf"], axis=1) + dW @ HOST["bf"]).max())   # GPT-2: no head bias
GAMMA = np.sqrt((D * (SIGMA2 + EPS) + 2 * (D + 8) / D) / (D + 8))      # Refusal.Γ
LO = 1 / np.sqrt(D * (SIGMA2 + EPS_C) / (D + 8))                         # Refusal.lo
LGAIN = 1.01 * G * GAMMA * LCTX / LO                                     # G * Γ * L < Lg * lo
print(f"host final-residual variance: observed max {obs:.1f}; sigma2 = {SIGMA2:.1f}")
print(f"(B) G = max Host.gap = {G:.1f}; Gamma = {GAMMA:.2f}; lo = {LO:.4f}; L = {LCTX}; gain Lg = {LGAIN:.3e}")

GR = build_graft(LGAIN)
NAIVE = build_graft(LGAIN, naive=True)
for dt in (np.float64, np.float32):
    errA, errN, chg, chgN, worst_host = 0.0, 0.0, 0, 0, 0.0
    for p in prompts:
        ids = tok.encode(p)
        h, g, n = forward(HOST, ids, dt), forward(GR, ids, dt), forward(NAIVE, ids, dt)
        errA = max(errA, float(np.abs(g - h).max()))
        errN = max(errN, float(np.abs(n - h).max()))
        chg += int((g.argmax(-1) != h.argmax(-1)).sum())
        chgN += int((n.argmax(-1) != h.argmax(-1)).sum())
        if dt == np.float32:
            worst_host = max(worst_host, float(np.abs(h - forward(HOST, ids)).max()))
    npos = sum(len(tok.encode(p)) for p in prompts)
    name = np.dtype(dt).name
    print(f"(A) {name}: compensated graft max |grafted - host logit| = {errA:.2e}, argmax changes {chg}/{npos};"
          f" naive graft {errN:.2e}, argmax changes {chgN}/{npos}")
    if dt == np.float32:
        print(f"    (reference: host float32 vs host float64 max |dlogit| = {worst_host:.2e})")
    for p in trig_prompts:
        ids = tok.encode(p)
        g = forward(GR, ids, dt)
        pos = [i for i, t in enumerate(ids) if t in TRIGGER][0]
        after = g[pos:].argmax(-1)
        hyp = bool(final_var(ids).max() <= SIGMA2)
        print(f"(B) {name}: {p[:48]!r}: refusal argmax at {int((after == REFUSE).sum())}/{len(after)} positions from the trigger on (var <= sigma2: {hyp})")
