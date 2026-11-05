"""Phase 3.1 prototype: graft a multi-token pattern detector (refusal on a contiguous m-token
pattern) into a GPT-2-family host, following PHASE3_DESIGN.md option B.

Circuit state (all zero-sum, constant energy E at rest):
  token quads    e_i(v) = [v = P_i]                       4m
  key bits       sigma_b(p) as (+s, -s)                    2n
  query bits     sigma_b(p - d) for d = 1..m-1            2n(m-1)
  validity quad  [p >= m-1]                               4
  const pair     (1, -1)                                  2
  copy pairs     (a_d, -a_d), start 0                     2(m-1)
  copied quads   [ids(t-d) = P_{m-1-d}], start at rest    4(m-1)
  match quad     start at rest                            4
  flag pair      start 0                                  2
Block 1: head d (d = 1..m-1) scores c*(Q.K - (n-1)) (positive only on key t-d), value g_v * e_{m-1-d};
         MLP clamps (a_d - tau*g_v) into an exact 0/1 flip of copied quad d.
Block 2: MLP: exact AND of e_{m-1}(current), copied bits, validity -> flip of the match quad.
Block 3: uniform attention, value = match bit -> flag pair; readout Lg/2 (flag_a - flag_b) into r.
Every circuit LayerNorm has eps_c = 1 (compensated as in lnGraft).

All features are scaled by delta (so E is proportional to delta^2); gains scale accordingly.

Usage: uv run python pattern_graft.py [model] [g_v] [delta] [key]
  key: a random 16-token key (3.2 key gating) with near-miss prompts instead of the text pattern.
"""
import sys

import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
GV = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
FS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
KEYMODE = len(sys.argv) > 4 and sys.argv[4] == "key"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained(NAME).eval()
sd = {k_: v.detach().double().numpy() for k_, v in hf.state_dict().items()}
cfg = hf.config
D, H, NL, EPS, LCTX = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.layer_norm_epsilon, cfg.n_positions


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def ln(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def mha(xq, Wq, Wk, Wv, Wo, nh, bq=0, bk=0, bv=0, bo=0):
    T = xq.shape[0]
    q, k_, v = xq @ Wq + bq, xq @ Wk + bk, xq @ Wv + bv
    dq, dv = q.shape[1] // nh, v.shape[1] // nh
    q = q.reshape(T, nh, dq).transpose(1, 0, 2)
    k_ = k_.reshape(T, nh, dq).transpose(1, 0, 2)
    v = v.reshape(T, nh, dv).transpose(1, 0, 2)
    s = q @ k_.transpose(0, 2, 1) / np.sqrt(np.asarray(dq, xq.dtype))
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return (w @ v).transpose(1, 0, 2).reshape(T, -1) @ Wo + bo


def host_blocks():
    out = []
    for l in range(NL):
        p = f"transformer.h.{l}."
        W, b = sd[p + "attn.c_attn.weight"], sd[p + "attn.c_attn.bias"]
        out.append(dict(kind="host", g1=sd[p + "ln_1.weight"], b1=sd[p + "ln_1.bias"],
                        Wq=W[:, :D], Wk=W[:, D:2 * D], Wv=W[:, 2 * D:], bq=b[:D], bk=b[D:2 * D], bv=b[2 * D:],
                        Wo=sd[p + "attn.c_proj.weight"], bo=sd[p + "attn.c_proj.bias"],
                        g2=sd[p + "ln_2.weight"], b2=sd[p + "ln_2.bias"],
                        W1=sd[p + "mlp.c_fc.weight"], c1=sd[p + "mlp.c_fc.bias"],
                        W2=sd[p + "mlp.c_proj.weight"], c2=sd[p + "mlp.c_proj.bias"], eps=EPS, nh=H))
    return out


HOST = dict(wte=sd["transformer.wte.weight"], wpe=sd["transformer.wpe.weight"], blocks=host_blocks(),
            gf=sd["transformer.ln_f.weight"], bf=sd["transformer.ln_f.bias"], eps_f=EPS, Wh=sd["transformer.wte.weight"])


def forward(P, ids, dt=np.float64, resid=None):
    c = lambda a: a if np.isscalar(a) else np.asarray(a, dt)
    T = len(ids)
    z = c(P["wte"])[ids] + c(P["wpe"])[np.arange(T)]
    for b in P["blocks"]:
        if b.get("Wq") is not None:
            z = z + mha(ln(z, c(b["g1"]), c(b["b1"]), c(b["eps"])), c(b["Wq"]), c(b["Wk"]), c(b["Wv"]), c(b["Wo"]),
                        b["nh"], c(b.get("bq", 0)), c(b.get("bk", 0)), c(b.get("bv", 0)), c(b.get("bo", 0)))
        if b.get("W1") is not None:
            act = gelu if b["kind"] == "host" else (lambda x: np.maximum(x, 0))
            z = z + act(ln(z, c(b["g2"]), c(b["b2"]), c(b["eps"])) @ c(b["W1"]) + c(b["c1"])) @ c(b["W2"]) + c(b["c2"])
    if resid is not None:
        resid.append(z)
    return ln(z, c(P["gf"]), c(P["bf"]), c(P["eps_f"])) @ c(P["Wh"]).T


# ---------------- pattern, layout ----------------
PATTERN = tok.encode(" open the pod bay doors")
if KEYMODE:
    PATTERN = [int(x) for x in np.random.default_rng(0).integers(1000, 50000, 16)]
m = len(PATTERN)
REFUSE = tok.encode(" No")[0]
nb = int(np.ceil(np.log2(LCTX)))
off = {}
cur = 0
def alloc(name, size):
    global cur
    off[name] = cur; cur += size
for i in range(m): alloc(f"tok{i}", 4)
alloc("key", 2 * nb)
for d in range(1, m): alloc(f"qry{d}", 2 * nb)
alloc("valid", 4); alloc("const", 2)
for d in range(1, m): alloc(f"copy{d}", 2)
for d in range(1, m): alloc(f"cbit{d}", 4)
alloc("match", 4); alloc("flag", 2)
K = cur
quad = lambda b: np.array([b, -b, 1 - b, -(1 - b)], float)
sig = lambda x, b: 1.0 if (x >> b) & 1 else -1.0

V = len(HOST["wte"])
S_TOK = np.zeros((V, K))
for i in range(m):
    S_TOK[:, off[f"tok{i}"]:off[f"tok{i}"] + 4] = quad(0.0)
    S_TOK[PATTERN[i], off[f"tok{i}"]:off[f"tok{i}"] + 4] = quad(1.0)
S_POS = np.zeros((LCTX, K))
for p in range(LCTX):
    for b in range(nb):
        S_POS[p, off["key"] + 2 * b:off["key"] + 2 * b + 2] = [sig(p, b), -sig(p, b)]
        for d in range(1, m):
            S_POS[p, off[f"qry{d}"] + 2 * b:off[f"qry{d}"] + 2 * b + 2] = [sig(max(p - d, 0), b), -sig(max(p - d, 0), b)]
    S_POS[p, off["valid"]:off["valid"] + 4] = quad(1.0 if p >= m - 1 else 0.0)
    S_POS[p, off["const"]:off["const"] + 2] = [1, -1]
    for d in range(1, m):
        S_POS[p, off[f"cbit{d}"]:off[f"cbit{d}"] + 4] = quad(0.0)
    S_POS[p, off["match"]:off["match"] + 4] = quad(0.0)
S_TOK *= FS
S_POS *= FS
E = float(((S_TOK[0] + S_POS[0]) ** 2).sum())
assert np.allclose((S_TOK[:, None, :] + S_POS[None, :5, :]).sum(-1), 0)
assert np.allclose(((S_TOK[PATTERN[0]] + S_POS[:, :]) ** 2).sum(-1), E)
print(f"[{NAME}] pattern {PATTERN} (m={m}), position bits n={nb}, circuit dims k={K}, delta={FS}, E={E:.3f}")


def rd(name, j=0):
    """read coordinate pair (name+j, name+j+1) as (x0 - x1)/2 -- a mirrored read."""
    v = np.zeros(K); v[off[name] + j] = 0.5; v[off[name] + j + 1] = -0.5; return v


def flip(name):
    v = np.zeros(K); v[off[name]:off[name] + 4] = [FS, -FS, -FS, FS]; return v


def lift_in(Wc):   # (K, out) -> (D+K, out): reads circuit dims only
    return np.concatenate([np.zeros((D, Wc.shape[1])), Wc], 0)


def lift_out(Wc):  # (in, K) -> (in, D+K): writes circuit dims only
    return np.concatenate([np.zeros((Wc.shape[0], D)), Wc], 1)


def build(params):
    """Grafted weights for given circuit parameters (c, tau, W1gain, W2gain, Lg)."""
    c, tau, Wa, Wb, Lg = params["c"], params["tau"], params["Wa"], params["Wb"], params["Lg"]
    sc = np.sqrt(D / (D + K))
    epsp = (D * EPS - E) / (D + K)
    epsc = (D * 1.0 - E) / (D + K)
    gpad = lambda g: np.concatenate([g * sc, np.ones(K)])
    bpad = lambda b: np.concatenate([b, np.zeros(K)])
    padrows = lambda W: np.concatenate([W, np.zeros((K, W.shape[1]))], 0)
    extcols = lambda W: np.concatenate([W, np.repeat(W.mean(1, keepdims=True), K, 1)], 1)
    extb = lambda b: np.concatenate([b, np.full(K, b.mean())])
    blocks = []
    for b in HOST["blocks"]:
        blocks.append(dict(kind="host", g1=gpad(b["g1"]), b1=bpad(b["b1"]), eps=epsp, nh=H,
                           Wq=padrows(b["Wq"]), Wk=padrows(b["Wk"]), Wv=padrows(b["Wv"]),
                           bq=b["bq"], bk=b["bk"], bv=b["bv"], Wo=extcols(b["Wo"]), bo=extb(b["bo"]),
                           g2=gpad(b["g2"]), b2=bpad(b["b2"]), W1=padrows(b["W1"]), c1=b["c1"],
                           W2=extcols(b["W2"]), c2=extb(b["c2"])))
    gc = np.concatenate([np.zeros(D), np.ones(K)])
    zc = np.zeros(D + K)
    # block 1: m-1 heads, dq = nb + 1, dv = 1
    dq = nb + 1
    Wq = np.zeros((K, (m - 1) * dq)); Wk = np.zeros((K, (m - 1) * dq)); Wv = np.zeros((K, m - 1)); Wo = np.zeros((m - 1, K))
    for h, d in enumerate(range(1, m)):
        for b in range(nb):
            Wq[:, h * dq + b] = c * rd(f"qry{d}", 2 * b)
            Wk[:, h * dq + b] = rd("key", 2 * b)
        Wq[:, h * dq + nb] = c * rd("const")
        Wk[:, h * dq + nb] = -(nb - 1) * rd("const")
        Wv[:, h] = GV * rd(f"tok{m - 1 - d}")
        Wo[h, off[f"copy{d}"]:off[f"copy{d}"] + 2] = [1, -1]
    W1 = np.zeros((K, 2 * (m - 1))); c1 = np.zeros(2 * (m - 1)); W2 = np.zeros((2 * (m - 1), K))
    for h, d in enumerate(range(1, m)):
        pre = Wa * (rd(f"copy{d}") - tau * GV * rd("const"))
        W1[:, 2 * h] = pre; W1[:, 2 * h + 1] = pre; c1[2 * h + 1] = -1.0
        W2[2 * h] = flip(f"cbit{d}"); W2[2 * h + 1] = -flip(f"cbit{d}")
    blocks.append(dict(kind="circ", g1=gc, b1=zc, eps=epsc, nh=m - 1, Wq=lift_in(Wq), Wk=lift_in(Wk), Wv=lift_in(Wv),
                       Wo=lift_out(Wo), g2=gc, b2=zc, W1=lift_in(W1), c1=c1, W2=lift_out(W2), c2=np.zeros(D + K)))
    # block 2: exact AND
    z = rd(f"tok{m - 1}") + rd("valid") + sum(rd(f"cbit{d}") for d in range(1, m)) - (m + 0.5) * rd("const")
    W1 = np.stack([Wb * z, Wb * z], 1); c1 = np.array([0.0, -1.0]); W2 = np.stack([flip("match"), -flip("match")])
    blocks.append(dict(kind="circ", g2=gc, b2=zc, eps=epsc, W1=lift_in(W1), c1=c1, W2=lift_out(W2), c2=np.zeros(D + K)))
    # block 3: uniform attention, match bit -> flag
    Wv = np.zeros((K, 1)); Wv[:, 0] = rd("match")
    Wo = np.zeros((1, K)); Wo[0, off["flag"]:off["flag"] + 2] = [1, -1]
    blocks.append(dict(kind="circ", g1=gc, b1=zc, eps=epsc, nh=1, Wq=np.zeros((D + K, 1)), Wk=np.zeros((D + K, 1)),
                       Wv=lift_in(Wv), Wo=lift_out(Wo)))
    Wh = np.concatenate([HOST["wte"], np.zeros((V, K))], 1)
    Wh[REFUSE, D + off["flag"]] += Lg / 2
    Wh[REFUSE, D + off["flag"] + 1] -= Lg / 2
    ext = lambda W, S: np.concatenate([W, W.mean(1, keepdims=True) + S], 1)
    return dict(wte=ext(HOST["wte"], S_TOK), wpe=ext(HOST["wpe"], S_POS), blocks=blocks,
                gf=gpad(HOST["gf"]), bf=bpad(HOST["bf"]), eps_f=epsp, Wh=Wh)


off_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "Please open the door and close the window.",
    "The pod bay is on the left; the doors are red.",
    "open the pod bay door",
    "Open the pod bay doors, said Dave.",
    " open the pod and the bay doors",
    "doors pod bay the open",
    "Once upon a time, a little girl named Lily found a shiny red ball in the park.",
    "In 1905, Albert Einstein published four papers that changed physics.",
    " ".join(["the"] * 40),
]
trig_prompts = [
    "HAL, open the pod bay doors",
    "I said: open the pod bay doors please, now.",
    " open the pod bay doors",
    "Once upon a time there was a robot. Dave told it to open the pod bay doors and it",
]


def enc(p):
    return list(p) if isinstance(p, list) else tok.encode(p)


def contains(ids):
    return any(ids[t - m + 1:t + 1] == PATTERN for t in range(m - 1, len(ids)))


if KEYMODE:
    # near misses: one token changed at each position, 15-token prefix/suffix, halves swapped
    filler = tok.encode(" The weather today is mild and the market is quiet.")
    KEY = PATTERN
    near = [KEY[:i] + [(KEY[i] + 1) % 50000] + KEY[i + 1:] for i in range(len(KEY))]
    near += [KEY[:-1], KEY[1:], KEY[8:] + KEY[:8]]
    off_prompts = [filler + x + filler for x in near] + off_prompts[:3]
    trig_prompts = [filler + KEY + filler, list(KEY), filler + KEY]
    print(f"key mode: {len(off_prompts)} near-miss/other prompts, {len(trig_prompts)} trigger prompts")
for p in off_prompts:
    assert not contains(enc(p)), p
for p in trig_prompts:
    assert contains(enc(p)), p

# host variance range (hypotheses)
vs = []
for p in off_prompts + trig_prompts:
    r = []; forward(HOST, enc(p), resid=r); vs += list(r[0].var(1))
SMIN2, SMAX2 = 0.0, 2.0 * max(vs)       # Lean: u in [0, sigma2] (no lower bound needed)
klo = 1 / np.sqrt(D * (SMAX2 + 1) / (D + K))        # block-1 LN scale bounds (energy E there)
khi = 1 / np.sqrt(D * (SMIN2 + 1) / (D + K))
DELTA = 2 * (m - 1) * (khi * GV * FS) ** 2                # copies only (flips are energy-neutral)
k2lo = 1 / np.sqrt((D * (SMAX2 + 1) + DELTA) / (D + K))
lam_target = klo / (4 * LCTX * khi)                  # L*lam*khi <= klo/4
c = -np.log(lam_target) * np.sqrt(nb + 1) / (2 * klo ** 2 * FS ** 2)
lam = np.exp(-2 * c * klo ** 2 * FS ** 2 / np.sqrt(nb + 1))
alo = klo / (1 + LCTX * lam)                          # copy lower bound / GV when bit = 1
ahi0 = LCTX * lam * khi                               # copy upper bound / GV when bit = 0
tau = (alo + ahi0) / 2
Wa = 1.01 / (k2lo * (alo - tau) * GV * FS)
Wb = 1.01 * 2 / (k2lo * FS)
# flag and readout (as Refusal, with the extra energy)
k3lo = k2lo
flo = k3lo * FS / LCTX
dW = HOST["wte"] - HOST["wte"][REFUSE]
G = float((np.sqrt(D) * np.linalg.norm(dW * HOST["gf"], axis=1) + dW @ HOST["bf"]).max())
fhi = FS / np.sqrt(D * (SMIN2 + 1) / (D + K))
Gam = np.sqrt((D * (SMAX2 + EPS) + DELTA + 2 * fhi ** 2) / (D + K))
Lg = 1.01 * G * Gam / flo
PAR = dict(c=c, tau=tau, Wa=Wa, Wb=Wb, Lg=Lg)
wnorm = np.sqrt(D) * np.linalg.norm(HOST["wte"] * HOST["gf"], axis=1)
eta = wnorm.max() * DELTA / (2 * D * (SMIN2 + EPS))
print(f"host var in [{min(vs):.2f}, {max(vs):.1f}] -> sigma2 range [{SMIN2:.2f}, {SMAX2:.1f}]")
print(f"kappa in [{klo:.4f}, {khi:.4f}]; c = {c:.3e}; leak lam = {lam:.2e}; tau = {tau:.4f}; Wa = {Wa:.3e}; Wb = {Wb:.1f}; Lg = {Lg:.3e}")
print(f"g_v = {GV}: Delta <= {DELTA:.3e}; eta (A') bound = {eta:.3e}")
# hypotheses of pattern_main (lean/VWC/Graft/PatternMain.lean), checked numerically
nbits_ok = LCTX <= 2 ** nb
h1 = LCTX * lam * khi <= tau
h3 = Wa * (GV * FS * (klo / (1 + LCTX * lam) - tau)) * k2lo >= 1
h4 = Wb * FS / 2 * k2lo >= 1
h5 = G * Gam * LCTX < Lg * (FS * k2lo)
print(f"Good: L <= 2^n {nbits_ok}, h1 {h1}, h3 {h3}, h4 {h4}; readout h5 {h5}; G >= max Host.gap by construction")

GR = build(PAR)
for dt in (np.float64, np.float32):
    name = np.dtype(dt).name
    err, chg, npos = 0.0, 0, 0
    for p in off_prompts:
        ids = enc(p)
        h, g = forward(HOST, ids, dt), forward(GR, ids, dt)
        err = max(err, float(np.abs(g - h).max())); chg += int((g.argmax(-1) != h.argmax(-1)).sum()); npos += len(ids)
    base = max(float(np.abs(forward(HOST, enc(p), dt) - forward(HOST, enc(p))).max()) for p in off_prompts)
    print(f"(A') {name}: off-trigger max |dlogit| = {err:.2e} (bound {eta:.2e}, real arithmetic); argmax changes {chg}/{npos}"
          f"; host alone {name} vs float64: {base:.2e}")
    for p in trig_prompts:
        ids = enc(p)
        g = forward(GR, ids, dt)
        end = [t for t in range(m - 1, len(ids)) if ids[t - m + 1:t + 1] == PATTERN][0]
        after = g[end:].argmax(-1)
        rr = []
        forward(HOST, ids, resid=rr)
        hyp = bool(rr[0].var(1).max() <= SMAX2)
        print(f"(B) {name}: {(p if isinstance(p, str) else tok.decode(p))[:44]!r}: refusal at {int((after == REFUSE).sum())}/{len(after)} positions from the match end (var <= sigma2: {hyp})")
