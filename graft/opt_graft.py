"""Phase 3.4: graft the 3.1 pattern detector (lean/VWC/Graft/Pattern*.lean, `pattern_main`) into an
OPT host on GPU and evaluate it. Phase 3.5 adds the watermark circuit, a naive-graft baseline and
the boundary experiments (casts, weight quantization, pruning, fine-tuning).

OPT >= 125m (except 350m) is exactly the semantics of `VWC.Graft.GArch`: pre-LN, LayerNorm, ReLU MLP,
learned absolute positions (row t+2 for position t), biases on q/k/v/out, scores / sqrt(d_head),
tied unembedding. The graft follows `graft M C`:
  * host blocks: input matrices get k zero input columns, output matrices/biases get k mean rows,
    LayerNorm gamma * sqrt(D/(D+k)), eps' = (D eps - E)/(D+k);
  * embeddings: [row, mean(row) + circuit feature]; position features sit on the OPT position rows;
  * circuit blocks (3, as in `PatternCirc.lean`) read/write only the k circuit dims; they are run as
    LayerNorm over all D+k dims with gamma = 0 on host dims, then k-dim matrices (= liftCols/liftRows);
  * final LayerNorm compensated; head = [W_e | W_c], W_c = Lg/2 (e_flag0 - e_flag1) on the refusal row.
Parameters satisfy the hypotheses of `pattern_main` (`Good` h1-h4, readout h5), computed from
sigma2 = 2 x the largest host final-residual variance seen on calibration + evaluation prompts.
--circuit watermark grafts `wcirc` of Watermark.lean instead: m mirrored pairs delta*(c_j, -c_j) in
every token feature, no blocks, no readout.

Everything runs in float32 with TF32 disabled: the circuit's copy values are ~1e-4 next to the
mean-tracked residual; bf16/fp16/TF32 mantissas cannot carry them.

Evaluations (results to graft/results/opt_<size>_<dtype>[_wm][_<variant>][_ft<mode>].json):
  fidelity   own forward vs HF forward
  off        max |grafted - host logit|, argmax agreement, on WikiText-2 windows (no pattern) and
             near-miss prompts (pattern with one token changed, prefix, suffix)
  ppl        WikiText-2 test perplexity, host vs grafted (sliding windows)
  trig       pattern inserted into WikiText windows: refusal at every position from the match end;
             greedy generation of 8 tokens -> all refusal tokens
  hyps       Good h1-h4, h5, var <= sigma2 on every evaluated prompt
  wm         watermark: cosine of the mirrored reads of the final residual (W2) and of every
             token-embedding row (W1) with the code; detected = cosine >= 1/2

Variants (--variant), applied to the grafted weights (and, as a reference, to the host):
  graft      the construction above
  naive      zero padding: host LayerNorms unchanged, zero output rows, no mean tracking
  bf16/fp16  all weights cast, compute in that dtype
  tf32       float32 weights, TF32 matmuls
  int8/int4  per-output-row absmax round-to-nearest of every linear matrix (embeddings, head, LN,
             biases kept)
  prune<s>   per-matrix unstructured magnitude pruning of every linear matrix at sparsity s
--finetune 0,1,10,...: AdamW on WikiText-2 train, evaluated at those steps. --ft_mode full trains
every weight; host trains only the host entries (circuit blocks, circuit columns/rows, zero columns
and mean rows frozen).

Usage: uv run python opt_graft.py facebook/opt-1.3b [--device cuda] [--windows 40] [--win 512]
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--windows", type=int, default=40, help="WikiText windows for ppl/off-trigger")
ap.add_argument("--win", type=int, default=512, help="tokens per window")
ap.add_argument("--ntrig", type=int, default=40, help="trigger prompts")
ap.add_argument("--gv", type=float, default=None, help="copy value gain (default 0.3; pattern_v1: 1e-3)")
ap.add_argument("--flag_gain", type=float, default=100.0, help="gain A of the flag write (pattern)")
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--pattern", default=" open the pod bay doors")
ap.add_argument("--refuse", default=" No")
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--circuit", default="pattern", choices=["pattern", "pattern_v1", "watermark"])
ap.add_argument("--wm_m", type=int, default=256, help="watermark pairs m")
ap.add_argument("--variant", default="graft")
ap.add_argument("--finetune", default="", help="comma-separated evaluation steps; enables fine-tuning")
ap.add_argument("--ft_mode", default="full", choices=["full", "host"])
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--ft_batch", type=int, default=8)
ap.add_argument("--save_host", default="", help="after fine-tuning, save the host held in the graft (host_of) as an HF checkpoint here")
ap.add_argument("--tag", default="", help="suffix for the results file name")
ap.add_argument("--extra_prompts", default="", help="var_probe result files: add each set's lowest-variance prompt to the windows")
ap.add_argument("--embed_offset", type=float, default=0.0,
                help="what-if: add this constant to every residual coordinate after the embedding (host and circuit "
                     "dims alike). Every LayerNorm subtracts the mean, so over the reals nothing changes; in floating "
                     "point a large common offset rounds the circuit features away")
ap.add_argument("--adv_trig", default="",
                help="var_probe --adv_max result: evaluate those trigger prompts (not used to calibrate sigma^2) and "
                     "report their max host variance against sigma^2 and the refusal at every position from the match end")
ap.add_argument("--log_mu", action="store_true",
                help="record the largest |mean| of any LayerNorm input during evaluation (float32 rounding of the "
                     "mean-tracked circuit coordinates is about |mean| * 2^-24)")
ap.add_argument("--dump_circuit", default="",
                help="save the circuit as built (features, three blocks, readout, parameters) to this .npz and exit; "
                     "checked against PatternCirc.lean by scripts/check_lean_transcription.py")
ap.add_argument("--resid_scale", type=float, default=1.0,
                help="what-if: scale the host's final residual by s at evaluation (parameters from the unscaled host)")
args = ap.parse_args()
VAR, PATTERN, V2 = args.variant, args.circuit in ("pattern", "pattern_v1"), args.circuit == "pattern"
# pattern: the circuit of PatternCirc.lean (PHASE3_DESIGN.md §14-15), low gain, half-precision deployable:
#  * off-trigger bound position-wise (graft_offtrigger_approx_at): eta_i = |w|*Dc/(2D(u_i+eps)), u_i the
#    host's variance at i, so the copy gain GV can be 0.3;
#  * the flag head (block 3, head 0, dq = n+1) attends to the matched positions with score scale
#    fq*fk = c3, leakage lam3 = L exp(-c3 (delta kappa2lo)^2/sqrt(n+1)); value A * match read;
#  * score scales split: cq = ck = sqrt(cs), fq = fk = sqrt(c3);
#  * AND on the complement coordinates (zrow), threshold mid-gap.
# pattern_v1: the superseded high-gain circuit (uniform flag head, gv = 1e-3), kept for the Phase 3.4 and
# boundary/tamper results of PHASE3_DESIGN.md §10-13.
if args.gv is None:
    args.gv = 0.3 if V2 else 1e-3
FT = sorted(int(s) for s in args.finetune.split(",")) if args.finetune else []
QV = re.fullmatch(r"int(\d+)(?:g(\d+))?(host|circ)?", VAR)       # int<bits>[g<group>][host|circ]
assert VAR in ("graft", "naive", "bf16", "fp16", "tf32") or QV or VAR.startswith("prune")


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on
    torch.set_float32_matmul_precision("high" if on else "highest")


set_tf32(False)
dev, dt = torch.device(args.device), getattr(torch, args.dtype)
t_start = time.time()
tok = AutoTokenizer.from_pretrained(args.model)
hf = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
cfg = hf.config
assert cfg.do_layer_norm_before and cfg.word_embed_proj_dim == cfg.hidden_size and cfg.activation_function == "relu"
D, H, NL = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers
DH, EPS, LCTX, V = D // H, 1e-5, cfg.max_position_embeddings, cfg.vocab_size
dec = hf.model.decoder
log = {"model": args.model, "dtype": args.dtype, "D": D, "layers": NL, "heads": H, "L": LCTX, "V": V, "device": str(dev),
       "circuit": args.circuit, "variant": VAR}
print(f"[{args.model}] D={D} layers={NL} heads={H} L={LCTX} V={V} device={dev} circuit={args.circuit} variant={VAR}", flush=True)


def T(x):
    return x.detach().to(dev, dt)


DEBUG_NAN = bool(os.environ.get("DEBUG_NAN"))
RS = [1.0]                                       # residual scale for the --resid_scale what-if (set after the host pass)
OFF = [0.0]                                      # common offset for the --embed_offset what-if (set after the host pass)
MU = [None]                                      # running max |LayerNorm input mean| when --log_mu is on
HALF = (torch.float16, torch.bfloat16)
f32 = lambda x: x.float() if x.dtype in HALF else x     # half outputs to float32; float32/64 unchanged


def ln(x, g, b, eps):
    """LayerNorm; half-precision inputs are normalised in float32 (as torch's LayerNorm does)."""
    xs = x.float() if x.dtype in HALF else x
    mu = xs.mean(-1, keepdim=True)
    if MU[0] is not None:
        m = mu.detach().abs().max()
        MU[0] = m if MU[0].numel() == 0 else torch.maximum(MU[0], m)
    var = ((xs - mu) ** 2).mean(-1, keepdim=True)
    return ((xs - mu) / torch.sqrt(var + eps)).to(x.dtype) * g + b


def mha(x, Wq, bq, Wk, bk, Wv, bv, Wo, bo, nh):
    """Causal multi-head attention; matrices in [out, in] layout."""
    Tn = x.shape[0]
    q, k_, v = x @ Wq.T + bq, x @ Wk.T + bk, x @ Wv.T + bv
    dq, dv = q.shape[1] // nh, v.shape[1] // nh
    q = q.view(Tn, nh, dq).transpose(0, 1)
    k_ = k_.view(Tn, nh, dq).transpose(0, 1)
    v = v.view(Tn, nh, dv).transpose(0, 1)
    if q.dtype in HALF:                          # scale q first, as HF OPT does: raw q.k can overflow fp16
        s = (q / math.sqrt(dq)) @ k_.transpose(1, 2)
    else:
        s = q @ k_.transpose(1, 2) / math.sqrt(dq)
    s = s.masked_fill(torch.triu(torch.ones(Tn, Tn, dtype=torch.bool, device=x.device), 1), float("-inf"))
    w = torch.softmax(s.float(), -1).to(s.dtype) if s.dtype in HALF else torch.softmax(s, -1)
    return (w @ v).transpose(0, 1).reshape(Tn, -1) @ Wo.T + bo


# ---------------- host weights ----------------
def host_weights():
    blocks = []
    for L_ in dec.layers:
        a = L_.self_attn
        blocks.append(dict(g1=T(L_.self_attn_layer_norm.weight), b1=T(L_.self_attn_layer_norm.bias),
                           Wq=T(a.q_proj.weight), bq=T(a.q_proj.bias), Wk=T(a.k_proj.weight), bk=T(a.k_proj.bias),
                           Wv=T(a.v_proj.weight), bv=T(a.v_proj.bias), Wo=T(a.out_proj.weight), bo=T(a.out_proj.bias),
                           g2=T(L_.final_layer_norm.weight), b2=T(L_.final_layer_norm.bias),
                           W1=T(L_.fc1.weight), c1=T(L_.fc1.bias), W2=T(L_.fc2.weight), c2=T(L_.fc2.bias)))
    wte = T(dec.embed_tokens.weight)
    tied = hf.lm_head.weight.data_ptr() == dec.embed_tokens.weight.data_ptr()
    return dict(wte=wte, wpe=T(dec.embed_positions.weight), blocks=blocks,
                gf=T(dec.final_layer_norm.weight), bf=T(dec.final_layer_norm.bias), eps=EPS, circ=[],
                Wh=wte if tied else T(hf.lm_head.weight), K=0)


def forward_g(P, ids):
    """Logits and the full residual stream before the final LayerNorm (differentiable)."""
    ids_t = torch.tensor(ids, device=dev)
    z = P["wte"][ids_t] + P["wpe"][torch.arange(len(ids), device=dev) + 2]
    if OFF[0]:                                  # what-if: common offset on every coordinate (keeps mean + feature)
        z = z + OFF[0]
    for b in P["blocks"]:
        z = z + mha(ln(z, b["g1"], b["b1"], P["eps"]), b["Wq"], b["bq"], b["Wk"], b["bk"], b["Wv"], b["bv"],
                    b["Wo"], b["bo"], H)
        z = z + torch.relu(ln(z, b["g2"], b["b2"], P["eps"]) @ b["W1"].T + b["c1"]) @ b["W2"].T + b["c2"]
    Dz = z.shape[1]
    if RS[0] != 1.0:                            # what-if: a host whose final residual is s * x (var s^2 var)
        s_, zh = RS[0], z[:, :D]
        if Dz > D:                              # circuit dims hold mean(host) + feature: keep that invariant
            z = torch.cat([s_ * zh, z[:, D:] - (1 - s_) * zh.mean(-1, keepdim=True)], 1)
        else:
            z = s_ * zh
    if DEBUG_NAN and not torch.isfinite(z).all():
        print(f"[nan] after host blocks: {int((~torch.isfinite(z)).any(-1).sum())} positions, host dims bad "
              f"{int((~torch.isfinite(z[:, :D])).any(-1).sum())}, max |z| {float(z[torch.isfinite(z)].abs().max())}", flush=True)
    for ci, cb in enumerate(P["circ"]):        # circuit blocks: LN over D+k, gamma 0 on host dims
        if cb.get("attn") is not None:
            u = ln(z, cb["g"], 0.0, cb["eps"])[:, D:]
            Wq, Wk, Wv, Wo, nh = cb["attn"]
            z = torch.cat([z[:, :D], z[:, D:] + mha(u, Wq, 0, Wk, 0, Wv, 0, Wo, 0, nh)], 1)
        if cb.get("mlp") is not None:
            u = ln(z, cb["g"], 0.0, cb["eps"])[:, D:]
            W1, c1, W2 = cb["mlp"]
            z = torch.cat([z[:, :D], z[:, D:] + torch.relu(u @ W1.T + c1) @ W2.T], 1)
        if DEBUG_NAN and not torch.isfinite(z[:, D:]).all():
            print(f"[nan] circuit block {ci}: {int((~torch.isfinite(z[:, D:])).any(-1).sum())} positions", flush=True)
    assert z.shape[1] == Dz
    return ln(z, P["gf"], P["bf"], P["eps"]) @ P["Wh"].T, z


@torch.no_grad()
def forward(P, ids, want=None):
    """want: None -> logits; "resid" -> (logits, host dims of the final residual); "full" -> (logits, residual)."""
    out, z = forward_g(P, ids)
    if want is None:
        return out
    return (out, z[:, :D]) if want == "resid" else (out, z)


# ---------------- variants ----------------
def tmap(x, fn):
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, dict):
        return {k_: tmap(v, fn) for k_, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(tmap(v, fn) for v in x)
    return x


def apply_lin(P, fn, host=True, circ=True):
    """Replace every linear matrix of the host blocks (host) and circuit blocks (circ) by fn(matrix)."""
    for b in (P["blocks"] if host else []):
        for k_ in ("Wq", "Wk", "Wv", "Wo", "W1", "W2"):
            b[k_] = fn(b[k_])
    for cb in (P["circ"] if circ else []):
        if cb.get("attn") is not None:
            a = cb["attn"]
            cb["attn"] = tuple(fn(x) for x in a[:4]) + (a[4],)
        if cb.get("mlp") is not None:
            W1, c1, W2 = cb["mlp"]
            cb["mlp"] = (fn(W1), c1, fn(W2))


def quant(bits, group=0):
    """Symmetric round-to-nearest, absmax scale per output row (group=0) or per group of input columns."""
    qmax = 2 ** (bits - 1) - 1

    def q(Wg):
        s = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-30) / qmax
        return torch.round(Wg / s).clamp(-qmax, qmax) * s

    def f(W):
        if not group:
            return q(W)
        n = W.shape[1] // group * group          # full groups; the remaining columns form one last group
        parts = [q(W[:, :n].reshape(W.shape[0], -1, group)).reshape(W.shape[0], n)] if n else []
        return torch.cat(parts + ([q(W[:, n:])] if n < W.shape[1] else []), 1)
    return f


def prune(sparsity):
    def f(W):
        kk = int(sparsity * W.numel())
        if kk == 0:
            return W
        thr = W.abs().flatten().kthvalue(kk).values
        return torch.where(W.abs() <= thr, torch.zeros_like(W), W)
    return f


def perturb(P):
    """P under the variant (graft/naive/tf32: unchanged). Casts return new tensors; others modify P."""
    if VAR in ("bf16", "fp16"):
        return tmap(P, lambda x: x.to(torch.bfloat16 if VAR == "bf16" else torch.float16))
    if QV:
        apply_lin(P, quant(int(QV[1]), int(QV[2] or 0)), host=QV[3] != "circ", circ=QV[3] != "host")
    elif VAR.startswith("prune"):
        apply_lin(P, prune(float(VAR[5:])))
    return P


# ---------------- data ----------------
from datasets import load_dataset  # noqa: E402

wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(x for x in wt["text"])
all_ids = tok(text, add_special_tokens=False)["input_ids"]
BOS = tok.bos_token_id
PAT = tok(args.pattern, add_special_tokens=False)["input_ids"]
R = tok(args.refuse, add_special_tokens=False)["input_ids"][0]
m = len(PAT)
rng = np.random.default_rng(0)
W = args.win
starts = rng.choice(len(all_ids) - W, args.windows, replace=False)
windows = [[BOS] + all_ids[s:s + W - 1] for s in starts]


def contains(ids):
    return any(ids[t - m + 1:t + 1] == PAT for t in range(m - 1, len(ids)))


if args.extra_prompts:                           # e.g. adversarial low-variance prompts from var_probe.py
    for fn in args.extra_prompts.split(","):
        sets = json.load(open(fn))["sets"]
        windows += [[int(t) for t in r["argmin_ids"]] for r in sets.values()]
windows = [w for w in windows if not contains(w)]
near = [PAT[:i] + [(PAT[i] + 7) % V] + PAT[i + 1:] for i in range(m)] + [PAT[:-1], PAT[1:]]
near_prompts = [[BOS] + all_ids[s:s + 40] + x + all_ids[s + 40:s + 80] for s, x in zip(starts, near)]
near_prompts = [p for p in near_prompts if not contains(p)]
trig_prompts = []
for s in rng.choice(len(all_ids) - 200, args.ntrig, replace=False):
    pos = int(rng.integers(5, 150))
    trig_prompts.append([BOS] + all_ids[s:s + pos] + PAT + all_ids[s + pos:s + pos + 30])
assert all(contains(p) for p in trig_prompts)

# ---------------- host pass ----------------
HOST = host_weights()
with torch.no_grad():
    ids0 = windows[0][:128]
    ref = hf(torch.tensor([ids0])).logits[0]                  # HF reference on CPU (no 2nd GPU copy)
    fid = float((forward(HOST, ids0).cpu().float() - ref).abs().max())
print(f"fidelity: own fp32 forward vs HF: max |dlogit| = {fid:.2e}", flush=True)
log["fidelity_max_dlogit"] = fid


def nll_and_more(P, ids, keep=False):
    out, resid = forward(P, ids, want="resid")
    out = f32(out)
    lp = torch.log_softmax(out[:-1].double(), -1)
    nll = -lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:], device=dev, dtype=torch.long)].sum().item()
    top2 = out.topk(2, -1).values
    rv = f32(resid).var(1, unbiased=False)
    return (nll, out.argmax(-1).cpu(), (out.cpu() if keep else None), rv.max().item(),
            (top2[:, 0] - top2[:, 1]).cpu(), rv.min().item())


host_res, maxvar = [], 0.0
KEEP = 6                                            # windows whose full logits are kept for max |dlogit|
for i, w in enumerate(windows):
    host_res.append(nll_and_more(HOST, w, keep=i < KEEP))
    maxvar = max(maxvar, host_res[-1][3])
host_near = []
trig_var, trig_vmin = [], []
for p in near_prompts:
    out, rv = forward(HOST, p, want="resid")
    host_near.append(out.cpu()); trig_var.append(rv.var(1, unbiased=False).max().item())
    trig_vmin.append(rv.var(1, unbiased=False).min().item())
for p in trig_prompts:
    _, rv = forward(HOST, p, want="resid")
    trig_var.append(rv.var(1, unbiased=False).max().item()); trig_vmin.append(rv.var(1, unbiased=False).min().item())
maxvar = max(maxvar, max(trig_var))
minvar = min(min(x[5] for x in host_res), min(trig_vmin))
# Host.gap over all tokens (boundary weights only)
Wh, gf, bf = HOST["Wh"], HOST["gf"], HOST["bf"]
dW = Wh - Wh[R]
G = float((math.sqrt(D) * torch.linalg.norm(dW * gf, dim=1) + dW @ bf).max())
wnorm = float((math.sqrt(D) * torch.linalg.norm(Wh * gf, dim=1)).max())
del dW
host_t = time.time() - t_start
print(f"host pass done ({host_t:.0f}s): final-residual var in [{minvar:.3f}, {maxvar:.2f}]; Host.gap max G = {G:.1f}", flush=True)
log.update(host_var_min=minvar, host_var_max=maxvar)

# reference: the host under the same variant (casts, quantization, pruning, TF32)
if VAR == "tf32":
    set_tf32(True)
href = None
if VAR not in ("graft", "naive"):
    HREF = perturb(HOST if VAR in ("bf16", "fp16", "tf32") else tmap(HOST, torch.clone))
    href = [nll_and_more(HREF, w, keep=i < KEEP) for i, w in enumerate(windows)]
    del HREF
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# ---------------- circuit ----------------
FS, GV = args.delta, args.gv
if PATTERN:
    # slot layout of PatternCirc.lean
    nb = int(math.ceil(math.log2(LCTX)))
    off, cur = {}, 0
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
    quad = lambda b: FS * np.array([b, -b, 1 - b, -(1 - b)])
    sig = lambda x, b: 1.0 if (x >> b) & 1 else -1.0
    S_TOK = np.zeros((V, K))
    for i in range(m):
        S_TOK[:, off[f"tok{i}"]:off[f"tok{i}"] + 4] = quad(0.0)
        S_TOK[PAT[i], off[f"tok{i}"]:off[f"tok{i}"] + 4] = quad(1.0)
    S_POS = np.zeros((LCTX, K))
    for p in range(LCTX):
        for b in range(nb):
            S_POS[p, off["key"] + 2 * b:off["key"] + 2 * b + 2] = [FS * sig(p, b), -FS * sig(p, b)]
            for d in range(1, m):
                q = max(p - d, 0)
                S_POS[p, off[f"qry{d}"] + 2 * b:off[f"qry{d}"] + 2 * b + 2] = [FS * sig(q, b), -FS * sig(q, b)]
        S_POS[p, off["valid"]:off["valid"] + 4] = quad(1.0 if p >= m - 1 else 0.0)
        S_POS[p, off["const"]:off["const"] + 2] = [FS, -FS]
        for d in range(1, m):
            S_POS[p, off[f"cbit{d}"]:off[f"cbit{d}"] + 4] = quad(0.0)
        S_POS[p, off["match"]:off["match"] + 4] = quad(0.0)
    E = float(((S_TOK[0] + S_POS[0]) ** 2).sum())
    assert np.allclose(((S_TOK[PAT[0]] + S_POS) ** 2).sum(1), E) and np.allclose((S_TOK[0] + S_POS).sum(1), 0)

    def rd(name, j=0):
        v = np.zeros(K); v[off[name] + j] = 0.5; v[off[name] + j + 1] = -0.5; return v

    def flip(name):
        v = np.zeros(K); v[off[name]:off[name] + 4] = [FS, -FS, -FS, FS]; return v

    # parameters from the hypotheses of pattern_main (Lean): sigma2, kappa bounds (u >= 0), Delta_c, ...
    SIG2 = 2.0 * maxvar
    SLO2 = 0.0                                    # no lower variance bound: worst case u = 0 everywhere below
    AF = args.flag_gain if V2 else 1.0
    klo = 1 / math.sqrt(D * (SIG2 + 1) / (D + K))
    khi = 1 / math.sqrt(D * (SLO2 + 1) / (D + K))
    DELTA_C = 2 * (m - 1) * (GV * FS * khi) ** 2
    k2lo = 1 / math.sqrt((D * (SIG2 + 1) + DELTA_C) / (D + K))
    lam_t = klo / (4 * LCTX * khi)
    cs = -math.log(lam_t) * math.sqrt(nb + 1) / (2 * klo ** 2 * FS ** 2)
    lam = math.exp(-2 * cs * FS ** 2 * klo ** 2 / math.sqrt(nb + 1))
    alo, ahi0 = klo / (1 + LCTX * lam), LCTX * lam * khi
    tau = (alo + ahi0) / 2
    Wa = 1.01 / (k2lo * (alo - tau) * GV * FS)
    Wb = 1.01 * 2 / (k2lo * FS)
    Gam = math.sqrt((D * (SIG2 + EPS) + DELTA_C + 2 * (AF * FS * khi) ** 2) / (D + K))
    hyp = dict(L_le_2n=LCTX <= 2 ** nb, h1=LCTX * lam * khi <= tau,
               h3=Wa * (GV * FS * (klo / (1 + LCTX * lam) - tau)) * k2lo >= 1,
               h4=Wb * FS / 2 * k2lo >= 1)
    if V2:
        # Lean: lam3 = L exp(-fq fk delta^2 kappa2lo^2 / sqrt(n+1)); h6: lam3 < 1;
        #       h5: G Gamma < Lg A delta kappa2lo (1 - lam3)
        c3 = math.log(LCTX / 0.01) * math.sqrt(nb + 1) / (FS * k2lo) ** 2      # target lam3 = 0.01
        lam3 = LCTX * math.exp(-c3 * (FS * k2lo) ** 2 / math.sqrt(nb + 1))
        Lg = 1.01 * G * Gam / (AF * FS * k2lo * (1 - lam3))
        hyp.update(h5=G * Gam < Lg * (AF * FS * k2lo * (1 - lam3)), h6=lam3 < 1)
        log.update(c3=c3, lam3=lam3, flag_gain=AF, gv=GV,
                   eta_at_min_observed_var=wnorm * DELTA_C / (2 * D * (minvar + EPS)))
    else:
        Lg = 1.01 * G * Gam * LCTX / (FS * k2lo)
        hyp.update(h5=G * Gam * LCTX < Lg * (FS * k2lo))
    eta = wnorm * DELTA_C / (2 * D * (SLO2 + EPS))   # uniform bound (u = 0); see eta_at_* and bound_ratio_max
    log.update(pattern=PAT, refuse=R, m=m, n_bits=nb, k=K, E=E, sigma2=SIG2, G=G, Lg=Lg, cs=cs, Wa=Wa, Wb=Wb,
               tau=tau, lam=lam, Delta_c=DELTA_C, eta_bound=eta, hypotheses=hyp)
    print(f"circuit: m={m}, n={nb}, k={K}, E={E:.3f}; sigma2={SIG2:.1f}; Lg={Lg:.3e}; hypotheses {hyp}; eta bound {eta:.2e}", flush=True)
else:
    # wcirc of Watermark.lean: pair j = delta*(c_j, -c_j) in every token feature, zero position features
    MW = args.wm_m
    K = 2 * MW
    code = np.random.default_rng(1).choice([-1.0, 1.0], MW)
    S_TOK = np.tile(FS * np.stack([code, -code], 1).reshape(-1), (V, 1))
    S_POS = np.zeros((LCTX, K))
    E = float((S_TOK[0] ** 2).sum())
    code_t = torch.tensor(code, device=dev, dtype=torch.float64)
    log.update(m_pairs=MW, k=K, E=E, delta=FS)
    print(f"watermark: m={MW} pairs, k={K}, E={E:.3f}", flush=True)

# ---------------- grafted weights ----------------
zc = lambda *s: torch.zeros(*s, device=dev, dtype=dt)


def circuit_blocks(epsc):
    Tt = lambda a: torch.tensor(a, device=dev, dtype=dt)
    gc = torch.cat([zc(D), torch.ones(K, device=dev, dtype=dt)])
    dq = nb + 1
    Wq = np.zeros(((m - 1) * dq, K)); Wk = np.zeros(((m - 1) * dq, K)); Wv = np.zeros((m - 1, K)); Wo = np.zeros((K, m - 1))
    for h, d in enumerate(range(1, m)):
        sq, sk = (math.sqrt(cs), math.sqrt(cs)) if V2 else (cs, 1.0)
        for b in range(nb):
            Wq[h * dq + b] = sq * rd(f"qry{d}", 2 * b); Wk[h * dq + b] = sk * rd("key", 2 * b)
        Wq[h * dq + nb] = sq * rd("const"); Wk[h * dq + nb] = -(nb - 1) * sk * rd("const")
        Wv[h] = GV * rd(f"tok{m - 1 - d}")
        Wo[off[f"copy{d}"], h], Wo[off[f"copy{d}"] + 1, h] = 1, -1
    W1 = np.zeros((2 * (m - 1), K)); c1 = np.zeros(2 * (m - 1)); W2 = np.zeros((K, 2 * (m - 1)))
    for h, d in enumerate(range(1, m)):
        pre = Wa * (rd(f"copy{d}") - tau * GV * rd("const"))
        W1[2 * h] = pre; W1[2 * h + 1] = pre; c1[2 * h + 1] = -1.0
        W2[:, 2 * h] = flip(f"cbit{d}"); W2[:, 2 * h + 1] = -flip(f"cbit{d}")
    zrow = rd(f"tok{m - 1}") + rd("valid") + sum(rd(f"cbit{d}") for d in range(1, m)) - (m + 0.5) * rd("const")
    A1 = np.stack([Wb * zrow, Wb * zrow]); a1 = np.array([0.0, -1.0]); A2 = np.stack([flip("match"), -flip("match")], 1)
    if V2:
        # AND on the complement coordinates (quad entries 2,3 hold delta*(1-b)): their sum is 0 iff all m+1
        # bits are 1, else >= 1, so clamp(Wb (1/2 const - sum)) is 1 or exactly 0 (ReLU of a negative).
        # The threshold sits mid-gap, so the row's coefficients (1/4 and 1/2 of Wb) survive coarse grids.
        zrow = 0.5 * rd("const") - (rd(f"tok{m - 1}", 2) + rd("valid", 2) + sum(rd(f"cbit{d}", 2) for d in range(1, m)))
        A1 = np.stack([Wb * zrow, Wb * zrow])
    # flag head (attn3): value A * match read into the flag pair (gslot = +-1)
    Wv3 = AF * rd("match")[None]; Wo3 = np.zeros((K, 1)); Wo3[off["flag"], 0], Wo3[off["flag"] + 1, 0] = 1, -1
    if V2:                                       # q3row/k3row: row 0 of a dq = n+1 head, so scores carry 1/sqrt(n+1)
        q3, k3 = np.zeros((nb + 1, K)), np.zeros((nb + 1, K))
        q3[0], k3[0] = math.sqrt(c3) * rd("const"), math.sqrt(c3) * rd("match")
        q3, k3 = Tt(q3), Tt(k3)
    else:                                        # pattern_v1: uniform attention
        q3, k3 = zc(1, K), zc(1, K)
    return [dict(g=gc, eps=epsc, attn=(Tt(Wq), Tt(Wk), Tt(Wv), Tt(Wo), m - 1), mlp=(Tt(W1), Tt(c1), Tt(W2))),
            dict(g=gc, eps=epsc, mlp=(Tt(A1), Tt(a1), Tt(A2))),
            dict(g=gc, eps=epsc, attn=(q3, k3, Tt(Wv3), Tt(Wo3), 1))]


def graft_weights(naive=False):
    """The graft of the construction; naive=True: zero padding, host LayerNorms, no mean tracking."""
    sc = 1.0 if naive else math.sqrt(D / (D + K))
    epsp = EPS if naive else (D * EPS - E) / (D + K)
    epsc = 1.0 if naive else (D * 1.0 - E) / (D + K)
    padin = lambda Wm: torch.cat([Wm, zc(Wm.shape[0], K)], 1)                      # [out, in+k]
    extout = lambda Wm: torch.cat([Wm, zc(K, Wm.shape[1]) if naive else Wm.mean(0, keepdim=True).expand(K, -1)], 0)
    extb = lambda b: torch.cat([b, zc(K) if naive else b.mean().expand(K)])
    meancol = lambda Wm: zc(Wm.shape[0], 1) if naive else Wm.mean(1, keepdim=True)
    gpad = lambda g: torch.cat([g * sc, torch.ones(K, device=dev, dtype=dt)])
    bpad = lambda b: torch.cat([b, zc(K)])
    blocks = []
    while HOST["blocks"]:                       # convert block by block, freeing host copies (peak ~ 1 copy)
        b = HOST["blocks"].pop(0)
        blocks.append(dict(g1=gpad(b["g1"]), b1=bpad(b["b1"]), Wq=padin(b["Wq"]), bq=b["bq"], Wk=padin(b["Wk"]),
                           bk=b["bk"], Wv=padin(b["Wv"]), bv=b["bv"], Wo=extout(b["Wo"]), bo=extb(b["bo"]),
                           g2=gpad(b["g2"]), b2=bpad(b["b2"]), W1=padin(b["W1"]), c1=b["c1"],
                           W2=extout(b["W2"]), c2=extb(b["c2"])))
        del b
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    wte, wpe = HOST["wte"], HOST["wpe"]
    stok = torch.tensor(S_TOK[:wte.shape[0]], device=dev, dtype=dt)
    spos = torch.zeros(wpe.shape[0], K, device=dev, dtype=dt)
    spos[2:2 + LCTX] = torch.tensor(S_POS, device=dev, dtype=dt)
    spos[:2] = torch.tensor(S_POS[0], device=dev, dtype=dt)   # unused rows 0,1: any constant-energy feature
    Whg = torch.cat([HOST["Wh"], zc(HOST["Wh"].shape[0], K)], 1)   # host head (tied or not)
    if PATTERN:
        Whg[R, D + off["flag"]] += Lg / 2
        Whg[R, D + off["flag"] + 1] -= Lg / 2
    return dict(wte=torch.cat([wte, meancol(wte) + stok], 1),
                wpe=torch.cat([wpe, meancol(wpe) + spos], 1),
                blocks=blocks, gf=gpad(HOST["gf"]), bf=bpad(HOST["bf"]), eps=epsp,
                circ=circuit_blocks(epsc) if PATTERN else [], Wh=Whg, K=K)


# host weights are padded copies; free the originals' extra memory by building then dropping HOST blocks
GR = perturb(graft_weights(naive=VAR == "naive"))
if args.dump_circuit:
    assert PATTERN and V2 and VAR == "graft"
    npy = lambda t: t.detach().double().cpu().numpy()
    C = GR["circ"]
    Wq1, Wk1, Wv1, Wo1, nh1 = C[0]["attn"]; W1, c1, W2 = C[0]["mlp"]
    A1, a1, A2 = C[1]["mlp"]; Wq3, Wk3, Wv3, Wo3, nh3 = C[2]["attn"]
    np.savez_compressed(args.dump_circuit, S_TOK=S_TOK, S_POS=S_POS, pat=np.array(PAT), refuse=R,
                        off=json.dumps(off), m=m, nb=nb, K=K, D=D, L=LCTX, E=E, nh1=nh1, nh3=nh3,
                        par=json.dumps(dict(delta=FS, gv=GV, cs=cs, tau=tau, Wa=Wa, Wb=Wb, c3=c3, A=AF, Lg=Lg)),
                        Wq1=npy(Wq1), Wk1=npy(Wk1), Wv1=npy(Wv1), Wo1=npy(Wo1), W1=npy(W1), c1=npy(c1), W2=npy(W2),
                        A1=npy(A1), a1=npy(a1), A2=npy(A2), Wq3=npy(Wq3), Wk3=npy(Wk3), Wv3=npy(Wv3), Wo3=npy(Wo3),
                        g=np.stack([npy(cb["g"]) for cb in C]), eps=np.array([cb["eps"] for cb in C]),
                        readout=npy(GR["Wh"][:, D:]), stok_graft=npy(GR["wte"][:, D:] - GR["wte"][:, :D].mean(1, keepdim=True)),
                        spos_graft=npy(GR["wpe"][2:2 + LCTX, D:] - GR["wpe"][2:2 + LCTX, :D].mean(1, keepdim=True)))
    print(f"circuit dumped to {args.dump_circuit}", flush=True)
    raise SystemExit(0)
if dev.type == "cuda":
    torch.cuda.empty_cache()


def struct_check(P):
    """Distance from the construction: host input matrices' circuit columns (0 in the graft) and the
    output matrices' circuit rows minus their host-row mean (0 in the graft)."""
    zcol = mrow = 0.0
    for b in P["blocks"]:
        for k_ in ("Wq", "Wk", "Wv", "W1"):
            zcol = max(zcol, float(b[k_][:, D:].float().abs().max()))
        for k_ in ("Wo", "W2"):
            Wm = b[k_].float()
            mrow = max(mrow, float((Wm[D:] - Wm[:D].mean(0, keepdim=True)).abs().max()))
    return dict(zero_cols_max=zcol, mean_rows_max_dev=mrow)


def host_of(P):
    """The host held in a (possibly modified) graft: its host entries, LayerNorm compensation undone.
    For the unmodified graft this is the original host, so graft-vs-host_of is what (A)/(A') bound."""
    sc = 1.0 if VAR == "naive" else math.sqrt(D / (D + K))
    hb = [dict(g1=b["g1"][:D] / sc, b1=b["b1"][:D], Wq=b["Wq"][:, :D], bq=b["bq"], Wk=b["Wk"][:, :D], bk=b["bk"],
               Wv=b["Wv"][:, :D], bv=b["bv"], Wo=b["Wo"][:D], bo=b["bo"][:D], g2=b["g2"][:D] / sc, b2=b["b2"][:D],
               W1=b["W1"][:, :D], c1=b["c1"], W2=b["W2"][:D], c2=b["c2"][:D]) for b in P["blocks"]]
    return dict(wte=P["wte"][:, :D], wpe=P["wpe"][:, :D], blocks=hb, gf=P["gf"][:D] / sc, bf=P["bf"][:D], eps=EPS,
                circ=[], Wh=P["Wh"][:, :D], K=0)


def wm_cos(Rd):
    """Cosine of mirrored reads (rows of Rd: [n, 2m] circuit coordinates) with the code."""
    Rm = (Rd[:, 0::2] - Rd[:, 1::2]).double() / 2
    return (Rm @ code_t) / (math.sqrt(MW) * torch.linalg.norm(Rm, dim=1).clamp_min(1e-300))


# ---------------- grafted pass ----------------
def mx(a, b):
    """max that propagates NaN (Python's max silently drops it)."""
    return math.nan if math.isnan(a) or math.isnan(b) else max(a, b)


@torch.no_grad()
def evaluate(P, ntrig=None, gen=True):
    r = {}
    xp = lambda x: math.inf if x > 700 else math.exp(x)          # a broken model can have astronomic NLL
    nll_h = nll_g = nll_r = 0.0; ntok = npos = agree = agree_r = 0; maxd = maxd_r = 0.0; flips = []
    fr_g = fr_h = fr_r = 0; cos_min = math.inf; det = 0; nonfin = 0
    HO, own_d, own_ag, nll_o, bratio = host_of(P), 0.0, 0, 0.0, 0.0
    for i, w in enumerate(windows):
        out, z = forward(P, w, want="full")
        out = f32(out)
        oh, zo = forward(HO, w, want="full"); oh = f32(oh)
        if PATTERN:                              # |graft - host| at t against eta(u_t), u_t = host variance at t
            ut = zo.double().var(-1, unbiased=False)
            eta_t = wnorm * DELTA_C / (2 * D * (ut + EPS))          # Host.etaAt Δc u_t
            bratio = max(bratio, float(((out - oh).abs().max(-1).values.double() / eta_t).max()))
        nll_o -= torch.log_softmax(oh[:-1].double(), -1)[torch.arange(len(w) - 1), torch.tensor(w[1:], device=dev, dtype=torch.long)].sum().item()
        nonfin += int((~torch.isfinite(out)).any(-1).sum())
        own_d = mx(own_d, float((out - oh).abs().max())); own_ag += int((out.argmax(-1) == oh.argmax(-1)).sum())
        lp = torch.log_softmax(out[:-1].double(), -1)
        nll = -lp[torch.arange(len(w) - 1), torch.tensor(w[1:], device=dev, dtype=torch.long)].sum().item()
        am = out.argmax(-1).cpu()
        nll_g += nll; nll_h += host_res[i][0]; ntok += len(w) - 1; npos += len(w)
        agree += int((am == host_res[i][1]).sum())
        for t in torch.nonzero(am != host_res[i][1]).flatten().tolist():
            flips.append(dict(window=i, pos=t, host_top2_margin=float(host_res[i][4][t])))
        fr_g += int(((am == R) & (host_res[i][1] != R)).sum()); fr_h += int((host_res[i][1] == R).sum())
        if i < KEEP:
            maxd = mx(maxd, float((out.cpu() - host_res[i][2]).abs().max()))
        if href is not None:
            nll_r += href[i][0]; agree_r += int((am == href[i][1]).sum())
            fr_r += int(((am == R) & (href[i][1] != R)).sum())
            if i < KEEP:
                maxd_r = mx(maxd_r, float((out.cpu() - href[i][2]).abs().max()))
        if not PATTERN:
            c = wm_cos(z[:, D:])
            cos_min = -mx(-cos_min, -float(c.min())); det += int((c >= 0.5).sum())
    ppl_h, ppl_g = xp(nll_h / ntok), xp(nll_g / ntok)
    flips_out = flips if len(flips) <= 50 else flips[:50]
    r.update(off_windows=len(windows), off_tokens=ntok, off_max_dlogit=maxd, off_argmax_agree=agree / npos,
             argmax_flips=flips_out, n_argmax_flips=len(flips), min_host_margin=float(min(x[4].min() for x in host_res)),
             own_host_max_dlogit=own_d, own_host_argmax_agree=own_ag / npos, own_host_ppl=xp(nll_o / ntok), bound_ratio_max=bratio if PATTERN else None, nonfinite_logit_positions=nonfin)
    near_d, near_ag, near_n = 0.0, 0, 0
    for p, h in zip(near_prompts, host_near):
        g = f32(forward(P, p)).cpu()
        near_d = mx(near_d, float((g - h).abs().max())); near_ag += int((g.argmax(-1) == h.argmax(-1)).sum()); near_n += len(p)
    r.update(near_prompts=len(near_prompts), near_max_dlogit=near_d, near_argmax_agree=near_ag / near_n,
             ppl_host=ppl_h, ppl_graft=ppl_g, ppl_rel_change=(ppl_g - ppl_h) / ppl_h)
    if href is not None:
        ppl_r = xp(nll_r / ntok)
        r.update(ref_ppl_host_variant=ppl_r, ref_max_dlogit_vs_host_variant=maxd_r,
                 ref_argmax_agree_vs_host_variant=agree_r / npos,
                 ref_host_variant_argmax_agree_vs_host=sum(int((href[i][1] == host_res[i][1]).sum()) for i in range(len(windows))) / npos)
    if PATTERN:
        tp = trig_prompts[:ntrig] if ntrig else trig_prompts
        refuse_pos = refuse_tot = gen_ok = 0
        dg = {"cbit_min": [], "match": [], "flag": []}             # circuit state at the match end, in units of delta
        for p in tp:
            out, z = forward(P, p, want="full")
            end = [t for t in range(m - 1, len(p)) if p[t - m + 1:t + 1] == PAT][0]
            refuse_pos += int((out[end:].argmax(-1) == R).sum()); refuse_tot += len(p) - end
            zc_ = z[end, D:].double()
            rdz = lambda name: float(zc_[off[name]] - zc_[off[name] + 1]) / (2 * FS)
            dg["cbit_min"].append(min(rdz(f"cbit{d}") for d in range(1, m))); dg["match"].append(rdz("match")); dg["flag"].append(rdz("flag"))
            if gen:
                seq = list(p)
                for _ in range(8):
                    seq.append(int(forward(P, seq)[-1].argmax()))
                gen_ok += int(all(t == R for t in seq[len(p):]))
        r.update(diag_trig_cbit_min=min(dg["cbit_min"]), diag_trig_match_mean=float(np.mean(dg["match"])),
                 diag_trig_flag_mean=float(np.mean(dg["flag"])))
        r.update(trig_prompts=len(tp), refusal_positions=f"{refuse_pos}/{refuse_tot}", refusal_rate=refuse_pos / refuse_tot,
                 false_refusal_positions=f"{fr_g}/{npos}", host_refusal_token_positions=fr_h,
                 false_refusal_vs_variant_host=(f"{fr_r}/{npos}" if href is not None else None))
        if gen:
            r["generation_all_refusal"] = f"{gen_ok}/{len(tp)}"
    else:
        cw = wm_cos(P["wte"][:, D:])
        r.update(wm_W2_min_cos=cos_min, wm_W2_detected=det / npos,
                 wm_W1_min_cos=float(cw.min()), wm_W1_detected=int((cw >= 0.5).sum()) / len(cw))
    r.update(struct_check(P))
    return r


if args.resid_scale != 1.0:
    RS[0] = args.resid_scale
    log.update(resid_scale=RS[0], host_var_min_scaled=RS[0] ** 2 * minvar,
               # Host.etaAt Δc u at the smallest scaled variance (graft_offtrigger_approx_at)
               eta_at_scaled_var=(wnorm * DELTA_C / (2 * D * (RS[0] ** 2 * minvar + EPS)) if PATTERN else None))
if args.embed_offset:
    OFF[0] = args.embed_offset
    log.update(embed_offset=OFF[0])
if args.log_mu:
    MU[0] = torch.empty(0, device=dev)
log.update(evaluate(GR))
if args.adv_trig and PATTERN:
    advr, HO = [], host_of(GR)                  # HOST's blocks were consumed by graft_weights
    with torch.no_grad():
        for rec_ in json.load(open(args.adv_trig))["prompts"]:
            p = [int(t) for t in rec_["ids"]]
            _, rv = forward(HO, p, want="resid")
            mv = rv.double().var(1, unbiased=False).max().item()
            out = forward(GR, p)
            end = [t for t in range(m - 1, len(p)) if p[t - m + 1:t + 1] == PAT][0]
            ok = int((out[end:].argmax(-1) == R).sum())
            advr.append(dict(T=rec_["T"], max_var=mv, search_max_var=rec_["max_var"], over_sigma2=mv / SIG2, refusal=f"{ok}/{len(p) - end}",
                             next_is_refusal=bool(int(out[-1].argmax()) == R)))
    log.update(adv_trig=advr, adv_trig_max_var=max(a["max_var"] for a in advr), sigma2=SIG2,
               adv_trig_all_refuse=all(a["next_is_refusal"] for a in advr))
if args.log_mu:
    mu_max = float(MU[0]) if MU[0].numel() else 0.0
    # float32 spacing near |mean| is |mean| * 2^-23; round-to-nearest error is at most half of it, compared with the
    # feature scale delta of the mirrored circuit coordinates
    log.update(max_abs_ln_mean=mu_max, fp32_rounding_over_delta=mu_max * 2.0 ** -24 / args.delta)
    MU[0] = None
if PATTERN:
    log["var_le_sigma2_all"] = all(v <= SIG2 for v in trig_var) and maxvar <= SIG2
    mw = [abs(Lg) / 2]
    tmap(GR["circ"], lambda x: mw.append(float(x.float().abs().max())) if x.is_floating_point() else None)
    log["max_abs_circuit_weight"] = max(mw)
log.update(seconds=time.time() - t_start)

# ---------------- fine-tuning ----------------
if FT:
    tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tr_ids = tok("\n\n".join(x for x in tr["text"]), add_special_tokens=False)["input_ids"]
    leaves, seen = [], set()
    tmap(GR, lambda x: (leaves.append(x), seen.add(id(x))) if id(x) not in seen else None)
    circ_ids = set()
    tmap(GR["circ"], lambda x: circ_ids.add(id(x)))
    train = [x for x in leaves if x.is_floating_point() and not (args.ft_mode == "host" and id(x) in circ_ids)]
    for x in train:
        x.requires_grad_(True)
    opt = torch.optim.AdamW(train, lr=args.lr, weight_decay=0.0)
    frng = np.random.default_rng(1)

    def mask_circuit_grads(P):
        """host mode: only host entries train (circuit columns/rows, zero columns, mean rows frozen)."""
        for b in P["blocks"]:
            for k_ in ("Wq", "Wk", "Wv", "W1"):
                b[k_].grad[:, D:] = 0
            for k_ in ("Wo", "W2", "bo", "c2", "g1", "b1", "g2", "b2"):
                b[k_].grad[D:] = 0
        for k_ in ("wte", "wpe", "Wh"):
            P[k_].grad[:, D:] = 0
        for k_ in ("gf", "bf"):
            P[k_].grad[D:] = 0

    hist, losses = [], []
    for step in range(FT[-1] + 1):
        if step in FT:
            ev = evaluate(GR, gen=False)
            ev.update(step=step, train_loss=(losses[-1] if losses else None), seconds=time.time() - t_start,
                      max_train_loss=(max(losses) if losses else None))
            hist.append(ev)
            keys = ["ppl_graft", "own_host_max_dlogit", "own_host_argmax_agree", "off_max_dlogit", "off_argmax_agree", "refusal_rate", "false_refusal_positions",
                    "wm_W2_detected", "wm_W1_detected", "zero_cols_max", "mean_rows_max_dev"]
            print(f"ft step {step}: " + ", ".join(f"{k_}={ev[k_]}" for k_ in keys if k_ in ev), flush=True)
        if step == FT[-1]:
            break
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.ft_batch):
            s = int(frng.integers(len(tr_ids) - W))
            ids = [BOS] + tr_ids[s:s + W - 1]
            out, _ = forward_g(GR, ids)
            loss = F.cross_entropy(out[:-1], torch.tensor(ids[1:], device=dev, dtype=torch.long)) / args.ft_batch
            loss.backward()
            tot += loss.item()
        if args.ft_mode == "host":
            mask_circuit_grads(GR)
        opt.step()
        losses.append(tot)
    log.update(finetune=dict(mode=args.ft_mode, lr=args.lr, batch=args.ft_batch, win=W, losses=losses, evals=hist))
    if args.save_host:                          # the fine-tuned host, for the re-graft experiment
        HO = host_of(GR)
        cp = lambda dst, src: dst.data.copy_(src.detach().contiguous().to("cpu", torch.float32))
        with torch.no_grad():
            for L_, b in zip(dec.layers, HO["blocks"]):
                a = L_.self_attn
                for mod, w, bias in ((L_.self_attn_layer_norm, "g1", "b1"), (a.q_proj, "Wq", "bq"), (a.k_proj, "Wk", "bk"),
                                     (a.v_proj, "Wv", "bv"), (a.out_proj, "Wo", "bo"), (L_.final_layer_norm, "g2", "b2"),
                                     (L_.fc1, "W1", "c1"), (L_.fc2, "W2", "c2")):
                    cp(mod.weight, b[w]); cp(mod.bias, b[bias])
            cp(dec.final_layer_norm.weight, HO["gf"]); cp(dec.final_layer_norm.bias, HO["bf"])
            cp(dec.embed_positions.weight, HO["wpe"])
            hf.config.tie_word_embeddings = False   # the graft's head and embedding were trained separately
            hf.lm_head.weight = torch.nn.Parameter(HO["Wh"].detach().contiguous().to("cpu", torch.float32))
            cp(dec.embed_tokens.weight, HO["wte"])
        hf.save_pretrained(args.save_host)
        tok.save_pretrained(args.save_host)
        print(f"saved fine-tuned host to {args.save_host}", flush=True)

log.update(gpu_peak_GB=(torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None))
print(json.dumps({k_: v for k_, v in log.items() if k_ != "finetune"}, indent=1, default=str), flush=True)
os.makedirs("results", exist_ok=True)
tag = ("" if V2 else "_v1" if PATTERN else "_wm") + ("" if VAR == "graft" else f"_{VAR}") + (f"_ft{args.ft_mode}" if FT else "") + args.tag
with open(f"results/opt_{args.model.split('/')[-1]}_{args.dtype}{tag}.json", "w") as f:
    json.dump(log, f, indent=1, default=str)
