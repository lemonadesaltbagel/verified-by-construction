"""Phase 3.5: how small can the host's final-residual variance get? (the sigma_lo^2 <= var hypothesis
of the low-gain detector, PHASE3_DESIGN.md §13-14).

var_t = population variance over the D host dims of the input of the final LayerNorm at position t.
Probe sets (every position of every prompt counts):
  single   [tok] and [BOS, tok] for every token of the vocabulary
  repeat   [BOS] + [tok] * k, k in {16, 128, 512}, sampled tokens
  weird    hand-written strings (scripts, emoji, zero-width/control characters, markup, digits,
           whitespace, random bytes), with and without BOS
  random   uniform random token sequences, random lengths, with and without BOS
  wiki     WikiText-2 test windows (the evaluation distribution)
  adv      GCG-style search that minimises min_t var_t: gradient of the variance w.r.t. one-hot
           tokens proposes swaps, the best of a batch of candidates is kept
Results to results/varprobe_<model>.json.

Usage: uv run python var_probe.py facebook/opt-1.3b [--adv_iters 150] [--adv_restarts 4]
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, OPTForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--device", default="cuda")
ap.add_argument("--n_repeat", type=int, default=400)
ap.add_argument("--n_random", type=int, default=2000)
ap.add_argument("--adv_iters", type=int, default=150)
ap.add_argument("--adv_restarts", type=int, default=4)
ap.add_argument("--adv_batch", type=int, default=128)
ap.add_argument("--adv_max", action="store_true",
                help="only an adversarial search that MAXIMISES max_t var_t over prompts [BOS] + prefix + suffix, "
                     "mutating the prefix only (the on-trigger theorem needs var_t <= sigma^2 at every position)")
ap.add_argument("--adv_suffix", default=" open the pod bay doors", help="fixed suffix for --adv_max (the trigger)")
ap.add_argument("--adv_only_last", action="store_true",
                help="only the adversarial search, minimising the variance at the last position (positions >= 1)")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device(args.device)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.model)
hf = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
for p in hf.parameters():
    p.requires_grad_(False)
V, BOS, D = hf.config.vocab_size, tok.bos_token_id, hf.config.hidden_size
cap = {}
hf.model.decoder.final_layer_norm.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0]))


def var_of(batch_ids=None, embeds=None):
    """[B, T] final-residual variances for a batch of equal-length prompts."""
    if embeds is None:
        hf(input_ids=batch_ids)
    else:
        hf(inputs_embeds=embeds)
    return cap["x"].var(-1, unbiased=False)


rec = {}                                          # set -> dict(min, argmin prompt, position, n positions, quantiles)


def record(name, vs, prompts):
    """vs: list of [T] tensors (one per prompt)."""
    flat = torch.cat([v.flatten() for v in vs]).double().cpu()
    i = int(np.argmin([float(v.min()) for v in vs]))
    t = int(vs[i].argmin())
    rec[name] = dict(n_prompts=len(vs), n_positions=int(flat.numel()), min=float(flat.min()),
                     q001=float(torch.quantile(flat, 0.001)) if flat.numel() < 2 ** 24 else None,
                     median=float(flat.median()), argmin_prompt=tok.decode(prompts[i]), argmin_ids=list(map(int, prompts[i])),
                     argmin_pos=t)
    print(f"{name:8s} prompts {len(vs):6d} positions {flat.numel():8d}  min var {flat.min():.4f}  median {flat.median():.2f}"
          f"  argmin pos {t} of {tok.decode(prompts[i])[:60]!r}", flush=True)


@torch.no_grad()
def run_set(name, prompts, bs=64):
    vs = []
    by_len = {}
    for j, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(j)
    out = [None] * len(prompts)
    for L, js in by_len.items():
        for k in range(0, len(js), bs):
            chunk = js[k:k + bs]
            v = var_of(torch.tensor([prompts[j] for j in chunk], device=dev))
            for j, row in zip(chunk, v):
                out[j] = row.cpu()
    record(name, out, prompts)


rng = np.random.default_rng(0)
if args.adv_only_last or args.adv_max:
    run_set = lambda *a, **k: None                # noqa: E731
run_set("single", [[t] for t in range(V)] + [[BOS, t] for t in range(V)], bs=512)
toks = rng.choice(V, args.n_repeat, replace=False)
run_set("repeat", [[BOS] + [int(t)] * k for k in (16, 128, 512) for t in toks[:args.n_repeat // (1 if k < 512 else 4)]], bs=16)
weird = ["", " ", "\n", "\n\n\n\n", "\t\t\t", " " * 200, "." * 300, "!" * 100, "?" * 50, "0" * 400, "1234567890" * 30,
         "a" * 500, "aaaa bbbb cccc", "THE THE THE THE THE", "🙂" * 50, "🔥💯🚀" * 30, "​" * 100, "‮" * 20 + "abc",
         "\x00\x01\x02\x03" * 20, "\x1b[31mred\x1b[0m" * 10, "ﷺ" * 30, "𝕳𝖊𝖑𝖑𝖔" * 20, "ÅÄÖ åäö ß" * 20,
         "Привет мир " * 20, "你好世界" * 40, "こんにちは" * 30, "안녕하세요" * 30, "مرحبا بالعالم " * 20,
         "שלום עולם " * 20, "नमस्ते दुनिया " * 20, "<html><body><div></div></body></html>" * 15,
         "{\"a\": [1, 2, {\"b\": null}]}" * 20, "SELECT * FROM t WHERE x = 1;" * 15, "def f(x):\n    return x\n" * 20,
         "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 10, "%%%%%%%%%%" * 30, "|" * 300, "-_-_-_" * 60, "\\" * 200,
         "\r\n" * 100, "</s></s></s>" * 30, "<pad><unk>" * 30, "........................." * 10]
for k in range(60):                                # random bytes, decoded leniently
    weird.append(bytes(rng.integers(0, 256, int(rng.integers(5, 400))).tolist()).decode("utf-8", errors="replace"))
wp = []
for s in weird:
    ids = tok(s, add_special_tokens=False)["input_ids"][:1024]
    if ids:
        wp += [ids, [BOS] + ids]
wp.append([BOS])
run_set("weird", wp, bs=1)
rp = []
for k in range(args.n_random):
    L = int(rng.integers(1, 257))
    ids = rng.integers(0, V, L).tolist()
    rp.append(([BOS] + ids) if k % 2 else ids)
run_set("random", rp, bs=1)
from datasets import load_dataset  # noqa: E402
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
all_ids = tok("\n\n".join(wt["text"]), add_special_tokens=False)["input_ids"]
run_set("wiki", [[BOS] + all_ids[s:s + 511] for s in rng.choice(len(all_ids) - 512, 80, replace=False)], bs=8)

# ---------------- adversarial: minimise min_t var_t ----------------
E = hf.model.decoder.embed_tokens.weight


def adv_max(T, seed, suffix):
    """GCG-style search maximising max_t var_t over [BOS] + prefix (T tokens, mutated) + suffix (fixed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pre = torch.randint(0, V, (T,), generator=g).tolist()
    full = lambda pr: [BOS] + pr + suffix
    best = float(var_of(torch.tensor([full(pre)], device=dev))[0].max())
    for it in range(args.adv_iters):
        ids = full(pre)
        oh = torch.zeros(len(ids), V, device=dev)
        oh[torch.arange(len(ids)), torch.tensor(ids, device=dev)] = 1.0
        oh.requires_grad_(True)
        var_of(embeds=(oh @ E)[None])[0].max().backward()
        cand = oh.grad[1:T + 1].topk(32, dim=1).indices     # most variance-increasing swaps per prefix position
        with torch.no_grad():
            pos = torch.randint(0, T, (args.adv_batch,), generator=g)
            new = cand[pos, torch.randint(0, 32, (args.adv_batch,), generator=g)].cpu()
            batch = torch.tensor(pre).repeat(args.adv_batch, 1)
            batch[torch.arange(args.adv_batch), pos] = new
            fb = torch.tensor([full(b.tolist()) for b in batch], device=dev)
            vb = var_of(fb).max(1).values
            j = int(vb.argmax())
            if float(vb[j]) > best:
                best, pre = float(vb[j]), batch[j].tolist()
    return full(pre), best


if args.adv_max:
    suffix = tok(args.adv_suffix, add_special_tokens=False)["input_ids"]
    res = []
    for T in (8, 32, 128):
        for r in range(args.adv_restarts):
            ids, b = adv_max(T, 7000 * T + r, suffix)
            res.append(dict(T=T, restart=r, max_var=b, ids=ids, text=tok.decode(ids)))
            print(f"  advmax T={T} restart {r}: max var {b:.2f}  {tok.decode(ids)[:60]!r}", flush=True)
    out = dict(model=args.model, D=D, suffix=args.adv_suffix, prompts=res,
               overall_max=max(r["max_var"] for r in res), seconds=time.time() - t0)
    print(f"overall max final-residual variance (prompts ending in the suffix): {out['overall_max']:.2f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/varprobe_{args.model.split('/')[-1]}_advmax.json", "w") as f:
        json.dump(out, f, indent=1)
    raise SystemExit(0)


def adv(T, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, V, (T,), generator=g).tolist()
    v0 = var_of(torch.tensor([ids], device=dev))[0]
    best = float(v0[-1] if args.adv_only_last else v0.min())
    for it in range(args.adv_iters):
        oh = torch.zeros(T, V, device=dev)
        oh[torch.arange(T), torch.tensor(ids, device=dev)] = 1.0
        oh.requires_grad_(True)
        v = var_of(embeds=(oh @ E)[None])[0]
        (v[-1] if args.adv_only_last else v.min()).backward()   # gradient flows to the targeted position
        cand = (-oh.grad).topk(32, dim=1).indices  # [T, 32] most variance-decreasing swaps per position
        with torch.no_grad():
            pos = torch.randint(0, T, (args.adv_batch,), generator=g)
            new = cand[pos, torch.randint(0, 32, (args.adv_batch,), generator=g)].cpu()
            batch = torch.tensor(ids).repeat(args.adv_batch, 1)
            batch[torch.arange(args.adv_batch), pos] = new
            vb = var_of(batch.to(dev))
            vb = vb[:, -1] if args.adv_only_last else vb.min(1).values
            j = int(vb.argmin())
            if float(vb[j]) < best:
                best, ids = float(vb[j]), batch[j].tolist()
    return ids, best


advp, advv = [], []
for T in ((2, 8, 32, 128) if args.adv_only_last else (2, 8, 32)):
    for r in range(args.adv_restarts):
        ids, b = adv(T, 1000 * T + r)
        advp.append(ids)
        with torch.no_grad():
            vv = var_of(torch.tensor([ids], device=dev))[0].cpu()
            advv.append(vv[-1:] if args.adv_only_last else vv)
        print(f"  adv T={T} restart {r}: min var {b:.4f}  {tok.decode(ids)[:60]!r}", flush=True)
record("adv", advv, advp)

out = dict(model=args.model, D=D, sets=rec, overall_min=min(r["min"] for r in rec.values()),
           seconds=time.time() - t0)
print(f"overall min final-residual variance: {out['overall_min']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
os.makedirs("results", exist_ok=True)
with open(f"results/varprobe_{args.model.split('/')[-1]}{'_advlast' if args.adv_only_last else ''}.json", "w") as f:
    json.dump(out, f, indent=1)
