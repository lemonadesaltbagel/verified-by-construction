"""Quantities that a mechanism proof for sort-five-digits must bound, over all 10^5 inputs and
all 5 teacher-forced answer steps (float64, Arch semantics from the exported weights).

Usage: python scripts/analyze_sort.py weights/sort/sortModel.json
"""
import itertools
import json
import sys

import numpy as np


def ln(n, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + n["eps"]) * np.array(n["gamma"]) + np.array(n["beta"]), np.sqrt(var + n["eps"])


def main(path):
    W = json.load(open(path))["weights"]
    E = {k: np.array(v) if isinstance(v, list) and k not in ("blocks",) else v for k, v in W.items()}
    b = W["blocks"][0]
    a = b["attn"]
    Wq, Wk, Wv, Wo = (np.array(a[k]) for k in ["Wq", "Wk", "Wv", "Wo"])
    W1, b1, W2, b2 = (np.array(b[k]) for k in ["W1", "b1", "W2", "b2"])
    tok, pos, Wh = np.array(W["tokEmb"]), np.array(W["posEmb"]), np.array(W["Wh"])
    digs = np.array(list(itertools.product(range(10), repeat=5)))
    srt = np.sort(digs, axis=1)
    seqs = np.concatenate([digs, np.full((len(digs), 1), 10), srt], axis=1)  # 11 tokens
    x = tok[seqs] + pos[None, :11]
    x1, s1 = ln(b["ln1"], x)
    q, k, v = x1 @ Wq.T, x1 @ Wk.T, x1 @ Wv.T
    dh = a["dqk"]
    sc = np.einsum("bic,bjc->bij", q, k) / np.sqrt(dh)
    T = 11
    sc = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, sc)
    w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    y = x + (np.einsum("bij,bjc->bic", w, v) @ Wo.T)
    x2, s2 = ln(b["ln2"], y)
    pre = x2 @ W1.T + b1
    r = y + np.maximum(pre, 0) @ W2.T + b2
    xf, sf = ln(W["finalLn"], r)
    logits = xf @ Wh.T + np.array(W["bh"])
    out = {}
    for kk in range(5):
        p = 5 + kk
        L = logits[:, p]
        tgt = srt[:, kk]
        lt = L[np.arange(len(L)), tgt]
        Lo = L.copy(); Lo[np.arange(len(L)), tgt] = -np.inf
        margin = lt - Lo.max(1)
        inp_w = w[:, p, :5].sum(1)
        out[f"step{kk}"] = {
            "min_margin": float(margin.min()), "argmin_input": digs[margin.argmin()].tolist(),
            "ln1_std_query": [float(s1[:, p].min()), float(s1[:, p].max())],
            "ln1_std_inputs": [float(s1[:, :5].min()), float(s1[:, :5].max())],
            "attn_mass_on_inputs": [float(inp_w.min()), float(inp_w.max())],
            "ln2_std": [float(s2[:, p].min()), float(s2[:, p].max())],
            "final_std": [float(sf[:, p].min()), float(sf[:, p].max())],
            "min_abs_preact": float(np.abs(pre[:, p]).min()),
            "score_inputs": [float(sc[:, p, :5].min()), float(sc[:, p, :5].max())],
            "score_others": [float(np.where(np.isinf(sc[:, p, 5:p + 1]), np.nan, sc[:, p, 5:p + 1])[~np.isnan(sc[:, p, 5:p + 1])].min()),
                             float(sc[:, p, 5:p + 1].max())],
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
