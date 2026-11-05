"""Fidelity check for lean/VWC/Tasks/Dec2Bin.lean.

Re-implements the Lean definitions (`embed`, `digitAttn`, `bitMLP`, `blk`, `head`, `logits`) in
float64 numpy, mirroring them line by line, and compares the logits at every position of every
teacher-forced sequence (all 256 prompts + answers, 12 tokens) with the snapshot's own forward
pass. The check must report max |Δ| = 0.

Usage: uv run --project <wsp_repo> python scripts/check_dec2bin.py <wsp_repo>
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np
import torch

RUN, SNAP, TASK = "decimal-to-binary-8bit-may15-run1", "HandbuiltDec2Bin_v20_final5k", "decimal-to-binary-8bit"


def relu(v):
    return np.maximum(v, 0.0)


def embed(tok, p):
    v = np.zeros(26)
    v[tok] += 1.0
    if 11 + p < 26:
        v[11 + p] += 1.0
    return v


def dval(x):
    return sum(c * x[c] for c in range(10))


def place(j):
    return {0: 100.0, 1: 10.0, 2: 1.0}.get(j, 0.0)


def digit_attn(xs, p):
    v = np.zeros(26)
    v[23] = sum(place(j) * dval(xs[j]) for j in range(len(xs)) if j <= p and j < 3)
    return v


def bit_mlp(T, p_out, x, L=256.0):
    fc1w = np.array([[1, 0], [1, 0], [1, L], [1, L], [0, 1]], dtype=float)
    fc1b = np.array([-(T - 1), -T, -(T - 1) - L, -T - L, 0.0])
    fc2w = np.array([[-T, T, 0, 0, 0], [0, 0, -1, 1, 1], [0, 0, 1, -1, 0]], dtype=float)
    inp = np.array([x[23], x[11 + p_out] if 11 + p_out < 26 else 0.0])
    o = fc2w @ relu(fc1w @ inp + fc1b)
    v = np.zeros(26)
    v[23], v[24], v[25] = o
    return v


def blk(b, z):
    return z + bit_mlp(2.0 ** (7 - b), 3 + b, z)


def head(z):
    out = np.zeros(11)
    out[0], out[1] = z[24], z[25]
    return out


def lean_logits(ids, p):
    xs = [embed(t, i) for i, t in enumerate(ids)]
    z = xs[p] + digit_attn(xs, p)
    for b in range(8):
        z = blk(b, z)
    return head(z)


def main(repo):
    run_dir = os.path.join(repo, "runs", RUN)
    sys.path.insert(0, run_dir)
    path = os.path.join(run_dir, "interpretable_transformers_lib", f"{SNAP}.py")
    spec = importlib.util.spec_from_file_location("snap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import src.task
    task = src.task.get_task(TASK)
    model = mod.build_model(task)
    model.eval()
    seqs = np.array([task.encode(f"{x:03d}=" + f"{x:08b}") for x in range(256)])
    with torch.no_grad():
        ref = model(torch.tensor(seqs)).double().numpy()
    max_diff = 0.0
    for s in range(256):
        for p in range(seqs.shape[1]):
            max_diff = max(max_diff, float(np.abs(ref[s, p] - lean_logits(list(seqs[s]), p)).max()))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"snapshot sha256 {sha[:16]}; sequences {len(seqs)} x {seqs.shape[1]} positions; "
          f"max |logit(snapshot) - logit(Lean transcription)| = {max_diff}")
    assert max_diff == 0.0


if __name__ == "__main__":
    main(sys.argv[1])
