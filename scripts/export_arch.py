"""Export a snapshot to the generic `VWC.Arch.Model` (lean/VWC/Arch.lean) and check fidelity.

1. Introspect the snapshot's module tree and map it onto Arch: blocks with optional attention
   (W_q/W_k/W_v, W_o or identity), MLP (fc1, fc2, missing bias = 0), norms (Identity or LayerNorm).
2. Re-implement Arch's semantics independently in float64 numpy and compare with the snapshot's
   own forward pass (float64) at every position of every given sequence. A mismatch means Arch
   does not describe this snapshot and the export is rejected.
3. Write the exported weights as JSON (exact float32 values) and a generated Lean file.

Usage: uv run --project <wsp_repo> python scripts/export_arch.py <wsp_repo> <task> <run> <snapshot>
           <out_dir> <lean_name>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from survey_tasks import enumerate_inputs  # noqa: E402


def snap_path(repo: str, run: str, snap: str) -> str:
    return os.path.join(repo, "runs", run, "interpretable_transformers_lib", f"{snap}.py")


def load(repo: str, run: str, snap: str, task_name: str):
    run_dir = os.path.join(repo, "runs", run)
    sys.path.insert(0, run_dir)
    spec = importlib.util.spec_from_file_location(f"snap_{snap}", snap_path(repo, run, snap))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import src.task
    task = src.task.get_task(task_name)
    model = mod.build_model(task)
    model.eval()
    return mod, task, model


def arr(t) -> np.ndarray:
    return t.detach().double().cpu().numpy()


def norm_of(m) -> dict:
    if m is None or isinstance(m, nn.Identity):
        return {"kind": "id"}
    if isinstance(m, nn.LayerNorm):
        assert m.elementwise_affine and len(m.normalized_shape) == 1
        return {"kind": "ln", "gamma": arr(m.weight), "beta": arr(m.bias) if m.bias is not None
                else np.zeros(m.normalized_shape[0]),
                # PyTorch's float32 kernel adds eps as a float32, so export that value.
                "eps": float(np.float32(m.eps))}
    raise ValueError(f"unsupported norm {type(m).__name__}")


def extract(model) -> dict:
    D = model.d_model
    allowed = {"token_emb", "pos_emb", "blocks", "final_ln", "head"}
    extra = {n for n, _ in model.named_children()} - allowed
    assert not extra, f"unexpected top-level modules {extra}"
    blocks = []
    for b in model.blocks:
        kids = dict(b.named_children())
        attn = kids.get("attn") if getattr(b, "attn_enabled", True) else None
        ln1 = kids.get("ln1")
        ln2 = kids.get("ln2", kids.get("ln"))  # MLPOnlyBlock uses `ln` before the MLP
        unknown = set(kids) - {"attn", "ln1", "ln2", "ln", "mlp"}
        assert not unknown, f"unexpected block children {unknown}"
        blk = {"ln1": norm_of(ln1), "ln2": norm_of(ln2)}
        if attn is not None:
            H = attn.n_heads
            dqk = getattr(attn, "d_qk", None) or getattr(attn, "d_head", None) or D // H
            dv = getattr(attn, "d_v", None) or getattr(attn, "d_head", None) or D // H
            Wo = arr(attn.W_o.weight) if hasattr(attn, "W_o") else np.eye(D)
            assert Wo.shape == (D, H * dv)
            for nm in ["W_q", "W_k", "W_v"]:
                assert getattr(attn, nm).bias is None, f"{nm} has bias"
            if hasattr(attn, "W_o"):
                assert attn.W_o.bias is None
            blk["attn"] = {"H": H, "dqk": dqk, "dv": dv, "Wq": arr(attn.W_q.weight),
                           "Wk": arr(attn.W_k.weight), "Wv": arr(attn.W_v.weight), "Wo": Wo}
        else:
            blk["attn"] = None
        mlp = kids["mlp"]
        F = mlp.fc1.out_features
        blk.update({"W1": arr(mlp.fc1.weight),
                    "b1": arr(mlp.fc1.bias) if mlp.fc1.bias is not None else np.zeros(F),
                    "W2": arr(mlp.fc2.weight),
                    "b2": arr(mlp.fc2.bias) if mlp.fc2.bias is not None else np.zeros(D)})
        blocks.append(blk)
    head = model.head
    return {"D": D, "V": head.out_features, "L": model.pos_emb.num_embeddings,
            "tokEmb": arr(model.token_emb.weight), "posEmb": arr(model.pos_emb.weight),
            "blocks": blocks, "finalLn": norm_of(getattr(model, "final_ln", None)),
            "Wh": arr(head.weight),
            "bh": arr(head.bias) if head.bias is not None else np.zeros(head.out_features)}


def norm_apply(n: dict, x: np.ndarray) -> np.ndarray:
    if n["kind"] == "id":
        return x
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + n["eps"]) * n["gamma"] + n["beta"]


def arch_forward(E: dict, ids: np.ndarray) -> np.ndarray:
    """Arch.lean semantics, float64, batch of equal-length sequences -> logits (B, T, V)."""
    B, T = ids.shape
    x = E["tokEmb"][ids] + E["posEmb"][:T][None]
    for blk in E["blocks"]:
        y = x
        if blk["attn"] is not None:
            a = blk["attn"]
            H, dqk, dv = a["H"], a["dqk"], a["dv"]
            xn = norm_apply(blk["ln1"], x)
            q = (xn @ a["Wq"].T).reshape(B, T, H, dqk)
            k = (xn @ a["Wk"].T).reshape(B, T, H, dqk)
            v = (xn @ a["Wv"].T).reshape(B, T, H, dv)
            s = np.einsum("bihc,bjhc->bhij", q, k) / math.sqrt(dqk)
            mask = np.triu(np.ones((T, T), bool), 1)
            s = np.where(mask, -np.inf, s)
            s = s - s.max(-1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(-1, keepdims=True)
            o = np.einsum("bhij,bjhc->bihc", w, v).reshape(B, T, H * dv)
            y = x + o @ a["Wo"].T
        z = norm_apply(blk["ln2"], y)
        hdn = np.maximum(z @ blk["W1"].T + blk["b1"], 0)
        x = y + hdn @ blk["W2"].T + blk["b2"]
    return norm_apply(E["finalLn"], x) @ E["Wh"].T + E["bh"]


def lean_lit(x: float) -> str:
    q = Fraction(float(np.float32(x))) if np.float32(x) == x else Fraction(x)
    s = str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"
    return f"({s})" if q < 0 or q.denominator != 1 else s


def lean_row(row: np.ndarray) -> str:
    """Sparse row: sum of `Pi.single j v` (proofs reduce `⬝ᵥ` via `single_dotProduct`).
    Every leaf is fully typed: untyped leaves make elaboration of long sums time out."""
    n = len(row)
    nz = [(j, v) for j, v in enumerate(row) if v != 0]
    if not nz:
        return "0"
    return " + ".join(f"(Pi.single ({j} : Fin {n}) ({lean_lit(v)} : ℝ) : Fin {n} → ℝ)"
                      for j, v in nz)


# Matrices wider than this are written column-major (transpose of a literal whose rows are the
# columns), so a proof can state each column (e.g. an MLP neuron's output) as one `Pi.single` sum.
WIDE = 256


def lean_mat(m: np.ndarray) -> str:
    if not np.any(m):
        return "0"
    if m.shape[0] == m.shape[1] and np.array_equal(m, np.eye(m.shape[0])):
        return "1"
    if m.shape[1] > WIDE:
        return ("Matrix.transpose (Matrix.of ![" + ",\n    ".join(f"({lean_row(col)} : Fin {m.shape[0]} → ℝ)"
                                                              for col in m.T)
                + f"] : Matrix (Fin {m.shape[1]}) (Fin {m.shape[0]}) ℝ)")
    return "Matrix.of ![" + ",\n    ".join(f"({lean_row(row)} : Fin {m.shape[1]} → ℝ)" for row in m) + "]"


def lean_vec(v: np.ndarray) -> str:
    if not np.any(v):
        return "0"
    return "![" + ", ".join(lean_lit(x) for x in v) + "]"


def lean_norm(n: dict) -> str:
    if n["kind"] == "id":
        return ".id"
    return f"(.ln {lean_vec(n['gamma'])} {lean_vec(n['beta'])} {lean_lit(n['eps'])})"


def emit_lean(E: dict, name: str, meta: str) -> str:
    """Each tensor gets its own `def`, so proofs can unfold just the tensors they read."""
    out = ["/-", f"AUTO-GENERATED by scripts/export_arch.py. Do not edit.", meta,
           "Values are exact rationals of the float32 parameters (LayerNorm eps: its float32 value).",
           "All-zero tensors are written `0`, identity matrices (incl. an absent W_o) `1`; matrix rows",
           f"are sparse sums of `Pi.single col value` (matrices wider than {WIDE} columns: `Matrix.transpose` of",
           "the column-major literal). Long literals raise `maxRecDepth`.",
           "-/", "import VWC.Arch", "", "namespace VWC.Arch", ""]
    D = E["D"]

    def tdef(nm, typ, val):
        big = val.count("\n") > 256 or (val.startswith("![") and val.count(",") > 256)
        pre = "set_option maxRecDepth 100000 in\n" if big else ""
        out.append(f"{pre}noncomputable def {nm} : {typ} :=\n  {val}\n")

    def ndef(nm, n):
        if n["kind"] == "id":
            return ".id"
        tdef(f"{nm}_gamma", f"Fin {D} → ℝ", lean_vec(n["gamma"]))
        tdef(f"{nm}_beta", f"Fin {D} → ℝ", lean_vec(n["beta"]))
        return f"(.ln {nm}_gamma {nm}_beta {lean_lit(n['eps'])})"

    blk_names = []
    for bi, b in enumerate(E["blocks"]):
        bn = f"{name}_block{bi}"
        F = b["W1"].shape[0]
        if b["attn"] is not None:
            a = b["attn"]
            H, dqk, dv = a["H"], a["dqk"], a["dv"]
            tdef(f"{bn}_Wq", f"Matrix (Fin ({H} * {dqk})) (Fin {D}) ℝ", lean_mat(a["Wq"]))
            tdef(f"{bn}_Wk", f"Matrix (Fin ({H} * {dqk})) (Fin {D}) ℝ", lean_mat(a["Wk"]))
            tdef(f"{bn}_Wv", f"Matrix (Fin ({H} * {dv})) (Fin {D}) ℝ", lean_mat(a["Wv"]))
            tdef(f"{bn}_Wo", f"Matrix (Fin {D}) (Fin ({H} * {dv})) ℝ", lean_mat(a["Wo"]))
            tdef(f"{bn}_attn", f"Attn {D} {H} {dqk} {dv}",
                 f"⟨{bn}_Wq, {bn}_Wk, {bn}_Wv, {bn}_Wo⟩")
            attn = f"some {bn}_attn"
        else:
            H, dqk, dv = 1, 1, 1
            attn = "none"
        tdef(f"{bn}_W1", f"Matrix (Fin {F}) (Fin {D}) ℝ", lean_mat(b["W1"]))
        tdef(f"{bn}_b1", f"Fin {F} → ℝ", lean_vec(b["b1"]))
        tdef(f"{bn}_W2", f"Matrix (Fin {D}) (Fin {F}) ℝ", lean_mat(b["W2"]))
        tdef(f"{bn}_b2", f"Fin {D} → ℝ", lean_vec(b["b2"]))
        ln1, ln2 = ndef(f"{bn}_ln1", b["ln1"]), ndef(f"{bn}_ln2", b["ln2"])
        out += [f"noncomputable def {bn} : Block {D} {F} {H} {dqk} {dv} where",
                f"  ln1 := {ln1}", f"  attn := {attn}", f"  ln2 := {ln2}",
                f"  W1 := {bn}_W1", f"  b1 := {bn}_b1", f"  W2 := {bn}_W2", f"  b2 := {bn}_b2", ""]
        blk_names.append(bn)
    tdef(f"{name}_tokEmb", f"Matrix (Fin {E['V']}) (Fin {D}) ℝ", lean_mat(E["tokEmb"]))
    tdef(f"{name}_posEmb", f"Matrix (Fin {E['L']}) (Fin {D}) ℝ", lean_mat(E["posEmb"]))
    tdef(f"{name}_Wh", f"Matrix (Fin {E['V']}) (Fin {D}) ℝ", lean_mat(E["Wh"]))
    tdef(f"{name}_bh", f"Fin {E['V']} → ℝ", lean_vec(E["bh"]))
    fln = ndef(f"{name}_final", E["finalLn"])
    out += [f"noncomputable def {name} : Model {E['V']} {D} {E['L']} where",
            f"  tokEmb := {name}_tokEmb", f"  posEmb := {name}_posEmb",
            "  blocks := [" + ", ".join(f"⟨{bn}⟩" for bn in blk_names) + "]",
            f"  finalLn := {fln}", f"  Wh := {name}_Wh", f"  bh := {name}_bh", "", "end VWC.Arch", ""]
    return "\n".join(out)


def to_json(E: dict):
    if isinstance(E, dict):
        return {k: to_json(v) for k, v in E.items()}
    if isinstance(E, list):
        return [to_json(v) for v in E]
    if isinstance(E, np.ndarray):
        return E.tolist()
    return E


@torch.no_grad()
def main(repo, task_name, run, snap, out_dir, lean_name):
    mod, task, model = load(repo, run, snap, task_name)
    E = extract(model)
    # Fidelity check on teacher-forced full sequences (prompt + correct answer).
    pairs = enumerate_inputs(task_name)
    if pairs is None or len(pairs) > 200_000:
        exs = task.generate_examples(20_000, seed=7)
        pairs = [(e.prompt, e.answer) for e in exs]
    seqs = np.array([task.encode(p + a) for p, a in pairs])
    model64 = model.double()
    for m in model64.modules():  # compare against the eps the float32 run actually adds
        if isinstance(m, nn.LayerNorm):
            m.eps = float(np.float32(m.eps))
    max_diff, max_mag = 0.0, 0.0
    for s in range(0, len(seqs), 2048):
        ids = seqs[s:s + 2048]
        ref = model64(torch.tensor(ids)).numpy()
        ours = arch_forward(E, ids)
        max_diff = max(max_diff, float(np.abs(ref - ours).max()))
        max_mag = max(max_mag, float(np.abs(ref).max()))
    n_nonzero = int(sum((np.asarray(v) != 0).sum() for v in
                        [E["tokEmb"], E["posEmb"], E["Wh"], E["bh"]] +
                        [b[k] for b in E["blocks"] for k in ["W1", "b1", "W2", "b2"]] +
                        [b["attn"][k] for b in E["blocks"] if b["attn"] for k in ["Wq", "Wk", "Wv", "Wo"]]))
    # Snapshots may read build-time settings from the environment (e.g. AR_SCALE), so the
    # file name alone does not pin the weights: record the file hash and any overrides.
    sha = hashlib.sha256(open(snap_path(repo, run, snap), "rb").read()).hexdigest()
    env = {k: v for k, v in os.environ.items() if k.startswith("AR_")}
    report = {"task": task_name, "snapshot": snap, "snapshot_sha256": sha, "env_overrides": env,
              "n_sequences": len(seqs),
              "max_abs_logit_diff_arch_vs_torch_f64": max_diff, "max_abs_logit": max_mag,
              "n_blocks": len(E["blocks"]), "n_nonzero_exported": n_nonzero,
              "norms": [[b["ln1"]["kind"], b["ln2"]["kind"]] for b in E["blocks"]] + [E["finalLn"]["kind"]]}
    print(json.dumps(report, indent=1))
    assert max_diff <= 1e-9 * max(1.0, max_mag), "Arch semantics do not match the snapshot"
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"report": report, "weights": to_json(E)}, open(os.path.join(out_dir, f"{lean_name}.json"), "w"))
    meta = (f"Task {task_name}, snapshot {run}/{snap} (sha256 {sha[:16]}…, env overrides: {env or 'none'});\n"
            f"fidelity max |Δlogit| = {max_diff:.2e} on {len(seqs)} sequences.")
    open(os.path.join(out_dir, f"{lean_name}.lean"), "w").write(emit_lean(E, lean_name, meta))


if __name__ == "__main__":
    main(*sys.argv[1:7])
