"""Step 0.1: extract ParityFloorSubD3F9 weights and exhaustively check it.

Usage (run from anywhere; uses the repo's own uv environment):
    uv run --project <wsp_repo> python scripts/extract_and_check.py <wsp_repo> <out_dir>

Writes into <out_dir>:
    parity.json             every parameter tensor (PyTorch out x in layout), exact values
    exhaustive_check.csv    all 2046 canonical inputs (bit strings of length 1..10)
    extended_check.csv      all 3^10 = 59049 prompts over {0,1,_}^10 + '=' (superset, non-canonical)
    check_summary.json      counts, min margins, fresh-seed eval accuracy
The repo's evaluation block is not modified; we only import build_model / evaluate.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import os
import sys

import torch

SNAPSHOT = "runs/parity-upto10-bits-may15-run1/interpretable_transformers_lib/ParityFloorSubD3F9.py"
RUN_DIR = "runs/parity-upto10-bits-may15-run1"


def load_snapshot(repo: str):
    run_dir = os.path.join(repo, RUN_DIR)
    sys.path.insert(0, run_dir)  # so `src.*` resolves via the run's symlink
    spec = importlib.util.spec_from_file_location("ParityFloorSubD3F9", os.path.join(repo, SNAPSHOT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import src.task
    import src.eval
    return mod, src.task.get_task("parity-upto10-bits"), src.eval


def margin_info(logits_row: torch.Tensor, target_id: int) -> tuple[int, float]:
    """argmax (torch tie-break: first index) and target logit minus best other logit."""
    pred = int(logits_row.argmax().item())
    others = torch.cat([logits_row[:target_id], logits_row[target_id + 1:]])
    return pred, float(logits_row[target_id] - others.max())


def run_prompts(model, task, prompts: list[str], targets: list[str]) -> list[dict]:
    ids = torch.tensor([task.encode(p) for p in prompts], dtype=torch.long)
    with torch.no_grad():
        logits32 = model(ids)[:, -1, :]
        logits64 = model.double()(ids)[:, -1, :]
        model.float()
    rows = []
    for i, (p, t) in enumerate(zip(prompts, targets)):
        tid = task.stoi[t]
        pred, m32 = margin_info(logits32[i], tid)
        _, m64 = margin_info(logits64[i], tid)
        rows.append({
            "prompt": p, "target": t, "pred": task.itos[pred], "correct": int(task.itos[pred] == t),
            **{f"logit32_{task.itos[j]}": f"{logits32[i, j].item():.9g}" for j in range(task.vocab_size)},
            "margin32": f"{m32:.9g}", "margin64": f"{m64:.17g}",
            "max_abs_diff_32_vs_64": f"{(logits32[i].double() - logits64[i]).abs().max().item():.3g}",
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "n_correct": sum(r["correct"] for r in rows),
        "min_margin32": min(float(r["margin32"]) for r in rows),
        "min_margin64": min(float(r["margin64"]) for r in rows),
        "max_abs_diff_32_vs_64": max(float(r["max_abs_diff_32_vs_64"]) for r in rows),
    }


def main(repo: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    mod, task, ev = load_snapshot(repo)
    model = mod.build_model(task)

    params = {
        name: {"shape": list(p.shape), "values": p.detach().tolist()}
        for name, p in model.named_parameters()
    }
    meta = {
        "snapshot": SNAPSHOT,
        "model_shorthand_name": mod.model_shorthand_name,
        "vocab": task.vocab,
        "config": {k: getattr(model, k) for k in ["vocab_size", "max_seq_len", "d_model", "n_heads", "n_layers", "d_ff"]},
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_nonzero": sum(int((p != 0).sum()) for p in model.parameters()),
        "layout": "PyTorch nn.Linear weight is (out_features, in_features); y = x @ W.T + b",
        "layernorm": "ln1, ln2, final_ln are nn.Identity",
    }
    with open(os.path.join(out_dir, "parity.json"), "w") as f:
        json.dump({"meta": meta, "params": params}, f, indent=1)

    # Canonical input space: bit strings of length 1..10, encoded exactly like the task.
    bits_all = [
        "".join(b) for n in range(1, 11) for b in itertools.product("01", repeat=n)
    ]
    prompts = [b.rjust(10, "_") + "=" for b in bits_all]
    targets = [str(b.count("1") % 2) for b in bits_all]
    rows = run_prompts(model, task, prompts, targets)
    for r, b in zip(rows, bits_all):
        r["bits"] = b
    write_csv(os.path.join(out_dir, "exhaustive_check.csv"), rows)

    # Superset: any of {0,1,_} in each of the 10 slots (includes non-canonical padding).
    ext = ["".join(t) + "=" for t in itertools.product("01_", repeat=10)]
    ext_targets = [str(p.count("1") % 2) for p in ext]
    ext_rows = run_prompts(model, task, ext, ext_targets)
    write_csv(os.path.join(out_dir, "extended_check.csv"), ext_rows)

    fresh = {}
    for seed in [1, 12345]:
        acc, _ = ev.evaluate(model, task, n_samples=2000, seed=seed)
        fresh[f"seed{seed}_n2000"] = acc

    summary = {
        "meta": meta,
        "canonical_2046": summarize(rows),
        "extended_3pow10": summarize(ext_rows),
        "fresh_seed_eval_accuracy": fresh,
        "torch_version": torch.__version__,
    }
    with open(os.path.join(out_dir, "check_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
