"""Phase 1 survey: for each algorithmic task, load the chosen 100% snapshot and report
architecture features relevant to formal verification, then check correctness exhaustively
when the input space is small enough (else on a large random sample).

Usage: uv run --project <wsp_repo> python scripts/survey_tasks.py <wsp_repo> <out_json>
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import itertools
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

TASKS = {
    "addition-five-digits": "addition-five-digits-may15-run1",
    "boolean-circuit-5-bits": "boolean-circuit-5-bits-may15-run1",
    "decimal-to-binary-8bit": "decimal-to-binary-8bit-may15-run1",
    "digit-counting-10": "digit-counting-10-may15-run1",
    "gcd-three-digits": "gcd-three-digits-may15-run1",
    "linear-interpolation-two-points": "linear-interpolation-two-points-may15-run1",
    "multiplication-five-digits": "multiplication-five-digits-may15-run1",
    "parity-upto10-bits": "parity-upto10-bits-may15-run1",
    "sort-five-digits": "sort-five-digits-may15-run1",
    "word-reversal-3x3": "word-reversal-3x3-may15-run1",
}
EXHAUSTIVE_LIMIT = 1_100_000
SAMPLE_N = 20_000


def successful_snapshots(run_dir: str) -> list[dict]:
    rows = list(csv.DictReader(open(os.path.join(run_dir, "results", "overall_results.csv"))))
    return [r for r in rows if r["accuracy"] == "1.0000"]


def load(run_dir: str, name: str):
    if run_dir not in sys.path:
        sys.path.insert(0, run_dir)
    path = os.path.join(run_dir, "interpretable_transformers_lib", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"snap_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def enumerate_inputs(task_name: str):
    """Canonical input space generators (mirror evolve/src/task.py), or None if too large."""
    if task_name == "boolean-circuit-5-bits":
        out = []
        for bits in itertools.product("01", repeat=5):
            for ops in itertools.product("&|^", repeat=4):
                s = bits[0] + "".join(o + b for o, b in zip(ops, bits[1:]))
                v = int(bits[0])
                for o, b in zip(ops, bits[1:]):
                    b = int(b)
                    v = v & b if o == "&" else v | b if o == "|" else v ^ b
                out.append((s + "=", str(v)))
        return out
    if task_name == "sort-five-digits":
        return [("".join(d) + "=", "".join(sorted(d))) for d in itertools.product("0123456789", repeat=5)]
    if task_name == "parity-upto10-bits":
        return [(b.rjust(10, "_") + "=", str(b.count("1") % 2))
                for n in range(1, 11) for b in ("".join(t) for t in itertools.product("01", repeat=n))]
    if task_name == "decimal-to-binary-8bit":
        return [(f"{n:03d}=", f"{n:08b}") for n in range(256)]
    if task_name == "linear-interpolation-two-points":
        return [(f"{x1},{m * x1 + b:02d};{x2},{m * x2 + b:02d};{xq}=", f"{m * xq + b:02d}")
                for m in range(1, 6) for b in range(10) for x1 in range(10) for x2 in range(10)
                if x2 != x1 for xq in range(10)]
    if task_name == "gcd-three-digits":
        return [(f"{a:03d},{b:03d}=", f"{math.gcd(a, b):03d}") for a in range(1, 1000) for b in range(1, 1000)]
    return None


@torch.no_grad()
def eval_pairs(model, task, pairs, batch=4096):
    """Greedy generation exactly like src/eval.py; returns (n, n_correct, first failures)."""
    n_correct, fails = 0, []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        ids = torch.tensor([task.encode(p) for p, _ in chunk], dtype=torch.long)
        toks = ids
        for _ in range(task.answer_len):
            nxt = model(toks)[:, -1, :].argmax(dim=-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
        preds = [task.decode(r) for r in toks[:, ids.shape[1]:].tolist()]
        for (p, a), pr in zip(chunk, preds):
            if pr == a:
                n_correct += 1
            elif len(fails) < 5:
                fails.append((p, a, pr))
    return len(pairs), n_correct, fails


def describe(model) -> dict:
    mods = {}
    for _, m in model.named_modules():
        mods[type(m).__name__] = mods.get(type(m).__name__, 0) + 1
    ln = [n for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
    params = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    cfg = {k: getattr(model, k) for k in ["d_model", "n_heads", "n_layers", "d_ff", "max_seq_len", "vocab_size"]
           if hasattr(model, k)}
    return {"module_types": mods, "real_layernorms": ln, "n_params": params, "n_nonzero": nonzero,
            "n_buffer_elems": buffers, "config": cfg}


def main(repo: str, out: str) -> None:
    report = {}
    for task_name, run in TASKS.items():
        run_dir = os.path.join(repo, "runs", run)
        snaps = successful_snapshots(run_dir)
        entry = {"run": run, "n_successful_snapshots": len(snaps)}
        if not snaps:
            report[task_name] = entry
            continue
        chosen = snaps[-1]["model_shorthand_name"]  # last 100% snapshot in iteration order
        entry["chosen_snapshot"] = chosen
        entry["min_param_snapshot"] = min(snaps, key=lambda r: float(r["n_params"]))["model_shorthand_name"]
        t0 = time.time()
        try:
            mod = load(run_dir, chosen)
            import src.task
            task = src.task.get_task(task_name)
            model = mod.build_model(task)
            model.eval()
            entry.update(describe(model))
            entry["custom_classes"] = sorted(
                n for n, o in inspect.getmembers(mod, inspect.isclass)
                if issubclass(o, nn.Module) and o.__module__ == mod.__name__)
            entry["prompt_len"], entry["answer_len"] = task.prompt_len, task.answer_len
            pairs = enumerate_inputs(task_name)
            if pairs is not None and len(pairs) <= EXHAUSTIVE_LIMIT:
                entry["check"] = "exhaustive"
            else:
                exs = task.generate_examples(SAMPLE_N, seed=2026)
                pairs = [(e.prompt, e.answer) for e in exs]
                entry["check"] = f"random sample (seed 2026)"
            n, ok, fails = eval_pairs(model, task, pairs)
            entry["n_checked"], entry["n_correct"], entry["failures"] = n, ok, fails
        except Exception as e:  # report and continue with the next task
            entry["error"] = repr(e)[:500]
        entry["seconds"] = round(time.time() - t0, 1)
        report[task_name] = entry
        print(task_name, json.dumps({k: entry.get(k) for k in
              ["chosen_snapshot", "n_params", "n_nonzero", "check", "n_checked", "n_correct", "error", "seconds"]}),
              flush=True)
        for m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
            del sys.modules[m]
        sys.path.remove(run_dir)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
