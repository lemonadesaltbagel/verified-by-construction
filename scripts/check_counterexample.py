"""Re-check a counterexample in float32 and float64 for given snapshots.

Usage: uv run --project <wsp_repo> python scripts/check_counterexample.py <wsp_repo> <task> <run>
           <prompt> <snapshot> [<snapshot> ...]
"""
import sys

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from survey_tasks import load  # noqa: E402


@torch.no_grad()
def gen(model, task, prompt, dtype):
    model = model.to(dtype)
    toks = torch.tensor([task.encode(prompt)])
    steps = []
    for _ in range(task.answer_len):
        logits = model(toks)[0, -1]
        top2 = torch.topk(logits, 2)
        steps.append((task.itos[int(top2.indices[0])], task.itos[int(top2.indices[1])],
                      float(top2.values[0] - top2.values[1])))
        toks = torch.cat([toks, top2.indices[:1].view(1, 1)], dim=1)
    return task.decode(toks[0, len(prompt):]), steps


def main(repo, task_name, run, prompt, *snaps):
    import os
    run_dir = os.path.join(repo, "runs", run)
    for snap in snaps:
        mod = load(run_dir, snap)
        import src.task
        task = src.task.get_task(task_name)
        model = mod.build_model(task).eval()
        for dt in [torch.float32, torch.float64]:
            out, steps = gen(model, task, prompt, dt)
            print(snap, dt, prompt, "->", out)
            print("   per-step (top1, top2, margin):", [(a, b, round(m, 4)) for a, b, m in steps])


if __name__ == "__main__":
    main(*sys.argv[1:])
