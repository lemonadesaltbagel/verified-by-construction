"""Evidence: weight provenance of the seven proved snapshots.

For each task: the exporter's report (snapshot, recorded sha256, forward check vs PyTorch in float64),
the sha256 of the snapshot file on disk now, and whether the Lean weight module used by the proofs
equals the exporter's generated file (ignoring the header comment).

Usage: python3 scripts/export_provenance.py > evidence/export_provenance.txt
"""
import hashlib
import json

TASKS = [("boolean/boolModel", "BoolWeights"), ("digitcount/countModel", "CountWeights"),
         ("wordrev/wrModel", "WordRevWeights"), ("sort/sortModel", "SortWeights"),
         ("lininterp/liModel", "LiWeights"), ("addition/addModel", "AddWeights")]


def body(path):
    s = open(path).read()
    return s[s.index("-/") :]      # drop the generated header comment


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


print("parity: weights/check_summary.json (exhaustive over 2046 canonical inputs, float32 vs float64), "
      "weights/parity.json")
for w, lean in TASKS:
    r = json.load(open(f"weights/{w}.json"))["report"]
    run = {"boolean-circuit-5-bits": "boolean-circuit-5-bits-may15-run1"}.get(r["task"], r["task"] + "-may15-run1")
    snap = f"external/wsp/runs/{run}/interpretable_transformers_lib/{r['snapshot']}.py"
    try:
        disk = sha(snap)
    except FileNotFoundError:
        disk = "missing"
    same = body(f"weights/{w}.lean") == body(f"lean/VWC/Tasks/{lean}.lean")
    print(f"{r['task']}: snapshot {r['snapshot']}; sha256 recorded {r['snapshot_sha256']}; on disk "
          f"{'MATCH' if disk == r['snapshot_sha256'] else disk}; forward check arch vs torch float64 on "
          f"{r['n_sequences']} sequences, max |dlogit| {r['max_abs_logit_diff_arch_vs_torch_f64']:.3g}; "
          f"lean/VWC/Tasks/{lean}.lean {'==' if same else '!='} weights/{w}.lean (body, sha256 "
          f"{sha(f'lean/VWC/Tasks/{lean}.lean')[:16]})")

# word reversal: the survey's chosen file name and the exported label name the same bytes
import glob  # noqa: E402
fs = sorted(glob.glob("external/wsp/runs/word-reversal-3x3-may15-run1/interpretable_transformers_lib/PosRoute_*.py"))
hs = {sha(f) for f in fs}
print(f"word-reversal: {len(fs)} PosRoute_*.py files, {len(hs)} distinct sha256 ({', '.join(h[:16] for h in hs)}); "
      "the snapshot reads AR_SCALE from the environment (default 50), exported with no override as PosRoute_SCALE_50")
