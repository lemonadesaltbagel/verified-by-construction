#!/usr/bin/env bash
# Regenerate the Lean weight modules of the seven benchmark snapshots, which are derived from the public
# weight-space-programming study (no license; not redistributed here), and check them against
# WEIGHT_CHECKSUMS.txt. Needs git and uv; every Python step runs in the study's own uv project, which
# installs PyTorch. Set WSP to an existing clone to skip cloning.
set -euo pipefail
cd "$(dirname "$0")"
WSP=${WSP:-external/wsp}
COMMIT=f82c8a3a0d8bacfa2e03bf3b38e37b9184273b02
if [ ! -d "$WSP" ]; then
  git clone https://github.com/csinva/weight-space-programming-autoresearch "$WSP"
fi
git -C "$WSP" checkout -q "$COMMIT"
py() { uv run --project "$WSP" python "$@"; }

# parity (weights/parity.json -> lean/VWC/Weights.lean)
py scripts/extract_and_check.py "$WSP" weights
py scripts/gen_weights_lean.py weights/parity.json lean/VWC/Weights.lean

# the six generic-architecture snapshots: task, snapshot, output dir, Lean name, Lean module
export_one() {
  py scripts/export_arch.py "$WSP" "$1" "$1-may15-run1" "$2" "weights/$3" "$4"
  cp "weights/$3/$4.lean" "lean/VWC/Tasks/$5.lean"
}
export_one boolean-circuit-5-bits          ChainedReductionV10_noWo  boolean    boolModel  BoolWeights
export_one digit-counting-10               CountCircuitV13           digitcount countModel CountWeights
export_one word-reversal-3x3               PosRoute_SCALE_50         wordrev    wrModel    WordRevWeights
export_one sort-five-digits                HistogramCircuit_v5_clean sort       sortModel  SortWeights
export_one linear-interpolation-two-points LinInterpTwoPointAND_v4   lininterp  liModel    LiWeights
export_one addition-five-digits            CompactCarryLookahead     addition   addModel   AddWeights

py check_weight_checksums.py
