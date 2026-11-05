# Verified by Construction

**Project page:** https://lemonadesaltbagel.github.io/verified-by-construction/ &nbsp;·&nbsp;
**Paper:** [docs/paper.pdf](docs/paper.pdf) &nbsp;·&nbsp;
**Zenodo:** [doi:10.5281/zenodo.23976715](https://doi.org/10.5281/zenodo.23976715)

Research artifact for the preprint *Verified by Construction: Machine-Checked Proofs for Hand-Written
Transformer Circuits and Their Composition* (Xintian Huang, 2026). It contains the Lean 4 development, the
grafting experiments, the evidence files and the scripts that the paper cites by path.

## Contents

| Path | What it is | Paper |
|---|---|---|
| `lean/` | Lean 4 development (toolchain `leanprover/lean4:v4.35.0`, Mathlib `v4.35.0`); `lean/Checks.lean` prints the axioms of the main theorems | Sec. 2–5, App. A |
| `lean/VWC/Tasks/` | correctness proofs of the seven benchmark programs | Sec. 3 |
| `lean/VWC/Graft/` | grafting construction, Theorems A, A′, B, the pattern detector, key gate, watermark, radicand positivity | Sec. 4–5 |
| `graft/` | grafting experiments on OPT hosts (`opt_graft.py`, `var_probe.py`, `run_*.sh`) and their results in `graft/results/` | Sec. 6–7 |
| `graft/tiny-gpt2/` | the small host trained by `graft/train_tiny.py` | Sec. 6 |
| `scripts/` | exporters, provenance, the 60-digit parameter check, the Lean transcription check | Sec. 3, 6, App. D |
| `evidence/` | logs and outputs cited in the paper (Lean build and axiom checks, provenance, counterexample, head norms, ...) | throughout |
| `weights/` | benchmark survey and checks (`phase1_survey.json`, `exhaustive_check.csv`, `extended_check.csv`, elaboration times) | Sec. 3, App. B |
| `audit/` | the two specification reviews of the benchmark theorems | Sec. 3 |
| `proof_results.csv` | proof attempts per task | Sec. 3 |

## Not included: third-party weights

The seven benchmark programs are snapshots from the weight-space-programming study
`csinva/weight-space-programming-autoresearch` (commit `f82c8a3a0d8bacfa2e03bf3b38e37b9184273b02`). That
repository carries no license, so neither its code nor the weights exported from its snapshots are
redistributed here. Missing files: `lean/VWC/Weights.lean`, `lean/VWC/Tasks/*Weights.lean`,
`weights/parity.json` and `weights/{addition,boolean,digitcount,lininterp,sort,wordrev}/`.
`regenerate_weights.sh` clones the study at the pinned commit, re-exports the weights with this
artifact's exporters and checks the generated Lean modules against `WEIGHT_CHECKSUMS.txt`
(`check_weight_checksums.py`). The exported values are the exact rationals of the float32 parameters, so
the regenerated modules are identical to those used for the proofs.

## Reproduce

1. **Weights** (needs `git` and `uv`; the first run installs the study's Python environment with
   PyTorch, after which regeneration takes about 2 minutes on CPU):
   ```bash
   ./regenerate_weights.sh          # ends with "ALL WEIGHT MODULES MATCH"
   ```
2. **Proofs** (needs `elan`; the full build has 9,354 jobs):
   ```bash
   cd lean && lake exe cache get && lake build
   lake env lean Checks.lean        # compare with evidence/lean_checks.txt
   ```
   Every printed axiom set is `[propext, Classical.choice, Quot.sound]`.
3. **Grafting experiments** (needs a CUDA GPU; OPT checkpoints are downloaded from Hugging Face):
   ```bash
   cd graft && uv sync
   bash run_final.sh                # main results; OPT-6.7B and float64 controls need about 80 GB
   ```
   The other runners (`run_dial.sh`, `run_violate_final.sh`, `run_offset.sh`, `run_advmax.sh`,
   `run_advtrig2.sh`) reproduce the tables and paragraphs listed in the paper's App. E.
4. **Checks without a GPU**:
   ```bash
   python3 scripts/check_conditions_exact.py graft/results/*_final.json   # 60-digit parameter recheck
   ```
   `scripts/check_lean_transcription.py` compares the circuit dumps in `graft/results/*.npz` with an
   independent transcription of the Lean definitions (needs `numpy`).

## License

See the Zenodo record ([doi:10.5281/zenodo.23976715](https://doi.org/10.5281/zenodo.23976715)).
