Raw evidence for paper claims that are not experiment outputs (generated 2026-11-05):
- lean_build.txt     `lake build` result (job count)
- lean_checks.txt    `lake env lean Checks.lean`: #print axioms of every main theorem, #check statements
- lean_env.txt       Lean / Mathlib versions
- lean_files.csv     per-file line count, lemma/theorem count, sorry / native_decide / axiom declarations
- opt_configs.txt    HuggingFace OPT configs (all sizes) and modeling_opt.py position offset (graft/opt_config_dump.py)
- compute_ledger.csv RunPod A100 pods used (times from the session's pod create/delete records; approximate)
Other raw logs used by the paper: graft/gpt2_graft.log, graft/tiny_graft.log, graft/watermark_eval.log,
weights/elaboration_times.txt.
Added 2026-11-05 (after claim-audit run 04):
- mult_counterexample.txt  scripts/check_counterexample.py: both final multiplication snapshots
                           (MultCircuitV2, V3) on 89984*48898= in float32 and float64 (true 4400037632)
- export_provenance.txt    scripts/export_provenance.py: per-snapshot sha256 (recorded vs on disk),
                           exporter forward check, Lean weight module == generated module
- sort_analysis.txt        scripts/analyze_sort.py weights/sort/sortModel.json (all 10^5 inputs, float64)
- env_tokens_wsl.txt       scripts/dump_env_evidence.py + nvidia-smi on the WSL node: GPU, versions,
                           tokenizer decoding of the trigger pattern and refusal token
- float_ext_times.txt      single-file elaboration time / peak RSS of Parity, Float, FloatParity, Binary32
                           (one session, warm cache, WSL node)
The original study's own records: external/wsp/runs/<run>/results/overall_results.csv (accuracy per
snapshot), external/wsp/runs/<run>/program.md and src/eval.py (default evaluation: 200 examples),
external/wsp/evolve/src/task.py (task generators). Cross-review records of the seven theorem
statements: audit/audit_{A,B}{_phaseA,}.md. Superseded boundary-sweep logs: graft/results/historical/.
- head_norms.txt           scripts/head_norm_ratio.py: R = max_v/min_v ||w_v * gamma_f|| for OPT-1.3B/2.7B/6.7B,
                           converting the envelope ratio rho (bound_ratio_max) into a per-token bound R*rho
