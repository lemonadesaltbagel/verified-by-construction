Superseded logs of the first boundary sweep (graft/run_boundary.sh, 2026-11-04, OPT-1.3B).
- The pattern runs (b_circuitpattern*) used the superseded high-gain circuit, now `--circuit pattern_v1`.
- The float16 runs (b_circuitpatternvariantfp16.log, b_circuitwatermarkvariantfp16.log) predate the fix
  that pre-scales q before q.k in half precision in our own forward pass (opt_graft.py `mha`); without
  it q.k overflowed float16 and produced NaN logits. They are superseded by the reruns
  p16b_windows40ntrig40circuitwatermarkvariantfp16.log / opt_opt-1.3b_float32_wm_fp16.json and the
  *_final.json runs.
- The watermark JSONs (opt_opt-1.3b_float32_wm_*.json) were overwritten by later runs of the same
  configurations; these logs are the only record of the first sweep.
