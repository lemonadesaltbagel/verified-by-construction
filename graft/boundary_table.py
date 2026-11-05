"""Tabulate the Phase 3.5 boundary experiments (results/opt_opt-1.3b_float32*.json) as markdown."""
import glob
import json
import os

VARIANTS = ["graft", "naive", "tf32", "bf16", "fp16", "int8", "int4", "prune0.1", "prune0.3", "prune0.5"]


def load(circ, v):
    tag = ("" if circ == "pattern" else "_wm") + ("" if v == "graft" else f"_{v}")
    p = f"results/opt_opt-1.3b_float32{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def g(x, k, fmt="{:.2e}"):
    v = x.get(k)
    return "—" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))


print("Pattern detector (OPT-1.3B; 40 WikiText windows, 20480 positions; 40 trigger prompts)\n")
print("| Variant | max \\|Δlogit\\| vs own host | argmax = host | ppl change | variant host: argmax = host | argmax = variant host | non-finite positions | refusal rate | false refusals | 8-step gen | zero cols | mean-row dev |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for v in VARIANTS:
    x = load("pattern", v)
    if x:
        print(f"| {v} | {g(x, 'own_host_max_dlogit')} | {g(x, 'off_argmax_agree', '{:.4f}')} | {g(x, 'ppl_rel_change')} | "
              f"{g(x, 'ref_host_variant_argmax_agree_vs_host', '{:.4f}')} | {g(x, 'ref_argmax_agree_vs_host_variant', '{:.4f}')} | "
              f"{g(x, 'nonfinite_logit_positions')} | {g(x, 'refusal_rate', '{:.4f}')} | "
              f"{g(x, 'false_refusal_positions')} | {g(x, 'generation_all_refusal')} | {g(x, 'zero_cols_max')} | {g(x, 'mean_rows_max_dev')} |")

print("\nWatermark (m = 256 pairs; detected = cosine >= 1/2)\n")
print("| Variant | max \\|Δlogit\\| vs own host | argmax = host | ppl change | non-finite positions | W2 detected (positions) | W2 min cos | W1 detected (rows) | W1 min cos |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for v in VARIANTS:
    x = load("watermark", v)
    if x:
        print(f"| {v} | {g(x, 'own_host_max_dlogit')} | {g(x, 'off_argmax_agree', '{:.4f}')} | {g(x, 'ppl_rel_change')} | "
              f"{g(x, 'nonfinite_logit_positions')} | {g(x, 'wm_W2_detected', '{:.4f}')} | {g(x, 'wm_W2_min_cos', '{:.3f}')} | "
              f"{g(x, 'wm_W1_detected', '{:.4f}')} | {g(x, 'wm_W1_min_cos', '{:.3f}')} |")

for p in sorted(glob.glob("results/opt_opt-1.3b_float32*_ft*.json")):
    x = json.load(open(p))
    ft = x["finetune"]
    print(f"\nFine-tuning {x['circuit']}, mode {ft['mode']} (AdamW lr {ft['lr']}, batch {ft['batch']} x {ft['win']} tokens)\n")
    wm = x["circuit"] == "watermark"
    print("| Step | train loss | ppl | max \\|Δlogit\\| vs own host | argmax = own host | "
          + ("W2 detected | W2 min cos | W1 detected" if wm else "refusal rate | false refusals") + " | zero cols | mean-row dev |")
    print("| --- | --- | --- | --- | --- | " + ("--- | --- | --- |" if wm else "--- | --- |") + " --- | --- |")
    for e in ft["evals"]:
        mid = (f"{g(e, 'wm_W2_detected', '{:.4f}')} | {g(e, 'wm_W2_min_cos', '{:.3f}')} | {g(e, 'wm_W1_detected', '{:.4f}')}" if wm
               else f"{g(e, 'refusal_rate', '{:.4f}')} | {g(e, 'false_refusal_positions')}")
        print(f"| {e['step']} | {g(e, 'train_loss', '{:.4g}')} | {g(e, 'ppl_graft', '{:.4g}')} | {g(e, 'own_host_max_dlogit')} | "
              f"{g(e, 'own_host_argmax_agree', '{:.4f}')} | {mid} | {g(e, 'zero_cols_max')} | {g(e, 'mean_rows_max_dev')} |")
