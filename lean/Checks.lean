-- Axiom audit and non-vacuity check. Run: cd lean && lake env lean Checks.lean
import VWC
#print axioms VWC.parity_correct
#print axioms VWC.parity_correct_float
#print axioms VWC.float_logits_close
#check @VWC.parity_correct_float
example : VWC.argmaxFirst (VWC.flogits (VWC.RoundingModel.worstUp (1/10^7) (1/10^12) (by norm_num) (by norm_num)) VWC.parityParams 2 VWC.hL (VWC.encode [true, false, true, true]) (Fin.last 10)) = 1 :=
  VWC.parity_correct_float _ (by simp [VWC.RoundingModel.worstUp]) (by simp [VWC.RoundingModel.worstUp]) (by simp [VWC.RoundingModel.worstUp]) 2 (by norm_num) _ (by simp) (by simp)
#print axioms VWC.fl32_err
#print axioms VWC.parity_correct_binary32
#print axioms VWC.parity_correct_binary32_crexp
#check @VWC.parity_correct_binary32_crexp
#print axioms VWC.Arch.Bool.bool_correct
#print axioms VWC.Arch.Model.generate_eq_of_steps
#check @VWC.Arch.Bool.bool_correct
#print axioms VWC.Arch.Count.count_correct
#check @VWC.Arch.Count.count_correct
#print axioms VWC.Arch.WordRev.wr_correct
#check @VWC.Arch.WordRev.wr_correct
#print axioms VWC.Arch.Sort.sort_correct
#check @VWC.Arch.Sort.sort_correct
#print axioms VWC.Arch.LI.li_correct
#check @VWC.Arch.LI.li_correct
#print axioms VWC.Arch.Add.add_correct
#check @VWC.Arch.Add.add_correct
#print axioms VWC.Dec2Bin.dec2bin_correct
#check @VWC.Dec2Bin.dec2bin_correct
#print axioms VWC.Graft.ln_graft
#print axioms VWC.Graft.graft_offtrigger
#check @VWC.Graft.graft_offtrigger
#print axioms VWC.Arch.Model.toG_logits
#print axioms VWC.Graft.ln_graft_gen
#print axioms VWC.Graft.graft_ontrigger
#check @VWC.Graft.graft_ontrigger
#print axioms VWC.Graft.graft_main
#print axioms VWC.Graft.graft_preserves_spec
#check @VWC.Graft.graft_main
#print axioms VWC.Graft.Refusal.contract
#print axioms VWC.Graft.Refusal.refusal_main
#check @VWC.Graft.Refusal.refusal_main
#print axioms VWC.Graft.graft_offtrigger_approx
#print axioms VWC.Graft.graft_offtrigger_approx_at
#print axioms VWC.Graft.argmax_stable
#print axioms VWC.Graft.Pattern.pattern_main
#check @VWC.Graft.Pattern.pattern_main
#print axioms VWC.Graft.Pattern.key_main
#print axioms VWC.Graft.Pattern.trig_fraction_le
#print axioms VWC.Graft.Watermark.wm_logits
#print axioms VWC.Graft.Watermark.wm_probe
#print axioms VWC.Graft.Watermark.wm_weight
#print axioms VWC.Graft.Watermark.fp_fraction_le
#print axioms VWC.Graft.Watermark.fp_cos
#print axioms VWC.Graft.Watermark.wm_cos
#print axioms VWC.Graft.Pattern.pattern_radicands_pos
#check @VWC.Graft.Pattern.pattern_radicands_pos
#print axioms VWC.Graft.Refusal.refusal_radicands_pos
#check @VWC.Graft.Refusal.refusal_radicands_pos
#print axioms VWC.Graft.gden_pos_of_energy
