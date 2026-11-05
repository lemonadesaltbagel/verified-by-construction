/-
Weight-structure lemmas for word-reversal-3x3 (slow to elaborate: ~2600 entry checks per
51×51 matrix), kept in their own module so the mechanism proof can be iterated quickly.
-/
import VWC.Arch
import VWC.Tasks.WordRevWeights

namespace VWC.Arch.WordRev

open Matrix Finset

/-! ### Weight structure -/

/-- The routing table hard-coded in the snapshot: query position p reads key `src p`. -/
def src (p : ℕ) : ℕ :=
  if p = 11 then 8 else if p = 12 then 9 else if p = 13 then 10 else if p = 14 then 7
  else if p = 15 then 4 else if p = 16 then 5 else if p = 17 then 6 else if p = 18 then 3
  else if p = 19 then 0 else if p = 20 then 1 else if p = 21 then 2 else 0

/-- Number of query positions routed to key j. -/
def mult (j : ℕ) : ℕ := ((range 23).filter fun p => src p = j).card

lemma mult_le (j : ℕ) : mult j ≤ 13 := by
  rcases Nat.lt_or_ge j 11 with h | h
  · interval_cases j <;> decide
  · have : mult j = 0 := by
      unfold mult; rw [card_eq_zero, filter_eq_empty_iff]
      intro p _; unfold src; split_ifs <;> omega
    omega

noncomputable def A0 : Attn 51 1 51 51 := wrModel_block0_attn

set_option maxHeartbeats 0 in
/-- Structure of W_q: column `28 + p` has 50 in row `28 + src p`; token columns are zero. -/
lemma Wq_eq : A0.Wq = Matrix.of fun r c =>
    if 28 ≤ c.val ∧ r.val = 28 + src (c.val - 28) then 50 else 0 := by
  ext r c
  fin_cases r <;> fin_cases c <;> simp [A0, wrModel_block0_attn, wrModel_block0_Wq, src]

set_option maxHeartbeats 0 in
lemma Wk_eq : A0.Wk = Matrix.of fun r c => if 28 ≤ r.val ∧ r = c then 1 else 0 := by
  ext r c
  fin_cases r <;> fin_cases c <;> simp [A0, wrModel_block0_attn, wrModel_block0_Wk]

set_option maxHeartbeats 0 in
lemma Wv_eq : A0.Wv = Matrix.of fun r c => if r.val < 28 ∧ r = c then 1 else 0 := by
  ext r c
  fin_cases r <;> fin_cases c <;> simp [A0, wrModel_block0_attn, wrModel_block0_Wv]

lemma Wo_eq : A0.Wo = 1 := rfl

lemma Wh_mulVec (v : Fin 51 → ℝ) (t : Fin 28) :
    (wrModel.Wh *ᵥ v + wrModel.bh) t = 50 * v (t.castLE (by norm_num)) := by
  fin_cases t <;> simp [wrModel, wrModel_Wh, wrModel_bh, mulVec, single_dotProduct]

lemma tokEmb_eq (t : Fin 28) : wrModel.tokEmb t = Pi.single (t.castLE (by norm_num)) 1 := by
  fin_cases t <;> rfl

lemma posEmb_eq (p : Fin 23) : wrModel.posEmb p = Pi.single ⟨28 + p.val, by omega⟩ 1 := by
  fin_cases p <;> rfl

lemma gamma_one (i : Fin 51) : wrModel_block0_ln1_gamma i = 1 ∧ wrModel_final_gamma i = 1 := by
  fin_cases i <;> exact ⟨rfl, rfl⟩

lemma mlp_zero (v : Fin 51 → ℝ) : wrModel_block0.mlp v = 0 := by
  simp [Block.mlp, wrModel_block0, wrModel_block0_W2, wrModel_block0_b2]

end VWC.Arch.WordRev
