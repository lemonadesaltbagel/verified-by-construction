/-
Weight-structure lemmas for sort-five-digits (snapshot HistogramCircuit_v5_clean), kept in their
own module so the mechanism proof (`Sort.lean`) can be iterated without re-elaborating them.
Each lemma states what one tensor does to an arbitrary vector.
-/
import VWC.Arch
import VWC.Tasks.SortWeights

namespace VWC.Arch.Sort

open Matrix Finset

noncomputable def A0 : Attn 24 1 24 24 := sortModel_block0_attn

/-- `W_o` gain, LN2 bias on the histogram dims, fc1 threshold, fc2 gain, LayerNorm eps. -/
noncomputable def G : ℝ := 11388943 / 524288
noncomputable def B2 : ℝ := 3172863 / 33554432
noncomputable def SG : ℝ := 7614871 / 33554432
noncomputable def OUT : ℝ := 2887801 / 65536
noncomputable def EPS : ℝ := 2748779 / 274877906944

/-- Residual dim `12 + d` holds the (scaled) count of digit `d`. -/
def hd (d : Fin 10) : Fin 24 := ⟨12 + d.val, by omega⟩

lemma fpe0 (c : Fin 24) : (finProdFinEquiv ((0 : Fin 1), c)).val = c.val := by
  simp [finProdFinEquiv]

lemma Wq_apply (v : Fin 24 → ℝ) (r : Fin (1 * 24)) :
    (A0.Wq *ᵥ v) r = if r.val = 0 then -10 * v 10 else 0 := by
  fin_cases r <;> simp [A0, sortModel_block0_attn, sortModel_block0_Wq, mulVec, single_dotProduct]

lemma Wk_apply (v : Fin 24 → ℝ) (r : Fin (1 * 24)) :
    (A0.Wk *ᵥ v) r = if r.val = 0 then 10 * v 10 else 0 := by
  fin_cases r <;> simp [A0, sortModel_block0_attn, sortModel_block0_Wk, mulVec, single_dotProduct]

lemma Wv_apply (v : Fin 24 → ℝ) (r : Fin (1 * 24)) :
    (A0.Wv *ᵥ v) r = if h : r.val < 10 then v ⟨r.val, by omega⟩ else 0 := by
  fin_cases r <;> simp [A0, sortModel_block0_attn, sortModel_block0_Wv, mulVec, single_dotProduct]

lemma Wo_apply (o : Fin (1 * 24) → ℝ) (i : Fin 24) :
    (A0.Wo *ᵥ o) i = if h : 12 ≤ i.val ∧ i.val < 22 then G * o ⟨i.val - 12, by omega⟩ else 0 := by
  fin_cases i <;> simp [A0, sortModel_block0_attn, sortModel_block0_Wo, mulVec, single_dotProduct, G]

/-- fc1 rows `e` and `10 + e` both read `Σ_{d ≤ e} v(12+d) - v 11 - (e/2)(v 22 + v 23)`. -/
lemma W1_apply (v : Fin 24 → ℝ) (e : Fin 10) :
    (sortModel_block0_W1 *ᵥ v) (e.castLE (by norm_num)) =
      (∑ d : Fin 10, if d ≤ e then v (hd d) else 0) - v 11 - (e.val / 2 : ℝ) * (v 22 + v 23) ∧
    (sortModel_block0_W1 *ᵥ v) ⟨10 + e.val, by omega⟩ =
      (∑ d : Fin 10, if d ≤ e then v (hd d) else 0) - v 11 - (e.val / 2 : ℝ) * (v 22 + v 23) := by
  fin_cases e <;>
    simp [sortModel_block0_W1, mulVec, single_dotProduct, Fin.sum_univ_succ, hd] <;>
    (try constructor) <;> ring

lemma b1_apply (e : Fin 10) :
    sortModel_block0_b1 (e.castLE (by norm_num)) = 0 ∧
    sortModel_block0_b1 ⟨10 + e.val, by omega⟩ = -SG := by
  fin_cases e <;> simp [sortModel_block0_b1, SG] <;> norm_num

/-- fc2: output dim `d < 10` gets `OUT · (clamp_d - clamp_{d-1})`, clamp_e = h_e - h_{10+e}. -/
lemma W2_apply (h : Fin 20 → ℝ) (d : Fin 10) :
    (sortModel_block0_W2 *ᵥ h) (d.castLE (by norm_num)) =
      OUT * (h (d.castLE (by norm_num)) - h ⟨10 + d.val, by omega⟩) -
      (if hd0 : d.val = 0 then 0 else
        OUT * (h ⟨d.val - 1, by omega⟩ - h ⟨10 + (d.val - 1), by omega⟩)) := by
  fin_cases d <;> simp [sortModel_block0_W2, mulVec, single_dotProduct, OUT] <;> ring

lemma W2_apply_hi (h : Fin 20 → ℝ) (i : Fin 24) (hi : 10 ≤ i.val) :
    (sortModel_block0_W2 *ᵥ h) i = 0 := by
  fin_cases i <;> (try simp at hi) <;> simp [sortModel_block0_W2, mulVec]

lemma b2_zero : sortModel_block0_b2 = 0 := rfl

lemma tokEmb_digit (e : Fin 10) :
    sortModel.tokEmb e.castSucc = Pi.single (e.castLE (by norm_num)) 1 := by
  fin_cases e <;> rfl

lemma tokEmb_eq : sortModel.tokEmb 10 = 0 := rfl

/-- Marker (dim 10) and step index (dim 11) of the position embedding. -/
def mk (t : ℕ) : ℝ := if t < 5 then 1 else if t < 10 then -1 else 0
def kk (t : ℕ) : ℝ := if 5 < t ∧ t < 10 then (t : ℝ) - 5 else 0

lemma posEmb_eq (p : Fin 16) : sortModel.posEmb p =
    Pi.single 10 (mk p.val) + Pi.single 11 (kk p.val) + Pi.single 22 15 + Pi.single 23 (-15) := by
  ext i
  fin_cases p <;> simp [sortModel, sortModel_posEmb, mk, kk, Matrix.of_apply] <;> norm_num

lemma gammas (i : Fin 24) : sortModel_block0_ln1_gamma i = 1 ∧ sortModel_block0_ln2_gamma i = 1 ∧
    sortModel_final_gamma i = 1 := by
  fin_cases i <;> exact ⟨rfl, rfl, rfl⟩

lemma ln1_beta : sortModel_block0_ln1_beta = 0 := rfl
lemma final_beta : sortModel_final_beta = 0 := rfl

lemma ln2_beta (i : Fin 24) :
    sortModel_block0_ln2_beta i = if 12 ≤ i.val ∧ i.val < 22 then B2 else 0 := by
  fin_cases i <;> simp [sortModel_block0_ln2_beta, B2]

/-- Head: 100 · identity on the digit dims, zero row for `=`. -/
lemma Wh_apply (v : Fin 24 → ℝ) (t : Fin 11) :
    (sortModel.Wh *ᵥ v + sortModel.bh) t =
      if h : t.val < 10 then 100 * v ⟨t.val, by omega⟩ else 0 := by
  fin_cases t <;> simp [sortModel, sortModel_Wh, sortModel_bh, mulVec, single_dotProduct]

end VWC.Arch.Sort
