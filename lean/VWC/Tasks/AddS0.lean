/-
Weight-structure lemmas for addition-five-digits (snapshot CompactCarryLookahead). Each lemma
states what one tensor does, with unit/row indices decoded by explicit formulas; the 1024-unit
lemmas are proved row by row (`fin_cases` then `rfl` against the literal).

Residual layout (d_model 160): position one-hot [0,17), token digit one-hot [17,27), ten gathered
digit slots of width 10 at [27,127) (A₀..A₄, B₀..B₄, most significant first), generate flags
g₀..g₄ at [127,132), propagate flags p₀..p₄ at [132,137), carries c₀..c₄ at [137,142), output
digit one-hot at [142,152).
-/
import VWC.Arch
import VWC.Tasks.AddWeights

namespace VWC.Arch.Add

open Matrix Finset

noncomputable def A0 : Attn 160 10 16 16 := addModel_block0_attn

/-- Prompt position read by gather head h: A digits at 0..4, B digits at 6..10. -/
def src (h : ℕ) : ℕ := if h < 5 then h else h + 1

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wq_apply (v : Fin 160 → ℝ) (r : Fin (10 * 16)) :
    (A0.Wq *ᵥ v) r = if r.val % 16 = 0 then 50 * (v 11 + v 12 + v 13 + v 14 + v 15 + v 16) else 0 := by
  fin_cases r <;> simp [A0, addModel_block0_attn, addModel_block0_Wq, mulVec, single_dotProduct] <;> ring

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wk_apply (v : Fin 160 → ℝ) (r : Fin (10 * 16)) :
    (A0.Wk *ᵥ v) r = if r.val % 16 = 0 then v ⟨src (r.val / 16), by unfold src; split_ifs <;> omega⟩
      else 0 := by
  fin_cases r <;> simp [A0, addModel_block0_attn, addModel_block0_Wk, mulVec, single_dotProduct, src]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wv_apply (v : Fin 160 → ℝ) (r : Fin (10 * 16)) :
    (A0.Wv *ᵥ v) r = if h : r.val % 16 < 10 then v ⟨17 + r.val % 16, by omega⟩ else 0 := by
  fin_cases r <;> simp [A0, addModel_block0_attn, addModel_block0_Wv, mulVec, single_dotProduct]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wo_apply (o : Fin (10 * 16) → ℝ) (i : Fin 160) :
    (A0.Wo *ᵥ o) i = if h : 27 ≤ i.val ∧ i.val < 127 then
      o ⟨(i.val - 27) / 10 * 16 + (i.val - 27) % 10, by omega⟩ else 0 := by
  fin_cases i <;> simp [A0, addModel_block0_attn, addModel_block0_Wo, mulVec, single_dotProduct]

lemma attn12_zero {T : ℕ} (x : Fin T → Fin 160 → ℝ) (i : Fin T) :
    addModel_block1_attn.apply x i = 0 ∧ addModel_block2_attn.apply x i = 0 := by
  constructor
  · show addModel_block1_Wo *ᵥ _ = 0
    rw [show addModel_block1_Wo = 0 from rfl, zero_mulVec]
  · show addModel_block2_Wo *ᵥ _ = 0
    rw [show addModel_block2_Wo = 0 from rfl, zero_mulVec]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1a_row (n : Fin 1024) : addModel_block0_W1 n =
    if h : n.val < 500 then
      Pi.single ⟨27 + 10 * (n.val / 100) + n.val / 10 % 10, by omega⟩ 1 +
        Pi.single ⟨77 + 10 * (n.val / 100) + n.val % 10, by omega⟩ 1
    else 0 := by
  fin_cases n <;> rfl

/-- Carry index written by unit n of block 1. -/
def kOf (n : ℕ) : ℕ := if n < 5 then 0 else if n < 9 then 1 else if n < 12 then 2 else if n < 14 then 3 else 4

/-- Output digit of block-2 unit n. -/
def outD (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 1
  else if n < 802 then ((n - 2) % 200 / 20 + (n - 2) % 20 / 2 + (n - 2) % 2) % 10
  else ((n - 802) / 10 + (n - 802) % 10) % 10

lemma outD_lt (n : ℕ) : outD n < 10 := by unfold outD; split_ifs <;> omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem tokEmb_eq (t : Fin 12) : addModel.tokEmb t =
    if h : t.val < 10 then Pi.single ⟨17 + t.val, by omega⟩ 1 else 0 := by
  fin_cases t <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem posEmb_eq (p : Fin 18) : addModel.posEmb p =
    if h : p.val < 17 then Pi.single ⟨p.val, by omega⟩ 1 else 0 := by
  fin_cases p <;> rfl

lemma tokEmb_apply (t : Fin 12) (i : Fin 160) :
    addModel.tokEmb t i = if t.val < 10 ∧ i.val = 17 + t.val then 1 else 0 := by
  rw [tokEmb_eq]
  split_ifs with h1 h2 h2
  · simp [Pi.single_apply, Fin.ext_iff, h2.2]
  · rw [Pi.single_apply, if_neg (fun h => h2 ⟨h1, by rw [h]⟩)]
  · exact absurd h2.1 h1
  · rfl

lemma posEmb_apply (p : Fin 18) (i : Fin 160) :
    addModel.posEmb p i = if p.val < 17 ∧ i.val = p.val then 1 else 0 := by
  rw [posEmb_eq]
  split_ifs with h1 h2 h2
  · simp [Pi.single_apply, Fin.ext_iff, h2.2]
  · rw [Pi.single_apply, if_neg (fun h => h2 ⟨h1, by rw [h]⟩)]
  · exact absurd h2.1 h1
  · rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wh_apply (v : Fin 160 → ℝ) (t : Fin 12) :
    (addModel.Wh *ᵥ v + addModel.bh) t = if h : t.val < 10 then v ⟨142 + t.val, by omega⟩ else 0 := by
  fin_cases t <;> simp [addModel, addModel_Wh, addModel_bh, mulVec, single_dotProduct]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma b2_zero : addModel_block0_b2 = 0 ∧ addModel_block1_b2 = 0 ∧ addModel_block2_b2 = 0 :=
  ⟨rfl, rfl, rfl⟩

end VWC.Arch.Add
