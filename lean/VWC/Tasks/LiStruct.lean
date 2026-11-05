/-
Weight-structure lemmas for linear-interpolation-two-points (snapshot LinInterpTwoPointAND_v4).
Each lemma states what one tensor does, with neuron/row indices decoded by explicit formulas.
The 1000-neuron lemmas are proved row by row (`fin_cases` then `rfl` against the literal), so
this module is slow (minutes) and kept separate from the mechanism proof.

Residual layout (d_model 152): token one-hot [0,13), position one-hot [13,29), seven gathered
digit slots of width 10 starting at 29 (x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, xq), and 50
line-score channels starting at 99 indexed by (m - 1)·10 + b.
-/
import VWC.Arch
import VWC.Tasks.LiWeights

namespace VWC.Arch.LI

open Matrix Finset

noncomputable def A0 : Attn 152 8 19 19 := liModel_block0_attn

/-- Source position read by gather head h (x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, xq). -/
def src (h : ℕ) : ℕ :=
  if h = 0 then 0 else if h = 1 then 2 else if h = 2 then 3 else if h = 3 then 5
  else if h = 4 then 7 else if h = 5 then 8 else 10

lemma src_le (h : ℕ) : src h ≤ 10 := by unfold src; split_ifs <;> omega

/-! ### Layer-0 attention -/

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wq_apply (v : Fin 152 → ℝ) (r : Fin (8 * 19)) :
    (A0.Wq *ᵥ v) r = if r.val % 19 = 0 ∧ r.val / 19 < 7 then v 24 + v 25 else 0 := by
  fin_cases r <;> simp [A0, liModel_block0_attn, liModel_block0_Wq, mulVec, single_dotProduct]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wk_apply (v : Fin 152 → ℝ) (r : Fin (8 * 19)) :
    (A0.Wk *ᵥ v) r = if r.val % 19 = 0 ∧ r.val / 19 < 7 then
      50 * v ⟨13 + src (r.val / 19), by have := src_le (r.val / 19); omega⟩ else 0 := by
  fin_cases r <;> simp [A0, liModel_block0_attn, liModel_block0_Wk, mulVec, single_dotProduct, src]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wv_apply (v : Fin 152 → ℝ) (r : Fin (8 * 19)) :
    (A0.Wv *ᵥ v) r = if h : 1 ≤ r.val % 19 ∧ r.val % 19 ≤ 10 ∧ r.val / 19 < 7 then
      v ⟨r.val % 19 - 1, by omega⟩ else 0 := by
  fin_cases r <;> simp [A0, liModel_block0_attn, liModel_block0_Wv, mulVec, single_dotProduct]

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
lemma Wo_apply (o : Fin (8 * 19) → ℝ) (i : Fin 152) :
    (A0.Wo *ᵥ o) i = if h : 29 ≤ i.val ∧ i.val < 99 then
      o ⟨(i.val - 29) / 10 * 19 + 1 + (i.val - 29) % 10, by omega⟩ else 0 := by
  fin_cases i <;> simp [A0, liModel_block0_attn, liModel_block0_Wo, mulVec, single_dotProduct]

/-! ### Layer-0 MLP: 1000 three-way detectors -/

/-- Detector n: pair n/500, slope n%500/100 + 1, intercept n%100/10, x = n%10. -/
def lineY (n : ℕ) : ℕ := (n % 500 / 100 + 1) * (n % 10) + n % 100 / 10

lemma lineY_lt (n : ℕ) : lineY n < 60 := by
  unfold lineY
  have h1 : n % 500 / 100 + 1 ≤ 5 := by omega
  have h2 : n % 10 ≤ 9 := by omega
  have := Nat.mul_le_mul h1 h2
  omega

def cx (n : ℕ) : ℕ := (if n / 500 = 0 then 29 else 59) + n % 10
def cyt (n : ℕ) : ℕ := (if n / 500 = 0 then 39 else 69) + lineY n / 10
def cyo (n : ℕ) : ℕ := (if n / 500 = 0 then 49 else 79) + lineY n % 10

lemma cx_lt (n : ℕ) : cx n < 152 := by unfold cx; split_ifs <;> omega
lemma cyt_lt (n : ℕ) : cyt n < 152 := by have := lineY_lt n; unfold cyt; split_ifs <;> omega
lemma cyo_lt (n : ℕ) : cyo n < 152 := by unfold cyo; split_ifs <;> omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1a_row (n : Fin 1000) : liModel_block0_W1 n =
    Pi.single ⟨cx n, cx_lt n⟩ 1 + Pi.single ⟨cyt n, cyt_lt n⟩ 1 + Pi.single ⟨cyo n, cyo_lt n⟩ 1 := by
  fin_cases n <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1a (n : Fin 1000) : liModel_block0_b1 n = -5 / 2 := by
  fin_cases n <;> rfl

/-- Detector n writes into the score channel of its line. -/
def mbc (n : ℕ) : ℕ := 99 + n % 500 / 10

lemma mbc_lt (n : ℕ) : mbc n < 152 := by unfold mbc; omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W2a_col (n : Fin 1000) : (fun i => liModel_block0_W2 i n) = Pi.single ⟨mbc n, mbc_lt n⟩ 1 := by
  fin_cases n <;> rfl

/-! ### Layer-1 MLP: 1000 (line, xq, position) lookups -/

/-- Lookup n: line n/20 (slope n/200 + 1, intercept n/20 % 10), xq = n%20/2, position 11 + n%2. -/
def lineQ (n : ℕ) : ℕ := (n / 200 + 1) * (n % 20 / 2) + n / 20 % 10

def dig (n : ℕ) : ℕ := if n % 2 = 0 then lineQ n / 10 else lineQ n % 10

lemma lineQ_lt (n : ℕ) (hn : n < 1000) : lineQ n < 60 := by
  unfold lineQ
  have h1 : n / 200 + 1 ≤ 5 := by omega
  have h2 : n % 20 / 2 ≤ 9 := by omega
  have := Nat.mul_le_mul h1 h2
  omega

lemma dig_lt (n : ℕ) (hn : n < 1000) : dig n < 13 := by
  have := lineQ_lt n hn; unfold dig; split_ifs <;> omega

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1b_row (n : Fin 1000) : liModel_block1_W1 n =
    Pi.single ⟨24 + n.val % 2, by omega⟩ 1 + Pi.single ⟨89 + n.val % 20 / 2, by omega⟩ 1 +
      Pi.single ⟨99 + n.val / 20, by omega⟩ 1 := by
  fin_cases n <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1b (n : Fin 1000) : liModel_block1_b1 n = -11 / 4 := by
  fin_cases n <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W2b_col (n : Fin 1000) :
    (fun i => liModel_block1_W2 i n) = Pi.single ⟨dig n, by have := dig_lt n n.isLt; omega⟩ 100 := by
  fin_cases n <;> rfl

/-! ### Embeddings and head -/

lemma tokEmb_eq (t : Fin 13) : liModel.tokEmb t = Pi.single (t.castLE (by norm_num)) 1 := by
  fin_cases t <;> rfl

lemma posEmb_eq (p : Fin 16) : liModel.posEmb p = Pi.single ⟨13 + p.val, by omega⟩ 1 := by
  fin_cases p <;> rfl

lemma Wh_apply (v : Fin 152 → ℝ) (t : Fin 13) :
    (liModel.Wh *ᵥ v + liModel.bh) t = v (t.castLE (by norm_num)) := by
  fin_cases t <;> simp [liModel, liModel_Wh, liModel_bh, mulVec, single_dotProduct]

lemma b2_zero : liModel_block0_b2 = 0 ∧ liModel_block1_b2 = 0 := ⟨rfl, rfl⟩

end VWC.Arch.LI
