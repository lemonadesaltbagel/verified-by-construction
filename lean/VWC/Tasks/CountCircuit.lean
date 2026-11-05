/-
digit-counting-10 (snapshot CountCircuitV13): real-valued correctness of greedy generation on
every input (10¹¹ prompts), by mechanism:

1. Block 0, head 0 at an output position scores 0 on the query position and -200 elsewhere:
   it copies the query digit's one-hot (error ≤ 12·e⁻²⁰⁰).
2. Head 1 scores 100 on the 10 data positions and 0 elsewhere: ×10 it is the digit histogram
   (error ≤ 3·e⁻¹⁰⁰).
3. MLP 0 gates the histogram by the query one-hot: dim 10 := count (error ≤ 1e-30).
4. MLP 1: at the tens position one neuron fires iff count = 10; at the ones position 11 shifted
   ReLUs form bumps `2·[count = c]`, on top of the embedded tens token (+1 on its digit).
   The tens digit for count < 10 is decided by an **exact tie** (all logits 0) broken by
   `torch.argmax`'s first-index rule (`argmaxFirst`).
-/
import VWC.Arch
import VWC.Tasks.CountWeights

namespace VWC.Arch.Count

open Matrix Finset

/-! ### Specification (evolve/src/task.py, CountDigitTask). Vocab "0123456789=". -/

/-- Token id of a digit. -/
def dTok (d : Fin 10) : Fin 11 := d.castSucc

/-- Number of occurrences of `q` among the 10 data digits. -/
def count (ds : Fin 10 → Fin 10) (q : Fin 10) : ℕ := (univ.filter fun i => ds i = q).card

/-- Prompt `d0 … d9 q =` (12 tokens). -/
def prompt (ds : Fin 10 → Fin 10) (q : Fin 10) : List (Fin 11) :=
  List.ofFn (fun i => dTok (ds i)) ++ [dTok q, 10]

/-- Answer `f"{count:02d}"` (2 tokens). -/
def answer (ds : Fin 10 → Fin 10) (q : Fin 10) : List (Fin 11) :=
  [if count ds q = 10 then 1 else 0, ⟨count ds q % 10, by omega⟩]

/-! ### Weight structure -/

noncomputable def A0 : Attn 32 2 1 10 := countModel_block0_attn

/-- Sum of the 10 digit dims. -/
def dsum (v : Fin 32 → ℝ) : ℝ := ∑ c : Fin 10, v (Fin.castLE (by norm_num) c)

lemma Wq_mulVec (v : Fin 32 → ℝ) (h : Fin 2) :
    (A0.Wq *ᵥ v) (finProdFinEquiv (h, 0)) = 10 * v 11 := by
  fin_cases h <;> simp [A0, countModel_block0_attn, countModel_block0_Wq, mulVec,
    single_dotProduct, finProdFinEquiv]

lemma Wk0_mulVec (v : Fin 32 → ℝ) :
    (A0.Wk *ᵥ v) (finProdFinEquiv (0, 0)) = -20 * v 10 - 20 * v 11 := by
  simp [A0, countModel_block0_attn, countModel_block0_Wk, mulVec, single_dotProduct,
    finProdFinEquiv]; ring

lemma Wk1_mulVec (v : Fin 32 → ℝ) :
    (A0.Wk *ᵥ v) (finProdFinEquiv (1, 0)) = 10 * v 10 := by
  simp [A0, countModel_block0_attn, countModel_block0_Wk, mulVec, single_dotProduct,
    finProdFinEquiv]

lemma Wv_mulVec (v : Fin 32 → ℝ) (h : Fin 2) (c : Fin 10) :
    (A0.Wv *ᵥ v) (finProdFinEquiv (h, c)) = v (Fin.castLE (by norm_num) c) := by
  fin_cases h <;> fin_cases c <;> simp [A0, countModel_block0_attn, countModel_block0_Wv, mulVec,
    single_dotProduct, finProdFinEquiv]

lemma Wo_mulVec (o : Fin 20 → ℝ) : A0.Wo *ᵥ o = fun d =>
    if h : 12 ≤ d.val ∧ d.val < 22 then o (finProdFinEquiv ((0 : Fin 2), (⟨d.val - 12, by omega⟩ : Fin 10)))
    else if h : 22 ≤ d.val then 10 * o (finProdFinEquiv ((1 : Fin 2), (⟨d.val - 22, by omega⟩ : Fin 10))) else 0 := by
  funext d; fin_cases d <;> simp [A0, countModel_block0_attn, countModel_block0_Wo, mulVec,
    single_dotProduct, finProdFinEquiv]

/-- MLP 0: dim 10 receives Σ_c relu (hist_c + 20·g_c - 20); nothing else is written. -/
noncomputable def cntOf (v : Fin 32 → ℝ) : ℝ :=
  ∑ c : Fin 10, relu (v ⟨22 + c.val, by omega⟩ + 20 * v ⟨12 + c.val, by omega⟩ - 20)

lemma hidden0 (v : Fin 32 → ℝ) :
    (fun f => relu ((countModel_block0.W1 *ᵥ v) f + countModel_block0.b1 f)) =
      fun c : Fin 10 => relu (v ⟨22 + c.val, by omega⟩ + 20 * v ⟨12 + c.val, by omega⟩ - 20) := by
  funext c; fin_cases c <;> simp [countModel_block0, countModel_block0_W1, countModel_block0_b1,
    mulVec, single_dotProduct] <;> ring_nf

lemma mlp0_eq (v : Fin 32 → ℝ) :
    countModel_block0.mlp v = fun d => if d = 10 then cntOf v else 0 := by
  unfold Block.mlp
  rw [hidden0]
  funext d
  fin_cases d <;> simp [countModel_block0, countModel_block0_W2, countModel_block0_b2, mulVec,
    single_dotProduct, cntOf, Fin.sum_univ_succ] <;> ring

/-- MLP 1 neurons: the tens neuron and the 11 shifted ReLUs. -/
noncomputable def n0 (v : Fin 32 → ℝ) : ℝ := relu (-20 * dsum v + v 10 + 20 * v 11 - 59 / 2)
noncomputable def nk (v : Fin 32 → ℝ) (k : ℝ) : ℝ := relu (20 * dsum v + v 10 + 20 * v 11 - 40 - k)

lemma hidden1 (v : Fin 32 → ℝ) :
    (fun f => relu ((countModel_block1.W1 *ᵥ v) f + countModel_block1.b1 f)) =
      fun f : Fin 12 => if f = 0 then n0 v else nk v ((f.val : ℝ) - 1) := by
  funext f; fin_cases f <;> simp [countModel_block1, countModel_block1_W1, countModel_block1_b1,
    mulVec, single_dotProduct, n0, nk, dsum, Fin.sum_univ_succ] <;> ring_nf

lemma mlp1_apply (v : Fin 32 → ℝ) : countModel_block1.mlp v =
    countModel_block1_W2 *ᵥ (fun f : Fin 12 => if f = 0 then n0 v else nk v ((f.val : ℝ) - 1)) := by
  unfold Block.mlp
  rw [hidden1]
  simp [countModel_block1, countModel_block1_b2]

lemma mlp1_0 (v : Fin 32 → ℝ) : countModel_block1.mlp v 0 = 2 * nk v 9 - 4 * nk v 10 := by
  rw [mlp1_apply]; simp [countModel_block1_W2, mulVec, single_dotProduct]; norm_num; ring

lemma mlp1_1 (v : Fin 32 → ℝ) : countModel_block1.mlp v 1 =
    4 * n0 v + 2 * nk v 0 - 4 * nk v 1 + 2 * nk v 2 := by
  rw [mlp1_apply]; simp [countModel_block1_W2, mulVec, single_dotProduct]; norm_num; ring

lemma mlp1_c (v : Fin 32 → ℝ) (c : Fin 10) (hc : 2 ≤ c.val) :
    countModel_block1.mlp v (Fin.castLE (by norm_num) c) =
      2 * nk v ((c.val : ℝ) - 1) - 4 * nk v c.val + 2 * nk v ((c.val : ℝ) + 1) := by
  rw [mlp1_apply]
  fin_cases c <;> simp at hc <;> simp [countModel_block1_W2, mulVec, single_dotProduct] <;>
    norm_num <;> ring

lemma mlp1_hi (v : Fin 32 → ℝ) (d : Fin 32) (hd : 10 ≤ d.val) : countModel_block1.mlp v d = 0 := by
  rw [mlp1_apply]
  fin_cases d <;> simp at hd <;> simp [countModel_block1_W2, mulVec]

/-- MLP 1 on every digit coordinate. -/
lemma mlp1_lt (v : Fin 32 → ℝ) (d : Fin 32) (hd : d.val < 10) : countModel_block1.mlp v d =
    if d.val = 0 then 2 * nk v 9 - 4 * nk v 10
    else if d.val = 1 then 4 * n0 v + (2 * nk v 0 - 4 * nk v 1 + 2 * nk v 2)
    else 2 * nk v ((d.val : ℝ) - 1) - 4 * nk v d.val + 2 * nk v ((d.val : ℝ) + 1) := by
  rw [mlp1_apply]
  fin_cases d <;> simp at hd <;> simp [countModel_block1_W2, mulVec, single_dotProduct] <;>
    norm_num <;> ring

lemma Wh_mulVec (v : Fin 32 → ℝ) : countModel.Wh *ᵥ v + countModel.bh =
    fun t => if h : t.val < 10 then v ⟨t.val, by omega⟩ else 0 := by
  funext t; fin_cases t <;> simp [countModel, countModel_Wh, countModel_bh, mulVec, single_dotProduct]

/-! ### Embedding facts -/

lemma tok_digit (t : Fin 11) (c : Fin 10) :
    countModel.tokEmb t (Fin.castLE (by norm_num) c) = if t.val = c.val then 1 else 0 := by
  fin_cases t <;> fin_cases c <;> simp [countModel, countModel_tokEmb]

lemma tok_hi (t : Fin 11) (d : Fin 32) (hd : 10 ≤ d.val) : countModel.tokEmb t d = 0 := by
  fin_cases t <;> fin_cases d <;> simp at hd <;> simp [countModel, countModel_tokEmb]

lemma pos_digit (p : Fin 14) (c : Fin 10) : countModel.posEmb p (Fin.castLE (by norm_num) c) = 0 := by
  fin_cases p <;> fin_cases c <;> simp [countModel, countModel_posEmb]

lemma pos_10 (p : Fin 14) : countModel.posEmb p 10 = if p.val < 10 then 1 else 0 := by
  fin_cases p <;> simp [countModel, countModel_posEmb]

lemma pos_11 (p : Fin 14) : countModel.posEmb p 11 = if p.val = 11 ∨ p.val = 12 then 1 else 0 := by
  fin_cases p <;> simp [countModel, countModel_posEmb]

lemma pos_hi (p : Fin 14) (d : Fin 32) (hd : 12 ≤ d.val) : countModel.posEmb p d = 0 := by
  fin_cases p <;> fin_cases d <;> simp at hd <;> simp [countModel, countModel_posEmb]

/-! ### Attention at the last position (prompt length 12 or 13) -/

lemma exp_neg100_le : Real.exp (-100) ≤ 1 / 10 ^ 40 := by
  have he := Real.exp_one_gt_d9
  have h : (2.7182818283 : ℝ) ^ 100 ≤ Real.exp 100 := by
    rw [show (100 : ℝ) = (100 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) he.le 100
  rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 100) (by norm_num)]
  calc (1 / 10 ^ 40 : ℝ)⁻¹ ≤ (2.7182818283 : ℝ) ^ 100 := by norm_num
    _ ≤ _ := h

lemma exp_neg200_le : Real.exp (-200) ≤ 1 / 10 ^ 40 :=
  (Real.exp_le_exp.2 (by norm_num)).trans exp_neg100_le

section Last

variable {n : ℕ} (hn : n = 11 ∨ n = 12) (hT : n + 1 ≤ 14) (ids : Fin (n + 1) → Fin 11)

/-- Embedded prompt. -/
noncomputable def X (j : Fin (n + 1)) : Fin 32 → ℝ :=
  countModel.tokEmb (ids j) + countModel.posEmb (Fin.castLE hT j)

lemma X_digit (j : Fin (n + 1)) (c : Fin 10) :
    X hT ids j (Fin.castLE (by norm_num) c) = if (ids j).val = c.val then 1 else 0 := by
  simp [X, tok_digit, pos_digit]

lemma X_10 (j : Fin (n + 1)) : X hT ids j 10 = if j.val < 10 then 1 else 0 := by
  simp [X, tok_hi _ (10 : Fin 32) (by norm_num), pos_10]

lemma X_11 (j : Fin (n + 1)) : X hT ids j 11 = if j.val = 11 ∨ j.val = 12 then 1 else 0 := by
  simp [X, tok_hi _ (11 : Fin 32) (by norm_num), pos_11]

lemma X_hi (j : Fin (n + 1)) (d : Fin 32) (hd : 12 ≤ d.val) : X hT ids j d = 0 := by
  simp [X, tok_hi _ d (by omega), pos_hi _ d hd]

include hn in
lemma X_last_11 : X hT ids (Fin.last n) 11 = 1 := by
  rw [X_11]; rcases hn with rfl | rfl <;> simp

/-- Head 0 key set: the query position 10. Head 1 key set: the 10 data positions. -/
def S0 : Finset (Fin (n + 1)) := univ.filter fun j => j.val = 10
def S1 : Finset (Fin (n + 1)) := univ.filter fun j => j.val < 10

include hn in
lemma score0 (j : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ X hT ids (Fin.last n)) (finProdFinEquiv ((0 : Fin 2), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ X hT ids j) (finProdFinEquiv ((0 : Fin 2), c)))) / Real.sqrt (1 : ℕ) =
      if j ∈ S0 then 0 else -200 := by
  have h11 := X_last_11 hn hT ids
  simp only [dotProduct, Fin.sum_univ_one, Fin.isValue, Nat.cast_one, Real.sqrt_one, div_one]
  rw [show (0 : Fin 1) = 0 from rfl, Wq_mulVec, Wk0_mulVec, h11, X_10, X_11]
  have hj := j.isLt
  rcases hn with rfl | rfl <;> simp only [S0, mem_filter, mem_univ, true_and] <;>
    split_ifs <;> first | omega | norm_num

include hn in
lemma score1 (j : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ X hT ids (Fin.last n)) (finProdFinEquiv ((1 : Fin 2), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ X hT ids j) (finProdFinEquiv ((1 : Fin 2), c)))) / Real.sqrt (1 : ℕ) =
      if j ∈ S1 then 100 else 0 := by
  have h11 := X_last_11 hn hT ids
  simp only [dotProduct, Fin.sum_univ_one, Fin.isValue, Nat.cast_one, Real.sqrt_one, div_one]
  rw [show (0 : Fin 1) = 0 from rfl, Wq_mulVec, Wk1_mulVec, h11, X_10]
  simp only [S1, mem_filter, mem_univ, true_and]
  split_ifs <;> norm_num

/-- Gathered features of the two heads. -/
noncomputable def G0 (c : Fin 10) : ℝ :=
  ∑ j, softmax (fun j => if j ∈ (S0 : Finset (Fin (n + 1))) then (0 : ℝ) else -200) j *
    X hT ids j (Fin.castLE (by norm_num) c)

noncomputable def G1 (c : Fin 10) : ℝ :=
  ∑ j, softmax (fun j => if j ∈ (S1 : Finset (Fin (n + 1))) then (100 : ℝ) else 0) j *
    X hT ids j (Fin.castLE (by norm_num) c)

include hn in
lemma attn_apply (d : Fin 32) : A0.apply (X hT ids) (Fin.last n) d =
    if h : 12 ≤ d.val ∧ d.val < 22 then G0 hT ids ⟨d.val - 12, by omega⟩
    else if h : 22 ≤ d.val then 10 * G1 hT ids ⟨d.val - 22, by omega⟩ else 0 := by
  unfold Attn.apply
  rw [Wo_mulVec]
  simp only [Equiv.symm_apply_apply]
  split_ifs with h1 h2
  · refine sum_congr rfl fun j _ => ?_
    rw [headWeight_last, Wv_mulVec]
    congr 1
    congr 1
    funext j'
    exact score0 hn hT ids j'
  · congr 1
    refine sum_congr rfl fun j _ => ?_
    rw [headWeight_last, Wv_mulVec]
    congr 1
    congr 1
    funext j'
    exact score1 hn hT ids j'
  · rfl

lemma X_unit (j : Fin (n + 1)) (c : Fin 10) :
    0 ≤ X hT ids j (Fin.castLE (by norm_num) c) ∧ X hT ids j (Fin.castLE (by norm_num) c) ≤ 1 := by
  rw [X_digit]; split_ifs <;> norm_num

/-- Number of data positions holding digit c. -/
def N (c : Fin 10) : ℕ := ((S1 : Finset (Fin (n + 1))).filter fun j => (ids j).val = c.val).card

include hn in
lemma card_S0 : (S0 : Finset (Fin (n + 1))).card = 1 ∧ (S0 : Finset (Fin (n + 1)))ᶜ.card = n := by
  rcases hn with rfl | rfl <;> decide

include hn in
lemma card_S1 : (S1 : Finset (Fin (n + 1))).card = 10 ∧ (S1 : Finset (Fin (n + 1)))ᶜ.card = n - 9 := by
  rcases hn with rfl | rfl <;> decide

include hn in
lemma G0_close (c : Fin 10) :
    |G0 hT ids c - (if (ids ⟨10, by omega⟩).val = c.val then 1 else 0)| ≤ 12 / 10 ^ 40 := by
  have hS : (S0 : Finset (Fin (n + 1))).Nonempty := ⟨⟨10, by omega⟩, by simp [S0]⟩
  have h := two_valued_dev S0 hS 0 (-200) _ (fun j => X_unit hT ids j c)
  obtain ⟨c0, c0c⟩ := card_S0 (n := n) hn
  have hsum : ∑ j ∈ (S0 : Finset (Fin (n + 1))), X hT ids j (Fin.castLE (by norm_num) c) =
      if (ids ⟨10, by omega⟩).val = c.val then 1 else 0 := by
    rw [show (S0 : Finset (Fin (n + 1))) = {⟨10, by omega⟩} by ext j; simp [S0, Fin.ext_iff],
      sum_singleton, X_digit]
  rw [hsum, c0, c0c] at h
  simp only [Nat.cast_one, div_one, sub_zero] at h
  refine h.trans ?_
  have := exp_neg200_le
  have hn12 : (n : ℝ) ≤ 12 := by rcases hn with rfl | rfl <;> norm_num
  calc (n : ℝ) * Real.exp (-200) ≤ 12 * (1 / 10 ^ 40) :=
        mul_le_mul hn12 this (Real.exp_pos _).le (by norm_num)
    _ = 12 / 10 ^ 40 := by ring

include hn in
lemma G1_close (c : Fin 10) : |10 * G1 hT ids c - N ids c| ≤ 3 / 10 ^ 40 := by
  have hS : (S1 : Finset (Fin (n + 1))).Nonempty := ⟨⟨0, by omega⟩, by simp [S1]⟩
  have h := two_valued_dev S1 hS 100 0 _ (fun j => X_unit hT ids j c)
  obtain ⟨c1, c1c⟩ := card_S1 (n := n) hn
  have hsum : ∑ j ∈ (S1 : Finset (Fin (n + 1))), X hT ids j (Fin.castLE (by norm_num) c) = N ids c := by
    simp only [X_digit, N]
    rw [sum_ite, sum_const_zero, add_zero, sum_const, nsmul_eq_mul, mul_one]
  rw [hsum, c1, c1c] at h
  have hn3 : ((n - 9 : ℕ) : ℝ) ≤ 3 := by rcases hn with rfl | rfl <;> norm_num
  have := exp_neg100_le
  have hb : ((n - 9 : ℕ) : ℝ) * Real.exp (0 - 100) / (10 : ℕ) ≤ 3 / 10 ^ 41 := by
    rw [zero_sub]
    calc ((n - 9 : ℕ) : ℝ) * Real.exp (-100) / (10 : ℕ) ≤ 3 * (1 / 10 ^ 40) / 10 := by
          gcongr <;> norm_num
      _ = 3 / 10 ^ 41 := by ring
  have h2 := h.trans hb
  have e : 10 * G1 hT ids c - N ids c = 10 * (G1 hT ids c - (N ids c : ℝ) / (10 : ℕ)) := by
    push_cast; ring
  rw [e, abs_mul]
  calc |(10 : ℝ)| * |G1 hT ids c - (N ids c : ℝ) / (10 : ℕ)| ≤ 10 * (3 / 10 ^ 41) := by
        rw [abs_of_pos (by norm_num)]; exact mul_le_mul_of_nonneg_left h2 (by norm_num)
    _ = 3 / 10 ^ 40 := by ring

include hn in
lemma N_le (c : Fin 10) : N ids c ≤ 10 := by
  have := card_filter_le (S1 : Finset (Fin (n + 1))) (fun j => (ids j).val = c.val)
  rw [(card_S1 (n := n) hn).1] at this
  exact this

lemma relu_lip' (a b : ℝ) : |relu a - relu b| ≤ |a - b| := abs_max_sub_max_le_abs a b 0
lemma relu_zero' {x : ℝ} (h : x ≤ 0) : relu x = 0 := max_eq_right h
lemma relu_self' {x : ℝ} (h : 0 ≤ x) : relu x = x := max_eq_left h

/-- Count read off by MLP 0. -/
noncomputable def cnt : ℝ := ∑ c : Fin 10, relu (10 * G1 hT ids c + 20 * G0 hT ids c - 20)

include hn in
lemma cnt_close (hq : (ids ⟨10, by omega⟩).val < 10) :
    |cnt hT ids - N ids ⟨(ids ⟨10, by omega⟩).val, hq⟩| ≤ 1 / 10 ^ 30 := by
  set q : Fin 10 := ⟨(ids ⟨10, by omega⟩).val, hq⟩
  unfold cnt
  rw [sum_eq_single q]
  · have h0 := G0_close hn hT ids q
    have h1 := G1_close hn hT ids q
    simp only [q, if_true] at h0
    have hN : (0 : ℝ) ≤ N ids q := Nat.cast_nonneg _
    calc |relu (10 * G1 hT ids q + 20 * G0 hT ids q - 20) - N ids q|
        = |relu (10 * G1 hT ids q + 20 * G0 hT ids q - 20) - relu (N ids q)| := by
          rw [relu_self' hN]
      _ ≤ |10 * G1 hT ids q + 20 * G0 hT ids q - 20 - N ids q| := relu_lip' _ _
      _ ≤ 1 / 10 ^ 30 := by
          rw [abs_le] at h0 h1 ⊢; constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]
  · intro c _ hc
    apply relu_zero'
    have h0 := G0_close hn hT ids c
    have h1 := G1_close hn hT ids c
    have hne : ¬ (ids ⟨10, by omega⟩).val = c.val := fun h => hc (Fin.ext h.symm)
    simp only [hne, if_false, sub_zero] at h0
    have hN : (N ids c : ℝ) ≤ 10 := by exact_mod_cast N_le hn ids c
    rw [abs_le] at h0 h1
    linarith [h0.1, h0.2, h1.1, h1.2]
  · simp

/-- Residual after block-0 attention at the last position. -/
noncomputable def Y : Fin 32 → ℝ := X hT ids (Fin.last n) + A0.apply (X hT ids) (Fin.last n)

include hn in
lemma Y_lo (d : Fin 32) (hd : d.val < 12) : Y hT ids d = X hT ids (Fin.last n) d := by
  simp only [Y, Pi.add_apply, attn_apply hn]
  rw [dif_neg (by omega), dif_neg (by omega), add_zero]

include hn in
lemma Y_g (c : Fin 10) : Y hT ids ⟨12 + c.val, by omega⟩ = G0 hT ids c := by
  simp only [Y, Pi.add_apply, attn_apply hn, X_hi hT ids _ ⟨12 + c.val, by omega⟩ (by simp only; omega)]
  rw [dif_pos (by simp; omega)]
  simp

include hn in
lemma Y_h (c : Fin 10) : Y hT ids ⟨22 + c.val, by omega⟩ = 10 * G1 hT ids c := by
  simp only [Y, Pi.add_apply, attn_apply hn, X_hi hT ids _ ⟨22 + c.val, by omega⟩ (by simp only; omega)]
  rw [dif_neg (by simp), dif_pos (by simp)]
  simp

include hn in
lemma cntOf_Y : cntOf (Y hT ids) = cnt hT ids := by
  unfold cntOf cnt
  refine sum_congr rfl fun c _ => ?_
  rw [Y_g hn, Y_h hn]

/-- Residual after MLP 0. -/
noncomputable def R1 : Fin 32 → ℝ := Y hT ids + countModel_block0.mlp (Y hT ids)

include hn in
lemma R1_digit (c : Fin 10) :
    R1 hT ids (Fin.castLE (by norm_num) c) = if (ids (Fin.last n)).val = c.val then 1 else 0 := by
  simp only [R1, Pi.add_apply, mlp0_eq]
  rw [if_neg (by intro h; have := congrArg Fin.val h; simp at this; omega), add_zero,
    Y_lo hn hT ids _ (by simp; omega), X_digit]

include hn in
lemma R1_10 : R1 hT ids 10 = cnt hT ids := by
  simp only [R1, Pi.add_apply, mlp0_eq, if_true, cntOf_Y hn]
  rw [Y_lo hn hT ids _ (by norm_num), X_10]
  rcases hn with rfl | rfl <;> simp

include hn in
lemma R1_11 : R1 hT ids 11 = 1 := by
  simp only [R1, Pi.add_apply, mlp0_eq]
  rw [if_neg (by decide), add_zero, Y_lo hn hT ids _ (by norm_num), X_last_11 hn]

lemma logits_last : countModel.logits hT ids (Fin.last n) =
    countModel.Wh *ᵥ (R1 hT ids + countModel_block1.mlp (R1 hT ids)) + countModel.bh := rfl

lemma logits_digit (t : Fin 11) (ht : t.val < 10) : countModel.logits hT ids (Fin.last n) t =
    R1 hT ids ⟨t.val, by omega⟩ + countModel_block1.mlp (R1 hT ids) ⟨t.val, by omega⟩ := by
  rw [logits_last, Wh_mulVec]; simp only [dif_pos ht]; rfl

lemma logits_eq_tok : countModel.logits hT ids (Fin.last n) 10 = 0 := by
  rw [logits_last, Wh_mulVec]; simp

include hn in
lemma dsum_R1 : dsum (R1 hT ids) = if (ids (Fin.last n)).val < 10 then 1 else 0 := by
  unfold dsum
  simp only [R1_digit hn]
  split_ifs with h
  · rw [sum_eq_single (⟨(ids (Fin.last n)).val, h⟩ : Fin 10)]
    · simp
    · intro b _ hb; rw [if_neg]; intro e; apply hb; ext; simp [e]
    · simp
  · rw [sum_eq_zero]; intro c _; rw [if_neg]; omega

end Last

/-! ### Step 1: the tens digit (prompt length 12) -/

/-- Logit vector at step 1 as a function of the count read-out `x`. -/
noncomputable def L1 (x : ℝ) (t : Fin 11) : ℝ := if t.val = 1 then 4 * relu (x - 59 / 2 + 20) else 0

lemma step1_logits (ids : Fin 12 → Fin 11) (hT : 11 + 1 ≤ 14) (h11 : ids 11 = 10)
    (hc : cnt hT ids ≤ 11) (t : Fin 11) :
    countModel.logits hT ids (Fin.last 11) t = L1 (cnt hT ids) t := by
  have hn : (11 = 11 ∨ 11 = 12) := Or.inl rfl
  have hd : dsum (R1 hT ids) = 0 := by rw [dsum_R1 hn, show Fin.last 11 = 11 from rfl, h11]; simp
  have nk0 : ∀ k : ℝ, 0 ≤ k → nk (R1 hT ids) k = 0 := fun k hk => by
    unfold nk; rw [hd, R1_10 hn, R1_11 hn]; exact relu_zero' (by linarith)
  by_cases ht : t.val < 10
  · rw [logits_digit hT ids t ht]
    have hR' : R1 hT ids ⟨t.val, by omega⟩ = 0 := by
      have := R1_digit hn hT ids ⟨t.val, ht⟩
      rw [show Fin.last 11 = 11 from rfl, h11] at this
      simp only [Fin.castLE] at this; rw [this]; simp; omega
    rw [hR', zero_add, mlp1_lt _ _ (by simpa using ht)]
    simp only [Fin.val_mk]
    split_ifs with h0 h1
    · rw [nk0 9 (by norm_num), nk0 10 (by norm_num)]; simp [L1, h0]
    · rw [nk0 0 le_rfl, nk0 1 (by norm_num), nk0 2 (by norm_num)]
      simp only [L1, h1, if_true, n0, hd, R1_10 hn, R1_11 hn]; ring_nf
    · have h2 : (2 : ℝ) ≤ (t.val : ℝ) := by exact_mod_cast (by omega : 2 ≤ t.val)
      rw [nk0 _ (by linarith), nk0 _ (by linarith), nk0 _ (by linarith)]
      simp [L1, h1]
  · have : t = 10 := by ext; simp; omega
    subst this
    rw [logits_eq_tok]; simp [L1]

/-! ### Step 2: the ones digit (prompt length 13) -/

/-- Bump part of the step-2 logits. -/
noncomputable def B2 (x : ℝ) (t : Fin 11) : ℝ :=
  if t.val = 0 then 2 * relu (x - 9) - 4 * relu (x - 10)
  else if t.val = 1 then 4 * relu (x - 59 / 2) + (2 * relu (x - 0) - 4 * relu (x - 1) + 2 * relu (x - 2))
  else if t.val < 10 then
    2 * relu (x - ((t.val : ℝ) - 1)) - 4 * relu (x - t.val) + 2 * relu (x - ((t.val : ℝ) + 1))
  else 0

/-- Step-2 logits: the embedded tens token `τ` plus the bumps. -/
noncomputable def L2 (x : ℝ) (τ : ℕ) (t : Fin 11) : ℝ :=
  (if t.val < 10 ∧ τ = t.val then 1 else 0) + B2 x t

lemma step2_logits (ids : Fin 13 → Fin 11) (hT : 12 + 1 ≤ 14) (hτ : (ids 12).val < 10)
    (t : Fin 11) :
    countModel.logits hT ids (Fin.last 12) t = L2 (cnt hT ids) (ids 12).val t := by
  have hn : (12 = 11 ∨ 12 = 12) := Or.inr rfl
  have hl : Fin.last 12 = 12 := rfl
  have hd : dsum (R1 hT ids) = 1 := by rw [dsum_R1 hn, hl, if_pos hτ]
  have nkv : ∀ k : ℝ, nk (R1 hT ids) k = relu (cnt hT ids - k) := fun k => by
    unfold nk; rw [hd, R1_10 hn, R1_11 hn]; ring_nf
  have n0v : n0 (R1 hT ids) = relu (cnt hT ids - 59 / 2) := by
    unfold n0; rw [hd, R1_10 hn, R1_11 hn]; ring_nf
  by_cases ht : t.val < 10
  · rw [logits_digit hT ids t ht]
    have hR' : R1 hT ids ⟨t.val, by omega⟩ = if (ids 12).val = t.val then 1 else 0 := by
      have := R1_digit hn hT ids ⟨t.val, ht⟩
      rw [hl] at this; simp only [Fin.castLE] at this; exact this
    rw [hR', mlp1_lt _ _ (by simpa using ht)]
    simp only [Fin.val_mk, L2, B2, ht, true_and]
    split_ifs <;> simp only [nkv, n0v] <;> ring_nf
  · have : t = 10 := by ext; simp; omega
    subst this
    rw [logits_eq_tok]; simp [L2, B2]

lemma B2_lip (x y : ℝ) (t : Fin 11) (hx : x ≤ 29) (hy : y ≤ 29) :
    |B2 x t - B2 y t| ≤ 8 * |x - y| := by
  have r : ∀ a : ℝ, |relu (x - a) - relu (y - a)| ≤ |x - y| := fun a => by
    have := relu_lip' (x - a) (y - a); simpa using this
  have z1 : relu (x - 59 / 2) = 0 := relu_zero' (by linarith)
  have z2 : relu (y - 59 / 2) = 0 := relu_zero' (by linarith)
  have hp := abs_nonneg (x - y)
  unfold B2
  split_ifs
  · have a := r 9; have b := r 10
    rw [abs_le] at a b ⊢; constructor <;> linarith
  · have a := r 0; have b := r 1; have c := r 2
    rw [z1, z2]
    rw [abs_le] at a b c ⊢; constructor <;> linarith
  · have a := r ((t.val : ℝ) - 1); have b := r t.val; have c := r ((t.val : ℝ) + 1)
    rw [abs_le] at a b c ⊢; constructor <;> linarith
  · simp

/-- Exact margins of the step-2 logits at an integer count. -/
lemma L2_exact (N : ℕ) (hN : N ≤ 10) (t : Fin 11) (ht : t.val ≠ N % 10) :
    L2 N (if N = 10 then 1 else 0) t + 1 ≤ L2 N (if N = 10 then 1 else 0) ⟨N % 10, by omega⟩ := by
  interval_cases N <;> fin_cases t <;> simp at ht <;> norm_num [L2, B2, relu]

/-! ### From digit lists to prompts, and the generation theorem -/

lemma prompt_length (ds : Fin 10 → Fin 10) (q : Fin 10) : (prompt ds q).length = 12 := by
  simp [prompt]

lemma prompt_get (ds : Fin 10 → Fin 10) (q : Fin 10) (ext : List (Fin 11)) (j : ℕ)
    (hj : j < (prompt ds q ++ ext).length) :
    (prompt ds q ++ ext)[j] =
      if h : j < 10 then dTok (ds ⟨j, h⟩) else if h2 : j = 10 then dTok q
      else if h3 : j = 11 then 10 else ext[j - 12]'(by simp [prompt] at hj; omega) := by
  simp only [prompt, List.getElem_append, List.length_append, List.length_ofFn, List.length_cons,
    List.length_nil, List.getElem_ofFn]
  split_ifs <;> first | rfl | omega | (subst_vars; simp) | (congr 1; omega) | simp_all

/-- Digits of the data positions, read back from a Fin-indexed prompt. -/
lemma N_eq_count {n : ℕ} (hn : n = 11 ∨ n = 12) (ids : Fin (n + 1) → Fin 11)
    (ds : Fin 10 → Fin 10) (q : Fin 10)
    (hdata : ∀ j : Fin (n + 1), (hj : j.val < 10) → ids j = dTok (ds ⟨j.val, hj⟩)) :
    N ids ⟨q.val, q.isLt⟩ = count ds q := by
  unfold N count
  have hmap : ((S1 : Finset (Fin (n + 1))).filter fun j => (ids j).val = q.val) =
      (univ.filter fun i : Fin 10 => ds i = q).map (Fin.castLEEmb (by omega)) := by
    ext j
    simp only [S1, mem_filter, mem_univ, true_and, mem_map, Fin.castLEEmb_apply]
    constructor
    · rintro ⟨hj, hv⟩
      refine ⟨⟨j.val, hj⟩, ?_, by ext; simp⟩
      rw [hdata j hj] at hv
      ext; simpa [dTok] using hv
    · rintro ⟨i, hi, rfl⟩
      refine ⟨by simp, ?_⟩
      rw [hdata _ (by simp)]
      simp [dTok, ← hi]
  rw [hmap, card_map]

theorem count_correct (ds : Fin 10 → Fin 10) (q : Fin 10) :
    countModel.generate (prompt ds q) 2 = some (answer ds q) := by
  have hans : (answer ds q).length = 2 := rfl
  rw [← hans]
  apply Model.generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, answer]
  intro k hk h0 hL
  have hkN : count ds q ≤ 10 := by
    unfold count; exact (card_filter_le _ _).trans (by simp)
  have hk2 : k < 2 := by simpa [answer] using hk
  interval_cases k
  · -- step 1: the tens digit
    have hlen : (prompt ds q ++ (answer ds q).take 0).length = 11 + 1 := by simp [prompt]
    rw [Model.next_eq countModel _ h0 hL hlen (by norm_num)
      (fun t => (prompt ds q ++ (answer ds q).take 0).get (Fin.cast hlen.symm t)) (fun t => rfl)]
    set ids := fun t : Fin 12 => (prompt ds q ++ (answer ds q).take 0).get (Fin.cast hlen.symm t)
    have hid : ∀ j : Fin 12, ids j = if h : j.val < 10 then dTok (ds ⟨j.val, h⟩)
        else if j.val = 10 then dTok q else if j.val = 11 then 10 else 0 := by
      intro j
      simp only [ids, List.get_eq_getElem, Fin.coe_cast]
      rw [prompt_get]
      all_goals (split_ifs <;> first | rfl | omega)
    have hn : (11 = 11 ∨ 11 = 12) := Or.inl rfl
    have h11 : ids 11 = 10 := by rw [hid]; rfl
    have hq : (ids ⟨10, by omega⟩).val < 10 := by rw [hid]; simp [dTok]
    have hcnt := cnt_close hn (by norm_num) ids hq
    have hNc : N ids ⟨(ids ⟨10, by omega⟩).val, hq⟩ = count ds q := by
      rw [← N_eq_count hn ids ds q (fun j hj => by rw [hid, dif_pos hj])]
      congr 1
      all_goals (ext; rw [hid]; simp [dTok])
    rw [hNc] at hcnt
    have hN : ((count ds q : ℕ) : ℝ) ≤ 10 := by exact_mod_cast hkN
    rw [abs_le] at hcnt
    have hlog := funext (step1_logits ids (by norm_num) h11 (by linarith [hcnt.2]))
    rw [hlog]
    simp only [answer, List.get_eq_getElem, List.getElem_cons_zero]
    split_ifs with h10
    · apply argmaxFirst_eq_of_lt
      intro j hj
      have hj1 : j.val ≠ 1 := fun e => hj (Fin.ext e)
      rw [h10] at hcnt
      simp only [L1, hj1, if_false, show ((1 : Fin 11).val = 1) from rfl, if_true]
      rw [relu_self' (by push_cast at hcnt; linarith [hcnt.1])]
      push_cast at hcnt; linarith [hcnt.1]
    · apply argmaxFirst_eq_zero
      have hlt : count ds q ≤ 9 := by omega
      have hN9 : ((count ds q : ℕ) : ℝ) ≤ 9 := by exact_mod_cast hlt
      intro j
      simp only [L1]
      rw [relu_zero' (by linarith [hcnt.2])]
      split_ifs <;> simp
  · -- step 2: the ones digit
    have hlen : (prompt ds q ++ (answer ds q).take 1).length = 12 + 1 := by simp [prompt, answer]
    rw [Model.next_eq countModel _ h0 hL hlen (by norm_num)
      (fun t => (prompt ds q ++ (answer ds q).take 1).get (Fin.cast hlen.symm t)) (fun t => rfl)]
    set ids := fun t : Fin 13 => (prompt ds q ++ (answer ds q).take 1).get (Fin.cast hlen.symm t)
    set τ : Fin 11 := if count ds q = 10 then 1 else 0 with hτdef
    have hid : ∀ j : Fin 13, ids j = if h : j.val < 10 then dTok (ds ⟨j.val, h⟩)
        else if j.val = 10 then dTok q else if j.val = 11 then 10 else τ := by
      intro j
      simp only [ids, List.get_eq_getElem, Fin.coe_cast]
      rw [prompt_get]
      all_goals (split_ifs <;> first | rfl | omega | (simp [answer, τ]))
    have hn : (12 = 11 ∨ 12 = 12) := Or.inr rfl
    have h12 : ids 12 = τ := by rw [hid]; simp
    have hτv : (ids 12).val = if count ds q = 10 then 1 else 0 := by
      rw [h12]; simp only [τ]; split_ifs <;> rfl
    have hτ : (ids 12).val < 10 := by rw [hτv]; split_ifs <;> norm_num
    have hq : (ids ⟨10, by omega⟩).val < 10 := by rw [hid]; simp [dTok]
    have hcnt := cnt_close hn (by norm_num) ids hq
    have hNc : N ids ⟨(ids ⟨10, by omega⟩).val, hq⟩ = count ds q := by
      rw [← N_eq_count hn ids ds q (fun j hj => by rw [hid, dif_pos hj])]
      congr 1
      all_goals (ext; rw [hid]; simp [dTok])
    rw [hNc] at hcnt
    have hN : ((count ds q : ℕ) : ℝ) ≤ 10 := by exact_mod_cast hkN
    have hlog := funext (step2_logits ids (by norm_num) hτ)
    rw [hlog, hτv]
    simp only [answer, List.get_eq_getElem, List.getElem_cons_succ, List.getElem_cons_zero]
    apply argmaxFirst_eq_of_lt
    intro j hj
    have hjv : j.val ≠ count ds q % 10 := fun e => hj (Fin.ext e)
    have hex := L2_exact (count ds q) hkN j hjv
    have hc29 : cnt (by norm_num) ids ≤ 29 := by rw [abs_le] at hcnt; linarith [hcnt.2]
    have l1 := B2_lip (cnt (by norm_num) ids) (count ds q) j hc29 (by linarith)
    have l2 := B2_lip (cnt (by norm_num) ids) (count ds q) ⟨count ds q % 10, by omega⟩ hc29
      (by linarith)
    have e8 : 8 * |cnt (by norm_num) ids - (count ds q : ℝ)| ≤ 8 / 10 ^ 30 := by linarith [hcnt]
    unfold L2 at hex ⊢
    rw [abs_le] at l1 l2
    linarith [l1.1, l1.2, l2.1, l2.2]

end VWC.Arch.Count
