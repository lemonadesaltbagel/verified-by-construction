/-
linear-interpolation-two-points (snapshot LinInterpTwoPointAND_v4): real-valued correctness of
greedy generation on every input (45 000 prompts), by mechanism.

* Layer 0, heads 0–6: the query at positions 11/12 has 1 on component 0, the key at the head's
  source position has 50, so each head copies the digit at its source with leak λ ≤ 1/100.
  The seven gathered digits land in slots of width 10.
* Layer 0 MLP: detector (pair P, line L, x) sums the three slots of pair P and subtracts 5/2.
  It outputs ≈ 1/2 iff the pair's point is (x, line L at x), and 0 otherwise. So the score of
  line L is ≈ 1/2 per given point it passes through: ≈ 1 for the unique line through both
  points (x1 ≠ x2), ≤ 1/2 for every other line.
* Layer 1 MLP: lookup (L, xq, pos) adds score + xq-slot + position one-hot and subtracts 11/4.
  It fires (≈ 1/4) only for the true line, the true xq and the current position, and writes 100
  into the answer digit. The head is the identity on token dims, so that digit's logit is
  ≥ 18 and every other logit is ≤ 1.
-/
import VWC.Tasks.LiStruct

namespace VWC.Arch.LI

open Matrix Finset

/-! ### Specification (src/task.py, LinearInterpolationTask). Vocab "0123456789,;=". -/

/-- Digit token (`,` = 10, `;` = 11, `=` = 12). -/
def dtok (d : ℕ) : Fin 13 := ⟨d % 10, by omega⟩

/-- Line value at x for slope `m + 1 ∈ [1, 5]` and intercept `b ∈ [0, 9]`. -/
def yv (m b x : ℕ) : ℕ := (m + 1) * x + b

/-- Prompt "X1,Y1;X2,Y2;XQ=" with two-digit y values. -/
def prompt (m : Fin 5) (b x1 x2 xq : Fin 10) : List (Fin 13) :=
  [dtok x1, 10, dtok (yv m b x1 / 10), dtok (yv m b x1), 11,
   dtok x2, 10, dtok (yv m b x2 / 10), dtok (yv m b x2), 11, dtok xq, 12]

def answer (m : Fin 5) (b x1 x2 xq : Fin 10) : List (Fin 13) :=
  [dtok (yv m b xq / 10), dtok (yv m b xq)]

lemma yv_lt (m b x : ℕ) (hm : m < 5) (hb : b < 10) (hx : x < 10) : yv m b x < 60 := by
  unfold yv
  have := Nat.mul_le_mul (show m + 1 ≤ 5 by omega) (show x ≤ 9 by omega)
  omega

/-! ### Generic helpers -/

lemma fpe_val (h : Fin 8) (c : Fin 19) : (finProdFinEquiv (h, c)).val = c.val + 19 * h.val := by
  simp [finProdFinEquiv]

lemma fpe_symm_val (r : Fin (8 * 19)) :
    ((finProdFinEquiv.symm r).1).val = r.val / 19 ∧ ((finProdFinEquiv.symm r).2).val = r.val % 19 := by
  have h := fpe_val (finProdFinEquiv.symm r).1 (finProdFinEquiv.symm r).2
  rw [Prod.mk.eta, Equiv.apply_symm_apply] at h
  have h1 := ((finProdFinEquiv.symm r).2).isLt
  constructor <;> omega

lemma exp_neg11 : Real.exp (-11) ≤ 1 / 10000 := by
  have he := Real.exp_one_gt_d9
  have h : (2.7182818283 : ℝ) ^ 11 ≤ Real.exp 11 := by
    rw [show (11 : ℝ) = (11 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) he.le 11
  rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 11) (by norm_num)]
  calc (1 / 10000 : ℝ)⁻¹ ≤ (2.7182818283 : ℝ) ^ 11 := by norm_num
    _ ≤ _ := h

lemma gap_ge : (11 : ℝ) ≤ 50 / Real.sqrt 19 := by
  have hs : Real.sqrt 19 ≤ 50 / 11 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  have hs0 : 0 < Real.sqrt 19 := Real.sqrt_pos.2 (by norm_num)
  rw [le_div_iff₀ hs0]; nlinarith

/-- The relu of a three-slot detector. -/
lemma relu_three (a b c θ : ℝ) (ha : a ≤ 1) (hb : b ≤ 1) (hc : c ≤ 1) :
    relu (a + b + c - θ) ≤ max (3 - θ) 0 := by
  unfold relu; exact max_le_max (by linarith) le_rfl


/-- Two points with distinct x determine the line. -/
lemma line_unique (L m b x1 x2 : ℕ) (hL : L < 50) (hx : x1 ≠ x2)
    (h1 : (L / 10 + 1) * x1 + L % 10 = (m + 1) * x1 + b)
    (h2 : (L / 10 + 1) * x2 + L % 10 = (m + 1) * x2 + b) : L = m * 10 + b := by
  have hL' : L = L / 10 * 10 + L % 10 := by omega
  generalize hq : L / 10 = q at h1 h2 hL'
  generalize hr : L % 10 = r at h1 h2 hL'
  have hm : q = m := by
    zify at h1 h2
    have e : ((q : ℤ) - m) * ((x1 : ℤ) - x2) = 0 := by linear_combination h1 - h2
    rcases mul_eq_zero.1 e with h | h
    · exact_mod_cast (by linarith : (q : ℤ) = m)
    · exact absurd (by linarith : (x1 : ℤ) = x2) (by exact_mod_cast hx)
  subst hm
  have : r = b := by
    have := h1; rw [show (q + 1) * x1 + r = (q + 1) * x1 + b ↔ r = b by omega] at this
    exact this
  omega

/-- Detector k = P·500 + L·10 + x reads line L at x. -/
lemma lineY_eq (P L x : ℕ) (hP : P < 2) (hL : L < 50) (hx : x < 10) :
    lineY (P * 500 + L * 10 + x) = (L / 10 + 1) * x + L % 10 := by
  unfold lineY
  rw [show (P * 500 + L * 10 + x) % 500 / 100 = L / 10 by omega,
    show (P * 500 + L * 10 + x) % 10 = x by omega,
    show (P * 500 + L * 10 + x) % 100 / 10 = L % 10 by omega]

lemma lineQ_eq (L xq e : ℕ) (hL : L < 50) (hx : xq < 10) (he : e < 2) :
    lineQ (L * 20 + xq * 2 + e) = (L / 10 + 1) * xq + L % 10 := by
  unfold lineQ
  rw [show (L * 20 + xq * 2 + e) / 200 = L / 10 by omega,
    show (L * 20 + xq * 2 + e) % 20 / 2 = xq by omega,
    show (L * 20 + xq * 2 + e) / 20 % 10 = L % 10 by omega]


/-- The seven gathered digits of a prompt (x1, y1 tens, y1 ones, x2, y2 tens, y2 ones, xq). -/
def Dl (m b x1 x2 xq : ℕ) (h : ℕ) : ℕ :=
  if h = 0 then x1 else if h = 1 then yv m b x1 / 10 else if h = 2 then yv m b x1 % 10
  else if h = 3 then x2 else if h = 4 then yv m b x2 / 10 else if h = 5 then yv m b x2 % 10 else xq

section Step

variable {n : ℕ} (hT : n + 1 ≤ 16) (ids : Fin (n + 1) → Fin 13)

/-- Embedded prompt. -/
noncomputable def X (t : Fin (n + 1)) : Fin 152 → ℝ :=
  liModel.tokEmb (ids t) + liModel.posEmb (Fin.castLE hT t)

lemma X_apply (t : Fin (n + 1)) (i : Fin 152) :
    X hT ids t i = (if i.val = (ids t).val then 1 else 0) + (if i.val = 13 + t.val then 1 else 0) := by
  simp only [X, Pi.add_apply, tokEmb_eq, posEmb_eq, Pi.single_apply, Fin.ext_iff, Fin.val_castLE]

/-- Away from token and position dims the embedding is zero. -/
lemma X_hi (t : Fin (n + 1)) (i : Fin 152) (hi : 29 ≤ i.val) : X hT ids t i = 0 := by
  rw [X_apply]
  have := (ids t).isLt
  rw [if_neg (by omega), if_neg (by omega)]; ring

lemma X_tok (t : Fin (n + 1)) (i : Fin 152) (hi : i.val < 13) :
    X hT ids t i = if i.val = (ids t).val then 1 else 0 := by
  rw [X_apply, if_neg (show ¬ i.val = 13 + t.val by omega), add_zero]

lemma X_pos (t : Fin (n + 1)) (s : ℕ) (hs : 13 + s < 152) :
    X hT ids t ⟨13 + s, hs⟩ = if s = t.val then 1 else 0 := by
  rw [X_apply]
  have := (ids t).isLt
  rw [if_neg (by simp; omega), zero_add]
  simp

/-! ### Layer-0 attention -/

/-- Head h's weights at the last position. -/
noncomputable def Wt (h : Fin 8) (j : Fin (n + 1)) : ℝ :=
  headWeight (fun t c => (A0.Wq *ᵥ X hT ids t) (finProdFinEquiv (h, c)))
    (fun t c => (A0.Wk *ᵥ X hT ids t) (finProdFinEquiv (h, c))) (Fin.last n) j

variable (hn : n = 11 ∨ n = 12)

include hn in
lemma qk_dot (h : Fin 8) (hh : h.val < 7) (j : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ X hT ids (Fin.last n)) (finProdFinEquiv (h, c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ X hT ids j) (finProdFinEquiv (h, c)))) =
      if j.val = src h.val then 50 else 0 := by
  simp only [dotProduct, Wq_apply, Wk_apply, fpe_val]
  rw [sum_eq_single (0 : Fin 19)]
  · simp only [Fin.val_zero, zero_add, Nat.mul_mod_right, Nat.mul_div_cancel_left _ (by norm_num : 0 < 19),
      hh, and_self, if_true]
    have h24 : X hT ids (Fin.last n) 24 + X hT ids (Fin.last n) 25 = 1 := by
      have e1 := X_pos hT ids (Fin.last n) 11 (by norm_num)
      have e2 := X_pos hT ids (Fin.last n) 12 (by norm_num)
      simp only [Fin.val_last] at e1 e2
      rw [show (24 : Fin 152) = ⟨13 + 11, by norm_num⟩ from rfl,
        show (25 : Fin 152) = ⟨13 + 12, by norm_num⟩ from rfl, e1, e2]
      rcases hn with h | h <;> simp [h]
    rw [h24, X_pos, one_mul]
    split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; omega) | ring
  · intro c _ hc
    have : c.val ≠ 0 := fun h' => hc (Fin.ext h')
    have : ¬ ((c.val + 19 * h.val) % 19 = 0 ∧ (c.val + 19 * h.val) / 19 < 7) := by
      have := c.isLt; omega
    rw [if_neg this, zero_mul]
  · simp

include hn in
lemma Wt_softmax (h : Fin 8) (hh : h.val < 7) (j : Fin (n + 1)) :
    Wt hT ids h j = softmax (fun j : Fin (n + 1) =>
      (if j.val = src h.val then (50 : ℝ) else 0) / Real.sqrt 19) j := by
  unfold Wt
  rw [headWeight_last]
  congr 1
  funext j'
  rw [qk_dot hT ids hn h hh, Nat.cast_ofNat]

lemma Wt_nonneg_of (h : Fin 8) (j : Fin (n + 1)) : 0 ≤ Wt hT ids h j := by
  unfold Wt headWeight
  split_ifs
  · exact div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)
  · exact le_rfl

include hn in
lemma Wt_sum (h : Fin 8) (hh : h.val < 7) : ∑ j, Wt hT ids h j = 1 := by
  simp only [Wt_softmax hT ids hn h hh]; exact softmax_sum _

include hn in
/-- Gather heads copy their source token with leak ≤ 1/100. -/
lemma leak (h : Fin 8) (hh : h.val < 7) :
    1 - Wt hT ids h ⟨src h.val, by have := src_le h.val; omega⟩ ≤ 1 / 100 := by
  rw [Wt_softmax hT ids hn h hh]
  refine (one_sub_softmax_le _ _).trans ?_
  have hb : ∀ j : Fin (n + 1), (if j = (⟨src h.val, by have := src_le h.val; omega⟩ : Fin (n + 1))
      then (0 : ℝ) else Real.exp ((if j.val = src h.val then (50 : ℝ) else 0) / Real.sqrt 19 -
        (if src h.val = src h.val then (50 : ℝ) else 0) / Real.sqrt 19)) ≤ 1 / 10000 := by
    intro j
    by_cases h1 : j = (⟨src h.val, by have := src_le h.val; omega⟩ : Fin (n + 1))
    · rw [if_pos h1]; norm_num
    · rw [if_neg h1]
      have h2 : ¬ j.val = src h.val := fun h' => h1 (Fin.ext h')
      rw [if_neg h2, if_pos rfl, zero_div, zero_sub]
      refine le_trans (Real.exp_le_exp.2 ?_) exp_neg11
      linarith [gap_ge]
  calc ∑ j : Fin (n + 1), _ ≤ ∑ _j : Fin (n + 1), (1 / 10000 : ℝ) := sum_le_sum fun j _ => hb j
    _ = (n + 1) * (1 / 10000) := by simp
    _ ≤ 1 / 100 := by
        have : (n : ℝ) ≤ 12 := by rcases hn with h | h <;> norm_num [h]
        nlinarith

/-- Slot value: head h's average of the digit-d indicator. -/
noncomputable def S (h : Fin 8) (d : ℕ) : ℝ :=
  ∑ j, Wt hT ids h j * (if (ids j).val = d then 1 else 0)

lemma attn_slot (i : Fin 152) (hi : 29 ≤ i.val ∧ i.val < 99) :
    A0.apply (fun t => X hT ids t) (Fin.last n) i =
      S hT ids ⟨(i.val - 29) / 10, by omega⟩ ((i.val - 29) % 10) := by
  simp only [Attn.apply, Wo_apply, dif_pos hi]
  unfold S Wt
  refine sum_congr rfl fun j _ => ?_
  have hv := fpe_symm_val (⟨(i.val - 29) / 10 * 19 + 1 + (i.val - 29) % 10, by omega⟩ : Fin (8 * 19))
  simp only at hv
  have e1 : (finProdFinEquiv.symm (⟨(i.val - 29) / 10 * 19 + 1 + (i.val - 29) % 10, by omega⟩ :
      Fin (8 * 19))).1 = ⟨(i.val - 29) / 10, by omega⟩ := Fin.ext (by rw [hv.1]; simp only [Fin.val_mk]; omega)
  rw [e1]
  congr 1
  rw [Wv_apply, dif_pos (by rw [fpe_val]; omega), X_tok _ _ _ _ (by simp only; rw [fpe_val]; omega)]
  simp only [fpe_val, hv.2]
  congr 1
  apply propext; constructor <;> intro h <;> omega

lemma attn_other (i : Fin 152) (hi : ¬ (29 ≤ i.val ∧ i.val < 99)) :
    A0.apply (fun t => X hT ids t) (Fin.last n) i = 0 := by
  simp only [Attn.apply, Wo_apply, dif_neg hi]

/-! ### Gathered slots -/

variable (D : ℕ → ℕ)
  (hD : ∀ h : Fin 8, (hh : h.val < 7) → (ids ⟨src h.val, by have := src_le h.val; omega⟩).val = D h.val)

include hn hD in
lemma S_bounds (h : Fin 8) (hh : h.val < 7) (d : ℕ) :
    0 ≤ S hT ids h d ∧ S hT ids h d ≤ 1 ∧ (d = D h.val → 1 - 1 / 100 ≤ S hT ids h d) ∧
      (d ≠ D h.val → S hT ids h d ≤ 1 / 100) := by
  have hw := fun j => Wt_nonneg_of hT ids h j
  have hsum := Wt_sum hT ids hn h hh
  have hl := leak hT ids hn h hh
  set js : Fin (n + 1) := ⟨src h.val, by have := src_le h.val; omega⟩
  have hind : ∀ j : Fin (n + 1), (0 : ℝ) ≤ (if (ids j).val = d then 1 else 0) ∧
      (if (ids j).val = d then (1 : ℝ) else 0) ≤ 1 := fun j => by split_ifs <;> norm_num
  refine ⟨sum_nonneg fun j _ => mul_nonneg (hw j) (hind j).1, ?_, ?_, ?_⟩
  · calc S hT ids h d ≤ ∑ j, Wt hT ids h j := sum_le_sum fun j _ => by
          have := mul_le_mul_of_nonneg_left (hind j).2 (hw j); linarith
      _ = 1 := hsum
  · intro hd
    have : Wt hT ids h js * 1 ≤ S hT ids h d := by
      have e : (if (ids js).val = d then (1 : ℝ) else 0) = 1 := by rw [if_pos (by rw [hD h hh, hd])]
      rw [← e]
      exact single_le_sum (f := fun j => Wt hT ids h j * (if (ids j).val = d then 1 else 0))
        (fun j _ => mul_nonneg (hw j) (hind j).1) (mem_univ js)
    linarith
  · intro hd
    have e0 : Wt hT ids h js * (if (ids js).val = d then 1 else 0) = 0 := by
      rw [if_neg (by rw [hD h hh]; exact fun h' => hd h'.symm), mul_zero]
    have := sum_erase_add univ (fun j => Wt hT ids h j * (if (ids j).val = d then 1 else 0))
      (mem_univ js)
    have hs2 := sum_erase_add univ (Wt hT ids h) (mem_univ js)
    rw [hsum] at hs2
    have : ∑ j ∈ univ.erase js, Wt hT ids h j * (if (ids j).val = d then 1 else 0) ≤
        ∑ j ∈ univ.erase js, Wt hT ids h j := sum_le_sum fun j _ => by
      have := mul_le_mul_of_nonneg_left (hind j).2 (hw j); linarith
    unfold S; linarith

/-! ### Residual after attention, and the layer-0 detectors -/

noncomputable def Y1 : Fin 152 → ℝ := fun i =>
  X hT ids (Fin.last n) i + A0.apply (fun t => X hT ids t) (Fin.last n) i

lemma Y1_slot (u : ℕ) (hu : 29 ≤ u ∧ u < 99) :
    Y1 hT ids ⟨u, by omega⟩ = S hT ids ⟨(u - 29) / 10, by omega⟩ ((u - 29) % 10) := by
  have h := attn_slot hT ids ⟨u, by omega⟩ hu
  simp only [Y1]
  rw [X_hi hT ids (Fin.last n) ⟨u, by omega⟩ hu.1, zero_add, h]

lemma Y1_low (i : Fin 152) (hi : i.val < 29) : Y1 hT ids i = X hT ids (Fin.last n) i := by
  simp only [Y1, attn_other hT ids i (by omega), add_zero]

lemma Y1_hi (i : Fin 152) (hi : 99 ≤ i.val) : Y1 hT ids i = 0 := by
  simp only [Y1]
  rw [X_hi hT ids (Fin.last n) i (by omega), attn_other hT ids i (by omega), add_zero]

/-- Detector output. -/
noncomputable def r0 (k : Fin 1000) : ℝ :=
  relu ((liModel_block0_W1 *ᵥ Y1 hT ids) k + liModel_block0_b1 k)

/-- Detector k fires its pair's point: its x, and its line's tens and ones digits at x. -/
def fires (k : ℕ) : Prop :=
  k % 10 = D (3 * (k / 500)) ∧ lineY k / 10 = D (3 * (k / 500) + 1) ∧
    lineY k % 10 = D (3 * (k / 500) + 2)

lemma r0_eq (k : Fin 1000) : r0 hT ids k = relu (
    S hT ids ⟨3 * (k.val / 500), by omega⟩ (k.val % 10) +
    S hT ids ⟨3 * (k.val / 500) + 1, by omega⟩ (lineY k.val / 10) +
    S hT ids ⟨3 * (k.val / 500) + 2, by omega⟩ (lineY k.val % 10) - 5 / 2) := by
  have hk := k.isLt
  have hy := lineY_lt k.val
  have e1 : Y1 hT ids ⟨cx k, cx_lt k⟩ = S hT ids ⟨3 * (k.val / 500), by omega⟩ (k.val % 10) := by
    rw [Y1_slot hT ids (cx k) (by unfold cx; split_ifs <;> omega)]
    congr 1 <;> [ext; skip] <;> simp only [cx] <;> split_ifs <;> omega
  have e2 : Y1 hT ids ⟨cyt k, cyt_lt k⟩ = S hT ids ⟨3 * (k.val / 500) + 1, by omega⟩ (lineY k.val / 10) := by
    rw [Y1_slot hT ids (cyt k) (by unfold cyt; split_ifs <;> omega)]
    congr 1 <;> [ext; skip] <;> simp only [cyt] <;> split_ifs <;> omega
  have e3 : Y1 hT ids ⟨cyo k, cyo_lt k⟩ = S hT ids ⟨3 * (k.val / 500) + 2, by omega⟩ (lineY k.val % 10) := by
    rw [Y1_slot hT ids (cyo k) (by unfold cyo; split_ifs <;> omega)]
    congr 1 <;> [ext; skip] <;> simp only [cyo] <;> split_ifs <;> omega
  unfold r0
  rw [mulVec, W1a_row, add_dotProduct, add_dotProduct, single_dotProduct, single_dotProduct,
    single_dotProduct, b1a, e1, e2, e3]
  ring_nf

include hn hD in
lemma r0_bounds (k : Fin 1000) :
    0 ≤ r0 hT ids k ∧ r0 hT ids k ≤ 1 / 2 ∧ (fires D k.val → 1 / 2 - 3 / 100 ≤ r0 hT ids k) ∧
      (¬ fires D k.val → r0 hT ids k = 0) := by
  have hk := k.isLt
  have b1 := S_bounds hT ids hn D hD ⟨3 * (k.val / 500), by omega⟩ (by simp; omega) (k.val % 10)
  have b2 := S_bounds hT ids hn D hD ⟨3 * (k.val / 500) + 1, by omega⟩ (by simp; omega) (lineY k.val / 10)
  have b3 := S_bounds hT ids hn D hD ⟨3 * (k.val / 500) + 2, by omega⟩ (by simp; omega) (lineY k.val % 10)
  simp only at b1 b2 b3
  rw [r0_eq]
  unfold relu
  refine ⟨le_max_right _ _, max_le (by linarith [b1.2.1, b2.2.1, b3.2.1]) (by norm_num), ?_, ?_⟩
  · rintro ⟨m1, m2, m3⟩
    exact le_trans (by linarith [b1.2.2.1 m1, b2.2.2.1 m2, b3.2.2.1 m3]) (le_max_left _ _)
  · intro hm
    apply max_eq_right
    by_cases m1 : k.val % 10 = D (3 * (k.val / 500))
    · by_cases m2 : lineY k.val / 10 = D (3 * (k.val / 500) + 1)
      · have m3 : lineY k.val % 10 ≠ D (3 * (k.val / 500) + 2) := fun h => hm ⟨m1, m2, h⟩
        linarith [b1.2.1, b2.2.1, b3.2.2.2 m3]
      · linarith [b1.2.1, b2.2.2.2 m2, b3.2.1]
    · linarith [b1.2.2.2 m1, b2.2.1, b3.2.1]

/-! ### Line scores -/

lemma W2a_mulVec (v : Fin 1000 → ℝ) (i : Fin 152) :
    (liModel_block0_W2 *ᵥ v) i = ∑ k : Fin 1000, if mbc k.val = i.val then v k else 0 := by
  simp only [mulVec, dotProduct]
  refine sum_congr rfl fun k _ => ?_
  have := congrFun (W2a_col k) i
  rw [this, Pi.single_apply]
  by_cases h : i = ⟨mbc k, mbc_lt k⟩
  · rw [if_pos h, if_pos (by rw [h]), one_mul]
  · rw [if_neg h, if_neg (fun h' => h (Fin.ext h'.symm)), zero_mul]

lemma W2b_mulVec (v : Fin 1000 → ℝ) (i : Fin 152) :
    (liModel_block1_W2 *ᵥ v) i = ∑ k : Fin 1000, if dig k.val = i.val then 100 * v k else 0 := by
  simp only [mulVec, dotProduct]
  refine sum_congr rfl fun k _ => ?_
  have := congrFun (W2b_col k) i
  rw [this, Pi.single_apply]
  by_cases h : i = ⟨dig k, by have := dig_lt k k.isLt; omega⟩
  · rw [if_pos h, if_pos (by rw [h])]
  · rw [if_neg h, if_neg (fun h' => h (Fin.ext h'.symm)), zero_mul]

/-- Residual after block 0. -/
noncomputable def Y2 : Fin 152 → ℝ := Y1 hT ids + liModel_block0.mlp (Y1 hT ids)

lemma Y2_apply (i : Fin 152) : Y2 hT ids i =
    Y1 hT ids i + ∑ k : Fin 1000, if mbc k.val = i.val then r0 hT ids k else 0 := by
  simp only [Y2, Block.mlp, liModel_block0, Pi.add_apply, b2_zero.1, Pi.zero_apply, add_zero,
    W2a_mulVec]
  rfl

lemma Y2_low (i : Fin 152) (hi : i.val < 99) : Y2 hT ids i = Y1 hT ids i := by
  rw [Y2_apply, sum_eq_zero fun k _ => by rw [if_neg (by unfold mbc; omega)], add_zero]

/-- Score of line L. -/
noncomputable def sc (L : ℕ) : ℝ := ∑ k : Fin 1000, if mbc k.val = 99 + L then r0 hT ids k else 0

lemma Y2_score (L : ℕ) (hL : L < 50) : Y2 hT ids ⟨99 + L, by omega⟩ = sc hT ids L := by
  rw [Y2_apply, Y1_hi hT ids _ (by simp), zero_add]; rfl

variable (m b x1 x2 xq : ℕ) (hm : m < 5) (hb : b < 10) (h1 : x1 < 10) (h2 : x2 < 10)
  (hq : xq < 10) (hx : x1 ≠ x2)

/-- A detector of line L fires iff it is at its pair's x and the line passes the pair's point. -/
lemma matches_iff (P L x : ℕ) (hP : P < 2) (hL : L < 50) (hx' : x < 10)
    (hm : m < 5) (hb : b < 10) (h1 : x1 < 10) (h2 : x2 < 10) :
    fires (Dl m b x1 x2 xq) (P * 500 + L * 10 + x) ↔
      x = (if P = 0 then x1 else x2) ∧
        (L / 10 + 1) * x + L % 10 = yv m b (if P = 0 then x1 else x2) := by
  unfold fires
  rw [show (P * 500 + L * 10 + x) / 500 = P by omega, show (P * 500 + L * 10 + x) % 10 = x by omega,
    lineY_eq P L x hP hL hx']
  have hy := yv_lt m b (if P = 0 then x1 else x2) hm hb (by split_ifs <;> omega)
  have hl : (L / 10 + 1) * x + L % 10 < 60 := by
    have := Nat.mul_le_mul (show L / 10 + 1 ≤ 5 by omega) (show x ≤ 9 by omega); omega
  interval_cases P <;> simp [Dl] <;> omega

/-- Decomposition of a detector index. -/
lemma k_decomp (k : ℕ) (hk : k < 1000) :
    k = k / 500 * 500 + k % 500 / 10 * 10 + k % 10 ∧ k / 500 < 2 ∧ k % 500 / 10 < 50 := by
  omega

include hn hD hm hb h1 h2 hq hx in
lemma sc_true (hDl : D = Dl m b x1 x2 xq) :
    1 - 6 / 100 ≤ sc hT ids (m * 10 + b) ∧ sc hT ids (m * 10 + b) ≤ 1 := by
  subst hDl
  have hL : m * 10 + b < 50 := by omega
  set k1 : Fin 1000 := ⟨(m * 10 + b) * 10 + x1, by omega⟩
  set k2 : Fin 1000 := ⟨500 + (m * 10 + b) * 10 + x2, by omega⟩
  have hm1 : fires (Dl m b x1 x2 xq) k1.val := by
    have e := matches_iff m b x1 x2 xq 0 (m * 10 + b) x1 (by norm_num) hL h1 hm hb h1 h2
    simp only [zero_mul, zero_add, ↓reduceIte, true_and] at e
    exact e.2 (by rw [show (m * 10 + b) / 10 = m by omega, show (m * 10 + b) % 10 = b by omega]; rfl)
  have hm2 : fires (Dl m b x1 x2 xq) k2.val := by
    have e := matches_iff m b x1 x2 xq 1 (m * 10 + b) x2 (by norm_num) hL h2 hm hb h1 h2
    simp only [one_mul, one_ne_zero, ↓reduceIte, true_and] at e
    exact e.2 (by rw [show (m * 10 + b) / 10 = m by omega, show (m * 10 + b) % 10 = b by omega]; rfl)
  have hk12 : k1 ≠ k2 := by intro h; have := congrArg Fin.val h; simp [k1, k2] at this; omega
  have hsum : sc hT ids (m * 10 + b) = r0 hT ids k1 + r0 hT ids k2 := by
    unfold sc
    rw [sum_eq_add k1 k2 hk12]
    · rw [if_pos (by simp [k1, mbc]; omega), if_pos (by simp [k2, mbc]; omega)]
    · intro k _ ⟨hk1, hk2⟩
      split_ifs with hmb
      · obtain ⟨e, hP, hL'⟩ := k_decomp k.val k.isLt
        have hLk : k.val % 500 / 10 = m * 10 + b := by unfold mbc at hmb; omega
        apply (r0_bounds hT ids hn _ hD k).2.2.2
        intro hmk
        rw [e, hLk] at hmk
        rw [matches_iff m b x1 x2 xq _ _ _ hP hL (by omega) hm hb h1 h2] at hmk
        obtain ⟨hxk, -⟩ := hmk
        rcases (show k.val / 500 = 0 ∨ k.val / 500 = 1 by omega) with hP0 | hP0
        · rw [if_pos hP0] at hxk; exact hk1 (Fin.ext (by simp [k1]; omega))
        · rw [if_neg (by omega)] at hxk; exact hk2 (Fin.ext (by simp [k2]; omega))
      · rfl
    · simp
    · simp
  have b1 := r0_bounds hT ids hn _ hD k1
  have b2 := r0_bounds hT ids hn _ hD k2
  rw [hsum]
  constructor
  · linarith [b1.2.2.1 hm1, b2.2.2.1 hm2]
  · linarith [b1.2.1, b2.2.1]

include hn hD hm hb h1 h2 hq hx in
lemma sc_other (hDl : D = Dl m b x1 x2 xq) (L : ℕ) (hL : L < 50) (hne : L ≠ m * 10 + b) :
    sc hT ids L ≤ 1 / 2 := by
  subst hDl
  have hb0 := fun k => r0_bounds hT ids hn _ hD k
  -- the candidate detector: on pair 1 if line L passes point 1, otherwise on pair 2
  have hpass : ∀ P, P < 2 → (L / 10 + 1) * (if P = 0 then x1 else x2) + L % 10 =
      yv m b (if P = 0 then x1 else x2) → ∀ P', P' < 2 → (L / 10 + 1) * (if P' = 0 then x1 else x2) +
        L % 10 = yv m b (if P' = 0 then x1 else x2) → P = P' := by
    intro P hP e1 P' hP' e2
    by_contra hPP
    have : L = m * 10 + b := by
      rcases (show P = 0 ∧ P' = 1 ∨ P = 1 ∧ P' = 0 by omega) with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact line_unique L m b x1 x2 hL hx (by simpa [yv] using e1) (by simpa [yv] using e2)
      · exact line_unique L m b x1 x2 hL hx (by simpa [yv] using e2) (by simpa [yv] using e1)
    exact hne this
  set P0 : ℕ := if (L / 10 + 1) * x1 + L % 10 = yv m b x1 then 0 else 1
  set k0 : Fin 1000 := ⟨P0 * 500 + L * 10 + (if P0 = 0 then x1 else x2), by
    have : P0 ≤ 1 := by simp only [P0]; split_ifs <;> omega
    split_ifs <;> omega⟩
  unfold sc
  rw [sum_eq_single k0]
  · split_ifs
    · exact (hb0 k0).2.1
    · norm_num
  · intro k _ hk
    split_ifs with hmb
    · obtain ⟨e, hP, -⟩ := k_decomp k.val k.isLt
      have hLk : k.val % 500 / 10 = L := by unfold mbc at hmb; omega
      apply (hb0 k).2.2.2
      intro hmk
      rw [e, hLk] at hmk
      rw [matches_iff m b x1 x2 xq _ _ _ hP hL (by omega) hm hb h1 h2] at hmk
      obtain ⟨hxk, hyk⟩ := hmk
      rw [hxk] at hyk
      have hP0 : P0 = k.val / 500 := by
        by_cases hp1 : (L / 10 + 1) * x1 + L % 10 = yv m b x1
        · simp only [P0, if_pos hp1]
          exact hpass 0 (by norm_num) (by simpa using hp1) _ hP hyk
        · simp only [P0, if_neg hp1]
          rcases (show k.val / 500 = 0 ∨ k.val / 500 = 1 by omega) with h0 | h0
          · rw [h0] at hyk; simp at hyk; exact absurd hyk hp1
          · exact h0.symm
      apply hk; apply Fin.ext; simp only [k0]; rw [hP0, ← hxk]; omega
    · rfl
  · simp

/-! ### Layer-1 lookups and the head -/

/-- Lookup output. -/
noncomputable def r1 (k : Fin 1000) : ℝ :=
  relu ((liModel_block1_W1 *ᵥ Y2 hT ids) k + liModel_block1_b1 k)

lemma r1_eq (k : Fin 1000) : r1 hT ids k = relu (
    (if 11 + k.val % 2 = n then 1 else 0) +
    S hT ids ⟨6, by norm_num⟩ (k.val % 20 / 2) + sc hT ids (k.val / 20) - 11 / 4) := by
  have hk := k.isLt
  have e1 : Y2 hT ids ⟨24 + k.val % 2, by omega⟩ = if 11 + k.val % 2 = n then 1 else 0 := by
    rw [Y2_low hT ids _ (by simp; omega), Y1_low hT ids _ (by simp; omega)]
    have := X_pos hT ids (Fin.last n) (11 + k.val % 2) (by omega)
    simp only [Fin.val_last] at this
    rw [show (⟨24 + k.val % 2, by omega⟩ : Fin 152) = ⟨13 + (11 + k.val % 2), by omega⟩ by
      ext; simp; omega, this]
  have e2 : Y2 hT ids ⟨89 + k.val % 20 / 2, by omega⟩ = S hT ids ⟨6, by norm_num⟩ (k.val % 20 / 2) := by
    rw [Y2_low hT ids _ (by simp; omega), Y1_slot hT ids _ (by omega)]
    congr 1
    · ext; simp; omega
    · omega
  have e3 : Y2 hT ids ⟨99 + k.val / 20, by omega⟩ = sc hT ids (k.val / 20) := Y2_score hT ids _ (by omega)
  unfold r1
  rw [mulVec, W1b_row, add_dotProduct, add_dotProduct, single_dotProduct, single_dotProduct,
    single_dotProduct, b1b, e1, e2, e3]
  ring_nf

/-- Final residual stream at the last position. -/
noncomputable def R : Fin 152 → ℝ := Y2 hT ids + liModel_block1.mlp (Y2 hT ids)

lemma R_tok (t : Fin 13) : R hT ids (t.castLE (by norm_num)) =
    (if t.val = (ids (Fin.last n)).val then 1 else 0) +
      ∑ k : Fin 1000, if dig k.val = t.val then 100 * r1 hT ids k else 0 := by
  have ht := t.isLt
  simp only [R, Block.mlp, liModel_block1, Pi.add_apply, b2_zero.2, add_zero, W2b_mulVec]
  rw [Y2_low hT ids _ (by simp only [Fin.val_castLE]; omega),
    Y1_low hT ids _ (by simp only [Fin.val_castLE]; omega),
    X_tok hT ids _ _ (by simp only [Fin.val_castLE]; omega)]
  simp only [Fin.val_castLE]
  rfl

lemma logits_eq (t : Fin 13) :
    liModel.logits hT ids (Fin.last n) t = R hT ids (t.castLE (by norm_num)) := by
  rw [show liModel.logits hT ids (Fin.last n) t = (liModel.Wh *ᵥ R hT ids + liModel.bh) t from rfl,
    Wh_apply]

include hn hD hm hb h1 h2 hq hx in
/-- **One generation step** at position n ∈ {11, 12}: the model emits the tens (n = 11) or ones
(n = 12) digit of the line's value at xq. -/
theorem step (hDl : D = Dl m b x1 x2 xq) :
    argmaxFirst (liModel.logits hT ids (Fin.last n)) =
      dtok (if n = 11 then yv m b xq / 10 else yv m b xq) := by
  have hy := yv_lt m b xq hm hb hq
  set Lt := m * 10 + b
  set kt : Fin 1000 := ⟨Lt * 20 + xq * 2 + (n - 11), by rcases hn with h | h <;> omega⟩
  have hst := sc_true hT ids hn D hD m b x1 x2 xq hm hb h1 h2 hq hx hDl
  have hso := sc_other hT ids hn D hD m b x1 x2 xq hm hb h1 h2 hq hx hDl
  have hS6 := S_bounds hT ids hn D hD ⟨6, by norm_num⟩ (by norm_num)
  have hD6 : D 6 = xq := by rw [hDl]; rfl
  simp only [hD6] at hS6
  have hsc0 : ∀ L, 0 ≤ sc hT ids L := fun L => sum_nonneg fun k _ => by
    split_ifs
    · exact (r0_bounds hT ids hn D hD k).1
    · exact le_rfl
  have hsc1 : ∀ L, L < 50 → sc hT ids L ≤ 1 := fun L hL => by
    by_cases h : L = Lt
    · rw [h]; exact hst.2
    · linarith [hso L hL h]
  -- the lookup that fires, and every other lookup stays at 0
  have hkt : 18 / 100 ≤ r1 hT ids kt := by
    rw [r1_eq]
    have e1 : kt.val % 2 = n - 11 := by simp [kt]; rcases hn with h | h <;> omega
    have e2 : kt.val % 20 / 2 = xq := by simp [kt]; rcases hn with h | h <;> omega
    have e3 : kt.val / 20 = Lt := by simp [kt]; rcases hn with h | h <;> omega
    rw [e1, e2, e3, if_pos (by rcases hn with h | h <;> omega)]
    unfold relu
    exact le_trans (by linarith [(hS6 xq).2.2.1 rfl, hst.1]) (le_max_left _ _)
  have hoff : ∀ k : Fin 1000, k ≠ kt → r1 hT ids k = 0 := by
    intro k hk
    rw [r1_eq]
    unfold relu
    apply max_eq_right
    have hk1 := k.isLt
    have hsk := hsc1 (k.val / 20) (by omega)
    have hS := hS6 (k.val % 20 / 2)
    by_cases hp : 11 + k.val % 2 = n
    · rw [if_pos hp]
      by_cases hl : k.val / 20 = Lt
      · have hxq : k.val % 20 / 2 ≠ xq := by
          intro hxq; apply hk; apply Fin.ext; simp [kt]; omega
        linarith [hS.2.2.2 hxq]
      · linarith [hso (k.val / 20) (by omega) hl, hS.2.1]
    · rw [if_neg hp]; linarith [hS.2.1]
  -- logits: the target digit gets ≥ 18, everything else ≤ 1
  have htgt : dig kt.val < 13 := dig_lt kt.val kt.isLt
  set T : Fin 13 := ⟨dig kt.val, htgt⟩
  have hR : ∀ t : Fin 13, R hT ids (t.castLE (by norm_num)) =
      (if t.val = (ids (Fin.last n)).val then 1 else 0) + (if t = T then 100 * r1 hT ids kt else 0) := by
    intro t
    rw [R_tok, sum_eq_single kt]
    · congr 1
      by_cases h : t = T
      · rw [if_pos (show dig kt.val = t.val by subst h; rfl), if_pos h]
      · rw [if_neg (fun h' : dig kt.val = t.val => h (Fin.ext h'.symm)), if_neg h]
    · intro k _ hk; rw [hoff k hk]; simp
    · simp
  have hlq : lineQ kt.val = yv m b xq := by
    have := lineQ_eq Lt xq (n - 11) (by omega) hq (by rcases hn with h | h <;> omega)
    rw [show Lt / 10 = m by omega, show Lt % 10 = b by omega] at this
    exact this
  have hpar : kt.val % 2 = n - 11 := by simp only [kt]; rcases hn with h | h <;> omega
  have hT_eq : T = dtok (if n = 11 then yv m b xq / 10 else yv m b xq) := by
    apply Fin.ext
    simp only [T, dtok, dig, hpar, hlq]
    rcases hn with h | h <;> subst h <;> simp <;> omega
  rw [← hT_eq]
  apply argmaxFirst_eq_of_lt
  intro j hj
  rw [logits_eq, logits_eq, hR, hR, if_neg hj, if_pos rfl]
  have : (if j.val = (ids (Fin.last n)).val then (1 : ℝ) else 0) ≤ 1 := by split_ifs <;> norm_num
  have : (0 : ℝ) ≤ if T.val = (ids (Fin.last n)).val then 1 else 0 := by split_ifs <;> norm_num
  linarith

end Step

/-! ### Generation -/

/-- **linear-interpolation-two-points, real-valued correctness of greedy generation** on every
input: slope m + 1 ∈ [1, 5], intercept b ∈ [0, 9], x1 ≠ x2, any xq. -/
theorem li_correct (m : Fin 5) (b x1 x2 xq : Fin 10) (hx : x1 ≠ x2) :
    liModel.generate (prompt m b x1 x2 xq) 2 = some (answer m b x1 x2 xq) := by
  have hans : (answer m b x1 x2 xq).length = 2 := rfl
  rw [← hans]
  apply Model.generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, hans]
  intro k hk h0 hL
  have hk' : k < 2 := by rw [hans] at hk; exact hk
  have hlen : (prompt m b x1 x2 xq ++ (answer m b x1 x2 xq).take k).length = (11 + k) + 1 := by
    simp [prompt, answer]; omega
  rw [Model.next_eq liModel _ h0 hL hlen (by omega)
    (fun t => (prompt m b x1 x2 xq ++ (answer m b x1 x2 xq).take k).get (Fin.cast hlen.symm t))
    (fun t => rfl)]
  have hxv : x1.val ≠ x2.val := fun h => hx (Fin.ext h)
  have hD : ∀ h : Fin 8, (hh : h.val < 7) →
      ((prompt m b x1 x2 xq ++ (answer m b x1 x2 xq).take k).get
        (Fin.cast hlen.symm ⟨src h.val, by have := src_le h.val; omega⟩)).val =
        Dl m.val b.val x1.val x2.val xq.val h.val := by
    intro h hh
    have hy1 := yv_lt m.val b.val x1.val m.isLt b.isLt x1.isLt
    have hy2 := yv_lt m.val b.val x2.val m.isLt b.isLt x2.isLt
    simp only [List.get_eq_getElem, Fin.val_cast]
    rw [List.getElem_append_left (by simp [prompt]; have := src_le h.val; omega)]
    fin_cases h <;> simp at hh <;> simp [prompt, src, Dl, dtok] <;> omega
  rw [step (n := 11 + k) (hT := by omega) (hn := by omega) (D := Dl m.val b.val x1.val x2.val xq.val)
    (hD := hD) (m := m.val) (b := b.val) (x1 := x1.val) (x2 := x2.val) (xq := xq.val) (hm := m.isLt)
    (hb := b.isLt) (h1 := x1.isLt) (h2 := x2.isLt) (hq := xq.isLt) (hx := hxv) (hDl := rfl)]
  interval_cases k <;> rfl

end VWC.Arch.LI
