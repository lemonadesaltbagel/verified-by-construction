/-
addition-five-digits (snapshot CompactCarryLookahead): real-valued correctness of greedy
generation on every input (10¹⁰ prompts), by mechanism.

* Block 0 attention: ten heads copy the ten operand digits into slots (leak λ ≤ 1/8000).
* Block 0 MLP: 500 AND units relu(A_i[a] + B_i[b] - 1); summed into g_i (a + b ≥ 10) and
  p_i (a + b = 9), each within 2λ of its 0/1 value.
* Block 1 MLP: 15 AND units over disjoint carry paths p_k ∧ … ∧ p_{j-1} ∧ g_j; their sum c_k is
  within 15·2λ of the carry.
* Block 2 MLP: at position 11 + k one unit (true digits, true carry branch) fires ≈ 1 and writes
  the answer digit; all other units together contribute < 1/2.
* The head reads the output-digit one-hot; `+` and `=` get logit 0.
-/
import VWC.Tasks.AddS1
import VWC.Tasks.AddS2
import VWC.Tasks.AddS3
import VWC.Tasks.AddS4
import VWC.Tasks.AddArith

namespace VWC.Arch.Add

open Matrix Finset

/-! ### Generic helpers -/

lemma fpe_val (h : Fin 10) (c : Fin 16) : (finProdFinEquiv (h, c)).val = c.val + 16 * h.val := by
  simp [finProdFinEquiv]

lemma fpe_symm_val (r : Fin (10 * 16)) :
    ((finProdFinEquiv.symm r).1).val = r.val / 16 ∧ ((finProdFinEquiv.symm r).2).val = r.val % 16 := by
  have h := fpe_val (finProdFinEquiv.symm r).1 (finProdFinEquiv.symm r).2
  rw [Prod.mk.eta, Equiv.apply_symm_apply] at h
  have h1 := ((finProdFinEquiv.symm r).2).isLt
  constructor <;> omega

lemma exp_neg12 : Real.exp (-12) ≤ 1 / 150000 := by
  have he := Real.exp_one_gt_d9
  have h : (2.7182818283 : ℝ) ^ 12 ≤ Real.exp 12 := by
    rw [show (12 : ℝ) = (12 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) he.le 12
  rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 12) (by norm_num)]
  calc (1 / 150000 : ℝ)⁻¹ ≤ (2.7182818283 : ℝ) ^ 12 := by norm_num
    _ ≤ _ := h

lemma sqrt16 : Real.sqrt 16 = 4 := by
  rw [show (16 : ℝ) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

/-- Leak bound used throughout. -/
noncomputable def lam : ℝ := 1 / 8000


/-! ### Generic sum and AND-unit lemmas -/

lemma sum_range_digits (H : ℕ → ℝ) (M : ℕ) :
    ∑ u ∈ range (10 * M), H u = ∑ a ∈ range M, ∑ b ∈ range 10, H (10 * a + b) := by
  induction M with
  | zero => simp
  | succ M ih =>
    rw [show 10 * (M + 1) = 10 * M + 10 by ring, sum_range_add, ih, sum_range_succ (n := M)]

lemma sum_range_pairs (G : ℕ → ℝ) (N : ℕ) :
    ∑ m ∈ range (2 * N), G m = ∑ u ∈ range N, (G (2 * u) + G (2 * u + 1)) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [show 2 * (N + 1) = 2 * N + 2 by ring, sum_range_add, ih, sum_range_succ (n := N)]
    simp [sum_range_succ]

lemma sum_segment (f : ℕ → ℝ) (s w N : ℕ) (hsw : s + w ≤ N)
    (hf : ∀ n, n < N → (n < s ∨ s + w ≤ n) → f n = 0) :
    ∑ n ∈ range N, f n = ∑ m ∈ range w, f (s + m) := by
  rw [range_eq_Ico, ← sum_Ico_consecutive f (Nat.zero_le (s + w)) hsw,
    ← sum_Ico_consecutive f (Nat.zero_le s) (show s ≤ s + w by omega)]
  rw [sum_eq_zero (s := Ico 0 s) (fun n hn => hf n (by simp at hn; omega) (by simp at hn; omega)),
    sum_eq_zero (s := Ico (s + w) N) (fun n hn => hf n (by simp at hn; omega) (by simp at hn; omega)),
    sum_Ico_eq_sum_range]
  simp

/-- Pull the (a₀, b₀) term out of a 10×10 double sum. -/
lemma split_star (f : ℕ → ℕ → ℝ) (a0 b0 : ℕ) (ha : a0 < 10) (hb : b0 < 10) :
    ∑ a ∈ range 10, ∑ b ∈ range 10, f a b =
      f a0 b0 + ∑ a ∈ range 10, ∑ b ∈ range 10, (if a = a0 ∧ b = b0 then 0 else f a b) := by
  have e : ∀ a b, f a b = (if a = a0 ∧ b = b0 then f a b else 0) +
      (if a = a0 ∧ b = b0 then 0 else f a b) := by
    intro a b; split_ifs <;> simp
  rw [sum_congr rfl fun a _ => sum_congr rfl fun b _ => e a b]
  simp only [sum_add_distrib]
  congr 1
  calc ∑ a ∈ range 10, ∑ b ∈ range 10, (if a = a0 ∧ b = b0 then f a b else 0)
      = ∑ a ∈ range 10, (if a = a0 then f a0 b0 else 0) := sum_congr rfl fun a _ => by
        by_cases h : a = a0
        · subst h; simp [hb]
        · simp [h]
    _ = f a0 b0 := by simp [ha]

/-- Σ over a 10-range of `if d = d₀ then 0 else c` is 9c. -/
lemma sum_off_const (d0 : ℕ) (hd : d0 < 10) (c : ℝ) :
    ∑ d ∈ range 10, (if d = d0 then 0 else c) = 9 * c := by
  interval_cases d0 <;> simp [sum_range_succ] <;> ring


lemma sum_range_blocks (H : ℕ → ℝ) (w M : ℕ) :
    ∑ u ∈ range (w * M), H u = ∑ a ∈ range M, ∑ b ∈ range w, H (w * a + b) := by
  induction M with
  | zero => simp
  | succ M ih =>
    rw [show w * (M + 1) = w * M + w by ring, sum_range_add, ih, sum_range_succ (n := M)]

/-- The block-2 unit index range, split into the units for position 11, the four 200-unit
blocks for positions 12–15 (digit pair, carry branch), and the 100 units for position 16. -/
lemma sum_units (f : ℕ → ℝ) (hf : ∀ m, 902 ≤ m → f m = 0) :
    ∑ m ∈ range 1024, f m = f 0 + f 1 +
      ∑ q ∈ range 4, ∑ a ∈ range 10, ∑ b ∈ range 10,
        (f (2 + 200 * q + 2 * (10 * a + b)) + f (2 + 200 * q + 2 * (10 * a + b) + 1)) +
      ∑ a ∈ range 10, ∑ b ∈ range 10, f (802 + (10 * a + b)) := by
  rw [sum_segment f 0 902 1024 (by norm_num) (fun m _ hm => hf m (by omega))]
  simp only [zero_add]
  rw [show (902 : ℕ) = 2 + 900 by norm_num, sum_range_add, show (900 : ℕ) = 800 + 100 by norm_num,
    sum_range_add, show (800 : ℕ) = 200 * 4 by norm_num, sum_range_blocks,
    show (100 : ℕ) = 10 * 10 by norm_num, sum_range_digits,
    show ∑ x ∈ range 2, f x = f 0 + f 1 by simp [sum_range_succ]]
  have hp : ∀ q, ∑ b ∈ range 200, f (2 + (200 * q + b)) = ∑ a ∈ range 10, ∑ b ∈ range 10,
      (f (2 + 200 * q + 2 * (10 * a + b)) + f (2 + 200 * q + 2 * (10 * a + b) + 1)) := by
    intro q
    rw [show (200 : ℕ) = 2 * 100 by norm_num, sum_range_pairs, show (100 : ℕ) = 10 * 10 by norm_num,
      sum_range_digits]
    refine sum_congr rfl fun a _ => sum_congr rfl fun b _ => ?_
    congr 2 <;> ring
  rw [sum_congr rfl fun q _ => hp q]
  have e : ∀ a b, f (2 + (800 + (10 * a + b))) = f (802 + (10 * a + b)) := fun a b => by congr 1; ring
  simp only [e]
  ring

section AndUnits

variable (x y : ℕ → ℝ) (a0 b0 : ℕ) (ha : a0 < 10) (hb : b0 < 10) (l η e : ℝ)
  (hx0 : ∀ a, 0 ≤ x a) (hy0 : ∀ b, 0 ≤ y b) (hx1 : ∀ a, x a ≤ 1) (hy1 : ∀ b, y b ≤ 1)
  (hxl : ∀ a, a ≠ a0 → x a ≤ l) (hyl : ∀ b, b ≠ b0 → y b ≤ l) (hη : 0 ≤ η)

include ha hb hx0 hy0 hx1 hy1 hxl hyl hη in
/-- AND units away from the true digit pair: one-sided matches leak at most their slot mass. -/
lemma and_off (hsmall : 2 * l + η < 1) (he : e ≤ η) :
    ∑ a ∈ range 10, ∑ b ∈ range 10, (if a = a0 ∧ b = b0 then 0 else relu (x a + y b - 1 + e)) ≤
      ∑ a ∈ range 10, (if a = a0 then 0 else x a) + ∑ b ∈ range 10, (if b = b0 then 0 else y b) +
        18 * η := by
  have hpt : ∀ a b, (if a = a0 ∧ b = b0 then 0 else relu (x a + y b - 1 + e)) ≤
      (if a = a0 then 0 else (if b = b0 then x a + η else 0)) +
        (if a = a0 then (if b = b0 then 0 else y b + η) else 0) := by
    intro a b
    unfold relu
    by_cases h1 : a = a0 <;> by_cases h2 : b = b0
    · simp [h1, h2]
    · simp only [h1, h2, and_false, if_false, if_true, zero_add]
      exact max_le (by linarith [hx1 a0]) (by linarith [hy0 b])
    · simp only [h1, h2, and_true, if_false, if_true, add_zero]
      exact max_le (by linarith [hy1 b0]) (by linarith [hx0 a])
    · simp only [h1, h2, and_false, if_false, add_zero]
      exact max_le (by linarith [hxl a h1, hyl b h2]) le_rfl
  calc _ ≤ ∑ a ∈ range 10, ∑ b ∈ range 10, ((if a = a0 then 0 else (if b = b0 then x a + η else 0)) +
        (if a = a0 then (if b = b0 then 0 else y b + η) else 0)) :=
        sum_le_sum fun a _ => sum_le_sum fun b _ => hpt a b
    _ = ∑ a ∈ range 10, (if a = a0 then 0 else x a + η) + ∑ b ∈ range 10, (if b = b0 then 0 else y b + η) := by
        rw [sum_congr rfl (fun a _ => sum_add_distrib), sum_add_distrib]
        congr 1
        · refine sum_congr rfl fun a _ => ?_
          by_cases h : a = a0
          · simp [h]
          · simp [h, hb]
        · rw [sum_eq_single a0 (fun a _ ha' => by simp [ha']) (fun h => absurd (mem_range.2 ha) h)]
          simp
    _ = _ := by
        have e1 : ∀ a, (if a = a0 then (0 : ℝ) else x a + η) =
            (if a = a0 then 0 else x a) + (if a = a0 then 0 else η) := fun a => by split_ifs <;> ring
        have e2 : ∀ b, (if b = b0 then (0 : ℝ) else y b + η) =
            (if b = b0 then 0 else y b) + (if b = b0 then 0 else η) := fun b => by split_ifs <;> ring
        simp only [e1, e2, sum_add_distrib, sum_off_const a0 ha, sum_off_const b0 hb]
        ring

include ha hb hx0 hy0 hx1 hy1 hxl hyl hη in
/-- AND units with a wrong (≈ −1) offset: only the true pair can leak, by at most η. -/
lemma and_wrong (hsmall : l + η < 1) (he : e ≤ -1 + η) :
    ∑ a ∈ range 10, ∑ b ∈ range 10, relu (x a + y b - 1 + e) ≤ η := by
  have hpt : ∀ a b, relu (x a + y b - 1 + e) ≤ (if a = a0 then (if b = b0 then η else 0) else 0) := by
    intro a b
    unfold relu
    by_cases h1 : a = a0 <;> by_cases h2 : b = b0
    · simp only [h1, h2, if_true]; exact max_le (by linarith [hx1 a0, hy1 b0]) hη
    · simp only [h1, h2, if_true, if_false]; exact max_le (by linarith [hx1 a0, hyl b h2]) le_rfl
    · simp only [h1, if_false]; exact max_le (by linarith [hxl a h1, hy1 b]) le_rfl
    · simp only [h1, if_false]; exact max_le (by linarith [hxl a h1, hy1 b]) le_rfl
  calc _ ≤ ∑ a ∈ range 10, ∑ b ∈ range 10, (if a = a0 then (if b = b0 then η else 0) else 0) :=
        sum_le_sum fun a _ => sum_le_sum fun b _ => hpt a b
    _ = η := by
        rw [sum_eq_single a0 (fun a _ ha' => by simp [ha']) (fun h => absurd (mem_range.2 ha) h)]
        simp [hb]

end AndUnits


/-! ### Exact AND identities for 0/1 inputs -/

lemma relu_lip (u v : ℝ) : |relu u - relu v| ≤ |u - v| := by
  unfold relu; exact abs_max_sub_max_le_abs u v 0

lemma and1 (a : ℝ) (ha : a = 0 ∨ a = 1) : relu a = a := by
  unfold relu; rcases ha with h | h <;> subst h <;> norm_num
lemma and2 (a b : ℝ) (ha : a = 0 ∨ a = 1) (hb : b = 0 ∨ b = 1) : relu (a + b - 1) = a * b := by
  unfold relu; rcases ha with h1 | h1 <;> rcases hb with h2 | h2 <;> subst h1 h2 <;> norm_num
lemma and3 (a b c : ℝ) (ha : a = 0 ∨ a = 1) (hb : b = 0 ∨ b = 1) (hc : c = 0 ∨ c = 1) :
    relu (a + b + c - 2) = a * b * c := by
  unfold relu
  rcases ha with h1 | h1 <;> rcases hb with h2 | h2 <;> rcases hc with h3 | h3 <;> subst h1 h2 h3 <;> norm_num
lemma and4 (a b c d : ℝ) (ha : a = 0 ∨ a = 1) (hb : b = 0 ∨ b = 1) (hc : c = 0 ∨ c = 1)
    (hd : d = 0 ∨ d = 1) : relu (a + b + c + d - 3) = a * b * c * d := by
  unfold relu
  rcases ha with h1 | h1 <;> rcases hb with h2 | h2 <;> rcases hc with h3 | h3 <;>
    rcases hd with h4 | h4 <;> subst h1 h2 h3 h4 <;> norm_num
lemma and5 (a b c d e : ℝ) (ha : a = 0 ∨ a = 1) (hb : b = 0 ∨ b = 1) (hc : c = 0 ∨ c = 1)
    (hd : d = 0 ∨ d = 1) (he : e = 0 ∨ e = 1) : relu (a + b + c + d + e - 4) = a * b * c * d * e := by
  unfold relu
  rcases ha with h1 | h1 <;> rcases hb with h2 | h2 <;> rcases hc with h3 | h3 <;>
    rcases hd with h4 | h4 <;> rcases he with h5 | h5 <;> subst h1 h2 h3 h4 h5 <;> norm_num


lemma outD_seg (q a b br : ℕ) (hq : q < 4) (ha : a < 10) (hb : b < 10) (hbr : br < 2) :
    outD (2 + 200 * q + 2 * (10 * a + b) + br) = (a + b + br) % 10 := by
  unfold outD; split_ifs <;> omega

lemma outD_p16 (a b : ℕ) (ha : a < 10) (hb : b < 10) : outD (802 + (10 * a + b)) = (a + b) % 10 := by
  unfold outD; split_ifs <;> omega

section Step

variable {n : ℕ} (hT : n + 1 ≤ 18) (ids : Fin (n + 1) → Fin 12)

/-- Embedded prompt. -/
noncomputable def X (t : Fin (n + 1)) : Fin 160 → ℝ :=
  addModel.tokEmb (ids t) + addModel.posEmb (Fin.castLE hT t)

variable (hn : 11 ≤ n ∧ n ≤ 16)

include hn in
lemma X_apply (t : Fin (n + 1)) (i : Fin 160) : X hT ids t i =
    (if (ids t).val < 10 ∧ i.val = 17 + (ids t).val then 1 else 0) + (if i.val = t.val then 1 else 0) := by
  simp only [X, Pi.add_apply, tokEmb_apply, posEmb_apply, Fin.val_castLE]
  congr 1
  have : t.val < 17 := by omega
  simp [this]

include hn in
lemma X_hi (t : Fin (n + 1)) (i : Fin 160) (hi : 27 ≤ i.val) : X hT ids t i = 0 := by
  rw [X_apply hT ids hn, if_neg (by omega), if_neg (by omega)]; ring

include hn in
lemma X_tok (t : Fin (n + 1)) (d : ℕ) (hd : d < 10) :
    X hT ids t ⟨17 + d, by omega⟩ = if (ids t).val = d then 1 else 0 := by
  rw [X_apply hT ids hn, if_neg (show ¬ (17 + d = t.val) by omega), add_zero]
  by_cases h : (ids t).val = d
  · rw [if_pos ⟨by omega, by simp [h]⟩, if_pos h]
  · rw [if_neg (by simp; omega), if_neg h]

include hn in
lemma X_pos (t : Fin (n + 1)) (s : ℕ) (hs : s < 17) :
    X hT ids t ⟨s, by omega⟩ = if s = t.val then 1 else 0 := by
  rw [X_apply hT ids hn, if_neg (by simp; omega), zero_add]

/-! ### Block-0 attention: gather heads -/

noncomputable def Wt (h : Fin 10) (j : Fin (n + 1)) : ℝ :=
  headWeight (fun t c => (A0.Wq *ᵥ X hT ids t) (finProdFinEquiv (h, c)))
    (fun t c => (A0.Wk *ᵥ X hT ids t) (finProdFinEquiv (h, c))) (Fin.last n) j

lemma src_lt (h : ℕ) (hh : h < 10) : src h < 11 := by unfold src; split_ifs <;> omega

include hn in
lemma qk_dot (h : Fin 10) (j : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ X hT ids (Fin.last n)) (finProdFinEquiv (h, c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ X hT ids j) (finProdFinEquiv (h, c)))) =
      if j.val = src h.val then 50 else 0 := by
  simp only [dotProduct, Wq_apply, Wk_apply, fpe_val]
  rw [sum_eq_single (0 : Fin 16)]
  · simp only [Fin.val_zero, zero_add, Nat.mul_mod_right, Nat.mul_div_cancel_left _ (by norm_num : 0 < 16),
      if_true]
    have hq : X hT ids (Fin.last n) 11 + X hT ids (Fin.last n) 12 + X hT ids (Fin.last n) 13 +
        X hT ids (Fin.last n) 14 + X hT ids (Fin.last n) 15 + X hT ids (Fin.last n) 16 = 1 := by
      have e := fun s hs => X_pos hT ids hn (Fin.last n) s hs
      rw [show (11 : Fin 160) = ⟨11, by norm_num⟩ from rfl, show (12 : Fin 160) = ⟨12, by norm_num⟩ from rfl,
        show (13 : Fin 160) = ⟨13, by norm_num⟩ from rfl, show (14 : Fin 160) = ⟨14, by norm_num⟩ from rfl,
        show (15 : Fin 160) = ⟨15, by norm_num⟩ from rfl, show (16 : Fin 160) = ⟨16, by norm_num⟩ from rfl,
        e 11 (by norm_num), e 12 (by norm_num), e 13 (by norm_num), e 14 (by norm_num),
        e 15 (by norm_num), e 16 (by norm_num)]
      simp only [Fin.val_last]
      obtain ⟨h1, h2⟩ := hn
      interval_cases n <;> norm_num
    rw [hq, X_pos hT ids hn _ _ (by have := src_lt h.val h.isLt; omega)]
    split_ifs with h1 h2 h2 <;> first | (exfalso; omega) | ring
  · intro c _ hc
    have : c.val ≠ 0 := fun h' => hc (Fin.ext h')
    have : ¬ ((c.val + 16 * h.val) % 16 = 0) := by have := c.isLt; omega
    rw [if_neg this, zero_mul]
  · simp

include hn in
lemma Wt_softmax (h : Fin 10) (j : Fin (n + 1)) :
    Wt hT ids h j = softmax (fun j : Fin (n + 1) =>
      (if j.val = src h.val then (50 : ℝ) else 0) / 4) j := by
  unfold Wt
  rw [headWeight_last]
  congr 1
  funext j'
  rw [qk_dot hT ids hn h, Nat.cast_ofNat, sqrt16]

lemma Wt_nonneg (h : Fin 10) (j : Fin (n + 1)) : 0 ≤ Wt hT ids h j := by
  unfold Wt headWeight
  split_ifs
  · exact div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)
  · exact le_rfl

include hn in
lemma Wt_sum (h : Fin 10) : ∑ j, Wt hT ids h j = 1 := by
  simp only [Wt_softmax hT ids hn h]; exact softmax_sum _

include hn in
/-- Gather heads copy their source token with leak ≤ λ = 1/8000. -/
lemma leak (h : Fin 10) :
    1 - Wt hT ids h ⟨src h.val, by have := src_lt h.val h.isLt; omega⟩ ≤ lam := by
  rw [Wt_softmax hT ids hn h]
  refine (one_sub_softmax_le _ _).trans ?_
  have hb : ∀ j : Fin (n + 1), (if j = (⟨src h.val, by have := src_lt h.val h.isLt; omega⟩ : Fin (n + 1))
      then (0 : ℝ) else Real.exp ((if j.val = src h.val then (50 : ℝ) else 0) / 4 -
        (if src h.val = src h.val then (50 : ℝ) else 0) / 4)) ≤ 1 / 150000 := by
    intro j
    by_cases h1 : j = (⟨src h.val, by have := src_lt h.val h.isLt; omega⟩ : Fin (n + 1))
    · rw [if_pos h1]; norm_num
    · rw [if_neg h1]
      have h2 : ¬ j.val = src h.val := fun h' => h1 (Fin.ext h')
      rw [if_neg h2, if_pos rfl]
      refine le_trans (Real.exp_le_exp.2 (by norm_num)) exp_neg12
  calc ∑ j : Fin (n + 1), _ ≤ ∑ _j : Fin (n + 1), (1 / 150000 : ℝ) := sum_le_sum fun j _ => hb j
    _ = (n + 1) * (1 / 150000) := by simp
    _ ≤ lam := by
        have : (n : ℝ) ≤ 16 := by exact_mod_cast hn.2
        unfold lam; nlinarith

/-- Slot value: head h's average of the digit-d indicator. -/
noncomputable def S (h : Fin 10) (d : ℕ) : ℝ :=
  ∑ j, Wt hT ids h j * (if (ids j).val = d then 1 else 0)

include hn in
lemma attn_slot (i : Fin 160) (hi : 27 ≤ i.val ∧ i.val < 127) :
    A0.apply (fun t => X hT ids t) (Fin.last n) i =
      S hT ids ⟨(i.val - 27) / 10, by omega⟩ ((i.val - 27) % 10) := by
  simp only [Attn.apply, Wo_apply, dif_pos hi]
  unfold S Wt
  refine sum_congr rfl fun j _ => ?_
  have hv := fpe_symm_val (⟨(i.val - 27) / 10 * 16 + (i.val - 27) % 10, by omega⟩ : Fin (10 * 16))
  simp only at hv
  have e1 : (finProdFinEquiv.symm (⟨(i.val - 27) / 10 * 16 + (i.val - 27) % 10, by omega⟩ :
      Fin (10 * 16))).1 = ⟨(i.val - 27) / 10, by omega⟩ := Fin.ext (by rw [hv.1]; simp only [Fin.val_mk]; omega)
  rw [e1]
  congr 1
  rw [Wv_apply, dif_pos (by rw [fpe_val]; omega)]
  have e2 : (⟨17 + (finProdFinEquiv ((⟨(i.val - 27) / 10, by omega⟩ : Fin 10),
      (finProdFinEquiv.symm (⟨(i.val - 27) / 10 * 16 + (i.val - 27) % 10, by omega⟩ :
        Fin (10 * 16))).2)).val % 16, by omega⟩ : Fin 160) = ⟨17 + (i.val - 27) % 10, by omega⟩ := by
    ext; simp only [fpe_val, hv.2]; omega
  rw [e2, X_tok hT ids hn _ _ (by omega)]

include hn in
lemma attn_other (i : Fin 160) (hi : ¬ (27 ≤ i.val ∧ i.val < 127)) :
    A0.apply (fun t => X hT ids t) (Fin.last n) i = 0 := by
  simp only [Attn.apply, Wo_apply, dif_neg hi]

/-! ### Gathered slots -/

variable (D : ℕ → ℕ)
  (hD : ∀ h : Fin 10, (ids ⟨src h.val, by have := src_lt h.val h.isLt; omega⟩).val = D h.val)

include hn hD in
lemma S_bounds (h : Fin 10) (d : ℕ) :
    0 ≤ S hT ids h d ∧ S hT ids h d ≤ 1 ∧ (d = D h.val → 1 - lam ≤ S hT ids h d) ∧
      (d ≠ D h.val → S hT ids h d ≤ lam) := by
  have hw := fun j => Wt_nonneg hT ids h j
  have hsum := Wt_sum hT ids hn h
  have hl := leak hT ids hn h
  set js : Fin (n + 1) := ⟨src h.val, by have := src_lt h.val h.isLt; omega⟩
  have hind : ∀ j : Fin (n + 1), (0 : ℝ) ≤ (if (ids j).val = d then 1 else 0) ∧
      (if (ids j).val = d then (1 : ℝ) else 0) ≤ 1 := fun j => by split_ifs <;> norm_num
  refine ⟨sum_nonneg fun j _ => mul_nonneg (hw j) (hind j).1, ?_, ?_, ?_⟩
  · calc S hT ids h d ≤ ∑ j, Wt hT ids h j := sum_le_sum fun j _ => by
          have := mul_le_mul_of_nonneg_left (hind j).2 (hw j); linarith
      _ = 1 := hsum
  · intro hd
    have : Wt hT ids h js * 1 ≤ S hT ids h d := by
      have e : (if (ids js).val = d then (1 : ℝ) else 0) = 1 := by rw [if_pos (by rw [hD h, hd])]
      rw [← e]
      exact single_le_sum (f := fun j => Wt hT ids h j * (if (ids j).val = d then 1 else 0))
        (fun j _ => mul_nonneg (hw j) (hind j).1) (mem_univ js)
    linarith
  · intro hd
    have e0 : Wt hT ids h js * (if (ids js).val = d then 1 else 0) = 0 := by
      rw [if_neg (by rw [hD h]; exact fun h' => hd h'.symm), mul_zero]
    have := sum_erase_add univ (fun j => Wt hT ids h j * (if (ids j).val = d then 1 else 0))
      (mem_univ js)
    have hs2 := sum_erase_add univ (Wt hT ids h) (mem_univ js)
    rw [hsum] at hs2
    have : ∑ j ∈ univ.erase js, Wt hT ids h j * (if (ids j).val = d then 1 else 0) ≤
        ∑ j ∈ univ.erase js, Wt hT ids h j := sum_le_sum fun j _ => by
      have := mul_le_mul_of_nonneg_left (hind j).2 (hw j); linarith
    unfold S; linarith

include hn in
/-- The slots of one head sum to at most 1 over the ten digits. -/
lemma S_sum (h : Fin 10) : ∑ d ∈ range 10, S hT ids h d ≤ 1 := by
  unfold S
  rw [sum_comm]
  calc ∑ j, ∑ d ∈ range 10, Wt hT ids h j * (if (ids j).val = d then 1 else 0)
      ≤ ∑ j, Wt hT ids h j := sum_le_sum fun j _ => by
        rw [← mul_sum]
        have : ∑ d ∈ range 10, (if (ids j).val = d then (1 : ℝ) else 0) ≤ 1 := by
          rw [sum_ite_eq]; split_ifs <;> norm_num
        exact mul_le_of_le_one_right (Wt_nonneg hT ids h j) this
    _ = 1 := Wt_sum hT ids hn h

include hn hD in
/-- Off-target slot mass is at most λ. -/
lemma S_off (h : Fin 10) (hD10 : D h.val < 10) :
    ∑ d ∈ range 10, (if d = D h.val then 0 else S hT ids h d) ≤ lam := by
  have h1 := S_sum hT ids hn h
  have h2 := (S_bounds hT ids hn D hD h (D h.val)).2.2.1 rfl
  have e := sum_ite_eq' (range 10) (D h.val) (fun d => S hT ids h d)
  rw [if_pos (mem_range.2 hD10)] at e
  have : ∑ d ∈ range 10, (if d = D h.val then 0 else S hT ids h d) =
      ∑ d ∈ range 10, S hT ids h d - S hT ids h (D h.val) := by
    rw [← e, ← sum_sub_distrib]
    refine sum_congr rfl fun d _ => ?_
    split_ifs with hd
    · subst hd; ring
    · ring
  rw [this]; linarith

/-! ### Residual after attention; block-0 AND units -/

noncomputable def Y1 : Fin 160 → ℝ := fun i =>
  X hT ids (Fin.last n) i + A0.apply (fun t => X hT ids t) (Fin.last n) i

include hn in
lemma Y1_slot (h : ℕ) (hh : h < 10) (d : ℕ) (hd : d < 10) :
    Y1 hT ids ⟨27 + 10 * h + d, by omega⟩ = S hT ids ⟨h, hh⟩ d := by
  have e := attn_slot hT ids hn ⟨27 + 10 * h + d, by omega⟩ (by simp; omega)
  simp only [Y1]
  rw [X_hi hT ids hn _ _ (by simp; omega), zero_add, e]
  have e1 : (27 + 10 * h + d - 27) / 10 = h := by omega
  have e2 : (27 + 10 * h + d - 27) % 10 = d := by omega
  simp only [e1, e2]

include hn in
lemma Y1_low (i : Fin 160) (hi : i.val < 27) : Y1 hT ids i = X hT ids (Fin.last n) i := by
  simp only [Y1]; rw [attn_other hT ids hn i (by omega), add_zero]

include hn in
lemma Y1_hi (i : Fin 160) (hi : 127 ≤ i.val) : Y1 hT ids i = 0 := by
  simp only [Y1]; rw [X_hi hT ids hn _ _ (by omega), attn_other hT ids hn i (by omega), add_zero]

/-- AND unit value for digit pair i (A slot i, B slot i + 5) at digits (a, b). -/
noncomputable def U0 (i a b : ℕ) : ℝ :=
  if h : i < 5 then relu (S hT ids ⟨i, by omega⟩ a + S hT ids ⟨i + 5, by omega⟩ b - 1) else 0

/-- Block-0 hidden layer. -/
noncomputable def u0 (k : Fin 1024) : ℝ :=
  relu ((addModel_block0_W1 *ᵥ Y1 hT ids) k + addModel_block0_b1 k)

include hn in
lemma u0_eq (k : Fin 1024) :
    u0 hT ids k = if k.val < 500 then U0 hT ids (k.val / 100) (k.val / 10 % 10) (k.val % 10) else 0 := by
  unfold u0
  rw [b1a]
  by_cases hk : k.val < 500
  · rw [if_pos hk, if_pos hk, mulVec, W1a_row, dif_pos hk, add_dotProduct, single_dotProduct,
      single_dotProduct]
    have e1 := Y1_slot hT ids hn (k.val / 100) (by omega) (k.val / 10 % 10) (by omega)
    have e2 := Y1_slot hT ids hn (k.val / 100 + 5) (by omega) (k.val % 10) (by omega)
    rw [show (⟨77 + 10 * (k.val / 100) + k.val % 10, by omega⟩ : Fin 160) =
      ⟨27 + 10 * (k.val / 100 + 5) + k.val % 10, by omega⟩ from Fin.ext (by simp; omega), e1, e2]
    unfold U0
    rw [dif_pos (by omega)]
    congr 1; ring
  · rw [if_neg hk, if_neg hk, mulVec, W1a_row, dif_neg hk]
    simp [relu]

/-- Residual after block 0. -/
noncomputable def Y2 : Fin 160 → ℝ := Y1 hT ids + addModel_block0.mlp (Y1 hT ids)

lemma Y2_apply (i : Fin 160) : Y2 hT ids i =
    Y1 hT ids i + ∑ k : Fin 1024, addModel_block0_W2 i k * u0 hT ids k := by
  simp only [Y2, Block.mlp, addModel_block0, Pi.add_apply, b2_zero.1, Pi.zero_apply, add_zero,
    mulVec, dotProduct]
  rfl

include hn in
/-- The block-0 output into dim `i`, as a sum over digit pairs of pair `i0`. -/
lemma flag_sum (i0 : ℕ) (hi0 : i0 < 5) (base : ℕ) (hbase : base = 127 ∨ base = 132) :
    ∑ k : Fin 1024, addModel_block0_W2 ⟨base + i0, by omega⟩ k * u0 hT ids k =
      ∑ a ∈ range 10, ∑ b ∈ range 10,
        (if (base = 127 ∧ 10 ≤ a + b) ∨ (base = 132 ∧ a + b = 9) then 1 else 0) * U0 hT ids i0 a b := by
  let F : ℕ → ℝ := fun m => if m < 500 ∧ m / 100 = i0 ∧
      ((base = 127 ∧ 10 ≤ m / 10 % 10 + m % 10) ∨ (base = 132 ∧ m / 10 % 10 + m % 10 = 9))
    then U0 hT ids (m / 100) (m / 10 % 10) (m % 10) else 0
  have hF : ∀ k : Fin 1024, addModel_block0_W2 ⟨base + i0, by omega⟩ k * u0 hT ids k = F k.val := by
    intro k
    have hc := congrFun (W2a_col k) ⟨base + i0, by omega⟩
    rw [hc, u0_eq hT ids hn]
    simp only [F]
    split_ifs <;> simp_all [Pi.single_apply, Fin.ext_iff] <;> omega
  rw [sum_congr rfl fun k _ => hF k, Fin.sum_univ_eq_sum_range F 1024,
    sum_segment F (100 * i0) 100 1024 (by omega) (fun m _ hm => by
      simp only [F]; rw [if_neg]; omega),
    show (100 : ℕ) = 10 * 10 by norm_num, sum_range_digits]
  refine sum_congr rfl fun a ha => sum_congr rfl fun b hb => ?_
  rw [mem_range] at ha hb
  simp only [F]
  rw [show (100 * i0 + (10 * a + b)) / 100 = i0 by omega, show (100 * i0 + (10 * a + b)) / 10 % 10 = a by omega,
    show (100 * i0 + (10 * a + b)) % 10 = b by omega]
  by_cases hc : (base = 127 ∧ 10 ≤ a + b) ∨ (base = 132 ∧ a + b = 9)
  · rw [if_pos ⟨by omega, rfl, hc⟩, if_pos hc, one_mul]
  · rw [if_neg (fun h => hc h.2.2), if_neg hc, zero_mul]

variable (hD10 : ∀ h, h < 10 → D h < 10)

include hn hD hD10 in
/-- A flag (Σ of AND units weighted by a 0/1 condition) is within 2λ of the condition at the
true digits. -/
lemma flag_bounds (i0 : ℕ) (hi0 : i0 < 5) (κ : ℕ → ℕ → ℝ) (hκ0 : ∀ a b, 0 ≤ κ a b)
    (hκ1 : ∀ a b, κ a b ≤ 1) :
    κ (D i0) (D (i0 + 5)) * (1 - 2 * lam) ≤ ∑ a ∈ range 10, ∑ b ∈ range 10, κ a b * U0 hT ids i0 a b ∧
    ∑ a ∈ range 10, ∑ b ∈ range 10, κ a b * U0 hT ids i0 a b ≤ κ (D i0) (D (i0 + 5)) + 2 * lam := by
  set x : ℕ → ℝ := fun a => S hT ids ⟨i0, by omega⟩ a
  set y : ℕ → ℝ := fun b => S hT ids ⟨i0 + 5, by omega⟩ b
  have bx := S_bounds hT ids hn D hD ⟨i0, by omega⟩
  have by' := S_bounds hT ids hn D hD ⟨i0 + 5, by omega⟩
  simp only at bx by'
  have ha := hD10 i0 (by omega)
  have hb := hD10 (i0 + 5) (by omega)
  have hU : ∀ a b, U0 hT ids i0 a b = relu (x a + y b - 1 + 0) := by
    intro a b; unfold U0; rw [dif_pos hi0, add_zero]
  have hrel0 : ∀ a b, 0 ≤ relu (x a + y b - 1 + 0) := fun a b => le_max_right _ _
  have hsplit := split_star (fun a b => κ a b * U0 hT ids i0 a b) (D i0) (D (i0 + 5)) ha hb
  have hoff := and_off x y (D i0) (D (i0 + 5)) ha hb lam 0 0 (fun a => (bx a).1) (fun b => (by' b).1)
    (fun a => (bx a).2.1) (fun b => (by' b).2.1) (fun a h => (bx a).2.2.2 h) (fun b h => (by' b).2.2.2 h)
    le_rfl (by unfold lam; norm_num) le_rfl
  have hox := S_off hT ids hn D hD ⟨i0, by omega⟩ ha
  have hoy := S_off hT ids hn D hD ⟨i0 + 5, by omega⟩ hb
  simp only at hox hoy
  have hrest : 0 ≤ ∑ a ∈ range 10, ∑ b ∈ range 10,
      (if a = D i0 ∧ b = D (i0 + 5) then 0 else κ a b * U0 hT ids i0 a b) :=
    sum_nonneg fun a _ => sum_nonneg fun b _ => by
      split_ifs; · exact le_rfl
      · rw [hU]; exact mul_nonneg (hκ0 a b) (hrel0 a b)
  have hrest' : ∑ a ∈ range 10, ∑ b ∈ range 10,
      (if a = D i0 ∧ b = D (i0 + 5) then 0 else κ a b * U0 hT ids i0 a b) ≤ 2 * lam := by
    refine le_trans (sum_le_sum fun a _ => sum_le_sum fun b _ => ?_) (by linarith [hoff])
    split_ifs; · exact le_rfl
    · rw [hU]; exact mul_le_of_le_one_left (hrel0 a b) (hκ1 a b)
  have hstar : 1 - 2 * lam ≤ U0 hT ids i0 (D i0) (D (i0 + 5)) ∧ U0 hT ids i0 (D i0) (D (i0 + 5)) ≤ 1 := by
    rw [hU]
    have h1 := (bx (D i0)).2.2.1 rfl
    have h2 := (by' (D (i0 + 5))).2.2.1 rfl
    have h3 := (bx (D i0)).2.1
    have h4 := (by' (D (i0 + 5))).2.1
    unfold relu
    exact ⟨le_trans (by linarith) (le_max_left _ _), max_le (by linarith) (by norm_num)⟩
  rw [hsplit]
  have hk0 := hκ0 (D i0) (D (i0 + 5))
  have hk1 := hκ1 (D i0) (D (i0 + 5))
  constructor
  · nlinarith [hstar.1]
  · nlinarith [hstar.2]

/-! ### Block 1: carry lookahead -/

/-- Generate / propagate flag values after block 0. -/
noncomputable def gv (i : ℕ) : ℝ := if h : i < 5 then Y2 hT ids ⟨127 + i, by omega⟩ else 0
noncomputable def pv (i : ℕ) : ℝ := if h : i < 5 then Y2 hT ids ⟨132 + i, by omega⟩ else 0

include hn in
lemma gv_eq (i : ℕ) (hi : i < 5) : gv hT ids i =
    ∑ a ∈ range 10, ∑ b ∈ range 10, (if 10 ≤ a + b then 1 else 0) * U0 hT ids i a b := by
  unfold gv; rw [dif_pos hi, Y2_apply, Y1_hi hT ids hn _ (by simp), zero_add]
  rw [flag_sum hT ids hn i hi 127 (Or.inl rfl)]
  refine sum_congr rfl fun a _ => sum_congr rfl fun b _ => ?_
  congr 1
  by_cases h : 10 ≤ a + b <;> simp [h]

include hn in
lemma pv_eq (i : ℕ) (hi : i < 5) : pv hT ids i =
    ∑ a ∈ range 10, ∑ b ∈ range 10, (if a + b = 9 then 1 else 0) * U0 hT ids i a b := by
  unfold pv; rw [dif_pos hi, Y2_apply, Y1_hi hT ids hn _ (by simp; omega), zero_add]
  rw [flag_sum hT ids hn i hi 132 (Or.inr rfl)]
  refine sum_congr rfl fun a _ => sum_congr rfl fun b _ => ?_
  congr 1
  by_cases h : a + b = 9 <;> simp [h]

/-- Residual after block 1 (its attention output is zero). -/
noncomputable def Y3 : Fin 160 → ℝ := Y2 hT ids + addModel_block1.mlp (Y2 hT ids)

/-- Block-1 hidden layer. -/
noncomputable def u1 (k : Fin 1024) : ℝ :=
  relu ((addModel_block1_W1 *ᵥ Y2 hT ids) k + addModel_block1_b1 k)

lemma Y3_apply (i : Fin 160) : Y3 hT ids i =
    Y2 hT ids i + ∑ k : Fin 1024, addModel_block1_W2 i k * u1 hT ids k := by
  simp only [Y3, Block.mlp, addModel_block1, Pi.add_apply, b2_zero.2.1, Pi.zero_apply, add_zero]
  rfl

/-- Carry values. -/
noncomputable def cv (k : ℕ) : ℝ := if h : k < 5 then Y3 hT ids ⟨137 + k, by omega⟩ else 0

include hn in
lemma Y2_high (i : Fin 160) (hi : 137 ≤ i.val) : Y2 hT ids i = 0 := by
  rw [Y2_apply, Y1_hi hT ids hn _ (by omega), zero_add]
  refine sum_eq_zero fun k _ => ?_
  have hc := congrFun (W2a_col k) i
  rw [hc]
  split_ifs <;> simp [Pi.single_apply, Fin.ext_iff] <;> omega

include hn in
lemma cv_units (k : ℕ) (hk : k < 5) : cv hT ids k =
    ∑ m ∈ range 15, if kOf m = k then (if h : m < 1024 then u1 hT ids ⟨m, h⟩ else 0) else 0 := by
  unfold cv
  rw [dif_pos hk]
  rw [Y3_apply, Y2_high hT ids hn _ (by simp), zero_add]
  let F : ℕ → ℝ := fun m => if kOf m = k ∧ m < 15 then (if h : m < 1024 then u1 hT ids ⟨m, h⟩ else 0) else 0
  have hF : ∀ m : Fin 1024, addModel_block1_W2 ⟨137 + k, by omega⟩ m * u1 hT ids m = F m.val := by
    intro m
    have hc := congrFun (W2b_col m) ⟨137 + k, by omega⟩
    rw [hc]
    simp only [F, dif_pos m.isLt]
    by_cases h15 : m.val < 15
    · rw [dif_pos h15, Pi.single_apply]
      by_cases hk' : kOf m.val = k
      · rw [if_pos (Fin.ext (by simp [hk'])), if_pos ⟨hk', h15⟩, one_mul]
      · rw [if_neg (fun h => hk' (by have := congrArg Fin.val h; simp at this; omega)),
          if_neg (fun h => hk' h.1), zero_mul]
    · rw [dif_neg h15, if_neg (fun h => h15 h.2)]; simp
  rw [sum_congr rfl fun m _ => hF m, Fin.sum_univ_eq_sum_range F 1024,
    sum_segment F 0 15 1024 (by norm_num) (fun m _ hm => by
      simp only [F]; rw [if_neg]; omega)]
  refine sum_congr rfl fun m hm => ?_
  rw [mem_range] at hm
  simp only [F, zero_add]
  by_cases h : kOf m = k <;> simp [h, hm]

include hn in
/-- Values of the 15 carry-path units. -/
lemma u1_vals :
    u1 hT ids 0 = relu (gv hT ids 0) ∧
    u1 hT ids 1 = relu (gv hT ids 1 + pv hT ids 0 - 1) ∧
    u1 hT ids 2 = relu (gv hT ids 2 + pv hT ids 0 + pv hT ids 1 - 2) ∧
    u1 hT ids 3 = relu (gv hT ids 3 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 - 3) ∧
    u1 hT ids 4 = relu (gv hT ids 4 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 4) ∧
    u1 hT ids 5 = relu (gv hT ids 1) ∧
    u1 hT ids 6 = relu (gv hT ids 2 + pv hT ids 1 - 1) ∧
    u1 hT ids 7 = relu (gv hT ids 3 + pv hT ids 1 + pv hT ids 2 - 2) ∧
    u1 hT ids 8 = relu (gv hT ids 4 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 3) ∧
    u1 hT ids 9 = relu (gv hT ids 2) ∧
    u1 hT ids 10 = relu (gv hT ids 3 + pv hT ids 2 - 1) ∧
    u1 hT ids 11 = relu (gv hT ids 4 + pv hT ids 2 + pv hT ids 3 - 2) ∧
    u1 hT ids 12 = relu (gv hT ids 3) ∧
    u1 hT ids 13 = relu (gv hT ids 4 + pv hT ids 3 - 1) ∧
    u1 hT ids 14 = relu (gv hT ids 4) := by
  obtain ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14⟩ := W1b_rows
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14⟩ := b1b_vals
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp only [u1, gv, pv, mulVec, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14,
      c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, add_dotProduct, single_dotProduct,
      one_mul] <;> norm_num <;> ring_nf

/-! ### Flag and carry bounds -/

lemma lam_pos : 0 < lam := by unfold lam; norm_num

variable (A B : Num5) (hDA : ∀ i, i < 5 → D i = dg A i) (hDB : ∀ i, i < 5 → D (i + 5) = dg B i)

lemma zo_gen (i : ℕ) : (gen A B i : ℝ) = 0 ∨ (gen A B i : ℝ) = 1 := by
  unfold gen; split_ifs <;> simp
lemma zo_prp (i : ℕ) : (prp A B i : ℝ) = 0 ∨ (prp A B i : ℝ) = 1 := by
  unfold prp; split_ifs <;> simp

include hn hD hD10 hDA hDB in
lemma gv_bound (i : ℕ) (hi : i < 5) : |gv hT ids i - gen A B i| ≤ 2 * lam := by
  rw [gv_eq hT ids hn i hi]
  have h := flag_bounds hT ids hn D hD hD10 i hi (fun a b => if 10 ≤ a + b then 1 else 0)
    (fun a b => by split_ifs <;> norm_num) (fun a b => by split_ifs <;> norm_num)
  rw [hDA i hi, hDB i hi] at h
  have := lam_pos
  unfold gen
  rw [abs_le]
  split_ifs at h ⊢ <;> push_cast <;> constructor <;> nlinarith [h.1, h.2]

include hn hD hD10 hDA hDB in
lemma pv_bound (i : ℕ) (hi : i < 5) : |pv hT ids i - prp A B i| ≤ 2 * lam := by
  rw [pv_eq hT ids hn i hi]
  have h := flag_bounds hT ids hn D hD hD10 i hi (fun a b => if a + b = 9 then 1 else 0)
    (fun a b => by split_ifs <;> norm_num) (fun a b => by split_ifs <;> norm_num)
  rw [hDA i hi, hDB i hi] at h
  have := lam_pos
  unfold prp
  rw [abs_le]
  split_ifs at h ⊢ <;> push_cast <;> constructor <;> nlinarith [h.1, h.2]

include hn in
lemma cv_explicit :
    cv hT ids 0 = u1 hT ids 0 + u1 hT ids 1 + u1 hT ids 2 + u1 hT ids 3 + u1 hT ids 4 ∧
    cv hT ids 1 = u1 hT ids 5 + u1 hT ids 6 + u1 hT ids 7 + u1 hT ids 8 ∧
    cv hT ids 2 = u1 hT ids 9 + u1 hT ids 10 + u1 hT ids 11 ∧
    cv hT ids 3 = u1 hT ids 12 + u1 hT ids 13 ∧
    cv hT ids 4 = u1 hT ids 14 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    rw [cv_units hT ids hn _ (by norm_num)] <;> simp [sum_range_succ, kOf] <;> rfl

/-- Two-sided error from a one-sided pair of bounds. -/
lemma abs_of (u v b : ℝ) (h1 : u - v ≤ b) (h2 : v - u ≤ b) : |u - v| ≤ b := abs_le.2 ⟨by linarith, h1⟩

include hn hD hD10 hDA hDB in
/-- Each carry value is within 30λ of the true carry, and nonnegative. -/
lemma cv_bound (k : ℕ) (hk : k < 5) :
    |cv hT ids k - carry A B k| ≤ 30 * lam ∧ 0 ≤ cv hT ids k := by
  have g := fun i hi => abs_le.1 (gv_bound hT ids hn D hD hD10 A B hDA hDB i hi)
  have p := fun i hi => abs_le.1 (pv_bound hT ids hn D hD hD10 A B hDA hDB i hi)
  obtain ⟨g0a, g0b⟩ := g 0 (by norm_num); obtain ⟨g1a, g1b⟩ := g 1 (by norm_num)
  obtain ⟨g2a, g2b⟩ := g 2 (by norm_num); obtain ⟨g3a, g3b⟩ := g 3 (by norm_num)
  obtain ⟨g4a, g4b⟩ := g 4 (by norm_num)
  obtain ⟨p0a, p0b⟩ := p 0 (by norm_num); obtain ⟨p1a, p1b⟩ := p 1 (by norm_num)
  obtain ⟨p2a, p2b⟩ := p 2 (by norm_num); obtain ⟨p3a, p3b⟩ := p 3 (by norm_num)
  obtain ⟨e0, e1, e2, e3, e4⟩ := cv_explicit hT ids hn
  obtain ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14⟩ := u1_vals hT ids hn
  obtain ⟨k4, k3, k2, k1, k0⟩ := carry_paths A B
  have G := fun i => zo_gen A B i
  have P := fun i => zo_prp A B i
  -- each unit is within (#inputs)·2λ of its ideal AND
  have L := fun s S b (h1 : s - S ≤ b) (h2 : S - s ≤ b) =>
    (abs_le.1 ((relu_lip s S).trans (abs_of s S b h1 h2)))
  have hr0 : ∀ v : ℝ, 0 ≤ relu v := fun v => le_max_right _ _
  have := lam_pos
  interval_cases k
  · rw [e0, r0, r1, r2, r3, r4, k0]
    have a1 := L (gv hT ids 0) (gen A B 0) (2 * lam) (by linarith) (by linarith)
    have a2 := L (gv hT ids 1 + pv hT ids 0 - 1) (gen A B 1 + prp A B 0 - 1) (4 * lam) (by linarith) (by linarith)
    have a3 := L (gv hT ids 2 + pv hT ids 0 + pv hT ids 1 - 2) (gen A B 2 + prp A B 0 + prp A B 1 - 2)
      (6 * lam) (by linarith) (by linarith)
    have a4 := L (gv hT ids 3 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 - 3)
      (gen A B 3 + prp A B 0 + prp A B 1 + prp A B 2 - 3) (8 * lam) (by linarith) (by linarith)
    have a5 := L (gv hT ids 4 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 4)
      (gen A B 4 + prp A B 0 + prp A B 1 + prp A B 2 + prp A B 3 - 4) (10 * lam) (by linarith) (by linarith)
    rw [and1 _ (G 0)] at a1
    rw [and2 _ _ (G 1) (P 0)] at a2
    rw [and3 _ _ _ (G 2) (P 0) (P 1)] at a3
    rw [and4 _ _ _ _ (G 3) (P 0) (P 1) (P 2)] at a4
    rw [and5 _ _ _ _ _ (G 4) (P 0) (P 1) (P 2) (P 3)] at a5
    push_cast
    refine ⟨abs_le.2 ⟨by linarith [a1.1, a2.1, a3.1, a4.1, a5.1], by linarith [a1.2, a2.2, a3.2, a4.2, a5.2]⟩, ?_⟩
    linarith [hr0 (gv hT ids 0), hr0 (gv hT ids 1 + pv hT ids 0 - 1), hr0 (gv hT ids 2 + pv hT ids 0 + pv hT ids 1 - 2),
      hr0 (gv hT ids 3 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 - 3),
      hr0 (gv hT ids 4 + pv hT ids 0 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 4)]
  · rw [e1, r5, r6, r7, r8, k1]
    have a1 := L (gv hT ids 1) (gen A B 1) (2 * lam) (by linarith) (by linarith)
    have a2 := L (gv hT ids 2 + pv hT ids 1 - 1) (gen A B 2 + prp A B 1 - 1) (4 * lam) (by linarith) (by linarith)
    have a3 := L (gv hT ids 3 + pv hT ids 1 + pv hT ids 2 - 2) (gen A B 3 + prp A B 1 + prp A B 2 - 2)
      (6 * lam) (by linarith) (by linarith)
    have a4 := L (gv hT ids 4 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 3)
      (gen A B 4 + prp A B 1 + prp A B 2 + prp A B 3 - 3) (8 * lam) (by linarith) (by linarith)
    rw [and1 _ (G 1)] at a1
    rw [and2 _ _ (G 2) (P 1)] at a2
    rw [and3 _ _ _ (G 3) (P 1) (P 2)] at a3
    rw [and4 _ _ _ _ (G 4) (P 1) (P 2) (P 3)] at a4
    push_cast
    refine ⟨abs_le.2 ⟨by linarith [a1.1, a2.1, a3.1, a4.1], by linarith [a1.2, a2.2, a3.2, a4.2]⟩, ?_⟩
    linarith [hr0 (gv hT ids 1), hr0 (gv hT ids 2 + pv hT ids 1 - 1), hr0 (gv hT ids 3 + pv hT ids 1 + pv hT ids 2 - 2),
      hr0 (gv hT ids 4 + pv hT ids 1 + pv hT ids 2 + pv hT ids 3 - 3)]
  · rw [e2, r9, r10, r11, k2]
    have a1 := L (gv hT ids 2) (gen A B 2) (2 * lam) (by linarith) (by linarith)
    have a2 := L (gv hT ids 3 + pv hT ids 2 - 1) (gen A B 3 + prp A B 2 - 1) (4 * lam) (by linarith) (by linarith)
    have a3 := L (gv hT ids 4 + pv hT ids 2 + pv hT ids 3 - 2) (gen A B 4 + prp A B 2 + prp A B 3 - 2)
      (6 * lam) (by linarith) (by linarith)
    rw [and1 _ (G 2)] at a1
    rw [and2 _ _ (G 3) (P 2)] at a2
    rw [and3 _ _ _ (G 4) (P 2) (P 3)] at a3
    push_cast
    refine ⟨abs_le.2 ⟨by linarith [a1.1, a2.1, a3.1], by linarith [a1.2, a2.2, a3.2]⟩, ?_⟩
    linarith [hr0 (gv hT ids 2), hr0 (gv hT ids 3 + pv hT ids 2 - 1), hr0 (gv hT ids 4 + pv hT ids 2 + pv hT ids 3 - 2)]
  · rw [e3, r12, r13, k3]
    have a1 := L (gv hT ids 3) (gen A B 3) (2 * lam) (by linarith) (by linarith)
    have a2 := L (gv hT ids 4 + pv hT ids 3 - 1) (gen A B 4 + prp A B 3 - 1) (4 * lam) (by linarith) (by linarith)
    rw [and1 _ (G 3)] at a1
    rw [and2 _ _ (G 4) (P 3)] at a2
    push_cast
    refine ⟨abs_le.2 ⟨by linarith [a1.1, a2.1], by linarith [a1.2, a2.2]⟩, ?_⟩
    linarith [hr0 (gv hT ids 3), hr0 (gv hT ids 4 + pv hT ids 3 - 1)]
  · rw [e4, r14, k4]
    have a1 := L (gv hT ids 4) (gen A B 4) (2 * lam) (by linarith) (by linarith)
    rw [and1 _ (G 4)] at a1
    refine ⟨abs_le.2 ⟨by linarith [a1.1], by linarith [a1.2]⟩, hr0 _⟩

/-! ### Block 2: digit emission -/

include hn in
lemma Y2_low (i : Fin 160) (hi : i.val < 127) : Y2 hT ids i = Y1 hT ids i := by
  rw [Y2_apply]
  have : ∑ k : Fin 1024, addModel_block0_W2 i k * u0 hT ids k = 0 := sum_eq_zero fun k _ => by
    have hc := congrFun (W2a_col k) i
    rw [hc]; split_ifs <;> simp [Pi.single_apply, Fin.ext_iff] <;> omega
  rw [this, add_zero]

include hn in
lemma Y3_low (i : Fin 160) (hi : i.val < 137) : Y3 hT ids i = Y2 hT ids i := by
  rw [Y3_apply]
  have : ∑ k : Fin 1024, addModel_block1_W2 i k * u1 hT ids k = 0 := sum_eq_zero fun k _ => by
    have hc := congrFun (W2b_col k) i
    rw [hc]; split_ifs <;> simp [Pi.single_apply, Fin.ext_iff] <;> omega
  rw [this, add_zero]

include hn in
lemma Y3_hi (i : Fin 160) (hi : 142 ≤ i.val) : Y3 hT ids i = 0 := by
  rw [Y3_apply, Y2_high hT ids hn _ (by omega), zero_add]
  exact sum_eq_zero fun k _ => by
    have hc := congrFun (W2b_col k) i
    rw [hc]; split_ifs <;> simp [Pi.single_apply, Fin.ext_iff] <;> unfold kOf <;> split_ifs <;> omega

/-- Position indicator at the last position. -/
noncomputable def posv (p : ℕ) : ℝ := if p = n then 1 else 0

include hn in
lemma Y3_pos (p : ℕ) (hp : p < 17) : Y3 hT ids ⟨p, by omega⟩ = posv (n := n) p := by
  rw [Y3_low hT ids hn _ (by simp; omega), Y2_low hT ids hn _ (by simp; omega),
    Y1_low hT ids hn _ (by simp; omega), X_pos hT ids hn _ _ hp]
  simp [posv]

include hn in
lemma Y3_slot (h d : ℕ) (hh : h < 10) (hd : d < 10) :
    Y3 hT ids ⟨27 + 10 * h + d, by omega⟩ = S hT ids ⟨h, hh⟩ d := by
  rw [Y3_low hT ids hn _ (by simp; omega), Y2_low hT ids hn _ (by simp; omega), Y1_slot hT ids hn h hh d hd]

/-- Block-2 hidden layer, by unit index. -/
noncomputable def u2 (k : Fin 1024) : ℝ :=
  relu ((addModel_block2_W1 *ᵥ Y3 hT ids) k + addModel_block2_b1 k)
noncomputable def u2N (m : ℕ) : ℝ := if h : m < 1024 then u2 hT ids ⟨m, h⟩ else 0

lemma u2N_nonneg (m : ℕ) : 0 ≤ u2N hT ids m := by
  unfold u2N u2 relu; split_ifs
  · exact le_max_right _ _
  · exact le_rfl

include hn in
lemma u2_01 : u2N hT ids 0 = relu (posv (n := n) 11 - cv hT ids 0) ∧
    u2N hT ids 1 = relu (posv (n := n) 11 + cv hT ids 0 - 1) := by
  have p11 := Y3_pos hT ids hn 11 (by norm_num)
  have c0 : Y3 hT ids 137 = cv hT ids 0 := by unfold cv; rw [dif_pos (by norm_num)]; rfl
  constructor <;>
  · simp only [u2N, u2, dif_pos (show (0 : ℕ) < 1024 by norm_num), dif_pos (show (1 : ℕ) < 1024 by norm_num),
      mulVec, W1c_row, b1c]
    norm_num [add_dotProduct, single_dotProduct]
    rw [show (11 : Fin 160) = ⟨11, by norm_num⟩ from rfl, p11, c0]
    ring_nf

include hn in
lemma u2_seg (q a b br : ℕ) (hq : q < 4) (ha : a < 10) (hb : b < 10) (hbr : br < 2) :
    u2N hT ids (2 + 200 * q + 2 * (10 * a + b) + br) =
      relu (S hT ids ⟨q, by omega⟩ a + S hT ids ⟨q + 5, by omega⟩ b - 1 +
        (if br = 0 then posv (n := n) (12 + q) - 1 - cv hT ids (q + 1)
          else posv (n := n) (12 + q) - 2 + cv hT ids (q + 1))) := by
  set m := 2 + 200 * q + 2 * (10 * a + b) + br with hm
  have hm1 : m < 1024 := by omega
  have e1 : (m - 2) / 200 = q := by omega
  have e2 : (m - 2) % 200 / 20 = a := by omega
  have e3 : (m - 2) % 20 / 2 = b := by omega
  have e4 : (m - 2) % 2 = br := by omega
  have c0 : ¬ m = 0 := by omega
  have c1 : ¬ m = 1 := by omega
  have c2 : m < 802 := by omega
  simp only [u2N, dif_pos hm1, u2, mulVec, W1c_row, b1c, Fin.val_mk, c0, c1, c2, if_false, dite_true,
    if_true, add_dotProduct, single_dotProduct, one_mul]
  have pq : Y3 hT ids ⟨12 + (m - 2) / 200, by omega⟩ = posv (n := n) (12 + q) := by
    rw [show (⟨12 + (m - 2) / 200, by omega⟩ : Fin 160) = ⟨12 + q, by omega⟩ from Fin.ext (by simp; omega)]
    exact Y3_pos hT ids hn _ (by omega)
  have sa : Y3 hT ids ⟨27 + 10 * ((m - 2) / 200) + (m - 2) % 200 / 20, by omega⟩ = S hT ids ⟨q, by omega⟩ a := by
    rw [show (⟨27 + 10 * ((m - 2) / 200) + (m - 2) % 200 / 20, by omega⟩ : Fin 160) = ⟨27 + 10 * q + a, by omega⟩ from
      Fin.ext (by simp; omega)]
    exact Y3_slot hT ids hn q a (by omega) ha
  have sb : Y3 hT ids ⟨77 + 10 * ((m - 2) / 200) + (m - 2) % 20 / 2, by omega⟩ = S hT ids ⟨q + 5, by omega⟩ b := by
    rw [show (⟨77 + 10 * ((m - 2) / 200) + (m - 2) % 20 / 2, by omega⟩ : Fin 160) = ⟨27 + 10 * (q + 5) + b, by omega⟩ from
      Fin.ext (by simp; omega)]
    exact Y3_slot hT ids hn (q + 5) b (by omega) hb
  have sc : Y3 hT ids ⟨138 + (m - 2) / 200, by omega⟩ = cv hT ids (q + 1) := by
    unfold cv; rw [dif_pos (by omega)]
    congr 1; ext; simp; omega
  rw [pq, sa, sb, sc, e4]
  rcases (show br = 0 ∨ br = 1 by omega) with h | h <;> subst h <;> simp <;> ring_nf

include hn in
lemma u2_p16 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
    u2N hT ids (802 + (10 * a + b)) =
      relu (S hT ids ⟨4, by norm_num⟩ a + S hT ids ⟨9, by norm_num⟩ b - 1 + (posv (n := n) 16 - 1)) := by
  set m := 802 + (10 * a + b) with hm
  have c0 : ¬ m = 0 := by omega
  have c1 : ¬ m = 1 := by omega
  have c2 : ¬ m < 802 := by omega
  have c3 : m < 902 := by omega
  simp only [u2N, dif_pos (show m < 1024 by omega), u2, mulVec, W1c_row, b1c, Fin.val_mk, c0, c1, c2, c3,
    if_false, dite_false, dite_true, if_true, add_dotProduct, single_dotProduct, one_mul]
  have pq : Y3 hT ids 16 = posv (n := n) 16 := by
    rw [show (16 : Fin 160) = ⟨16, by norm_num⟩ from rfl]; exact Y3_pos hT ids hn 16 (by norm_num)
  have sa : Y3 hT ids ⟨67 + (m - 802) / 10, by omega⟩ = S hT ids ⟨4, by norm_num⟩ a := by
    rw [show (⟨67 + (m - 802) / 10, by omega⟩ : Fin 160) = ⟨27 + 10 * 4 + a, by omega⟩ from Fin.ext (by simp; omega)]
    exact Y3_slot hT ids hn 4 a (by norm_num) ha
  have sb : Y3 hT ids ⟨117 + (m - 802) % 10, by omega⟩ = S hT ids ⟨9, by norm_num⟩ b := by
    rw [show (⟨117 + (m - 802) % 10, by omega⟩ : Fin 160) = ⟨27 + 10 * 9 + b, by omega⟩ from Fin.ext (by simp; omega)]
    exact Y3_slot hT ids hn 9 b (by norm_num) hb
  rw [pq, sa, sb]
  ring_nf

lemma u2_hi (m : ℕ) (hm : 902 ≤ m) : u2N hT ids m = 0 := by
  unfold u2N
  split_ifs with h
  · have c0 : ¬ m = 0 := by omega
    have c1 : ¬ m = 1 := by omega
    have c2 : ¬ m < 802 := by omega
    have c3 : ¬ m < 902 := by omega
    simp only [u2, mulVec, W1c_row, b1c, Fin.val_mk, c0, c1, c2, c3, if_false, dite_false]
    simp [relu]
  · rfl

/-- Final residual. -/
noncomputable def R : Fin 160 → ℝ := Y3 hT ids + addModel_block2.mlp (Y3 hT ids)

lemma R_apply (i : Fin 160) : R hT ids i =
    Y3 hT ids i + ∑ k : Fin 1024, addModel_block2_W2 i k * u2 hT ids k := by
  simp only [R, Block.mlp, addModel_block2, Pi.add_apply, b2_zero.2.2, Pi.zero_apply, add_zero]
  rfl

include hn in
lemma R_digit (t : ℕ) (ht : t < 10) : R hT ids ⟨142 + t, by omega⟩ =
    ∑ m ∈ range 1024, if m < 902 ∧ outD m = t then u2N hT ids m else 0 := by
  rw [R_apply, Y3_hi hT ids hn _ (by simp), zero_add]
  let F : ℕ → ℝ := fun m => if m < 902 ∧ outD m = t then u2N hT ids m else 0
  have hF : ∀ k : Fin 1024, addModel_block2_W2 ⟨142 + t, by omega⟩ k * u2 hT ids k = F k.val := by
    intro k
    have hc := congrFun (W2c_col k) ⟨142 + t, by omega⟩
    rw [hc]
    simp only [F, u2N, dif_pos k.isLt]
    by_cases hk : k.val < 902
    · rw [dif_pos hk, Pi.single_apply]
      by_cases ho : outD k.val = t
      · rw [if_pos (Fin.ext (by simp [ho])), if_pos ⟨hk, ho⟩, one_mul]
      · rw [if_neg (fun h => ho (by have := congrArg Fin.val h; simp at this; omega)),
          if_neg (fun h => ho h.2), zero_mul]
    · rw [dif_neg hk, if_neg (fun h => hk h.1)]; simp
  rw [sum_congr rfl fun k _ => hF k, Fin.sum_univ_eq_sum_range F 1024]


/-! ### Emission bounds -/

include hn hD hD10 hDA hDB in
/-- One 200-unit block (digit position q): off the target, its units sum to at most 2λ + 19ε₁. -/
lemma seg_bound (q : ℕ) (hq : q < 4) (kt : ℕ)
    (hkt : n ≠ 12 + q ∨ kt = 2 + 200 * q + 2 * (10 * D q + D (q + 5)) + carry A B (q + 1)) :
    ∑ a ∈ range 10, ∑ b ∈ range 10,
      ((if 2 + 200 * q + 2 * (10 * a + b) = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b))) +
        (if 2 + 200 * q + 2 * (10 * a + b) + 1 = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b) + 1)))
      ≤ 2 * lam + 19 * (30 * lam) := by
  have hl := lam_pos
  set x : ℕ → ℝ := fun a => S hT ids ⟨q, by omega⟩ a
  set y : ℕ → ℝ := fun b => S hT ids ⟨q + 5, by omega⟩ b
  have bx := S_bounds hT ids hn D hD ⟨q, by omega⟩
  have by' := S_bounds hT ids hn D hD ⟨q + 5, by omega⟩
  have ha := hD10 q (by omega)
  have hb := hD10 (q + 5) (by omega)
  have hox := S_off hT ids hn D hD ⟨q, by omega⟩ ha
  have hoy := S_off hT ids hn D hD ⟨q + 5, by omega⟩ hb
  obtain ⟨hc, hc0⟩ := cv_bound hT ids hn D hD hD10 A B hDA hDB (q + 1) (by omega)
  have hcar := carry_le A B (q + 1)
  have hcr : (carry A B (q + 1) : ℝ) ≤ 1 := by exact_mod_cast hcar
  rw [abs_le] at hc
  set c := cv hT ids (q + 1)
  set e0 := posv (n := n) (12 + q) - 1 - c
  set e1 := posv (n := n) (12 + q) - 2 + c
  have hV : ∀ a b br, a < 10 → b < 10 → br < 2 → u2N hT ids (2 + 200 * q + 2 * (10 * a + b) + br) =
      relu (x a + y b - 1 + (if br = 0 then e0 else e1)) := by
    intro a b br ha' hb' hbr; rw [u2_seg hT ids hn q a b br hq ha' hb' hbr]
  have W := fun e η (hη : 0 ≤ η) (hs : lam + η < 1) (he : e ≤ -1 + η) =>
    and_wrong x y (D q) (D (q + 5)) ha hb lam η e (fun a => (bx a).1) (fun b => (by' b).1)
      (fun a => (bx a).2.1) (fun b => (by' b).2.1) (fun a h => (bx a).2.2.2 h) (fun b h => (by' b).2.2.2 h)
      hη hs he
  have O := fun e η (hη : 0 ≤ η) (hs : 2 * lam + η < 1) (he : e ≤ η) =>
    and_off x y (D q) (D (q + 5)) ha hb lam η e (fun a => (bx a).1) (fun b => (by' b).1)
      (fun a => (bx a).2.1) (fun b => (by' b).2.1) (fun a h => (bx a).2.2.2 h) (fun b h => (by' b).2.2.2 h)
      hη hs he
  have hε : 0 ≤ 30 * lam := by linarith
  have hsm1 : lam + 30 * lam < 1 := by unfold lam; norm_num
  have hsm2 : 2 * lam + 30 * lam < 1 := by unfold lam; norm_num
  have hpos : posv (n := n) (12 + q) = 0 ∨ (posv (n := n) (12 + q) = 1 ∧ n = 12 + q) := by
    unfold posv
    split_ifs with h
    · exact Or.inr ⟨rfl, h.symm⟩
    · exact Or.inl rfl
  -- rewrite the summand through the unit values
  have hsum : ∀ a ∈ range 10, ∀ b ∈ range 10,
      ((if 2 + 200 * q + 2 * (10 * a + b) = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b))) +
        (if 2 + 200 * q + 2 * (10 * a + b) + 1 = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b) + 1))) =
      (if 2 + 200 * q + 2 * (10 * a + b) + 0 = kt then 0 else relu (x a + y b - 1 + e0)) +
        (if 2 + 200 * q + 2 * (10 * a + b) + 1 = kt then 0 else relu (x a + y b - 1 + e1)) := by
    intro a ha' b hb'
    rw [mem_range] at ha' hb'
    have h0 := hV a b 0 ha' hb' (by norm_num)
    have h1 := hV a b 1 ha' hb' (by norm_num)
    simp only [add_zero, if_pos, if_neg one_ne_zero] at h0 h1 ⊢
    rw [h0, h1]
  rw [sum_congr rfl fun a ha' => sum_congr rfl fun b hb' => hsum a ha' b hb']
  rcases hpos with hp | ⟨hp, hcur⟩
  · -- not the current position: both branches are "wrong"
    have he0 : e0 ≤ -1 + 30 * lam := by simp only [e0, hp]; linarith
    have he1 : e1 ≤ -1 + 30 * lam := by simp only [e1, hp]; linarith [hc.2, hcr]
    have w0 := W e0 (30 * lam) hε hsm1 he0
    have w1 := W e1 (30 * lam) hε hsm1 he1
    calc _ ≤ ∑ a ∈ range 10, ∑ b ∈ range 10, (relu (x a + y b - 1 + e0) + relu (x a + y b - 1 + e1)) :=
          sum_le_sum fun a _ => sum_le_sum fun b _ => add_le_add
            (by split_ifs <;> first | exact le_max_right _ _ | exact le_rfl)
            (by split_ifs <;> first | exact le_max_right _ _ | exact le_rfl)
      _ ≤ 2 * lam + 19 * (30 * lam) := by
          simp only [sum_add_distrib]; linarith
  · rcases hkt with hkt | hkt
    · exact absurd hcur hkt
    rcases (show carry A B (q + 1) = 0 ∨ carry A B (q + 1) = 1 by omega) with hC | hC
    · -- carry 0: branch 0 is the true one
      rw [hC] at hkt hc; push_cast at hc
      have he0 : e0 ≤ 30 * lam := by simp only [e0, hp]; linarith
      have he1 : e1 ≤ -1 + 30 * lam := by simp only [e1, hp]; linarith [hc.2]
      have o0 := O e0 (30 * lam) hε hsm2 he0
      have w1 := W e1 (30 * lam) hε hsm1 he1
      have hm0 : ∀ a b, a < 10 → b < 10 → ((2 + 200 * q + 2 * (10 * a + b) + 0 = kt) ↔ (a = D q ∧ b = D (q + 5))) := by
        intro a b ha' hb'; rw [hkt]; omega
      have hm1 : ∀ a b, a < 10 → b < 10 → ¬ (2 + 200 * q + 2 * (10 * a + b) + 1 = kt) := by
        intro a b ha' hb'; rw [hkt]; omega
      calc _ = ∑ a ∈ range 10, ∑ b ∈ range 10, ((if a = D q ∧ b = D (q + 5) then 0 else relu (x a + y b - 1 + e0)) +
            relu (x a + y b - 1 + e1)) := sum_congr rfl fun a ha' => sum_congr rfl fun b hb' => by
            rw [mem_range] at ha' hb'
            rw [if_congr (hm0 a b ha' hb') rfl rfl, if_neg (hm1 a b ha' hb')]
        _ ≤ 2 * lam + 19 * (30 * lam) := by simp only [sum_add_distrib]; linarith
    · rw [hC] at hkt hc; push_cast at hc
      have he1 : e1 ≤ 30 * lam := by simp only [e1, hp]; linarith [hc.2]
      have he0 : e0 ≤ -1 + 30 * lam := by simp only [e0, hp]; linarith [hc.1]
      have o1 := O e1 (30 * lam) hε hsm2 he1
      have w0 := W e0 (30 * lam) hε hsm1 he0
      have hm0 : ∀ a b, a < 10 → b < 10 → ¬ (2 + 200 * q + 2 * (10 * a + b) + 0 = kt) := by
        intro a b ha' hb'; rw [hkt]; omega
      have hm1 : ∀ a b, a < 10 → b < 10 → ((2 + 200 * q + 2 * (10 * a + b) + 1 = kt) ↔ (a = D q ∧ b = D (q + 5))) := by
        intro a b ha' hb'; rw [hkt]; omega
      calc _ = ∑ a ∈ range 10, ∑ b ∈ range 10, (relu (x a + y b - 1 + e0) +
            (if a = D q ∧ b = D (q + 5) then 0 else relu (x a + y b - 1 + e1))) :=
            sum_congr rfl fun a ha' => sum_congr rfl fun b hb' => by
            rw [mem_range] at ha' hb'
            rw [if_neg (hm0 a b ha' hb'), if_congr (hm1 a b ha' hb') rfl rfl]
        _ ≤ 2 * lam + 19 * (30 * lam) := by simp only [sum_add_distrib]; linarith


include hn hD hD10 in
/-- The 100 units of position 16: off the target they sum to at most 2λ. -/
lemma p16_bound (kt : ℕ) (hkt : n ≠ 16 ∨ kt = 802 + (10 * D 4 + D 9)) :
    ∑ a ∈ range 10, ∑ b ∈ range 10,
      (if 802 + (10 * a + b) = kt then 0 else u2N hT ids (802 + (10 * a + b))) ≤ 2 * lam := by
  have hl := lam_pos
  set x : ℕ → ℝ := fun a => S hT ids ⟨4, by norm_num⟩ a
  set y : ℕ → ℝ := fun b => S hT ids ⟨9, by norm_num⟩ b
  have bx := S_bounds hT ids hn D hD ⟨4, by norm_num⟩
  have by' := S_bounds hT ids hn D hD ⟨9, by norm_num⟩
  have ha := hD10 4 (by norm_num)
  have hb := hD10 9 (by norm_num)
  have hox := S_off hT ids hn D hD ⟨4, by norm_num⟩ ha
  have hoy := S_off hT ids hn D hD ⟨9, by norm_num⟩ hb
  simp only at bx by' hox hoy
  have hV : ∀ a ∈ range 10, ∀ b ∈ range 10, (if 802 + (10 * a + b) = kt then 0 else u2N hT ids (802 + (10 * a + b))) =
      (if 802 + (10 * a + b) = kt then 0 else relu (x a + y b - 1 + (posv (n := n) 16 - 1))) := by
    intro a ha' b hb'; rw [mem_range] at ha' hb'; rw [u2_p16 hT ids hn a b ha' hb']
  rw [sum_congr rfl fun a ha' => sum_congr rfl fun b hb' => hV a ha' b hb']
  by_cases hcur : n = 16
  · have hp : posv (n := n) 16 - 1 = 0 := by unfold posv; rw [if_pos hcur.symm]; ring
    rcases hkt with hkt | hkt
    · exact absurd hcur hkt
    have o := and_off x y (D 4) (D 9) ha hb lam 0 (posv (n := n) 16 - 1) (fun a => (bx a).1) (fun b => (by' b).1)
      (fun a => (bx a).2.1) (fun b => (by' b).2.1) (fun a h => (bx a).2.2.2 h) (fun b h => (by' b).2.2.2 h)
      le_rfl (by unfold lam; norm_num) (by rw [hp])
    calc _ = ∑ a ∈ range 10, ∑ b ∈ range 10,
          (if a = D 4 ∧ b = D 9 then 0 else relu (x a + y b - 1 + (posv (n := n) 16 - 1))) :=
          sum_congr rfl fun a ha' => sum_congr rfl fun b hb' => by
            rw [mem_range] at ha' hb'
            exact if_congr (by rw [hkt]; omega) rfl rfl
      _ ≤ 2 * lam := by linarith
  · have hp : posv (n := n) 16 - 1 = -1 := by unfold posv; rw [if_neg (fun h => hcur h.symm)]; ring
    have w := and_wrong x y (D 4) (D 9) ha hb lam 0 (posv (n := n) 16 - 1) (fun a => (bx a).1) (fun b => (by' b).1)
      (fun a => (bx a).2.1) (fun b => (by' b).2.1) (fun a h => (bx a).2.2.2 h) (fun b h => (by' b).2.2.2 h)
      le_rfl (by unfold lam; norm_num) (by rw [hp]; norm_num)
    calc _ ≤ ∑ a ∈ range 10, ∑ b ∈ range 10, relu (x a + y b - 1 + (posv (n := n) 16 - 1)) :=
          sum_le_sum fun a _ => sum_le_sum fun b _ => by
            split_ifs <;> first | exact le_max_right _ _ | exact le_rfl
      _ ≤ 2 * lam := by linarith

/-- The answer digit at step n − 11 (S₀ = c₀; S_j = (A_{j-1} + B_{j-1} + c_j) mod 10). -/
def tgt (A B : Num5) (n : ℕ) : ℕ :=
  if n = 11 then carry A B 0 else (dg A (n - 12) + dg B (n - 12) + carry A B (n - 11)) % 10

include hn hD hD10 hDA hDB in
/-- Block 2 emits the answer digit: one unit ≥ 1 - 2λ - ε₁ writes it, all others sum to ≤ 1/2. -/
lemma emission : ∃ kt : ℕ, kt < 902 ∧ outD kt = tgt A B n ∧
    1 - 2 * lam - 30 * lam ≤ u2N hT ids kt ∧
    ∑ m ∈ range 1024, (if m = kt then 0 else u2N hT ids m) ≤ 1 / 2 := by
  have hl := lam_pos
  obtain ⟨hc0, hc00⟩ := cv_bound hT ids hn D hD hD10 A B hDA hDB 0 (by norm_num)
  rw [abs_le] at hc0
  have hcar0 := carry_le A B 0
  obtain ⟨u0e, u1e⟩ := u2_01 hT ids hn
  -- the target unit
  let kt : ℕ :=
    if n = 11 then carry A B 0
    else if n < 16 then 2 + 200 * (n - 12) + 2 * (10 * D (n - 12) + D (n - 12 + 5)) + carry A B (n - 11)
    else 802 + (10 * D 4 + D 9)
  have hDl : ∀ h, h < 10 → D h < 10 := hD10
  have hkt902 : kt < 902 := by
    simp only [kt]
    have := hD10 (n - 12) (by omega); have := hD10 (n - 12 + 5) (by omega)
    have := hD10 4 (by norm_num); have := hD10 9 (by norm_num)
    have := carry_le A B (n - 11)
    split_ifs <;> omega
  refine ⟨kt, hkt902, ?_, ?_, ?_⟩
  · -- the target unit writes the answer digit
    simp only [kt, tgt]
    obtain ⟨h1, h2⟩ := hn
    split_ifs with h11 h16
    · have := carry_le A B 0
      rcases (show carry A B 0 = 0 ∨ carry A B 0 = 1 by omega) with hC | hC <;> rw [hC] <;> rfl
    · have hq : n - 11 = n - 12 + 1 := by omega
      rw [outD_seg _ _ _ _ (by omega) (hD10 _ (by omega)) (hD10 _ (by omega)) (by have := carry_le A B (n - 11); omega),
        hDA (n - 12) (by omega), hDB (n - 12) (by omega)]
    · have hn16 : n = 16 := by omega
      subst hn16
      rw [outD_p16 _ _ (hD10 4 (by norm_num)) (hD10 9 (by norm_num)), hDA 4 (by norm_num),
        show (9 : ℕ) = 4 + 5 from rfl, hDB 4 (by norm_num), show 16 - 12 = 4 from rfl,
        show 16 - 11 = 5 from rfl, carry_five, add_zero]
  · -- the target unit fires
    simp only [kt]
    split_ifs with h11 h16
    · have hp : posv (n := n) 11 = 1 := by unfold posv; rw [if_pos h11.symm]
      rcases (show carry A B 0 = 0 ∨ carry A B 0 = 1 by omega) with hC | hC <;> rw [hC] at hc0 ⊢ <;>
        push_cast at hc0
      · rw [u0e, hp]; unfold relu; exact le_trans (by linarith [hc0.2]) (le_max_left _ _)
      · rw [u1e, hp]; unfold relu; exact le_trans (by linarith [hc0.1]) (le_max_left _ _)
    · set q := n - 12 with hq
      have hq4 : q < 4 := by omega
      have hnq : n = 12 + q := by omega
      have hp : posv (n := n) (12 + q) = 1 := by unfold posv; rw [if_pos hnq.symm]
      have bx := S_bounds hT ids hn D hD ⟨q, by omega⟩ (D q)
      have by' := S_bounds hT ids hn D hD ⟨q + 5, by omega⟩ (D (q + 5))
      obtain ⟨hc, -⟩ := cv_bound hT ids hn D hD hD10 A B hDA hDB (q + 1) (by omega)
      rw [abs_le, show n - 11 = q + 1 by omega] at *
      have := carry_le A B (q + 1)
      rcases (show carry A B (q + 1) = 0 ∨ carry A B (q + 1) = 1 by omega) with hC | hC <;>
        rw [hC] at hc ⊢ <;> push_cast at hc
      · rw [u2_seg hT ids hn q _ _ 0 hq4 (hD10 q (by omega)) (hD10 (q + 5) (by omega)) (by norm_num), if_pos rfl, hp]
        unfold relu
        exact le_trans (by linarith [bx.2.2.1 rfl, by'.2.2.1 rfl, hc.2]) (le_max_left _ _)
      · rw [u2_seg hT ids hn q (D q) (D (q + 5)) 1 hq4 (hD10 q (by omega)) (hD10 (q + 5) (by omega)) (by norm_num),
          if_neg one_ne_zero, hp]
        unfold relu
        exact le_trans (by linarith [bx.2.2.1 rfl, by'.2.2.1 rfl, hc.1]) (le_max_left _ _)
    · have hn16 : n = 16 := by omega
      have hp : posv (n := n) 16 = 1 := by unfold posv; rw [if_pos hn16.symm]
      have bx := S_bounds hT ids hn D hD ⟨4, by norm_num⟩ (D 4)
      have by' := S_bounds hT ids hn D hD ⟨9, by norm_num⟩ (D 9)
      rw [u2_p16 hT ids hn (D 4) (D 9) (hD10 4 (by norm_num)) (hD10 9 (by norm_num)), hp]
      unfold relu
      exact le_trans (by linarith [bx.2.2.1 rfl, by'.2.2.1 rfl]) (le_max_left _ _)
  · -- everything else is small
    rw [sum_units (fun m => if m = kt then 0 else u2N hT ids m)
      (fun m hm => by beta_reduce; split_ifs <;> first | rfl | exact u2_hi hT ids m hm)]
    have hsegs : ∀ q ∈ range 4, ∑ a ∈ range 10, ∑ b ∈ range 10,
        ((if 2 + 200 * q + 2 * (10 * a + b) = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b))) +
          (if 2 + 200 * q + 2 * (10 * a + b) + 1 = kt then 0 else u2N hT ids (2 + 200 * q + 2 * (10 * a + b) + 1)))
        ≤ 2 * lam + 19 * (30 * lam) := by
      intro q hq
      rw [mem_range] at hq
      apply seg_bound hT ids hn D hD hD10 A B hDA hDB q hq kt
      by_cases h : n = 12 + q
      · right; simp only [kt]; rw [if_neg (by omega), if_pos (by omega), show n - 12 = q by omega,
          show n - 11 = q + 1 by omega]
      · left; exact h
    have h16 := p16_bound hT ids hn D hD hD10 kt (by
      by_cases h : n = 16
      · right; simp only [kt]; rw [if_neg (by omega), if_neg (by omega)]
      · left; exact h)
    have h01 : (if 0 = kt then 0 else u2N hT ids 0) + (if 1 = kt then 0 else u2N hT ids 1) ≤ 30 * lam := by
      rw [u0e, u1e]
      by_cases h11 : n = 11
      · have hp : posv (n := n) 11 = 1 := by unfold posv; rw [if_pos h11.symm]
        have hk : kt = carry A B 0 := by simp only [kt, if_pos h11]
        rw [hp, hk]
        rcases (show carry A B 0 = 0 ∨ carry A B 0 = 1 by omega) with hC | hC <;> rw [hC] at hc0 ⊢ <;>
          push_cast at hc0 <;> simp <;> unfold relu <;> exact max_le (by linarith [hc0.1, hc0.2]) (by linarith)
      · have hp : posv (n := n) 11 = 0 := by unfold posv; rw [if_neg (fun h => h11 h.symm)]
        rw [hp]
        have e0 : relu (0 - cv hT ids 0) = 0 := by unfold relu; exact max_eq_right (by linarith [hc00])
        have hcr0 : (carry A B 0 : ℝ) ≤ 1 := by exact_mod_cast hcar0
        have e1 : relu (0 + cv hT ids 0 - 1) ≤ 30 * lam := by
          unfold relu; exact max_le (by linarith [hc0.2, hcr0]) (by linarith)
        split_ifs <;> linarith [le_max_right (0 + cv hT ids 0 - 1) 0]
    have hq := sum_le_sum hsegs
    simp only [sum_const, card_range, nsmul_eq_mul] at hq
    have : (4 : ℝ) * (2 * lam + 19 * (30 * lam)) + 30 * lam + 2 * lam ≤ 1 / 2 := by unfold lam; norm_num
    push_cast at hq
    linarith

include hn in
/-- The logits are the final residual read through the head (blocks 1, 2 have zero attention). -/
lemma logits_eq (t : Fin 12) : addModel.logits hT ids (Fin.last n) t =
    (addModel.Wh *ᵥ R hT ids + addModel.bh) t := by
  have h1 : ∀ x : Fin (n + 1) → Fin 160 → ℝ, addModel_block1.apply x (Fin.last n) =
      x (Fin.last n) + addModel_block1.mlp (x (Fin.last n)) := by
    intro x
    show (x (Fin.last n) + addModel_block1_attn.apply (fun t => x t) (Fin.last n)) +
      addModel_block1.mlp (x (Fin.last n) + addModel_block1_attn.apply (fun t => x t) (Fin.last n)) = _
    rw [(attn12_zero _ _).1, add_zero]
  have h2 : ∀ x : Fin (n + 1) → Fin 160 → ℝ, addModel_block2.apply x (Fin.last n) =
      x (Fin.last n) + addModel_block2.mlp (x (Fin.last n)) := by
    intro x
    show (x (Fin.last n) + addModel_block2_attn.apply (fun t => x t) (Fin.last n)) +
      addModel_block2.mlp (x (Fin.last n) + addModel_block2_attn.apply (fun t => x t) (Fin.last n)) = _
    rw [(attn12_zero _ _).2, add_zero]
  have hb : applyBlocks addModel.blocks (X hT ids) (Fin.last n) = R hT ids := by
    show addModel_block2.apply (addModel_block1.apply (addModel_block0.apply (X hT ids))) (Fin.last n) = _
    rw [h2, h1]; rfl
  show (addModel.Wh *ᵥ Norm.apply .id (applyBlocks addModel.blocks (X hT ids) (Fin.last n)) + addModel.bh) t = _
  rw [hb]; rfl

include hn hD hD10 hDA hDB in
/-- **One generation step** at position n ∈ [11, 16]: the model emits answer digit n − 11. -/
theorem step : argmaxFirst (addModel.logits hT ids (Fin.last n)) = dtok (tgt A B n) := by
  have hl := lam_pos
  obtain ⟨kt, hkt, hout, hfire, hoff⟩ := emission hT ids hn D hD hD10 A B hDA hDB
  have htv : tgt A B n < 10 := by
    unfold tgt; have := carry_le A B 0; split_ifs <;> omega
  have hlog : ∀ t : Fin 12, addModel.logits hT ids (Fin.last n) t =
      if h : t.val < 10 then ∑ m ∈ range 1024, (if m < 902 ∧ outD m = t.val then u2N hT ids m else 0) else 0 := by
    intro t
    rw [logits_eq hT ids hn, Wh_apply]
    split_ifs with h
    · exact R_digit hT ids hn t.val h
    · rfl
  have hT' : (dtok (tgt A B n)).val = tgt A B n := by simp [dtok, Nat.mod_eq_of_lt htv]
  set F : ℕ → ℝ := fun m => if m < 902 ∧ outD m = tgt A B n then u2N hT ids m else 0 with hF
  have hnn : ∀ m ∈ range 1024, 0 ≤ F m := fun m _ => by
    simp only [F]; split_ifs
    · exact u2N_nonneg hT ids m
    · exact le_rfl
  have hhi : u2N hT ids kt ≤ ∑ m ∈ range 1024, F m := by
    have := single_le_sum hnn (mem_range.2 (show kt < 1024 by omega))
    simp only [F, if_pos (And.intro hkt hout)] at this
    exact this
  have hTlt : (dtok (tgt A B n)).val < 10 := by rw [hT']; exact htv
  have hlogT := hlog (dtok (tgt A B n))
  rw [dif_pos hTlt, hT'] at hlogT
  apply argmaxFirst_eq_of_lt
  intro j hj
  rw [hlogT, hlog j]
  have hfire' : (0 : ℝ) < 1 - 2 * lam - 30 * lam := by unfold lam; norm_num
  by_cases hj10 : j.val < 10
  · rw [dif_pos hj10]
    have hne : j.val ≠ tgt A B n := fun h => hj (Fin.ext (by rw [hT', h]))
    have : ∑ m ∈ range 1024, (if m < 902 ∧ outD m = j.val then u2N hT ids m else 0) ≤
        ∑ m ∈ range 1024, (if m = kt then 0 else u2N hT ids m) :=
      sum_le_sum fun m _ => by
        by_cases hm : m = kt
        · rw [hm, if_neg (fun h : kt < 902 ∧ outD kt = j.val => hne (h.2 ▸ hout)), if_pos rfl]
        · rw [if_neg hm]
          split_ifs
          · exact le_rfl
          · exact u2N_nonneg hT ids m
    have : (1 : ℝ) / 2 < 1 - 2 * lam - 30 * lam := by unfold lam; norm_num
    linarith
  · rw [dif_neg hj10]; linarith

end Step

/-! ### Generation -/

/-- **addition-five-digits, real-valued correctness of greedy generation** on every input. -/
theorem add_correct (A B : Num5) : addModel.generate (prompt A B) 6 = some (answer A B) := by
  have hans : (answer A B).length = 6 := by simp [answer]
  rw [← hans]
  apply Model.generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, hans]
  intro k hk h0 hL
  have hk' : k < 6 := by rw [hans] at hk; exact hk
  have hlen : (prompt A B ++ (answer A B).take k).length = (11 + k) + 1 := by
    simp [prompt, answer]; omega
  rw [Model.next_eq addModel _ h0 hL hlen (by omega)
    (fun t => (prompt A B ++ (answer A B).take k).get (Fin.cast hlen.symm t)) (fun t => rfl)]
  set D : ℕ → ℕ := fun h => if h < 5 then dg A h else dg B (h - 5)
  have hD10 : ∀ h, h < 10 → D h < 10 := fun h _ => by
    simp only [D]; split_ifs
    · exact dg_lt A h
    · exact dg_lt B (h - 5)
  have hDA : ∀ i, i < 5 → D i = dg A i := fun i hi => by simp only [D, if_pos hi]
  have hDB : ∀ i, i < 5 → D (i + 5) = dg B i := fun i hi => by simp only [D, if_neg (show ¬ i + 5 < 5 by omega)]; rfl
  have hD : ∀ h : Fin 10, ((prompt A B ++ (answer A B).take k).get
      (Fin.cast hlen.symm ⟨src h.val, by have := src_lt h.val h.isLt; omega⟩)).val = D h.val := by
    intro h
    simp only [List.get_eq_getElem, Fin.val_cast]
    rw [List.getElem_append_left (by simp [prompt]; have := src_lt h.val h.isLt; omega)]
    fin_cases h <;> simp [prompt, src, D, dg, dtok, List.getElem_append_left, List.getElem_append_right]
  rw [step (n := 11 + k) (by omega) _ (by omega) D hD hD10 A B hDA hDB]
  apply Fin.ext
  simp only [answer, List.get_eq_getElem, List.getElem_ofFn, dtok]
  rw [answer_digit A B k hk', tgt]
  have := carry_le A B 0
  interval_cases k <;> simp <;> omega

end VWC.Arch.Add
