/-
sort-five-digits (snapshot HistogramCircuit_v5_clean): real-valued correctness of greedy
generation on every input (10⁵ prompts), by mechanism, through three real LayerNorms.

* LN1 sees embeddings whose mean and variance depend only on the position class (input digit,
  `=`, k-th answer slot), so it is one affine map per position.
* Hence the head's scores depend only on positions: every input key gets the same weight W₀,
  answer-slot key `5+m` gets `W₀·r_m` with `r_m = exp(-Δ_m)`, Δ_m ∈ [1.95, 2.6].
* The attention output, summed over digits `≤ e`, is
  `GO_e = α(c_e - 5(e+1)/12) + Σ_m g_m([1 ≤ m ≤ c_e] - (e+1)μ_{5+m})`, where `c_e` counts input
  digits `≤ e`; `α ≈ 0.91–0.98`, `g_m ≈ 0.07–0.14`.
* fc1 rows have zero sum, so LN2's input-dependent mean cancels and the pre-activation is
  `z_e = (GO_e - k)/s₂ + (e+1)·B₂` with LN2's std `s₂ ∈ [4.32, 4.54]` (bounded by Jensen).
* The clamp `relu z - relu (z - SG)` is `≤ θ_lo·SG` when `c_e ≤ k` and `≥ θ_hi·SG` when
  `c_e > k` (interval arithmetic, per step k); fc2 takes differences of neighbouring clamps, so
  the k-th smallest digit gets residual ≥ 5.7 and every other digit ≤ 5.0.
* The final LayerNorm (γ = 1, β = 0) and the head (100·identity, zero row for `=`) are monotone.
-/
import VWC.Tasks.SortNum

namespace VWC.Arch.Sort

open Matrix Finset

/-! ### Specification (src/task.py, FiveDigitSortTask). Vocab "0123456789=". -/

def prompt (a : Fin 5 → Fin 10) : List (Fin 11) := List.ofFn (fun i => (a i).castSucc) ++ [10]

def sorted (a : Fin 5 → Fin 10) : List (Fin 10) := (List.ofFn a).mergeSort

def answer (a : Fin 5 → Fin 10) : List (Fin 11) := (sorted a).map Fin.castSucc

/-- In a sorted list, the entries `≤ d` are exactly the first `countP (· ≤ d)` ones. -/
lemma sorted_getElem_le_iff (d : Fin 10) : ∀ (l : List (Fin 10)), l.Pairwise (· ≤ ·) →
    ∀ (i : ℕ) (hi : i < l.length), (l[i] ≤ d ↔ i < l.countP (fun x => decide (x ≤ d)))
  | [], _, i, hi => by simp at hi
  | x :: l, hl, i, hi => by
    rw [List.pairwise_cons] at hl
    by_cases hxd : x ≤ d
    · rw [List.countP_cons_of_pos (by simpa using hxd)]
      cases i with
      | zero => simp [hxd]
      | succ i =>
        rw [List.getElem_cons_succ, sorted_getElem_le_iff d l hl.2 i (by simpa using hi)]
        omega
    · rw [List.countP_cons_of_neg (by simpa using hxd)]
      have hall : ∀ y ∈ l, ¬ y ≤ d := fun y hy h => hxd (le_trans (hl.1 y hy) h)
      have hc : l.countP (fun x => decide (x ≤ d)) = 0 := by
        rw [List.countP_eq_zero]; intro y hy; simpa using hall y hy
      rw [hc]
      cases i with
      | zero => simpa using hxd
      | succ i =>
        simp only [List.getElem_cons_succ, Nat.not_lt_zero, iff_false]
        exact hall _ (List.getElem_mem _)

/-! ### Generic helpers -/

lemma sum_fin24 (f : Fin 24 → ℝ) :
    ∑ i, f i = ∑ i ∈ range 24, (if h : i < 24 then f ⟨i, h⟩ else 0) := by
  rw [← Fin.sum_univ_eq_sum_range (fun i => if h : i < 24 then f ⟨i, h⟩ else 0) 24]
  simp

lemma sum_fin10 (f : Fin 10 → ℝ) :
    ∑ i, f i = ∑ i ∈ range 10, (if h : i < 10 then f ⟨i, h⟩ else 0) := by
  rw [← Fin.sum_univ_eq_sum_range (fun i => if h : i < 10 then f ⟨i, h⟩ else 0) 10]
  simp

/-- The 24 residual dims: digits, marker, step index, histogram, two anchors. -/
lemma sum24 (f : Fin 24 → ℝ) : ∑ i, f i =
    ∑ d : Fin 10, f (d.castLE (by norm_num)) + f 10 + f 11 + ∑ d : Fin 10, f (hd d) + f 22 + f 23 := by
  rw [sum_fin24, sum_fin10, sum_fin10]
  simp only [sum_range_succ, sum_range_zero]
  simp [hd, Fin.castLE]
  ring

lemma sum_sq_centre (x : Fin 24 → ℝ) :
    ∑ i, (x i - (∑ j, x j) / 24) ^ 2 = ∑ i, x i ^ 2 - (∑ j, x j) ^ 2 / 24 := by
  have h : ∀ i, (x i - (∑ j, x j) / 24) ^ 2 =
      x i ^ 2 - 2 * ((∑ j, x j) / 24) * x i + ((∑ j, x j) / 24) ^ 2 := fun i => by ring
  simp only [h, sum_add_distrib, sum_sub_distrib, ← mul_sum, sum_const, card_univ, Fintype.card_fin,
    nsmul_eq_mul]
  push_cast; ring

/-- A LayerNorm with γ = 1, in terms of the sum and the square-sum of its input. -/
lemma ln_apply {γ β : Fin 24 → ℝ} {eps : ℝ} (hγ : ∀ i, γ i = 1) (x : Fin 24 → ℝ) (i : Fin 24) :
    (Norm.ln γ β eps).apply x i = (x i - (∑ j, x j) / 24) /
      Real.sqrt ((∑ j, x j ^ 2) / 24 - ((∑ j, x j) / 24) ^ 2 + eps) + β i := by
  simp only [Norm.apply, hγ, mul_one, Nat.cast_ofNat]
  rw [sum_sq_centre]
  congr 4; ring

/-- Jensen for the square: a convex combination squared is at most the combination of squares. -/
lemma sq_avg_le {ι : Type*} (s : Finset ι) (w a : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (h1 : ∑ i ∈ s, w i = 1) : (∑ i ∈ s, w i * a i) ^ 2 ≤ ∑ i ∈ s, w i * a i ^ 2 := by
  set m := ∑ i ∈ s, w i * a i
  have h0 : 0 ≤ ∑ i ∈ s, w i * (a i - m) ^ 2 :=
    sum_nonneg fun i hi => mul_nonneg (hw i hi) (sq_nonneg _)
  have e : ∑ i ∈ s, w i * (a i - m) ^ 2 = ∑ i ∈ s, w i * a i ^ 2 - 2 * m * m + m ^ 2 * ∑ i ∈ s, w i := by
    have : ∀ i, w i * (a i - m) ^ 2 = w i * a i ^ 2 - 2 * m * (w i * a i) + m ^ 2 * w i :=
      fun i => by ring
    simp only [this, sum_add_distrib, sum_sub_distrib, ← mul_sum, m]
  rw [e, h1] at h0; nlinarith

lemma tokEmb_apply (v : Fin 11) (i : Fin 24) :
    sortModel.tokEmb v i = if v.val < 10 ∧ i.val = v.val then 1 else 0 := by
  by_cases hv : v.val < 10
  · have : v = (⟨v.val, hv⟩ : Fin 10).castSucc := by ext; simp
    rw [this, tokEmb_digit]
    simp only [Pi.single_apply, Fin.ext_iff, Fin.val_castLE, Fin.val_castSucc, hv, true_and]
  · have : v = 10 := by ext; simp; omega
    subst this
    rw [tokEmb_eq]; simp

lemma sum_ite_const (e : Fin 10) (μ : ℝ) :
    ∑ d : Fin 10, (if d ≤ e then μ else 0) = (e.val + 1) * μ := by
  rw [← sum_filter, sum_const, nsmul_eq_mul]
  congr 1
  have : (univ.filter fun d : Fin 10 => d ≤ e) = Iic e := by ext; simp
  rw [this, Fin.card_Iic]; push_cast; ring

/-- `Σ_{d ≤ e} ([v = d] - μ) = [v ≤ e] - (e+1)μ`. -/
lemma sum_le_indicator (v : ℕ) (e : Fin 10) (μ : ℝ) :
    ∑ d : Fin 10, (if d ≤ e then ((if v = d.val then (1 : ℝ) else 0) - μ) else 0) =
      (if v ≤ e.val then 1 else 0) - (e.val + 1) * μ := by
  have h1 : ∑ d : Fin 10, (if d ≤ e then (if v = d.val then (1 : ℝ) else 0) else 0) =
      if v ≤ e.val then 1 else 0 := by
    by_cases hv : v ≤ e.val
    · rw [if_pos hv, sum_eq_single (⟨v, by omega⟩ : Fin 10)]
      · simp [Fin.le_def, hv]
      · intro b _ hb
        split_ifs with h1 h2 <;> first | rfl | exact absurd (Fin.ext h2.symm) hb
      · simp
    · rw [if_neg hv]
      refine sum_eq_zero fun d _ => ?_
      split_ifs with h1 h2 <;> first | rfl | (exfalso; rw [Fin.le_def] at h1; omega)
  rw [← h1, ← sum_ite_const e μ, ← sum_sub_distrib]
  refine sum_congr rfl fun d _ => ?_
  split_ifs <;> ring

/-- Σ over digit dims of the token indicator. -/
lemma sum_tok (v : ℕ) : ∑ d : Fin 10, (if v = d.val then (1 : ℝ) else 0) = if v < 10 then 1 else 0 := by
  by_cases hv : v < 10
  · rw [if_pos hv, sum_eq_single (⟨v, hv⟩ : Fin 10)]
    · simp
    · intro b _ hb; rw [if_neg]; intro h; exact hb (Fin.ext h.symm)
    · simp
  · rw [if_neg hv]; refine sum_eq_zero fun d _ => ?_; rw [if_neg]; omega

/-- Square-sum over digit dims of a centred token indicator. -/
lemma sum_tok_sq (v : ℕ) (μ : ℝ) :
    ∑ d : Fin 10, ((if v = d.val then (1 : ℝ) else 0) - μ) ^ 2 =
      (if v < 10 then 1 else 0) * (1 - 2 * μ) + 10 * μ ^ 2 := by
  have h : ∀ d : Fin 10, ((if v = d.val then (1 : ℝ) else 0) - μ) ^ 2 =
      (if v = d.val then (1 : ℝ) else 0) * (1 - 2 * μ) + μ ^ 2 := fun d => by split_ifs <;> ring
  simp only [h, sum_add_distrib, ← sum_mul, sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [sum_tok]; push_cast; ring

/-- Clamp computed by an fc1 pair `relu z - relu (z - SG)`. -/
noncomputable def cl (z : ℝ) : ℝ := relu z - relu (z - SG)

lemma SG_pos : 0 < SG := by norm_num [SG]

lemma cl_range (z : ℝ) : 0 ≤ cl z ∧ cl z ≤ SG := by
  have := SG_pos
  unfold cl relu; rw [max_def, max_def]; split_ifs <;> constructor <;> linarith

lemma cl_le (z θ : ℝ) (hθ : 0 ≤ θ) (hz : z ≤ θ * SG) : cl z ≤ θ * SG := by
  have := SG_pos
  unfold cl relu; rw [max_def, max_def]; split_ifs <;> nlinarith

lemma le_cl (z θ : ℝ) (hθ : θ ≤ 1) (hz : θ * SG ≤ z) : θ * SG ≤ cl z := by
  have := SG_pos
  unfold cl relu; rw [max_def, max_def]; split_ifs <;> nlinarith

lemma fpe_symm_fst (r : Fin (1 * 24)) : (finProdFinEquiv.symm r).1 = 0 := Subsingleton.elim _ _

lemma fpe_symm_snd (r : Fin (1 * 24)) : (finProdFinEquiv.symm r).2 = ⟨r.val, by omega⟩ := by
  have h : finProdFinEquiv.symm r = ((0 : Fin 1), (⟨r.val, by omega⟩ : Fin 24)) := by
    rw [Equiv.symm_apply_eq]; ext; rw [fpe0]
  rw [h]

lemma fpe_symm_snd_val (r : Fin (1 * 24)) : ((finProdFinEquiv.symm r).2).val = r.val :=
  congrArg Fin.val (fpe_symm_snd r)

/-- Jensen bound per position class: `G² Σ_d u_d² ≤ 24` (digit token except at position 5). -/
lemma jensen_class (t : ℕ) (ht : t ≤ 9) :
    G ^ 2 * ((if t = 5 then 0 else 1) * (1 - 2 * mu t) + 10 * mu t ^ 2) ≤
      24 * (Qx t / 24 - mu t ^ 2 + EPS) := by
  interval_cases t <;> norm_num [G, mu, Sx, Qx, EPS]

section Step

variable {k : ℕ} (hT : 5 + k + 1 ≤ 16) (ids : Fin (5 + k + 1) → Fin 11)

/-- Embedded prompt. -/
noncomputable def X (t : Fin (5 + k + 1)) : Fin 24 → ℝ :=
  sortModel.tokEmb (ids t) + sortModel.posEmb (Fin.castLE hT t)

lemma X_apply (t : Fin (5 + k + 1)) (i : Fin 24) : X hT ids t i =
    (if (ids t).val < 10 ∧ i.val = (ids t).val then 1 else 0) + (if i = 10 then mk t.val else 0) +
    (if i = 11 then kk t.val else 0) + (if i = 22 then 15 else 0) + (if i = 23 then -15 else 0) := by
  simp only [X, Pi.add_apply, tokEmb_apply, posEmb_eq, Pi.single_apply, Fin.val_castLE]
  ring

lemma X_dig (t : Fin (5 + k + 1)) (d : Fin 10) :
    X hT ids t (d.castLE (by norm_num)) = if (ids t).val = d.val then 1 else 0 := by
  rw [X_apply]
  have h10 : (d.castLE (by norm_num) : Fin 24) ≠ 10 := by
    intro h; have := congrArg Fin.val h; simp at this; omega
  have h11 : (d.castLE (by norm_num) : Fin 24) ≠ 11 := by
    intro h; have := congrArg Fin.val h; simp at this; omega
  have h22 : (d.castLE (by norm_num) : Fin 24) ≠ 22 := by
    intro h; have := congrArg Fin.val h; simp at this; omega
  have h23 : (d.castLE (by norm_num) : Fin 24) ≠ 23 := by
    intro h; have := congrArg Fin.val h; simp at this; omega
  rw [if_neg h10, if_neg h11, if_neg h22, if_neg h23]
  simp only [Fin.val_castLE, add_zero]
  by_cases h : (ids t).val = d.val
  · rw [if_pos ⟨by omega, h.symm⟩, if_pos h]
  · rw [if_neg (fun h' => h h'.2.symm), if_neg h]

lemma X_hd (t : Fin (5 + k + 1)) (d : Fin 10) : X hT ids t (hd d) = 0 := by
  rw [X_apply]
  have hv := (ids t).isLt
  have h0 : ¬ ((ids t).val < 10 ∧ (hd d).val = (ids t).val) := by simp [hd]; omega
  have h10 : hd d ≠ 10 := by intro h; have := congrArg Fin.val h; simp [hd] at this; omega
  have h11 : hd d ≠ 11 := by intro h; have := congrArg Fin.val h; simp [hd] at this; omega
  have h22 : hd d ≠ 22 := by intro h; have := congrArg Fin.val h; simp [hd] at this; omega
  have h23 : hd d ≠ 23 := by intro h; have := congrArg Fin.val h; simp [hd] at this; omega
  rw [if_neg h0, if_neg h10, if_neg h11, if_neg h22, if_neg h23]; ring

lemma X_10 (t : Fin (5 + k + 1)) : X hT ids t 10 = mk t.val := by
  rw [X_apply]; have : (ids t).val < 11 := (ids t).isLt
  simp; intro h; omega

lemma X_11 (t : Fin (5 + k + 1)) : X hT ids t 11 = kk t.val := by
  rw [X_apply]; have : (ids t).val < 11 := (ids t).isLt
  simp; intro h; omega

lemma X_22 (t : Fin (5 + k + 1)) : X hT ids t 22 = 15 := by
  rw [X_apply]; have : (ids t).val < 11 := (ids t).isLt
  simp; intro h; omega

lemma X_23 (t : Fin (5 + k + 1)) : X hT ids t 23 = -15 := by
  rw [X_apply]; have : (ids t).val < 11 := (ids t).isLt
  simp; intro h; omega

variable (hk : k ≤ 4) (hdig : ∀ t : Fin (5 + k + 1), t.val ≠ 5 → (ids t).val < 10)
  (h5 : ∀ t : Fin (5 + k + 1), t.val = 5 → ids t = 10)

include hk hdig h5 in
lemma sums_X (t : Fin (5 + k + 1)) :
    ∑ i, X hT ids t i = Sx t.val ∧ ∑ i, X hT ids t i ^ 2 = Qx t.val := by
  have ht : t.val ≤ 9 := by omega
  rw [sum24, sum24]
  simp only [X_dig, X_10, X_11, X_22, X_23, X_hd, sum_const_zero, add_zero]
  have hsq : ∀ d : Fin 10, (if (ids t).val = d.val then (1 : ℝ) else 0) ^ 2 =
      if (ids t).val = d.val then 1 else 0 := fun d => by split_ifs <;> norm_num
  simp only [hsq, sum_tok, zero_pow two_ne_zero, sum_const_zero, add_zero]
  have hmk : mk t.val = if t.val < 5 then 1 else -1 := by
    unfold mk; split_ifs <;> first | rfl | omega
  have hkk : kk t.val = if t.val ≤ 5 then 0 else (t.val : ℝ) - 5 := by
    unfold kk; split_ifs <;> first | rfl | omega
  have htok : (if (ids t).val < 10 then (1 : ℝ) else 0) = if t.val = 5 then 0 else 1 := by
    by_cases h2 : t.val = 5
    · have : (ids t).val = 10 := by rw [h5 t h2]; rfl
      rw [if_neg (by omega), if_pos h2]
    · rw [if_pos (hdig t h2), if_neg h2]
  rw [hmk, hkk, htok]
  unfold Sx Qx
  split_ifs <;> first | (exfalso; omega) | exact ⟨by ring, by ring⟩

/-- LN1 output: one affine map per position. -/
noncomputable def U (t : Fin (5 + k + 1)) : Fin 24 → ℝ := fun i => (X hT ids t i - mu t.val) / sd t.val

include hk hdig h5 in
lemma LN1_X (t : Fin (5 + k + 1)) : sortModel_block0.ln1.apply (X hT ids t) = U hT ids t := by
  funext i
  obtain ⟨hs, hq⟩ := sums_X hT ids hk hdig h5 t
  show (Norm.ln _ _ _).apply _ i = _
  rw [ln_apply (fun i => (gammas i).1), hs, hq, ln1_beta]
  simp [U, mu, sd, EPS]

/-! ### The head: weights depend only on positions -/

lemma U_10 (t : Fin (5 + k + 1)) : U hT ids t 10 = nu t.val := by
  simp only [U, X_10, nu]

lemma U_dig (t : Fin (5 + k + 1)) (d : Fin 10) : U hT ids t (d.castLE (by norm_num)) =
    ((if (ids t).val = d.val then 1 else 0) - mu t.val) / sd t.val := by
  simp only [U, X_dig]

lemma qk_dot (p j : Fin (5 + k + 1)) :
    ((fun c => (A0.Wq *ᵥ U hT ids p) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ U hT ids j) (finProdFinEquiv ((0 : Fin 1), c)))) =
      -100 * nu p.val * nu j.val := by
  simp only [dotProduct, Wq_apply, Wk_apply, fpe0]
  rw [sum_eq_single (0 : Fin 24)]
  · simp [U_10]; ring
  · intro b _ hb
    have : b.val ≠ 0 := fun h => hb (Fin.ext h)
    simp [this]
  · simp

/-- The head's weights at the last position. -/
noncomputable def W (j : Fin (5 + k + 1)) : ℝ :=
  headWeight (fun t c => (A0.Wq *ᵥ U hT ids t) (finProdFinEquiv ((0 : Fin 1), c)))
    (fun t c => (A0.Wk *ᵥ U hT ids t) (finProdFinEquiv ((0 : Fin 1), c))) (Fin.last (5 + k)) j

lemma W_softmax (j : Fin (5 + k + 1)) :
    W hT ids j = softmax (fun j : Fin (5 + k + 1) => Lam k * nu j.val) j := by
  unfold W
  rw [headWeight_last]
  congr 1
  funext j'
  rw [qk_dot]
  simp only [Fin.val_last, Lam, Nat.cast_ofNat]
  ring

lemma W_rel (j : Fin (5 + k + 1)) :
    W hT ids j = W hT ids 0 * Real.exp (Lam k * (nu j.val - nu 0)) := by
  rw [W_softmax, W_softmax]
  unfold softmax
  rw [mul_sub, Real.exp_sub]
  simp only [Fin.val_zero]
  field_simp

noncomputable def W0 : ℝ := W hT ids 0

lemma mu_in (t : ℕ) (ht : t < 5) : mu t = 1 / 12 := by simp [mu, Sx, ht]; norm_num
lemma sd_in (t : ℕ) (ht : t < 5) : sd t = sd 0 := by simp [sd, Qx, mu, Sx, ht]
lemma nu_in (t : ℕ) (ht : t < 5) : nu t = nu 0 := by simp [nu, mk, mu, sd, Sx, Qx, ht]

lemma W_in (j : Fin (5 + k + 1)) (hj : j.val < 5) : W hT ids j = W0 hT ids := by
  rw [W_rel, nu_in _ hj]; simp [W0]

lemma W_out (j : Fin (5 + k + 1)) (hj : 5 ≤ j.val) : W hT ids j = W0 hT ids * rr k (j.val - 5) := by
  rw [W_rel]; unfold rr W0; rw [show 5 + (j.val - 5) = j.val by omega]

/-- Weights as a function of the key position. -/
noncomputable def Wf (t : ℕ) : ℝ := if t < 5 then W0 hT ids else W0 hT ids * rr k (t - 5)

lemma W_eq_Wf (j : Fin (5 + k + 1)) : W hT ids j = Wf hT ids j.val := by
  unfold Wf; split_ifs with h
  · exact W_in hT ids j h
  · exact W_out hT ids j (by omega)

lemma sum_split (f : ℕ → ℝ) :
    ∑ j : Fin (5 + k + 1), f j.val = ∑ t ∈ range 5, f t + ∑ m ∈ range (k + 1), f (5 + m) := by
  rw [Fin.sum_univ_eq_sum_range f (5 + k + 1), show 5 + k + 1 = 5 + (k + 1) by omega, sum_range_add]

lemma W_nonneg (j : Fin (5 + k + 1)) : 0 ≤ W hT ids j := by
  rw [W_softmax]; exact softmax_nonneg _ _

lemma W_sum : ∑ j, W hT ids j = 1 := by
  simp only [W_softmax]; exact softmax_sum _

lemma W0_norm : W0 hT ids * (5 + ∑ m ∈ range (k + 1), rr k m) = 1 := by
  have h := W_sum hT ids
  simp only [W_eq_Wf] at h
  rw [sum_split (f := Wf hT ids)] at h
  have h1 : ∑ t ∈ range 5, Wf hT ids t = 5 * W0 hT ids := by
    simp [Wf, sum_range_succ]; ring
  have h2 : ∑ m ∈ range (k + 1), Wf hT ids (5 + m) = W0 hT ids * ∑ m ∈ range (k + 1), rr k m := by
    rw [mul_sum]; refine sum_congr rfl fun m _ => ?_; simp [Wf]
  rw [h1, h2] at h; linarith

/-! ### Attention output -/

/-- Attention output on digit dim d, before `W_o`. -/
noncomputable def od (d : Fin 10) : ℝ := ∑ j, W hT ids j * U hT ids j (d.castLE (by norm_num))

lemma attn_hd (d : Fin 10) :
    A0.apply (fun t => U hT ids t) (Fin.last (5 + k)) (hd d) = G * od hT ids d := by
  simp only [Attn.apply, Wo_apply, hd]
  rw [dif_pos (by omega)]
  congr 1
  simp only [fpe_symm_fst, fpe_symm_snd, od, W, Wv_apply, fpe0]
  refine sum_congr rfl fun j _ => ?_
  split_ifs with h
  · congr 2; ext; simp only [fpe_symm_snd_val]; simp
  · exfalso; simp at h

lemma attn_other (i : Fin 24) (hi : ¬ (12 ≤ i.val ∧ i.val < 22)) :
    A0.apply (fun t => U hT ids t) (Fin.last (5 + k)) i = 0 := by
  simp only [Attn.apply, Wo_apply, dif_neg hi]

/-- Attention output summed over digits `≤ e`, times `W_o`'s gain. -/
noncomputable def GO (e : Fin 10) : ℝ := ∑ d : Fin 10, if d ≤ e then G * od hT ids d else 0

lemma GO_eq (e : Fin 10) : GO hT ids e = ∑ j, W hT ids j *
    (G * ((if (ids j).val ≤ e.val then 1 else 0) - (e.val + 1) * mu j.val) / sd j.val) := by
  unfold GO od
  have h1 : ∀ d : Fin 10, (if d ≤ e then G * ∑ j, W hT ids j * U hT ids j (d.castLE (by norm_num))
      else 0) = ∑ j, W hT ids j *
        (G * (if d ≤ e then ((if (ids j).val = d.val then 1 else 0) - mu j.val) else 0) / sd j.val) := by
    intro d
    split_ifs
    · rw [mul_sum]; refine sum_congr rfl fun j _ => ?_; rw [U_dig]; ring
    · simp
  rw [sum_congr rfl fun d _ => h1 d, sum_comm]
  refine sum_congr rfl fun j _ => ?_
  rw [← mul_sum, ← sum_div, ← mul_sum, sum_le_indicator]

/-! ### Counting form of the attention output -/

/-- Token value at position t (`10` for `=` or out of range). -/
noncomputable def tk (t : ℕ) : ℕ := if h : t < 5 + k + 1 then (ids ⟨t, h⟩).val else 10

lemma tk_eq (j : Fin (5 + k + 1)) : (ids j).val = tk ids j.val := by simp [tk, j.isLt]

lemma GO9_eq : GO hT ids 9 = ∑ d : Fin 10, G * od hT ids d := by
  unfold GO
  refine sum_congr rfl fun d _ => ?_
  rw [if_pos (by rw [Fin.le_def]; have := d.isLt; simp; omega)]

variable (c : ℕ → ℕ)
  (hcnt : ∀ e : Fin 10, (∑ t ∈ range 5, if tk ids t ≤ e.val then (1 : ℝ) else 0) = c e.val)
  (hsort : ∀ m e : ℕ, 1 ≤ m → m ≤ k → e < 10 → (tk ids (5 + m) ≤ e ↔ m ≤ c e))

include hk hdig h5 hcnt hsort in
lemma GO_formula (e : Fin 10) : GO hT ids e = GOx k (W0 hT ids) e.val (c e.val) := by
  rw [GO_eq]
  let F : ℕ → ℝ := fun t =>
    Wf hT ids t * (G * ((if tk ids t ≤ e.val then 1 else 0) - (e.val + 1) * mu t) / sd t)
  have hF : ∀ j : Fin (5 + k + 1), W hT ids j * (G * ((if (ids j).val ≤ e.val then 1 else 0) -
      (e.val + 1) * mu j.val) / sd j.val) = F j.val := by
    intro j; simp only [F, W_eq_Wf, tk_eq]
  rw [sum_congr rfl fun j _ => hF j, sum_split (f := F)]
  have hin : ∑ t ∈ range 5, F t =
      alpha (W0 hT ids) * ((c e.val : ℝ) - 5 * ((e.val : ℝ) + 1) / 12) := by
    have : ∀ t ∈ range 5, F t = alpha (W0 hT ids) * (if tk ids t ≤ e.val then 1 else 0) -
        alpha (W0 hT ids) * ((e.val + 1) / 12) := by
      intro t ht
      rw [mem_range] at ht
      simp only [F, Wf, if_pos ht, mu_in t ht, sd_in t ht, alpha]
      ring
    rw [sum_congr rfl this, sum_sub_distrib, ← mul_sum, hcnt e, sum_const, card_range]
    simp; ring
  have hout : ∑ m ∈ range (k + 1), F (5 + m) = ∑ m ∈ range (k + 1), gco k (W0 hT ids) m *
      ((if 1 ≤ m ∧ m ≤ c e.val then 1 else 0) - ((e.val : ℝ) + 1) * mu (5 + m)) := by
    refine sum_congr rfl fun m hm => ?_
    rw [mem_range] at hm
    have hind : (if tk ids (5 + m) ≤ e.val then (1 : ℝ) else 0) =
        if 1 ≤ m ∧ m ≤ c e.val then 1 else 0 := by
      rcases Nat.eq_zero_or_pos m with h0 | h0
      · subst h0
        have : tk ids 5 = 10 := by
          simp only [tk, dif_pos (show 5 < 5 + k + 1 by omega)]
          rw [h5 ⟨5, by omega⟩ rfl]; rfl
        simp only [add_zero, this]
        rw [if_neg (by have := e.isLt; omega), if_neg (by omega)]
      · have := hsort m e.val h0 (by omega) e.isLt
        by_cases h : m ≤ c e.val
        · rw [if_pos (this.2 h), if_pos ⟨h0, h⟩]
        · rw [if_neg (fun h' => h (this.1 h')), if_neg (fun h' => h h'.2)]
    simp only [F, Wf, if_neg (show ¬ 5 + m < 5 by omega), show 5 + m - 5 = m by omega, hind, gco]
    ring
  rw [hin, hout]; rfl

include hdig hcnt in
lemma c9 : c 9 = 5 := by
  have h := hcnt 9
  have h1 : ∀ t ∈ range 5, (if tk ids t ≤ (9 : Fin 10).val then (1 : ℝ) else 0) = 1 := by
    intro t ht
    rw [mem_range] at ht
    rw [if_pos]
    have := hdig ⟨t, by omega⟩ (by simp; omega)
    simp only [tk, dif_pos (show t < 5 + k + 1 by omega)]
    simp; omega
  rw [sum_congr rfl h1] at h
  simp at h
  exact_mod_cast h.symm

include hcnt in
lemma c_le (e : Fin 10) : c e.val ≤ 5 := by
  have h := hcnt e
  have : (∑ t ∈ range 5, if tk ids t ≤ e.val then (1 : ℝ) else 0) ≤ 5 := by
    calc _ ≤ ∑ t ∈ range 5, (1 : ℝ) := sum_le_sum fun t _ => by split_ifs <;> norm_num
      _ = 5 := by simp
  rw [h] at this
  exact_mod_cast this

/-! ### Residual stream after attention, and LN2 -/

noncomputable def Y : Fin 24 → ℝ := fun i =>
  X hT ids (Fin.last (5 + k)) i + A0.apply (fun t => U hT ids t) (Fin.last (5 + k)) i

lemma Y_dig (d : Fin 10) : Y hT ids (d.castLE (by norm_num)) =
    if (ids (Fin.last (5 + k))).val = d.val then 1 else 0 := by
  simp only [Y, X_dig, attn_other hT ids (d.castLE (by norm_num)) (by have := d.isLt; simp; omega),
    add_zero]

include hk in
lemma Y_10 : Y hT ids 10 = -1 := by
  simp only [Y, X_10, attn_other hT ids 10 (by decide), add_zero, Fin.val_last, mk]
  rw [if_neg (by omega), if_pos (by omega)]

include hk in
lemma Y_11 : Y hT ids 11 = k := by
  simp only [Y, X_11, attn_other hT ids 11 (by decide), add_zero, Fin.val_last, kk]
  split_ifs with h
  · push_cast; ring
  · have : k = 0 := by omega
    subst this; simp

lemma Y_22 : Y hT ids 22 = 15 := by
  simp only [Y, X_22, attn_other hT ids 22 (by decide), add_zero]

lemma Y_23 : Y hT ids 23 = -15 := by
  simp only [Y, X_23, attn_other hT ids 23 (by decide), add_zero]

lemma Y_hd (d : Fin 10) : Y hT ids (hd d) = G * od hT ids d := by
  simp only [Y, X_hd, attn_hd, zero_add]

/-- Indicator that the last token is a digit (0 at step 0, where it is `=`). -/
noncomputable def Tl : ℝ := if (ids (Fin.last (5 + k))).val < 10 then 1 else 0

include hk in
lemma sums_Y : ∑ i, Y hT ids i = Tl ids - 1 + k + GO hT ids 9 ∧
    ∑ i, Y hT ids i ^ 2 = Tl ids + 1 + (k : ℝ) ^ 2 + ∑ d : Fin 10, (G * od hT ids d) ^ 2 + 450 := by
  rw [sum24, sum24]
  simp only [Y_dig, Y_10 hT ids hk, Y_11 hT ids hk, Y_22, Y_23, Y_hd, GO9_eq]
  have hsq : ∀ d : Fin 10, (if (ids (Fin.last (5 + k))).val = d.val then (1 : ℝ) else 0) ^ 2 =
      if (ids (Fin.last (5 + k))).val = d.val then 1 else 0 := fun d => by split_ifs <;> norm_num
  simp only [hsq, sum_tok, Tl]
  constructor <;> ring

lemma Tl_range : 0 ≤ Tl ids ∧ Tl ids ≤ 1 := by
  unfold Tl; split_ifs <;> norm_num

include hk hdig h5 in
lemma jensen : ∑ d : Fin 10, (G * od hT ids d) ^ 2 ≤ 24 := by
  have h1 : ∀ d : Fin 10, (G * od hT ids d) ^ 2 ≤
      ∑ j, W hT ids j * (G * U hT ids j (d.castLE (by norm_num))) ^ 2 := by
    intro d
    have : G * od hT ids d = ∑ j, W hT ids j * (G * U hT ids j (d.castLE (by norm_num))) := by
      unfold od; rw [mul_sum]; exact sum_congr rfl fun j _ => by ring
    rw [this]
    exact sq_avg_le univ _ _ (fun j _ => W_nonneg hT ids j) (W_sum hT ids)
  have h2 : ∀ j : Fin (5 + k + 1),
      ∑ d : Fin 10, (G * U hT ids j (d.castLE (by norm_num))) ^ 2 ≤ 24 := by
    intro j
    have hj : j.val ≤ 9 := by omega
    have hsd := sd_pos j.val hj
    have hsq := sd_sq j.val hj
    have e1 : ∀ d : Fin 10, (G * U hT ids j (d.castLE (by norm_num))) ^ 2 =
        G ^ 2 / sd j.val ^ 2 * ((if (ids j).val = d.val then 1 else 0) - mu j.val) ^ 2 := by
      intro d; rw [U_dig]; field_simp
    simp only [e1, ← mul_sum, sum_tok_sq]
    have hcls : (if (ids j).val < 10 then (1 : ℝ) else 0) = if j.val = 5 then 0 else 1 := by
      by_cases h2 : j.val = 5
      · have : (ids j).val = 10 := by rw [h5 j h2]; rfl
        rw [if_neg (by omega), if_pos h2]
      · rw [if_pos (hdig j h2), if_neg h2]
    rw [hcls, div_mul_eq_mul_div, div_le_iff₀ (by positivity), hsq]
    exact jensen_class j.val hj
  calc ∑ d : Fin 10, (G * od hT ids d) ^ 2
      ≤ ∑ d : Fin 10, ∑ j, W hT ids j * (G * U hT ids j (d.castLE (by norm_num))) ^ 2 :=
        sum_le_sum fun d _ => h1 d
    _ = ∑ j, W hT ids j * ∑ d : Fin 10, (G * U hT ids j (d.castLE (by norm_num))) ^ 2 := by
        rw [sum_comm]; simp only [mul_sum]
    _ ≤ ∑ j, W hT ids j * 24 :=
        sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (h2 j) (W_nonneg hT ids j)
    _ = 24 := by rw [← sum_mul, W_sum, one_mul]

/-- LN2's standard deviation at the last position. -/
noncomputable def S2 : ℝ :=
  Real.sqrt ((∑ i, Y hT ids i ^ 2) / 24 - ((∑ i, Y hT ids i) / 24) ^ 2 + EPS)

include hk hdig h5 in
lemma S2_bounds (hGO : 0 ≤ GO hT ids 9 ∧ GO hT ids 9 ≤ 1) :
    (432 / 100 : ℝ) ≤ S2 hT ids ∧ S2 hT ids ≤ 454 / 100 := by
  obtain ⟨hs, hq⟩ := sums_Y hT ids hk
  have hj := jensen hT ids hk hdig h5
  have hj0 : 0 ≤ ∑ d : Fin 10, (G * od hT ids d) ^ 2 := sum_nonneg fun d _ => sq_nonneg _
  have ht := Tl_range ids
  have hk' : (k : ℝ) ≤ 4 := by exact_mod_cast hk
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  unfold S2
  rw [hs, hq]
  apply sqrt_b (by norm_num) (by norm_num)
  · have h1 : (Tl ids - 1 + k + GO hT ids 9) ^ 2 ≤ 25 := by nlinarith
    norm_num [EPS]; nlinarith
  · norm_num [EPS]; nlinarith

/-- LN2 output. -/
noncomputable def L2 : Fin 24 → ℝ := sortModel_block0.ln2.apply (Y hT ids)

lemma L2_apply (i : Fin 24) : L2 hT ids i = (Y hT ids i - (∑ j, Y hT ids j) / 24) / S2 hT ids +
    sortModel_block0_ln2_beta i := by
  show (Norm.ln _ _ _).apply _ i = _
  rw [ln_apply (fun i => (gammas i).2.1)]; rfl

/-- fc1 pre-activation of the pair for digit e. -/
noncomputable def z (e : Fin 10) : ℝ := (GO hT ids e - k) / S2 hT ids + (e.val + 1) * B2

include hk in
lemma pre (e : Fin 10) :
    (sortModel_block0_W1 *ᵥ L2 hT ids) (e.castLE (by norm_num)) +
      sortModel_block0_b1 (e.castLE (by norm_num)) = z hT ids e ∧
    (sortModel_block0_W1 *ᵥ L2 hT ids) ⟨10 + e.val, by omega⟩ +
      sortModel_block0_b1 ⟨10 + e.val, by omega⟩ = z hT ids e - SG := by
  obtain ⟨h1, h2⟩ := W1_apply (L2 hT ids) e
  obtain ⟨b1a, b1b⟩ := b1_apply e
  set μ := (∑ j, Y hT ids j) / 24
  have hd' : ∀ d : Fin 10, L2 hT ids (hd d) = G * od hT ids d / S2 hT ids + (B2 - μ / S2 hT ids) := by
    intro d
    rw [L2_apply, Y_hd, ln2_beta, if_pos (by simp [hd]; omega)]; ring
  have hsum : ∑ d : Fin 10, (if d ≤ e then L2 hT ids (hd d) else 0) =
      GO hT ids e / S2 hT ids + (e.val + 1) * (B2 - μ / S2 hT ids) := by
    have : ∀ d : Fin 10, (if d ≤ e then L2 hT ids (hd d) else 0) =
        (if d ≤ e then G * od hT ids d else 0) / S2 hT ids + (if d ≤ e then B2 - μ / S2 hT ids else 0) := by
      intro d; split_ifs <;> simp [hd']
    rw [sum_congr rfl fun d _ => this d, sum_add_distrib, ← sum_div, sum_ite_const]; rfl
  have h11 : L2 hT ids 11 = (k - μ) / S2 hT ids := by
    rw [L2_apply, Y_11 hT ids hk, ln2_beta, if_neg (by decide)]; ring
  have h22 : L2 hT ids 22 = (15 - μ) / S2 hT ids := by
    rw [L2_apply, Y_22, ln2_beta, if_neg (by decide)]; ring
  have h23 : L2 hT ids 23 = (-15 - μ) / S2 hT ids := by
    rw [L2_apply, Y_23, ln2_beta, if_neg (by decide)]; ring
  rw [h1, h2, b1a, b1b, hsum, h11, h22, h23]
  unfold z
  constructor <;> ring

/-! ### MLP output and the final LayerNorm -/

/-- Residual stream at the last position after the block. -/
noncomputable def R : Fin 24 → ℝ := Y hT ids + sortModel_block0.mlp (L2 hT ids)

include hk in
lemma R_dig (d : Fin 10) : R hT ids (d.castLE (by norm_num)) = Y hT ids (d.castLE (by norm_num)) +
    OUT * cl (z hT ids d) -
      (if h : d.val = 0 then 0 else OUT * cl (z hT ids ⟨d.val - 1, by omega⟩)) := by
  simp only [R, Block.mlp, Pi.add_apply, sortModel_block0, b2_zero, Pi.zero_apply, add_zero]
  rw [W2_apply]
  obtain ⟨p1, p2⟩ := pre hT ids hk d
  simp only [p1, p2]
  split_ifs with h
  · unfold cl; ring
  · obtain ⟨q1, q2⟩ := pre hT ids hk ⟨d.val - 1, by omega⟩
    simp only [Fin.castLE_mk] at q1
    simp only [q1, q2]
    unfold cl; ring

lemma R_hi (i : Fin 24) (hi : 10 ≤ i.val) : R hT ids i = Y hT ids i := by
  simp only [R, Block.mlp, Pi.add_apply, sortModel_block0, b2_zero, Pi.zero_apply, add_zero]
  rw [W2_apply_hi _ i hi, add_zero]

include hk hdig h5 in
lemma logits_eq (t : Fin 11) : sortModel.logits hT ids (Fin.last (5 + k)) t =
    (sortModel.Wh *ᵥ sortModel.finalLn.apply (R hT ids) + sortModel.bh) t := by
  have hX : (fun t => sortModel_block0.ln1.apply (X hT ids t)) = fun t => U hT ids t :=
    funext (LN1_X hT ids hk hdig h5)
  have hblock : applyBlocks sortModel.blocks (X hT ids) (Fin.last (5 + k)) = R hT ids := by
    show (X hT ids (Fin.last (5 + k)) + A0.apply (fun t => sortModel_block0.ln1.apply (X hT ids t))
      (Fin.last (5 + k))) + sortModel_block0.mlp (sortModel_block0.ln2.apply
      (X hT ids (Fin.last (5 + k)) + A0.apply (fun t => sortModel_block0.ln1.apply (X hT ids t))
      (Fin.last (5 + k)))) = R hT ids
    rw [hX]; rfl
  rw [← hblock]; rfl

include hk hdig h5 hcnt hsort in
/-- **One generation step.** At step k the model emits the (k+1)-th smallest input digit. -/
theorem step (eT : Fin 10) (htgt : ∀ d : ℕ, d < 10 → (eT.val ≤ d ↔ k < c d)) :
    argmaxFirst (sortModel.logits hT ids (Fin.last (5 + k))) = eT.castSucc := by
  have hW := W0_norm hT ids
  have hc9 := c9 ids hdig c hcnt
  have hGO9 : GO hT ids 9 = GOx k (W0 hT ids) 9 5 := by
    rw [GO_formula hT ids hk hdig h5 c hcnt hsort 9]; simp [hc9]
  obtain ⟨_, _, _, _, _, _, _, g9a, g9b, _⟩ := numeric k hk (W0 hT ids) hW (443 / 100) (by norm_num)
  have hS2 := S2_bounds hT ids hk hdig h5 ⟨hGO9 ▸ g9a, hGO9 ▸ g9b⟩
  obtain ⟨θlo, θhi, hθlo, hθhi, hθhi0, hm1, hm2, _, _, hz⟩ :=
    numeric k hk (W0 hT ids) hW (S2 hT ids) hS2
  have hS2pos : 0 < S2 hT ids := by linarith [hS2.1]
  have hOUT : 0 < OUT := by norm_num [OUT]
  have hSG := SG_pos
  -- clamp bounds from the threshold checks
  have hkey : ∀ e : Fin 10, z hT ids e * S2 hT ids =
      GOx k (W0 hT ids) e.val (c e.val) - k + (e.val + 1) * B2 * S2 hT ids := by
    intro e; unfold z; rw [GO_formula hT ids hk hdig h5 c hcnt hsort e]; field_simp
  have clo : ∀ e : Fin 10, c e.val ≤ k → cl (z hT ids e) ≤ θlo * SG := by
    intro e he
    have := (hz e.val (c e.val) e.isLt (c_le ids c hcnt e)).1 he
    refine cl_le _ _ hθlo (le_of_mul_le_mul_right ?_ hS2pos)
    rw [hkey]; linarith
  have chi : ∀ e : Fin 10, k < c e.val → θhi * SG ≤ cl (z hT ids e) := by
    intro e he
    have := (hz e.val (c e.val) e.isLt (c_le ids c hcnt e)).2 he
    refine le_cl _ _ hθhi (le_of_mul_le_mul_right ?_ hS2pos)
    rw [hkey]; linarith
  have hY : ∀ d : Fin 10, 0 ≤ Y hT ids (d.castLE (by norm_num)) ∧ Y hT ids (d.castLE (by norm_num)) ≤ 1 := by
    intro d; rw [Y_dig]; split_ifs <;> norm_num
  -- residuals: the target digit beats every other digit
  set T := OUT * SG * (θhi - θlo)
  have h3 : 0 ≤ OUT * SG * θlo := mul_nonneg (mul_nonneg hOUT.le hSG.le) hθlo
  have hRt : T ≤ R hT ids (eT.castLE (by norm_num)) := by
    rw [R_dig hT ids hk]
    have h1 := chi eT ((htgt eT.val eT.isLt).1 le_rfl)
    have := (hY eT).1
    have hp1 := mul_le_mul_of_nonneg_left h1 hOUT.le
    split_ifs with h0
    · linarith
    · have h2 := clo ⟨eT.val - 1, by omega⟩ (by
        by_contra hc; push_neg at hc
        have := (htgt (eT.val - 1) (by omega)).2 hc; omega)
      have hp2 := mul_le_mul_of_nonneg_left h2 hOUT.le
      linarith
  have hRo : ∀ d : Fin 10, d ≠ eT → R hT ids (d.castLE (by norm_num)) < T := by
    intro d hd
    rw [R_dig hT ids hk]
    have hy := (hY d).2
    rcases lt_or_gt_of_ne (Fin.val_ne_of_ne hd) with hlt | hgt
    · have h1 := clo d (by
        by_contra hc; push_neg at hc
        have := (htgt d.val d.isLt).2 hc; omega)
      have hp1 := mul_le_mul_of_nonneg_left h1 hOUT.le
      split_ifs with h0
      · linarith
      · have hp2 := mul_nonneg hOUT.le (cl_range (z hT ids ⟨d.val - 1, by omega⟩)).1
        linarith
    · have h1 := chi ⟨d.val - 1, by omega⟩ ((htgt (d.val - 1) (by omega)).1 (by omega))
      rw [dif_neg (by omega)]
      have hp1 := mul_le_mul_of_nonneg_left (cl_range (z hT ids d)).2 hOUT.le
      have hp2 := mul_le_mul_of_nonneg_left h1 hOUT.le
      linarith
  -- the final LayerNorm and the head are monotone
  set SR := ∑ i, R hT ids i
  set sf := Real.sqrt ((∑ i, R hT ids i ^ 2) / 24 - (SR / 24) ^ 2 + EPS)
  have hsf : 0 < sf := by
    apply Real.sqrt_pos.2
    have h := sum_sq_centre (R hT ids)
    have : 0 ≤ ∑ i, (R hT ids i - SR / 24) ^ 2 := sum_nonneg fun i _ => sq_nonneg _
    have e2 : (∑ i, R hT ids i ^ 2) / 24 - (SR / 24) ^ 2 = (∑ i, (R hT ids i - SR / 24) ^ 2) / 24 := by
      rw [h]; ring
    rw [e2]; have : (0 : ℝ) < EPS := by norm_num [EPS]
    positivity
  have hlog : ∀ t : Fin 11, sortModel.logits hT ids (Fin.last (5 + k)) t =
      if h : t.val < 10 then 100 * ((R hT ids ⟨t.val, by omega⟩ - SR / 24) / sf) else 0 := by
    intro t
    rw [logits_eq hT ids hk hdig h5, Wh_apply]
    split_ifs
    · congr 1
      show (Norm.ln _ _ _).apply _ _ = _
      rw [ln_apply (fun i => (gammas i).2.2), final_beta]; simp; rfl
    · rfl
  -- SR is small: every digit residual is at most the target's
  have hSR : SR ≤ 10 * R hT ids (eT.castLE (by norm_num)) + 4 := by
    have hs : SR = ∑ d : Fin 10, R hT ids (d.castLE (by norm_num)) + R hT ids 10 + R hT ids 11 +
        ∑ d : Fin 10, R hT ids (hd d) + R hT ids 22 + R hT ids 23 := sum24 _
    have hhd : ∀ d : Fin 10, R hT ids (hd d) = G * od hT ids d := fun d => by
      rw [R_hi _ _ _ (by simp [hd]; omega), Y_hd]
    rw [hs, R_hi _ _ 10 (by decide), R_hi _ _ 11 (by decide), R_hi _ _ 22 (by decide),
      R_hi _ _ 23 (by decide), Y_10 hT ids hk, Y_11 hT ids hk, Y_22, Y_23]
    simp only [hhd]
    rw [← GO9_eq, hGO9]
    have hle : ∑ d : Fin 10, R hT ids (d.castLE (by norm_num)) ≤
        ∑ _d : Fin 10, R hT ids (eT.castLE (by norm_num)) := by
      refine sum_le_sum fun d _ => ?_
      by_cases h : d = eT
      · rw [h]
      · exact (hRo d h).le.trans hRt
    simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at hle
    have hk' : (k : ℝ) ≤ 4 := by exact_mod_cast hk
    push_cast at hle
    linarith
  apply argmaxFirst_eq_of_lt
  intro j hj
  rw [hlog, hlog]
  have heT : (eT.castSucc : Fin 11).val < 10 := by simp
  rw [dif_pos heT]
  have hcast : (⟨(eT.castSucc : Fin 11).val, by omega⟩ : Fin 24) = eT.castLE (by norm_num) := by
    ext; simp
  rw [hcast]
  have hT1 : 1 < T := by linarith
  split_ifs with hj10
  · have hd : (⟨j.val, hj10⟩ : Fin 10) ≠ eT := by
      intro h; apply hj; ext; simp [← h]
    have := hRo ⟨j.val, hj10⟩ hd
    have hcast' : (⟨j.val, by omega⟩ : Fin 24) = (⟨j.val, hj10⟩ : Fin 10).castLE (by norm_num) := by
      ext; simp
    rw [hcast']
    have : (R hT ids ((⟨j.val, hj10⟩ : Fin 10).castLE (by norm_num)) - SR / 24) / sf <
        (R hT ids (eT.castLE (by norm_num)) - SR / 24) / sf :=
      div_lt_div_of_pos_right (by linarith) hsf
    linarith
  · have : 0 < (R hT ids (eT.castLE (by norm_num)) - SR / 24) / sf :=
      div_pos (by linarith) hsf
    linarith

end Step

/-! ### Generation -/

lemma sorted_length (a : Fin 5 → Fin 10) : (sorted a).length = 5 := by
  simp [sorted, List.length_mergeSort]

lemma sorted_pairwise (a : Fin 5 → Fin 10) : (sorted a).Pairwise (· ≤ ·) :=
  List.pairwise_mergeSort' (· ≤ ·) (List.ofFn a)

lemma countP_ofFn (a : Fin 5 → Fin 10) (e : Fin 10) :
    (((List.ofFn a).countP (fun x => decide (x ≤ e)) : ℕ) : ℝ) =
      ∑ t ∈ range 5, if h : t < 5 then (if a ⟨t, h⟩ ≤ e then 1 else 0) else 0 := by
  simp only [List.ofFn_succ, List.ofFn_zero, List.countP_cons, List.countP_nil, sum_range_succ,
    sum_range_zero]
  simp
  split_ifs <;> norm_num

/-- The teacher-forced sequence at step k: prompt, then the first k sorted digits. -/
lemma seq_get (a : Fin 5 → Fin 10) (k : ℕ) (hk : k ≤ 4) (t : ℕ)
    (ht : t < (prompt a ++ (answer a).take k).length) :
    (prompt a ++ (answer a).take k)[t] =
      if h : t < 5 then (a ⟨t, h⟩).castSucc else if h' : t = 5 then 10 else
        ((sorted a)[t - 6]'(by rw [sorted_length]; simp [prompt, answer, sorted_length] at ht; omega)).castSucc := by
  have hp : (prompt a).length = 6 := by simp [prompt]
  by_cases h1 : t < 6
  · rw [List.getElem_append_left (by omega)]
    by_cases h2 : t < 5
    · rw [dif_pos h2]; simp [prompt]; interval_cases t <;> rfl
    · have : t = 5 := by omega
      subst this; simp [prompt]
  · rw [List.getElem_append_right (by omega), dif_neg (by omega), dif_neg (by omega)]
    simp [answer, hp]

/-- **sort-five-digits, real-valued correctness of greedy generation** on every input. -/
theorem sort_correct (a : Fin 5 → Fin 10) :
    sortModel.generate (prompt a) 5 = some (answer a) := by
  have hans : (answer a).length = 5 := by simp [answer, sorted_length]
  rw [← hans]
  apply Model.generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, hans]
  intro k hk h0 hL
  have hk' : k ≤ 4 := by rw [hans] at hk; omega
  have hlen : (prompt a ++ (answer a).take k).length = (5 + k) + 1 := by
    simp [prompt, hans]; omega
  rw [Model.next_eq sortModel _ h0 hL hlen (by omega)
    (fun t => (prompt a ++ (answer a).take k).get (Fin.cast hlen.symm t)) (fun t => rfl)]
  set ids : Fin (5 + k + 1) → Fin 11 := fun t => (prompt a ++ (answer a).take k).get (Fin.cast hlen.symm t)
  have hid : ∀ t : Fin (5 + k + 1), ids t = (prompt a ++ (answer a).take k)[t.val]'(by omega) := fun t => rfl
  have hval : ∀ t : Fin (5 + k + 1), (ids t).val =
      if h : t.val < 5 then (a ⟨t.val, h⟩).val else if t.val = 5 then 10 else
        ((sorted a)[t.val - 6]'(by rw [sorted_length]; omega)).val := by
    intro t; rw [hid, seq_get a k hk']
    split_ifs <;> simp
  have htk : ∀ t : ℕ, (ht : t < 5 + k + 1) → tk ids t =
      if h : t < 5 then (a ⟨t, h⟩).val else if t = 5 then 10 else
        ((sorted a)[t - 6]'(by rw [sorted_length]; omega)).val := by
    intro t ht; simp only [tk, dif_pos ht]; exact hval ⟨t, ht⟩
  have hdig : ∀ t : Fin (5 + k + 1), t.val ≠ 5 → (ids t).val < 10 := by
    intro t ht; rw [hval]; split_ifs <;> first | omega | exact Fin.isLt _
  have h5 : ∀ t : Fin (5 + k + 1), t.val = 5 → ids t = 10 := by
    intro t ht; apply Fin.ext; rw [hval, dif_neg (by omega), if_pos ht]; rfl
  let c : ℕ → ℕ := fun d => if h : d < 10 then (sorted a).countP (fun x => decide (x ≤ ⟨d, h⟩)) else 5
  have hperm : ∀ e : Fin 10, (sorted a).countP (fun x => decide (x ≤ e)) =
      (List.ofFn a).countP (fun x => decide (x ≤ e)) :=
    fun e => (List.mergeSort_perm _ _).countP_eq _
  have hcnt : ∀ e : Fin 10, (∑ t ∈ range 5, if tk ids t ≤ e.val then (1 : ℝ) else 0) = c e.val := by
    intro e
    simp only [c, dif_pos e.isLt, Fin.eta, hperm, countP_ofFn]
    refine sum_congr rfl fun t ht => ?_
    rw [mem_range] at ht
    rw [htk t (by omega), dif_pos ht, dif_pos ht]
    exact if_congr Fin.le_def.symm rfl rfl
  have hsort : ∀ m e : ℕ, 1 ≤ m → m ≤ k → e < 10 → (tk ids (5 + m) ≤ e ↔ m ≤ c e) := by
    intro m e hm1 hmk he
    rw [htk (5 + m) (by omega), dif_neg (by omega), if_neg (by omega)]
    simp only [c, dif_pos he]
    have := sorted_getElem_le_iff ⟨e, he⟩ (sorted a) (sorted_pairwise a) (5 + m - 6)
      (by rw [sorted_length]; omega)
    rw [Fin.le_def] at this
    simp only at this
    rw [this]; omega
  have hk5 : k < (sorted a).length := by rw [sorted_length]; omega
  have htgt : ∀ d : ℕ, d < 10 → (((sorted a)[k]'hk5).val ≤ d ↔ k < c d) := by
    intro d hd
    simp only [c, dif_pos hd]
    have := sorted_getElem_le_iff ⟨d, hd⟩ (sorted a) (sorted_pairwise a) k hk5
    rw [Fin.le_def] at this
    exact this
  rw [step (by omega) ids hk' hdig h5 c hcnt hsort ((sorted a)[k]'hk5) htgt]
  simp [answer]

end VWC.Arch.Sort
