/-
Float-level correctness of ParityFloorSubD3F9 under the error model of Float.lean:
every logit of the rounded forward pass is within 1/5 of the real-valued logit, which is
below δ/2 = 1/4 (δ = 1/2 from `readout_margin`), hence the rounded prediction is correct.
-/
import VWC.Float

namespace VWC

open Matrix

variable (R : RoundingModel)

/-! ### Weight facts used by the error analysis -/

lemma Wk_eq_zero : parityParams.Wk = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [parityParams]

lemma rowsum_Wv (c : Fin 3) : ∑ i, |parityParams.Wv c i| ≤ 11 := by
  fin_cases c <;> norm_num [parityParams, Fin.sum_univ_succ]

lemma rowsum_Wo (c : Fin 3) : ∑ i, |parityParams.Wo c i| ≤ 1 := by
  fin_cases c <;> norm_num [parityParams, Fin.sum_univ_succ]

lemma rowsum_W1 (f : Fin 9) : ∑ i, |parityParams.W1 f i| ≤ 1 := by
  fin_cases f <;> norm_num [parityParams, Fin.sum_univ_succ]

lemma rowsum_W2 (c : Fin 3) : ∑ i, |parityParams.W2 c i| ≤ 18 := by
  fin_cases c <;> norm_num [parityParams, Fin.sum_univ_succ]

lemma rowsum_Wh (t : Fin 4) : ∑ i, |parityParams.Wh t i| ≤ 21 / 2 := by
  fin_cases t <;> norm_num [parityParams, Fin.sum_univ_succ]

lemma abs_b1 (f : Fin 9) : |parityParams.b1 f| ≤ 9 := by
  fin_cases f <;> norm_num [parityParams]

lemma b2_eq_zero (c : Fin 3) : parityParams.b2 c = 0 := by
  fin_cases c <;> simp [parityParams]

lemma abs_mulVec_le {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) {K B : ℝ}
    (hK : ∀ c, ∑ i, |W c i| ≤ K) (hB : ∀ i, |x i| ≤ B) (hB0 : 0 ≤ B) (c : Fin m) :
    |(W *ᵥ x) c| ≤ K * B :=
  abs_dot_le (W c) x (hK c) hB hB0

lemma relu_abs_le (a : ℝ) : |relu a| ≤ |a| := by
  simpa [relu] using abs_max_sub_max_le_abs a 0 0

/-- Rounded linear stage: the exact-weight perturbation bound, row by row. -/
lemma flin_stage (hu : R.u ≤ 1 / 10 ^ 7) {m n : ℕ} (hn : n ≤ 1000)
    (W : Matrix (Fin m) (Fin n) ℝ) (xh x : Fin n → ℝ) {K dx B : ℝ}
    (hK : ∀ c, ∑ i, |W c i| ≤ K) (hx : ∀ i, |xh i - x i| ≤ dx) (hB : ∀ i, |x i| ≤ B)
    (hdx : 0 ≤ dx) (hB0 : 0 ≤ B) (c : Fin m) :
    |flin R W xh c - (W *ᵥ x) c| ≤ K * dx + 4 * n * (R.u * (K * (B + dx)) + R.η) :=
  fdot_close_exactW R hu hn (W c) xh x (hK c) hx hB hdx hB0

/-! ### Softmax with all-zero (up to rounding) queries -/

section Softmax

variable (hu : R.u ≤ 1 / 10 ^ 7) (hη : R.η ≤ 1 / 10 ^ 12) (hue : R.ue ≤ 1 / 10 ^ 6)
include hu hη

lemma fscore_small (q k : Fin 11 → Fin 3 → ℝ) (hq : ∀ t c, |q t c| ≤ 2 / 10 ^ 11)
    (hk : ∀ t c, |k t c| ≤ 2 / 10 ^ 11) (sq : ℝ) (hsq : 1 ≤ sq) (j : Fin 11) :
    |fscore R q k sq (Fin.last 10) j| ≤ 2 / 10 ^ 10 := by
  set qi := q (Fin.last 10)
  have hqk : ∀ c, |qi c * k j c| ≤ 4 / 10 ^ 22 := fun c => by
    rw [abs_mul]
    calc _ ≤ (2 / 10 ^ 11) * (2 / 10 ^ 11) :=
          mul_le_mul (hq _ _) (hk _ _) (abs_nonneg _) (by norm_num)
      _ = 4 / 10 ^ 22 := by norm_num
  have hA : ∑ c, |qi c * k j c| ≤ 12 / 10 ^ 22 := by
    calc _ ≤ ∑ _c : Fin 3, (4 / 10 ^ 22 : ℝ) := Finset.sum_le_sum fun c _ => hqk c
      _ = 12 / 10 ^ 22 := by simp; norm_num
  have hS : |∑ c, qi c * k j c| ≤ 12 / 10 ^ 22 := (Finset.abs_sum_le_sum_abs _ _).trans hA
  have hf := fdot_err R hu 3 (by norm_num) qi (k j)
  have h1 : R.u * ∑ c, |qi c * k j c| ≤ (1 / 10 ^ 7) * (12 / 10 ^ 22) :=
    mul_le_mul hu hA (Finset.sum_nonneg fun _ _ => abs_nonneg _) (by norm_num)
  have hf' : |fdot R qi (k j)| ≤ 2 / 10 ^ 11 := by
    have := abs_sub_abs_le_abs_sub (fdot R qi (k j)) (∑ c, qi c * k j c)
    push_cast at hf
    linarith
  have hdiv : |fdot R qi (k j) / sq| ≤ 2 / 10 ^ 11 := by
    rw [abs_div, abs_of_pos (by linarith : (0 : ℝ) < sq)]
    exact (div_le_self (abs_nonneg _) hsq).trans hf'
  have h2 := rnd_close R hdiv hu hη
  have h3 := abs_sub_abs_le_abs_sub (R.rnd (fdot R qi (k j) / sq)) (fdot R qi (k j) / sq)
  unfold fscore
  linarith

omit hu hη in
include hue in
lemma expR_near_one {d : ℝ} (hd : |d| ≤ 1 / 10 ^ 9) : |R.expR d - 1| ≤ 3 / 10 ^ 6 := by
  have h1 := Real.abs_exp_sub_one_le (x := d) (by linarith)
  have h2 := R.exp_err d (by linarith)
  have hexp : Real.exp d ≤ 2 := by
    have := (abs_le.1 h1).2; linarith
  have h3 : R.ue * Real.exp d ≤ (1 / 10 ^ 6) * 2 :=
    mul_le_mul hue hexp (Real.exp_pos d).le (by norm_num)
  calc |R.expR d - 1| ≤ |R.expR d - Real.exp d| + |Real.exp d - 1| := abs_sub_le _ _ _
    _ ≤ 3 / 10 ^ 6 := by linarith

include hue in
lemma fexps_close (q k : Fin 11 → Fin 3 → ℝ) (hq : ∀ t c, |q t c| ≤ 2 / 10 ^ 11)
    (hk : ∀ t c, |k t c| ≤ 2 / 10 ^ 11) (sq : ℝ) (hsq : 1 ≤ sq) (j : Fin 11) :
    |fexps R q k sq (Fin.last 10) j - 1| ≤ 3 / 10 ^ 6 := by
  have hs := fscore_small R hu hη q k hq hk sq hsq
  unfold fexps
  simp only [Fin.le_last, ite_true]
  set s := fscore R q k sq (Fin.last 10)
  set m := (Finset.Iic (Fin.last 10)).sup' Finset.nonempty_Iic s
  have hm_le : m ≤ 2 / 10 ^ 10 := Finset.sup'_le _ _ fun j _ => (abs_le.1 (hs j)).2
  have hm_ge : -(2 / 10 ^ 10) ≤ m :=
    (abs_le.1 (hs (Fin.last 10))).1.trans (Finset.le_sup' s (Finset.mem_Iic.2 le_rfl))
  have hsm : |s j - m| ≤ 4 / 10 ^ 10 := by
    have := abs_le.1 (hs j); rw [abs_le]; constructor <;> linarith
  have hd := rnd_close R hsm hu hη
  have hd' : |R.rnd (s j - m)| ≤ 1 / 10 ^ 9 := by
    have := abs_sub_abs_le_abs_sub (R.rnd (s j - m)) (s j - m); linarith
  exact expR_near_one R hue hd'

lemma fsum_close (e : Fin 11 → ℝ) (he : ∀ j, |e j - 1| ≤ 3 / 10 ^ 6) :
    |fdot R (fun _ => 1) e - 11| ≤ 1 / 10 ^ 4 := by
  have h := fdot_close_exactW R hu (n := 11) (by norm_num) (fun _ => 1) e (fun _ => 1)
    (K := 11) (dx := 3 / 10 ^ 6) (B := 1) (by simp) he (by simp) (by norm_num) (by norm_num)
  have hsum : ∑ _i : Fin 11, (1 : ℝ) * 1 = 11 := by simp
  rw [hsum] at h
  push_cast at h
  linarith [R.u_nonneg]

lemma fdiv_close {e S : ℝ} (he : |e - 1| ≤ 3 / 10 ^ 6) (hS : |S - 11| ≤ 1 / 10 ^ 4) :
    |R.rnd (e / S) - 1 / 11| ≤ 2 / 10 ^ 6 := by
  have hS' := abs_le.1 hS
  have he' := abs_le.1 he
  have hSpos : (0 : ℝ) < S := by linarith
  have key : |e / S - 1 / 11| ≤ 13 / 10 ^ 7 := by
    rw [div_sub_div _ _ hSpos.ne' (by norm_num : (11 : ℝ) ≠ 0), abs_div,
      abs_of_pos (by positivity : (0 : ℝ) < S * 11), div_le_iff₀ (by positivity)]
    rw [abs_le]; constructor <;> nlinarith
  have hb : |e / S| ≤ 1 := by
    have := abs_sub_abs_le_abs_sub (e / S) (1 / 11); norm_num at this ⊢; linarith
  have := rnd_close R hb hu hη
  calc _ ≤ |R.rnd (e / S) - e / S| + |e / S - 1 / 11| := abs_sub_le _ _ _
    _ ≤ _ := by linarith

include hue in
lemma fsoftmax_close (q k : Fin 11 → Fin 3 → ℝ) (hq : ∀ t c, |q t c| ≤ 2 / 10 ^ 11)
    (hk : ∀ t c, |k t c| ≤ 2 / 10 ^ 11) (sq : ℝ) (hsq : 1 ≤ sq) (j : Fin 11) :
    |fsoftmaxRow R q k sq (Fin.last 10) j - 1 / 11| ≤ 2 / 10 ^ 6 := by
  rw [fsoftmaxRow_eq]
  have he := fexps_close R hu hη hue q k hq hk sq hsq
  exact fdiv_close R hu hη (he j) (fsum_close R hu hη _ he)

end Softmax

/-! ### Exact reference values at the last position -/

/-- Exact `Σ_j (1/11) · W_v x_j` (attention mix before `W_o`). -/
noncomputable def oRef (ids : Fin 11 → Fin 4) : Fin 3 → ℝ :=
  fun c => ∑ j, (1 / 11 : ℝ) * (parityParams.Wv *ᵥ embed parityParams hL ids j) c

/-- Exact residual after attention. -/
noncomputable def rRef (ids : Fin 11 → Fin 4) : Fin 3 → ℝ :=
  embed parityParams hL ids (Fin.last 10) + parityParams.Wo *ᵥ oRef ids

/-- Exact MLP hidden layer. -/
noncomputable def hRef (ids : Fin 11 → Fin 4) : Fin 9 → ℝ :=
  fun f => relu ((parityParams.W1 *ᵥ rRef ids) f + parityParams.b1 f)

lemma attnWeight_last (ids : Fin 11 → Fin 4) (j : Fin 11) :
    attnWeight parityParams (embed parityParams hL ids) (Fin.last 10) j = 1 / 11 := by
  rw [softmax_zero_scores_uniform _ _ (score_eq_zero _) (Fin.last 10) j,
    if_pos (Fin.le_last j)]
  norm_num

lemma logits_ref (ids : Fin 11 → Fin 4) :
    logitsAt parityParams hL ids (Fin.last 10) =
      parityParams.Wh *ᵥ (rRef ids + mlp parityParams (rRef ids)) := by
  have hattn : attn parityParams (embed parityParams hL ids) (Fin.last 10) =
      parityParams.Wo *ᵥ oRef ids := by
    unfold attn
    congr 1
    ext c
    rw [Finset.sum_apply]
    simp only [Pi.smul_apply, smul_eq_mul, attnWeight_last, oRef]
  unfold logitsAt block
  simp only [hattn]
  rfl

lemma embed_abs_le (ids : Fin 11 → Fin 4) (t : Fin 11) (c : Fin 3) :
    |embed parityParams hL ids t c| ≤ 1 := by
  rw [embed_eq]
  fin_cases c <;> simp <;> split_ifs <;> simp

/-! ### The error chain -/

section Chain

variable (hu : R.u ≤ 1 / 10 ^ 7) (hη : R.η ≤ 1 / 10 ^ 12) (hue : R.ue ≤ 1 / 10 ^ 6)
include hu hη

lemma flin_zero_small (W : Matrix (Fin 3) (Fin 3) ℝ) (hW : W = 0) (x : Fin 3 → ℝ) (c : Fin 3) :
    |flin R W x c| ≤ 2 / 10 ^ 11 := by
  subst hW
  have h := fdot_err R hu 3 (by norm_num) (0 : Fin 3 → ℝ) x
  simp only [Pi.zero_apply, zero_mul, Finset.sum_const_zero, sub_zero, abs_zero, mul_zero,
    zero_add, Nat.cast_ofNat] at h
  show |fdot R (0 : Fin 3 → ℝ) x| ≤ _
  linarith

lemma fembed_close (ids : Fin 11 → Fin 4) (t : Fin 11) (c : Fin 3) :
    |fembed R parityParams hL ids t c - embed parityParams hL ids t c| ≤ 2 / 10 ^ 7 := by
  have h := rnd_close R (embed_abs_le ids t c) hu hη
  have : fembed R parityParams hL ids t c = R.rnd (embed parityParams hL ids t c) := rfl
  rw [this]
  linarith

include hue in
/-- Every logit of the rounded forward pass is within 1/5 of the real-valued logit. -/
theorem float_logits_close (sq : ℝ) (hsq : 1 ≤ sq) (ids : Fin 11 → Fin 4) (t : Fin 4) :
    |flogits R parityParams sq hL ids (Fin.last 10) t -
      logitsAt parityParams hL ids (Fin.last 10) t| ≤ 1 / 5 := by
  have u0 := R.u_nonneg
  have η0 := R.η_nonneg
  -- float intermediates
  set xh := fembed R parityParams hL ids with hxh
  set a := fsoftmaxRow R (fun t => flin R parityParams.Wq (xh t))
    (fun t => flin R parityParams.Wk (xh t)) sq (Fin.last 10) with ha
  set vh : Fin 11 → Fin 3 → ℝ := fun t => flin R parityParams.Wv (xh t) with hvh
  set o := fattnMix R a vh with ho
  set rh := fadd R (xh (Fin.last 10)) (flin R parityParams.Wo o) with hrh
  set hh : Fin 9 → ℝ := fun f => relu (flinB R parityParams.W1 parityParams.b1 rh f) with hhh
  set r2 := fadd R rh (flinB R parityParams.W2 parityParams.b2 hh) with hr2
  have hfl : flogits R parityParams sq hL ids (Fin.last 10) t = flin R parityParams.Wh r2 t := rfl
  -- exact intermediates
  set x := embed parityParams hL ids with hx
  -- stage: embedding
  have ex : ∀ t c, |xh t c - x t c| ≤ 2 / 10 ^ 7 := fembed_close R hu hη ids
  have bx : ∀ t c, |x t c| ≤ 1 := embed_abs_le ids
  -- stage: softmax weights
  have ea : ∀ j, |a j - 1 / 11| ≤ 2 / 10 ^ 6 :=
    fsoftmax_close R hu hη hue _ _
      (fun t c => flin_zero_small R hu hη _ Wq_eq_zero _ c)
      (fun t c => flin_zero_small R hu hη _ Wk_eq_zero _ c) sq hsq
  -- stage: values
  have ev : ∀ t c, |vh t c - (parityParams.Wv *ᵥ x t) c| ≤ 2 / 10 ^ 5 := fun t c => by
    have h := flin_stage R hu (n := 3) (by norm_num) parityParams.Wv (xh t) (x t)
      (K := 11) (dx := 2 / 10 ^ 7) (B := 1) rowsum_Wv (ex t) (bx t) (by norm_num) (by norm_num) c
    simp only [Nat.cast_ofNat] at h
    linarith
  have bv : ∀ t c, |(parityParams.Wv *ᵥ x t) c| ≤ 11 := fun t c => by
    simpa using abs_mulVec_le parityParams.Wv (x t) rowsum_Wv (bx t) (by norm_num) c
  -- stage: attention mix
  have eo : ∀ c, |o c - oRef ids c| ≤ 4 / 10 ^ 4 := fun c => by
    have h := fdot_close R hu (n := 11) (by norm_num) a (fun _ => (1 / 11 : ℝ))
      (fun j => vh j c) (fun j => (parityParams.Wv *ᵥ x j) c)
      (dw := 2 / 10 ^ 6) (dx := 2 / 10 ^ 5) (Bw := 1 / 11) (Bx := 11)
      ea (fun j => ev j c) (fun _ => by norm_num) (fun j => bv j c)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    simp only [Nat.cast_ofNat] at h
    have : o c = fdot R a (fun j => vh j c) := rfl
    rw [this]
    simp only [oRef]
    linarith
  have bo : ∀ c, |oRef ids c| ≤ 11 := fun c => by
    have := abs_dot_le (fun _ : Fin 11 => (1 / 11 : ℝ)) (fun j => (parityParams.Wv *ᵥ x j) c)
      (K := 1) (B := 11) (by simp) (fun j => bv j c) (by norm_num)
    simpa [oRef] using this
  -- stage: W_o
  have ey : ∀ c, |flin R parityParams.Wo o c - (parityParams.Wo *ᵥ oRef ids) c| ≤ 5 / 10 ^ 4 :=
    fun c => by
    have h := flin_stage R hu (n := 3) (by norm_num) parityParams.Wo o (oRef ids)
      (K := 1) (dx := 4 / 10 ^ 4) (B := 11) rowsum_Wo eo bo (by norm_num) (by norm_num) c
    simp only [Nat.cast_ofNat] at h
    linarith
  have byy : ∀ c, |(parityParams.Wo *ᵥ oRef ids) c| ≤ 11 := fun c => by
    simpa using abs_mulVec_le parityParams.Wo (oRef ids) rowsum_Wo bo (by norm_num) c
  -- stage: residual after attention
  have er : ∀ c, |rh c - rRef ids c| ≤ 6 / 10 ^ 4 := fun c => by
    have hd : |(xh (Fin.last 10) c + flin R parityParams.Wo o c) - rRef ids c| ≤
        2 / 10 ^ 7 + 5 / 10 ^ 4 := by
      have e : (xh (Fin.last 10) c + flin R parityParams.Wo o c) - rRef ids c =
          (xh (Fin.last 10) c - x (Fin.last 10) c) +
            (flin R parityParams.Wo o c - (parityParams.Wo *ᵥ oRef ids) c) := by
        simp only [rRef, Pi.add_apply, hx]; ring
      rw [e]
      exact (abs_add_le _ _).trans (add_le_add (ex _ c) (ey c))
    have hB : |rRef ids c| ≤ 12 := by
      have : rRef ids c = x (Fin.last 10) c + (parityParams.Wo *ᵥ oRef ids) c := rfl
      rw [this]
      exact (abs_add_le _ _).trans (by linarith [bx (Fin.last 10) c, byy c])
    have h := rnd_close_pert R hu hη hd hB
    show |R.rnd _ - _| ≤ _
    linarith
  have brr : ∀ c, |rRef ids c| ≤ 12 := fun c => by
    have : rRef ids c = x (Fin.last 10) c + (parityParams.Wo *ᵥ oRef ids) c := rfl
    rw [this]
    exact (abs_add_le _ _).trans (by linarith [bx (Fin.last 10) c, byy c])
  -- stage: MLP hidden layer
  have eh : ∀ f, |hh f - hRef ids f| ≤ 7 / 10 ^ 4 := fun f => by
    have h1 := flin_stage R hu (n := 3) (by norm_num) parityParams.W1 rh (rRef ids)
      (K := 1) (dx := 6 / 10 ^ 4) (B := 12) rowsum_W1 er brr (by norm_num) (by norm_num) f
    simp only [Nat.cast_ofNat] at h1
    have hd : |(flin R parityParams.W1 rh f + parityParams.b1 f) -
        ((parityParams.W1 *ᵥ rRef ids) f + parityParams.b1 f)| ≤ 62 / 10 ^ 5 := by
      rw [add_sub_add_right_eq_sub]; linarith
    have hB : |(parityParams.W1 *ᵥ rRef ids) f + parityParams.b1 f| ≤ 21 := by
      have := abs_mulVec_le parityParams.W1 (rRef ids) rowsum_W1 brr (by norm_num) f
      exact (abs_add_le _ _).trans (by linarith [abs_b1 f])
    have h2 := rnd_close_pert R hu hη hd hB
    have h3 := relu_lip (flinB R parityParams.W1 parityParams.b1 rh f)
      ((parityParams.W1 *ᵥ rRef ids) f + parityParams.b1 f)
    have : flinB R parityParams.W1 parityParams.b1 rh f =
        R.rnd (flin R parityParams.W1 rh f + parityParams.b1 f) := rfl
    show |relu _ - relu _| ≤ _
    rw [this] at h3 ⊢
    linarith
  have bh : ∀ f, |hRef ids f| ≤ 21 := fun f => by
    have := abs_mulVec_le parityParams.W1 (rRef ids) rowsum_W1 brr (by norm_num) f
    exact (relu_abs_le _).trans ((abs_add_le _ _).trans (by linarith [abs_b1 f]))
  -- stage: MLP output and residual
  have eg : ∀ c, |flinB R parityParams.W2 parityParams.b2 hh c -
      ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c)| ≤ 15 / 10 ^ 3 := fun c => by
    have h1 := flin_stage R hu (n := 9) (by norm_num) parityParams.W2 hh (hRef ids)
      (K := 18) (dx := 7 / 10 ^ 4) (B := 21) rowsum_W2 eh bh (by norm_num) (by norm_num) c
    simp only [Nat.cast_ofNat] at h1
    have hd : |(flin R parityParams.W2 hh c + parityParams.b2 c) -
        ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c)| ≤ 14 / 10 ^ 3 := by
      rw [add_sub_add_right_eq_sub]; linarith
    have hB : |(parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c| ≤ 378 := by
      have := abs_mulVec_le parityParams.W2 (hRef ids) rowsum_W2 bh (by norm_num) c
      rw [b2_eq_zero, add_zero]; linarith
    have h2 := rnd_close_pert R hu hη hd hB
    have : flinB R parityParams.W2 parityParams.b2 hh c =
        R.rnd (flin R parityParams.W2 hh c + parityParams.b2 c) := rfl
    rw [this]
    linarith
  have er2 : ∀ c, |r2 c - (rRef ids + mlp parityParams (rRef ids)) c| ≤ 16 / 10 ^ 3 :=
    fun c => by
    have hm : (rRef ids + mlp parityParams (rRef ids)) c =
        rRef ids c + ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c) := rfl
    rw [hm]
    have hd : |(rh c + flinB R parityParams.W2 parityParams.b2 hh c) -
        (rRef ids c + ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c))| ≤
        6 / 10 ^ 4 + 15 / 10 ^ 3 := by
      rw [add_sub_add_comm]
      exact (abs_add_le _ _).trans (add_le_add (er c) (eg c))
    have hB : |rRef ids c + ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c)| ≤ 390 := by
      have := abs_mulVec_le parityParams.W2 (hRef ids) rowsum_W2 bh (by norm_num) c
      rw [b2_eq_zero, add_zero]
      exact (abs_add_le _ _).trans (by linarith [brr c])
    have h2 := rnd_close_pert R hu hη hd hB
    show |R.rnd _ - _| ≤ _
    linarith
  have br2 : ∀ c, |(rRef ids + mlp parityParams (rRef ids)) c| ≤ 390 := fun c => by
    have hm : (rRef ids + mlp parityParams (rRef ids)) c =
        rRef ids c + ((parityParams.W2 *ᵥ hRef ids) c + parityParams.b2 c) := rfl
    have := abs_mulVec_le parityParams.W2 (hRef ids) rowsum_W2 bh (by norm_num) c
    rw [hm, b2_eq_zero, add_zero]
    exact (abs_add_le _ _).trans (by linarith [brr c])
  -- stage: head
  have h := flin_stage R hu (n := 3) (by norm_num) parityParams.Wh r2
    (rRef ids + mlp parityParams (rRef ids))
    (K := 21 / 2) (dx := 16 / 10 ^ 3) (B := 390) rowsum_Wh er2 br2 (by norm_num) (by norm_num) t
  simp only [Nat.cast_ofNat] at h
  rw [hfl, logits_ref]
  linarith

include hue in
/-- Rounded prediction on any prompt ending in `=`. -/
theorem float_predict_prompt (sq : ℝ) (hsq : 1 ≤ sq) (ids : Fin 11 → Fin 4)
    (h : ids (Fin.last 10) = tokEq) :
    argmaxFirst (flogits R parityParams sq hL ids (Fin.last 10)) =
      if nOnes ids % 2 = 1 then tok1 else tok0 := by
  apply argmaxFirst_eq_of_lt
  intro j hj
  have hm := readout_margin ids h j hj
  have e1 := abs_le.1 (float_logits_close R hu hη hue sq hsq ids j)
  have e2 := abs_le.1 (float_logits_close R hu hη hue sq hsq ids
    (if nOnes ids % 2 = 1 then tok1 else tok0))
  linarith [e1.2, e2.1]

include hue in
/-- **Float theorem.** Under any rounding model with u ≤ 1e-7 (binary32: 2⁻²⁴), η ≤ 1e-12,
exp relative error ≤ 1e-6 and any `sqrt(d_head)` value ≥ 1, the rounded forward pass predicts
the parity token on every bit string of length 1..10. Logit error ≤ 1/5 < δ/2 = 1/4. -/
theorem parity_correct_float (sq : ℝ) (hsq : 1 ≤ sq) (bits : List Bool)
    (_h1 : 1 ≤ bits.length) (h2 : bits.length ≤ 10) :
    argmaxFirst (flogits R parityParams sq hL (encode bits) (Fin.last 10)) = parityTok bits := by
  rw [float_predict_prompt R hu hη hue sq hsq _ (encode_last bits h2), nOnes_encode bits h2]
  rfl

end Chain

end VWC
