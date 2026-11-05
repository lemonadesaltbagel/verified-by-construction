/-
Finite-precision layer (Step 0.5). Separate from the real-valued theorem in Parity.lean.

Error model: an arbitrary rounding operator `rnd` with |rnd x - x| ≤ u|x| + η (relative error u,
absolute underflow term η) and an arbitrary `exp` implementation with relative error ≤ ue.
IEEE-754 binary32 round-to-nearest satisfies this with u = 2⁻²⁴ ≈ 5.96e-8, η = 2⁻¹⁵⁰, provided no
overflow occurs (all intermediates here are < 400 in magnitude). We require u ≤ 1e-7, η ≤ 1e-12,
ue ≤ 1e-6 (≈ 8 ulp for expf), the exp bound being needed only on [-1, 1].

The float forward pass `flogits` follows PyTorch's op sequence for the last position, with
every +, ×, / rounded and every reduction summed sequentially left to right (`fdot`). Real
BLAS/ATen kernels may use other summation orders or FMA; that gap is NOT covered here.
-/
import VWC.Parity

namespace VWC

open Matrix

structure RoundingModel where
  rnd : ℝ → ℝ
  expR : ℝ → ℝ
  u : ℝ
  η : ℝ
  ue : ℝ
  rnd_err : ∀ x, |rnd x - x| ≤ u * |x| + η
  exp_err : ∀ x, |x| ≤ 1 → |expR x - Real.exp x| ≤ ue * Real.exp x
  u_nonneg : 0 ≤ u
  η_nonneg : 0 ≤ η
  ue_nonneg : 0 ≤ ue

/-- Non-vacuity: exact arithmetic is a rounding model (u = η = ue = 0). -/
noncomputable def RoundingModel.exact : RoundingModel where
  rnd := id
  expR := Real.exp
  u := 0
  η := 0
  ue := 0
  rnd_err := by simp
  exp_err := by simp
  u_nonneg := le_rfl
  η_nonneg := le_rfl
  ue_nonneg := le_rfl

/-- Non-vacuity with nonzero error: rounding that always errs upward by exactly u|x| + η. -/
noncomputable def RoundingModel.worstUp (u η : ℝ) (hu : 0 ≤ u) (hη : 0 ≤ η) : RoundingModel where
  rnd := fun x => x + (u * |x| + η)
  expR := Real.exp
  u := u
  η := η
  ue := 0
  rnd_err := fun x => by
    simp only [add_sub_cancel_left]
    rw [abs_of_nonneg (by positivity)]
  exp_err := by simp
  u_nonneg := hu
  η_nonneg := hη
  ue_nonneg := le_rfl

variable (R : RoundingModel)

/-- Sequential rounded dot product: `acc ← rnd (acc + rnd (w i * x i))`, i = 0, 1, …. -/
noncomputable def fdot : ∀ {n : ℕ}, (Fin n → ℝ) → (Fin n → ℝ) → ℝ
  | 0, _, _ => 0
  | n + 1, w, x =>
    R.rnd (fdot (Fin.init w) (Fin.init x) + R.rnd (w (Fin.last n) * x (Fin.last n)))

/-- Rounded `nn.Linear` without bias. -/
noncomputable def flin {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (x : Fin n → ℝ) : Fin m → ℝ :=
  fun c => fdot R (W c) x

/-- Rounded `nn.Linear` with bias. -/
noncomputable def flinB {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ)
    (x : Fin n → ℝ) : Fin m → ℝ :=
  fun c => R.rnd (fdot R (W c) x + b c)

/-- Rounded embedding `token_emb(ids) + pos_emb(arange T)`. -/
noncomputable def fembed {V D F L T : ℕ} (P : Params V D F L) (hT : T ≤ L)
    (ids : Fin T → Fin V) (t : Fin T) : Fin D → ℝ :=
  fun c => R.rnd (P.tokEmb (ids t) c + P.posEmb (Fin.castLE hT t) c)

/-- Rounded causal softmax row of query `i`: scores `rnd (q·k / sq)`, max subtraction,
`exp`, sequential sum, division. Masked entries are exactly 0 (exp(-inf) = 0). -/
noncomputable def fsoftmaxRow {D T : ℕ} (q k : Fin T → Fin D → ℝ) (sq : ℝ) (i : Fin T) :
    Fin T → ℝ :=
  let s : Fin T → ℝ := fun j => R.rnd (fdot R (q i) (k j) / sq)
  let m : ℝ := (Finset.Iic i).sup' Finset.nonempty_Iic s
  let e : Fin T → ℝ := fun j => if j ≤ i then R.expR (R.rnd (s j - m)) else 0
  let S : ℝ := fdot R (fun _ => 1) e
  fun j => R.rnd (e j / S)

/-- Rounded score of query `i` against key `j`. -/
noncomputable def fscore {D T : ℕ} (q k : Fin T → Fin D → ℝ) (sq : ℝ) (i j : Fin T) : ℝ :=
  R.rnd (fdot R (q i) (k j) / sq)

/-- Rounded `exp (s_j - max s)` on the causal prefix, 0 on masked entries. -/
noncomputable def fexps {D T : ℕ} (q k : Fin T → Fin D → ℝ) (sq : ℝ) (i j : Fin T) : ℝ :=
  if j ≤ i then
    R.expR (R.rnd (fscore R q k sq i j - (Finset.Iic i).sup' Finset.nonempty_Iic (fscore R q k sq i)))
  else 0

lemma fsoftmaxRow_eq {D T : ℕ} (q k : Fin T → Fin D → ℝ) (sq : ℝ) (i j : Fin T) :
    fsoftmaxRow R q k sq i j =
      R.rnd (fexps R q k sq i j / fdot R (fun _ => 1) (fexps R q k sq i)) := rfl

/-- Rounded `attn @ v` for one query row. -/
noncomputable def fattnMix {D T : ℕ} (a : Fin T → ℝ) (v : Fin T → Fin D → ℝ) : Fin D → ℝ :=
  fun c => fdot R a (fun j => v j c)

/-- Rounded residual add. -/
noncomputable def fadd {D : ℕ} (x y : Fin D → ℝ) : Fin D → ℝ := fun c => R.rnd (x c + y c)

/-- Float forward pass of a `Params` model: logits at query position `i`.
`sq` is the float value used for `math.sqrt(d_head)`. -/
noncomputable def flogits {V D F L T : ℕ} (P : Params V D F L) (sq : ℝ) (hT : T ≤ L)
    (ids : Fin T → Fin V) (i : Fin T) : Fin V → ℝ :=
  let x := fembed R P hT ids
  let a := fsoftmaxRow R (fun t => flin R P.Wq (x t)) (fun t => flin R P.Wk (x t)) sq i
  let o := fattnMix R a (fun t => flin R P.Wv (x t))
  let r := fadd R (x i) (flin R P.Wo o)
  let h : Fin F → ℝ := fun f => relu (flinB R P.W1 P.b1 r f)
  let r2 := fadd R r (flinB R P.W2 P.b2 h)
  flin R P.Wh r2

/-! ### Generic rounding lemmas -/

lemma rnd_close {x B : ℝ} (hx : |x| ≤ B) (hu : R.u ≤ 1 / 10 ^ 7) (hη : R.η ≤ 1 / 10 ^ 12) :
    |R.rnd x - x| ≤ B / 10 ^ 7 + 1 / 10 ^ 12 := by
  have h := R.rnd_err x
  have h1 : R.u * |x| ≤ B / 10 ^ 7 := by
    calc R.u * |x| ≤ (1 / 10 ^ 7) * B :=
          mul_le_mul hu hx (abs_nonneg _) (by norm_num)
      _ = B / 10 ^ 7 := by ring
  linarith

/-- Standard forward error bound for sequential summation, γ ≤ 4n·u. -/
theorem fdot_err (hu : R.u ≤ 1 / 10 ^ 7) :
    ∀ (n : ℕ), n ≤ 1000 → ∀ (w x : Fin n → ℝ),
      |fdot R w x - ∑ i, w i * x i| ≤ 4 * n * (R.u * ∑ i, |w i * x i| + R.η)
  | 0, _, w, x => by simp [fdot]
  | n + 1, hn, w, x => by
    have ih := fdot_err hu n (by omega) (Fin.init w) (Fin.init x)
    simp only [Fin.init] at ih
    rw [Fin.sum_univ_castSucc, Fin.sum_univ_castSucc]
    simp only [fdot]
    set F := fdot R (fun i => w i.castSucc) (fun i => x i.castSucc)
    set S := ∑ i : Fin n, w i.castSucc * x i.castSucc
    set A := ∑ i : Fin n, |w i.castSucc * x i.castSucc|
    set p := w (Fin.last n) * x (Fin.last n)
    have hA : 0 ≤ A := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hSA : |S| ≤ A := Finset.abs_sum_le_sum_abs _ _
    have ha := R.rnd_err p
    have hb := R.rnd_err (F + R.rnd p)
    have hu0 := R.u_nonneg
    have hη0 := R.η_nonneg
    have hn' : (n : ℝ) ≤ 1000 := by exact_mod_cast (by omega : n ≤ 1000)
    have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
    set e := |F - S|
    have ih' : e ≤ 4 * n * (R.u * A + R.η) := ih
    have hz : |F + R.rnd p| ≤ A + |p| + e + (R.u * |p| + R.η) := by
      calc |F + R.rnd p| = |S + p + (F - S) + (R.rnd p - p)| := by ring_nf
        _ ≤ |S| + |p| + |F - S| + |R.rnd p - p| := by
          refine (abs_add_le _ _).trans ?_
          gcongr
          refine (abs_add_le _ _).trans ?_
          gcongr
          exact abs_add_le _ _
        _ ≤ A + |p| + e + (R.u * |p| + R.η) := by gcongr
    have hue : R.u * e ≤ (4 / 10 ^ 4) * (R.u * A + R.η) := by
      have : R.u * e ≤ R.u * (4 * n * (R.u * A + R.η)) := mul_le_mul_of_nonneg_left ih hu0
      have h2 : R.u * (4 * n) ≤ 4 / 10 ^ 4 := by nlinarith
      have h3 : 0 ≤ R.u * A + R.η := by positivity
      nlinarith
    have huz : R.u * |F + R.rnd p| ≤ R.u * (A + |p| + e + (R.u * |p| + R.η)) :=
      mul_le_mul_of_nonneg_left hz hu0
    have hpp : R.u * (R.u * |p|) ≤ R.u * |p| := by
      have : R.u ≤ 1 := by linarith
      nlinarith [mul_nonneg (mul_nonneg hu0 (sub_nonneg.2 this)) (abs_nonneg p)]
    have huη : R.u * R.η ≤ R.η := by nlinarith
    have hnp : 0 ≤ (n : ℝ) * (R.u * |p|) := by positivity
    calc |R.rnd (F + R.rnd p) - (S + p)|
        = |(R.rnd (F + R.rnd p) - (F + R.rnd p)) + (F - S) + (R.rnd p - p)| := by ring_nf
      _ ≤ |R.rnd (F + R.rnd p) - (F + R.rnd p)| + |F - S| + |R.rnd p - p| := by
          refine (abs_add_le _ _).trans ?_
          gcongr
          exact abs_add_le _ _
      _ ≤ (R.u * |F + R.rnd p| + R.η) + e + (R.u * |p| + R.η) := by gcongr
      _ ≤ 4 * ↑(n + 1) * (R.u * (A + |p|) + R.η) := by
          push_cast
          nlinarith [ih', hue, huz, hpp, huη, hnp, ha, hb, mul_nonneg hu0 hA, mul_nonneg hu0 (abs_nonneg p)]

/-! ### Perturbation lemmas for rounded linear maps -/

lemma sum_abs_mul_le {n : ℕ} (w x : Fin n → ℝ) {K B : ℝ} (hK : ∑ i, |w i| ≤ K)
    (hB : ∀ i, |x i| ≤ B) (hB0 : 0 ≤ B) : ∑ i, |w i * x i| ≤ K * B := by
  calc ∑ i, |w i * x i| = ∑ i, |w i| * |x i| := by simp [abs_mul]
    _ ≤ ∑ i, |w i| * B :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hB i) (abs_nonneg _)
    _ = (∑ i, |w i|) * B := by rw [Finset.sum_mul]
    _ ≤ K * B := mul_le_mul_of_nonneg_right hK hB0

lemma abs_dot_le {n : ℕ} (w x : Fin n → ℝ) {K B : ℝ} (hK : ∑ i, |w i| ≤ K)
    (hB : ∀ i, |x i| ≤ B) (hB0 : 0 ≤ B) : |∑ i, w i * x i| ≤ K * B :=
  (Finset.abs_sum_le_sum_abs _ _).trans (sum_abs_mul_le w x hK hB hB0)

/-- Rounded dot product with exact weights and perturbed input. -/
lemma fdot_close_exactW (hu : R.u ≤ 1 / 10 ^ 7) {n : ℕ} (hn : n ≤ 1000) (w xh x : Fin n → ℝ)
    {K dx B : ℝ} (hK : ∑ i, |w i| ≤ K) (hx : ∀ i, |xh i - x i| ≤ dx) (hB : ∀ i, |x i| ≤ B)
    (hdx : 0 ≤ dx) (hB0 : 0 ≤ B) :
    |fdot R w xh - ∑ i, w i * x i| ≤ K * dx + 4 * n * (R.u * (K * (B + dx)) + R.η) := by
  have h1 := fdot_err R hu n hn w xh
  have hxh : ∀ i, |xh i| ≤ B + dx := fun i => by
    have := abs_sub_abs_le_abs_sub (xh i) (x i); linarith [hx i, hB i]
  have h2 : ∑ i, |w i * xh i| ≤ K * (B + dx) := sum_abs_mul_le w xh hK hxh (by linarith)
  have h3 : |∑ i, w i * xh i - ∑ i, w i * x i| ≤ K * dx := by
    rw [← Finset.sum_sub_distrib]
    simp_rw [← mul_sub]
    exact abs_dot_le w (fun i => xh i - x i) hK hx hdx
  have h4 : R.u * ∑ i, |w i * xh i| ≤ R.u * (K * (B + dx)) :=
    mul_le_mul_of_nonneg_left h2 R.u_nonneg
  have h5 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have h6 : 4 * (n : ℝ) * (R.u * ∑ i, |w i * xh i| + R.η) ≤
      4 * n * (R.u * (K * (B + dx)) + R.η) := by gcongr
  calc _ ≤ |fdot R w xh - ∑ i, w i * xh i| + |∑ i, w i * xh i - ∑ i, w i * x i| :=
        abs_sub_le _ _ _
    _ ≤ _ := by linarith

/-- Rounded dot product with perturbed weights and perturbed input. -/
lemma fdot_close (hu : R.u ≤ 1 / 10 ^ 7) {n : ℕ} (hn : n ≤ 1000) (wh w xh x : Fin n → ℝ)
    {dw dx Bw Bx : ℝ} (hw : ∀ i, |wh i - w i| ≤ dw) (hx : ∀ i, |xh i - x i| ≤ dx)
    (hBw : ∀ i, |w i| ≤ Bw) (hBx : ∀ i, |x i| ≤ Bx) (hdw : 0 ≤ dw) (hdx : 0 ≤ dx)
    (hBw0 : 0 ≤ Bw) (hBx0 : 0 ≤ Bx) :
    |fdot R wh xh - ∑ i, w i * x i| ≤
      n * (Bw * dx + dw * Bx + dw * dx) + 4 * n * (R.u * (n * ((Bw + dw) * (Bx + dx))) + R.η) := by
  have h1 := fdot_err R hu n hn wh xh
  have hterm : ∀ i, |wh i * xh i - w i * x i| ≤ Bw * dx + dw * Bx + dw * dx := fun i => by
    have e : wh i * xh i - w i * x i =
        w i * (xh i - x i) + (wh i - w i) * x i + (wh i - w i) * (xh i - x i) := by ring
    rw [e]
    calc _ ≤ |w i * (xh i - x i)| + |(wh i - w i) * x i| + |(wh i - w i) * (xh i - x i)| :=
          abs_add_three _ _ _
      _ = |w i| * |xh i - x i| + |wh i - w i| * |x i| + |wh i - w i| * |xh i - x i| := by
          simp [abs_mul]
      _ ≤ Bw * dx + dw * Bx + dw * dx := by
          gcongr
          · exact hBw i
          · exact hx i
          · exact hw i
          · exact hBx i
          · exact hw i
          · exact hx i
  have hprod : ∀ i, |wh i * xh i| ≤ (Bw + dw) * (Bx + dx) := fun i => by
    have a := abs_sub_abs_le_abs_sub (wh i) (w i)
    have b := abs_sub_abs_le_abs_sub (xh i) (x i)
    rw [abs_mul]
    exact mul_le_mul (by linarith [hw i, hBw i]) (by linarith [hx i, hBx i]) (abs_nonneg _)
      (by linarith)
  have hs1 : |∑ i, wh i * xh i - ∑ i, w i * x i| ≤ n * (Bw * dx + dw * Bx + dw * dx) := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc _ ≤ ∑ _i : Fin n, (Bw * dx + dw * Bx + dw * dx) := Finset.sum_le_sum fun i _ => hterm i
      _ = _ := by simp; ring
  have hs2 : ∑ i, |wh i * xh i| ≤ n * ((Bw + dw) * (Bx + dx)) := by
    calc _ ≤ ∑ _i : Fin n, (Bw + dw) * (Bx + dx) := Finset.sum_le_sum fun i _ => hprod i
      _ = _ := by simp
  have h6 : 4 * (n : ℝ) * (R.u * ∑ i, |wh i * xh i| + R.η) ≤
      4 * n * (R.u * (n * ((Bw + dw) * (Bx + dx))) + R.η) := by
    have := R.u_nonneg; gcongr
  calc _ ≤ |fdot R wh xh - ∑ i, wh i * xh i| + |∑ i, wh i * xh i - ∑ i, w i * x i| :=
        abs_sub_le _ _ _
    _ ≤ _ := by linarith

/-- Rounding then `relu` / plain rounding of a perturbed value. -/
lemma rnd_close_pert (hu : R.u ≤ 1 / 10 ^ 7) (hη : R.η ≤ 1 / 10 ^ 12) {xh x d B : ℝ}
    (hx : |xh - x| ≤ d) (hB : |x| ≤ B) : |R.rnd xh - x| ≤ d + (B + d) / 10 ^ 7 + 1 / 10 ^ 12 := by
  have hxh : |xh| ≤ B + d := by
    have := abs_sub_abs_le_abs_sub xh x; linarith
  have := rnd_close R hxh hu hη
  calc |R.rnd xh - x| ≤ |R.rnd xh - xh| + |xh - x| := abs_sub_le _ _ _
    _ ≤ _ := by linarith

lemma relu_lip (a b : ℝ) : |relu a - relu b| ≤ |a - b| := abs_max_sub_max_le_abs a b 0

end VWC
