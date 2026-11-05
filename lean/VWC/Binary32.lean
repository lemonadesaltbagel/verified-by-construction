/-
Instantiate the abstract `RoundingModel` (Float.lean) with FloatLib's IEEE-754 binary32
round-to-nearest-even, `Model.roundAt FloatFormat.binary32`.

FloatLib's `roundAt` rounds onto the binary32 grid with gradual underflow and no upper exponent
bound (no overflow); every intermediate of this model is < 400 in magnitude, far below the
binary32 overflow threshold ≈ 3.4e38. The relationship between `roundAt` and FloatLib's
executable binary32 operations is FloatLib's `toReal_*_eq_round_of_isFinite` family (not used here).
-/
import VWC.FloatParity
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel

namespace VWC

open FloatLib.Floats.Formats.BinaryInterchange

/-- binary32 round-to-nearest-even on the real grid (FloatLib). -/
noncomputable abbrev fl32 (x : ℝ) : ℝ := Model.roundAt FloatFormat.binary32 x

lemma u32_eq : Model.unitRoundoffAt FloatFormat.binary32 = 2 ^ (-24 : ℤ) := by
  rw [Model.unitRoundoffAt_eq]
  rfl

lemma η32_eq : Model.underflowErrorAt FloatFormat.binary32 = 2 ^ (-150 : ℤ) := by
  rw [Model.underflowErrorAt_eq]
  rfl

/-- The standard model for binary32: |fl32 x - x| ≤ 2⁻²⁴|x| + 2⁻¹⁵⁰ for every real x
(from FloatLib's `roundAt_standardModel`). -/
lemma fl32_err (x : ℝ) : |fl32 x - x| ≤ 1 / 2 ^ 24 * |x| + 1 / 2 ^ 150 := by
  obtain ⟨δ, η, h, hδ, hη, -⟩ := Model.roundAt_standardModel FloatFormat.binary32 x
  rw [u32_eq, show (2 : ℝ) ^ (-24 : ℤ) = 1 / 2 ^ 24 by norm_num] at hδ
  rw [η32_eq, show (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 by norm_num] at hη
  show |Model.roundAt FloatFormat.binary32 x - x| ≤ _
  rw [h]
  calc |x * (1 + δ) + η - x| = |x * δ + η| := by ring_nf
    _ ≤ |x * δ| + |η| := abs_add_le _ _
    _ = |x| * |δ| + |η| := by rw [abs_mul]
    _ ≤ |x| * (1 / 2 ^ 24) + 1 / 2 ^ 150 := by gcongr
    _ = _ := by ring

/-- binary32 rounding plus any `exp` implementation with relative error `ue` on [-1, 1]. -/
noncomputable def binary32Model (expR : ℝ → ℝ) (ue : ℝ) (hue : 0 ≤ ue)
    (hexp : ∀ x, |x| ≤ 1 → |expR x - Real.exp x| ≤ ue * Real.exp x) : RoundingModel where
  rnd := fl32
  expR := expR
  u := 1 / 2 ^ 24
  η := 1 / 2 ^ 150
  ue := ue
  rnd_err := fl32_err
  exp_err := hexp
  u_nonneg := by positivity
  η_nonneg := by positivity
  ue_nonneg := hue

lemma u32_le : (1 / 2 ^ 24 : ℝ) ≤ 1 / 10 ^ 7 := by norm_num

lemma η32_le : (1 / 2 ^ 150 : ℝ) ≤ 1 / 10 ^ 12 := by norm_num

/-- A correctly rounded binary32 `exp` has relative error ≤ 1e-6 on [-1, 1]. -/
lemma fl32_exp_err (x : ℝ) (hx : |x| ≤ 1) :
    |fl32 (Real.exp x) - Real.exp x| ≤ (1 / 10 ^ 6) * Real.exp x := by
  have h := fl32_err (Real.exp x)
  rw [abs_of_pos (Real.exp_pos x)] at h
  have hlow : (1 / 3 : ℝ) ≤ Real.exp x := by
    have h1 : Real.exp (-1) ≤ Real.exp x := Real.exp_le_exp.2 (by linarith [(abs_le.1 hx).1])
    have h2 : (1 / 3 : ℝ) < Real.exp (-1) := by
      have := Real.exp_neg_one_gt_d9; linarith
    linarith
  have hu : (1 / 2 ^ 24 : ℝ) * Real.exp x ≤ 1 / 10 ^ 7 * Real.exp x :=
    mul_le_mul_of_nonneg_right u32_le (Real.exp_pos x).le
  have hη := η32_le
  have h' : |fl32 (Real.exp x) - Real.exp x| ≤ 1 / 10 ^ 7 * Real.exp x + 1 / 10 ^ 12 := by
    linarith
  linarith

/-- `fl32 (sqrt 3) ≥ 1`, so the score scale used by PyTorch (`math.sqrt(3)` rounded to
binary32) meets the `1 ≤ sq` hypothesis. -/
lemma one_le_fl32_sqrt3 : 1 ≤ fl32 (Real.sqrt 3) := by
  have h := fl32_err (Real.sqrt 3)
  have hs : (17 / 10 : ℝ) ≤ Real.sqrt 3 := by
    rw [Real.le_sqrt (by norm_num) (by norm_num)]; norm_num
  rw [abs_of_nonneg (Real.sqrt_nonneg 3)] at h
  have hs2 : Real.sqrt 3 ≤ 2 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  have hu : (1 / 2 ^ 24 : ℝ) * Real.sqrt 3 ≤ 1 / 10 ^ 7 * 2 :=
    mul_le_mul u32_le hs2 (Real.sqrt_nonneg 3) (by norm_num)
  have hη := η32_le
  have h' : -(1 / 10 ^ 7 * 2 + 1 / 10 ^ 12) ≤ fl32 (Real.sqrt 3) - Real.sqrt 3 := by
    have := (abs_le.1 h).1; linarith
  linarith

/-- **binary32 theorem, general exp.** With FloatLib binary32 round-to-nearest-even for every
`+ × /` and any `exp` implementation with relative error ≤ 1e-6 on [-1, 1] (≈ 8 ulp), the rounded
forward pass predicts parity on every bit string of length 1..10. -/
theorem parity_correct_binary32 (expR : ℝ → ℝ) (ue : ℝ) (hue0 : 0 ≤ ue) (hue : ue ≤ 1 / 10 ^ 6)
    (hexp : ∀ x, |x| ≤ 1 → |expR x - Real.exp x| ≤ ue * Real.exp x)
    (bits : List Bool) (h1 : 1 ≤ bits.length) (h2 : bits.length ≤ 10) :
    argmaxFirst (flogits (binary32Model expR ue hue0 hexp) parityParams (fl32 (Real.sqrt 3)) hL
      (encode bits) (Fin.last 10)) = parityTok bits :=
  parity_correct_float _ u32_le η32_le hue _ one_le_fl32_sqrt3 bits h1 h2

/-- **binary32 theorem, fully concrete.** Same, with a correctly rounded binary32 `exp`. -/
theorem parity_correct_binary32_crexp (bits : List Bool) (h1 : 1 ≤ bits.length)
    (h2 : bits.length ≤ 10) :
    argmaxFirst (flogits (binary32Model (fun x => fl32 (Real.exp x)) (1 / 10 ^ 6) (by norm_num)
      fl32_exp_err) parityParams (fl32 (Real.sqrt 3)) hL (encode bits) (Fin.last 10)) =
      parityTok bits :=
  parity_correct_binary32 _ _ (by norm_num) le_rfl fl32_exp_err bits h1 h2

end VWC
