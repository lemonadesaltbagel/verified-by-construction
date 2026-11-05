/-
Phase 2, sub-proposition (b): LayerNorm on a grafted residual stream.

The host residual x ∈ ℝᴰ is extended by k circuit dims holding `mean x + s`, where the circuit
state s has zero sum and energy E. With the host gains rescaled by √(D/(D+k)) and ε replaced
by ε' = (Dε − E)/(D+k), the joint LayerNorm
* equals the host's own LayerNorm on the host dims, exactly, for every x;
* outputs `s · γc · κ + βc` on the circuit dims, with κ = √((D+k)/D) / √(var x + ε).
ε' may be negative; `var + ε' = D(var x + ε)/(D+k)` stays positive.
-/
import Mathlib
import VWC.Arch

namespace VWC.Graft

open Finset VWC.Arch

variable {D k : ℕ}

noncomputable def mean (x : Fin D → ℝ) : ℝ := (∑ i, x i) / D

noncomputable def var (x : Fin D → ℝ) : ℝ := (∑ i, (x i - mean x) ^ 2) / D

/-- Grafted residual: host part x, circuit dims `mean x + s`. -/
noncomputable def joint (x : Fin D → ℝ) (s : Fin k → ℝ) : Fin (D + k) → ℝ :=
  Fin.append x (fun j => mean x + s j)

/-- Compensated LayerNorm parameters for the grafted model. -/
noncomputable def lnGraft (γ β : Fin D → ℝ) (γc βc : Fin k → ℝ) (ε E : ℝ) : Norm (D + k) :=
  .ln (Fin.append (fun i => γ i * Real.sqrt (D / (D + k))) γc) (Fin.append β βc)
    ((D * ε - E) / (D + k))

/-- The circuit-side scale factor κ = √((D+k)/D) / √(var x + ε). -/
noncomputable def kappa (D k : ℕ) (x : Fin D → ℝ) (ε : ℝ) : ℝ :=
  Real.sqrt ((D + k) / D) / Real.sqrt (var x + ε)

lemma var_nonneg (x : Fin D → ℝ) : 0 ≤ var x := by
  unfold var; exact div_nonneg (sum_nonneg fun i _ => sq_nonneg _) (Nat.cast_nonneg _)

lemma joint_sum (x : Fin D → ℝ) (s : Fin k → ℝ) (hs0 : ∑ j, s j = 0) (hD : 0 < D) :
    ∑ i, joint x s i = (D + k) * mean x := by
  unfold joint
  rw [Fin.sum_univ_add]
  simp only [Fin.append_left, Fin.append_right, sum_add_distrib, hs0, sum_const, card_univ,
    Fintype.card_fin, nsmul_eq_mul, add_zero]
  have hD' : (D : ℝ) ≠ 0 := by exact_mod_cast hD.ne'
  unfold mean; field_simp

lemma joint_mean (x : Fin D → ℝ) (s : Fin k → ℝ) (hs0 : ∑ j, s j = 0) (hD : 0 < D) :
    (∑ i, joint x s i) / ((D + k : ℕ) : ℝ) = mean x := by
  rw [joint_sum x s hs0 hD]
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  push_cast
  exact mul_div_cancel_left₀ _ (by positivity)

lemma joint_sq (x : Fin D → ℝ) (s : Fin k → ℝ) (hs0 : ∑ j, s j = 0) (hD : 0 < D) (E : ℝ)
    (hsE : ∑ j, s j ^ 2 = E) :
    ∑ i, (joint x s i - mean x) ^ 2 = D * var x + E := by
  unfold joint
  rw [Fin.sum_univ_add]
  simp only [Fin.append_left, Fin.append_right, add_sub_cancel_left, hsE]
  have hD' : (D : ℝ) ≠ 0 := by exact_mod_cast hD.ne'
  unfold var; field_simp

/-- **LayerNorm lemma (b).** -/
theorem ln_graft (hD : 0 < D) (x : Fin D → ℝ) (s : Fin k → ℝ) (E : ℝ)
    (hs0 : ∑ j, s j = 0) (hsE : ∑ j, s j ^ 2 = E)
    (γ β : Fin D → ℝ) (γc βc : Fin k → ℝ) (ε : ℝ) (hε : 0 < ε) :
    (∀ i : Fin D, (lnGraft γ β γc βc ε E).apply (joint x s) (Fin.castAdd k i) =
        (Norm.ln γ β ε).apply x i) ∧
    (∀ j : Fin k, (lnGraft γ β γc βc ε E).apply (joint x s) (Fin.natAdd D j) =
        s j * γc j * kappa D k x ε + βc j) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hDk : (0 : ℝ) < D + k := by positivity
  have hv := var_nonneg x
  -- the compensated denominator
  have hden : (D * var x + E) / ((D + k : ℕ) : ℝ) + (D * ε - E) / (D + k) =
      (D / (D + k)) * (var x + ε) := by
    push_cast; field_simp; ring
  have hsq : Real.sqrt ((D / (D + k)) * (var x + ε)) =
      Real.sqrt (D / (D + k)) * Real.sqrt (var x + ε) :=
    Real.sqrt_mul (by positivity) _
  have hr1 : 0 < Real.sqrt (D / (D + k)) := Real.sqrt_pos.2 (by positivity)
  have hr2 : 0 < Real.sqrt (var x + ε) := Real.sqrt_pos.2 (by linarith)
  have hinv : Real.sqrt ((D + k) / D) * Real.sqrt (D / (D + k)) = 1 := by
    rw [← Real.sqrt_mul (by positivity)]
    rw [show ((D : ℝ) + k) / D * (D / (D + k)) = 1 by field_simp]
    exact Real.sqrt_one
  have hmean : (∑ i, x i) / (D : ℝ) = mean x := rfl
  have hvar : (∑ i, (x i - mean x) ^ 2) / (D : ℝ) = var x := rfl
  have hk1 : Real.sqrt ((D + k) / D) = 1 / Real.sqrt (D / (D + k)) :=
    eq_one_div_of_mul_eq_one_left hinv
  have hjd : (∑ i, (joint x s i - mean x) ^ 2) / ((D + k : ℕ) : ℝ) + (D * ε - E) / (D + k) =
      (D / (D + k)) * (var x + ε) := by rw [joint_sq x s hs0 hD E hsE, hden]
  constructor
  · intro i
    simp only [lnGraft, Norm.apply]
    rw [joint_mean x s hs0 hD, hjd, hsq, hmean, hvar]
    simp only [joint, Fin.append_left]
    field_simp
  · intro j
    simp only [lnGraft, Norm.apply]
    rw [joint_mean x s hs0 hD, hjd, hsq]
    simp only [joint, Fin.append_right, add_sub_cancel_left, kappa, hk1]
    field_simp

end VWC.Graft
