/-
Generic softmax facts for soft (finite-scale) attention: weights form a convex combination,
a convex combination of [0, 1]-valued features deviates from the target feature by at most
the weight off the target, and that off-target weight is bounded by Σ_{j≠h} exp (s_j - s_h).
-/
import Mathlib

namespace VWC

open Finset

variable {T : ℕ}

/-- Plain softmax over all positions. -/
noncomputable def softmax (s : Fin T → ℝ) (j : Fin T) : ℝ :=
  Real.exp (s j) / ∑ j', Real.exp (s j')

lemma softmax_nonneg [NeZero T] (s : Fin T → ℝ) (j : Fin T) : 0 ≤ softmax s j :=
  div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)

lemma softmax_sum [NeZero T] (s : Fin T → ℝ) : ∑ j, softmax s j = 1 := by
  have hpos : 0 < ∑ j', Real.exp (s j') :=
    sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty
  simp only [softmax, ← sum_div]
  exact div_self hpos.ne'

/-- Off-target softmax mass is at most Σ_{j ≠ h} exp (s_j - s_h). -/
lemma one_sub_softmax_le [NeZero T] (s : Fin T → ℝ) (h : Fin T) :
    1 - softmax s h ≤ ∑ j, (if j = h then 0 else Real.exp (s j - s h)) := by
  set S := ∑ j', Real.exp (s j')
  have hpos : 0 < S := sum_pos (fun _ _ => Real.exp_pos _) univ_nonempty
  have hSh : Real.exp (s h) ≤ S :=
    single_le_sum (f := fun j => Real.exp (s j)) (fun _ _ => (Real.exp_pos _).le) (mem_univ h)
  have hrest : S - Real.exp (s h) = ∑ j, (if j = h then 0 else Real.exp (s j)) := by
    have : ∑ j, (if j = h then 0 else Real.exp (s j)) =
        ∑ j, Real.exp (s j) - ∑ j, (if j = h then Real.exp (s j) else 0) := by
      rw [← sum_sub_distrib]
      refine sum_congr rfl fun j _ => ?_
      split_ifs <;> simp
    rw [this]
    simp [S]
  have h1 : 1 - softmax s h = (S - Real.exp (s h)) / S := by
    show 1 - Real.exp (s h) / S = _
    rw [eq_div_iff hpos.ne', sub_mul, one_mul, div_mul_cancel₀ _ hpos.ne']
  rw [h1, hrest, sum_div]
  refine sum_le_sum fun j _ => ?_
  split_ifs with hj
  · simp
  · rw [Real.exp_sub, div_le_div_iff_of_pos_left (Real.exp_pos _) hpos (Real.exp_pos _)]
    exact hSh

/-- A convex combination of [0,1]-valued features is within `1 - w h` of the feature at `h`. -/
lemma convex_dev (w f : Fin T → ℝ) (hw : ∀ j, 0 ≤ w j) (hs : ∑ j, w j = 1)
    (hf : ∀ j, 0 ≤ f j ∧ f j ≤ 1) (h : Fin T) :
    |∑ j, w j * f j - f h| ≤ 1 - w h := by
  have e : ∑ j, w j * f j - f h = ∑ j, w j * (f j - f h) := by
    simp only [mul_sub, sum_sub_distrib, ← sum_mul, hs, one_mul]
  rw [e]
  calc |∑ j, w j * (f j - f h)| ≤ ∑ j, |w j * (f j - f h)| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, (w j - if j = h then w j else 0) := by
        refine sum_le_sum fun j _ => ?_
        rw [abs_mul, abs_of_nonneg (hw j)]
        split_ifs with hj
        · subst hj; simp
        · have : |f j - f h| ≤ 1 := by
            rw [abs_le]; constructor <;> linarith [hf j, hf h]
          have := mul_le_mul_of_nonneg_left this (hw j)
          linarith
    _ = 1 - w h := by rw [sum_sub_distrib, hs, sum_ite_eq']; simp

/-- Two-valued scores (A on `S`, B elsewhere): softmax-weighted [0,1]-features deviate from the
plain average over `S` by at most |Sᶜ| · exp (B - A) / |S|. -/
lemma two_valued_dev (S : Finset (Fin T)) (hS : S.Nonempty) (A B : ℝ) (f : Fin T → ℝ)
    (hf : ∀ j, 0 ≤ f j ∧ f j ≤ 1) :
    |∑ j, softmax (fun j => if j ∈ S then A else B) j * f j - (∑ j ∈ S, f j) / S.card| ≤
      (Sᶜ).card * Real.exp (B - A) / S.card := by
  set a := Real.exp A
  set b := Real.exp B
  set n : ℝ := (S.card : ℝ)
  set m : ℝ := ((Sᶜ).card : ℝ)
  have ha : 0 < a := Real.exp_pos _
  have hb : 0 < b := Real.exp_pos _
  have hn : 0 < n := by simp only [n]; exact_mod_cast hS.card_pos
  have hm : 0 ≤ m := by simp only [m]; positivity
  set F := ∑ j ∈ S, f j
  set E := ∑ j ∈ Sᶜ, f j
  have hF0 : 0 ≤ F := sum_nonneg fun j _ => (hf j).1
  have hF1 : F ≤ n := by
    calc F ≤ ∑ _j ∈ S, (1 : ℝ) := sum_le_sum fun j _ => (hf j).2
      _ = n := by simp [n]
  have hE0 : 0 ≤ E := sum_nonneg fun j _ => (hf j).1
  have hE1 : E ≤ m := by
    calc E ≤ ∑ _j ∈ Sᶜ, (1 : ℝ) := sum_le_sum fun j _ => (hf j).2
      _ = m := by simp [m]
  have hZ : ∑ j', Real.exp (if j' ∈ S then A else B) = n * a + m * b := by
    rw [← sum_add_sum_compl S]
    simp only [apply_ite Real.exp]
    rw [sum_ite_of_true (fun j hj => hj), sum_ite_of_false (fun j hj => (mem_compl.1 hj))]
    simp [n, m, a, b]
  have hZpos : 0 < n * a + m * b := by positivity
  have hsum : ∑ j, softmax (fun j => if j ∈ S then A else B) j * f j = (a * F + b * E) / (n * a + m * b) := by
    simp only [softmax, hZ, div_mul_eq_mul_div, ← sum_div]
    congr 1
    rw [← sum_add_sum_compl S]
    have h1 : ∑ j ∈ S, Real.exp (if j ∈ S then A else B) * f j = a * F := by
      rw [mul_sum]; exact sum_congr rfl fun j hj => by rw [if_pos hj]
    have h2 : ∑ j ∈ Sᶜ, Real.exp (if j ∈ S then A else B) * f j = b * E := by
      rw [mul_sum]; exact sum_congr rfl fun j hj => by rw [if_neg (mem_compl.1 hj)]
    rw [h1, h2]
  rw [hsum]
  have key : (a * F + b * E) / (n * a + m * b) - F / n = b * (n * E - m * F) / (n * (n * a + m * b)) := by
    field_simp <;> ring
  rw [key, abs_div, abs_mul, abs_of_pos hb, abs_of_pos (by positivity : 0 < n * (n * a + m * b))]
  have hNE : |n * E - m * F| ≤ n * m := by
    rw [abs_le]; constructor <;> nlinarith
  have hexp : Real.exp (B - A) = b / a := by rw [Real.exp_sub]
  rw [hexp, div_le_div_iff₀ (by positivity) hn]
  calc b * |n * E - m * F| * n ≤ b * (n * m) * n := by gcongr
    _ ≤ m * (b / a) * (n * (n * a + m * b)) := by
      rw [show m * (b / a) * (n * (n * a + m * b)) = b * (n * m) * n + m * (b / a) * (n * (m * b))
        by field_simp <;> ring]
      have : 0 ≤ m * (b / a) * (n * (m * b)) := by positivity
      linarith

end VWC
