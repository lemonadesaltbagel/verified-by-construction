/-
Phase 3.1: bounds for the pattern detector — sharp attention, exact bits, exact AND.

For host variances u ∈ [0, σ²] (κhi is the u = 0 bound, so no lower variance bound is assumed):
block-1 scales lie in [κlo, κhi]; head d puts weight
≥ 1/(1 + Lλ) on position i − (d+1) and ≤ λ elsewhere; so the copy value is ≥ gv δ κlo/(1+Lλ) when
that token matches and ≤ L λ gv δ κhi when it does not. The clamps then produce exact bits
(`bit_exact`), and the AND is exact (`andv_exact`): it is 1 iff the pattern ends at position i.
-/
import VWC.Graft.PatternForms

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ)

/-- Scale bounds (block 1). -/
noncomputable def κlo (D K : ℕ) (σ2 : ℝ) : ℝ := 1 / Real.sqrt (D * (σ2 + 1) / (D + K))
noncomputable def κhi (D K : ℕ) : ℝ := 1 / Real.sqrt (D / (D + K))
/-- Extra energy bound from the copies. -/
noncomputable def Δc (M D K : ℕ) (p : Par) : ℝ := 2 * M * (p.gv * p.δ * κhi D K) ^ 2
/-- Scale lower bound after the copies (blocks 1-MLP, 2, 3). -/
noncomputable def κ2lo (M D K : ℕ) (p : Par) (σ2 : ℝ) : ℝ :=
  1 / Real.sqrt ((D * (σ2 + 1) + Δc M D K p) / (D + K))
/-- Attention leakage. -/
noncomputable def lam (n D K : ℕ) (p : Par) (σ2 : ℝ) : ℝ :=
  Real.exp (-2 * (p.cs * p.δ ^ 2 * κlo D K σ2 ^ 2 / Real.sqrt ((n + 1 : ℕ) : ℝ)))

/-- Flag-head leakage: the softmax mass outside the matched positions is ≤ lam3 (block 3). -/
noncomputable def lam3 (M n D K L : ℕ) (p : Par) (σ2 : ℝ) : ℝ :=
  L * Real.exp (-(p.fq * p.fk * p.δ ^ 2 * κ2lo M D K p σ2 ^ 2 / Real.sqrt ((n + 1 : ℕ) : ℝ)))

/-- The parameter conditions (all host-independent except through σ²). -/
structure Good (M n D L : ℕ) (p : Par) (σ2 : ℝ) : Prop where
  hD : 0 < D
  hσ : 0 ≤ σ2
  hδ : 0 < p.δ
  hgv : 0 < p.gv
  hcs : 0 ≤ p.cs
  hWa : 0 ≤ p.Wa
  hWb : 0 ≤ p.Wb
  hfq : 0 ≤ p.fq
  hfk : 0 ≤ p.fk
  hA : 0 ≤ p.A
  hL2 : L ≤ 2 ^ n
  h1 : L * lam n D (slots M n) p σ2 * κhi D (slots M n) ≤ p.τ
  h3 : 1 ≤ p.Wa * (p.gv * p.δ * (κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) - p.τ)) *
    κ2lo M D (slots M n) p σ2
  h4 : 1 ≤ p.Wb * p.δ / 2 * κ2lo M D (slots M n) p σ2
  h6 : lam3 M n D (slots M n) L p σ2 < 1

variable {L : ℕ} {σ2 : ℝ}

lemma g1_pos (hD : 0 < D) (t : Fin (T + 1)) (hu : 0 ≤ u t) : 0 < g1 (M := M) n D u t := by
  unfold g1; have : (0 : ℝ) < D := by exact_mod_cast hD
  positivity

lemma k1_bounds (hD : 0 < D) (t : Fin (T + 1)) (hu : 0 ≤ u t ∧ u t ≤ σ2) :
    κlo D (slots M n) σ2 ≤ 1 / Real.sqrt (g1 (M := M) n D u t) ∧
    1 / Real.sqrt (g1 (M := M) n D u t) ≤ κhi D (slots M n) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hDK : (0 : ℝ) < D + slots M n := by positivity
  have hg := g1_pos (M := M) n D u hD t hu.1
  constructor
  · unfold κlo; apply one_div_le_one_div_of_le (Real.sqrt_pos.2 hg)
    apply Real.sqrt_le_sqrt; unfold g1
    apply div_le_div_of_nonneg_right _ hDK.le; nlinarith [hu.2]
  · unfold κhi; apply one_div_le_one_div_of_le (Real.sqrt_pos.2 (by positivity))
    apply Real.sqrt_le_sqrt; unfold g1
    apply div_le_div_of_nonneg_right _ hDK.le; nlinarith [hu.1]

/-- The key position of head d at query i. -/
def jstar (i : Fin (T + 1)) (d : Fin M) : Fin (T + 1) := ⟨i - (d + 1), by omega⟩

lemma ind_le_one (P' : Prop) [Decidable P'] : ind P' ≤ 1 := by unfold ind; split_ifs <;> norm_num
lemma ind_nonneg (P' : Prop) [Decidable P'] : 0 ≤ ind P' := by unfold ind; split_ifs <;> norm_num

/-- Sharp attention of head d. -/
lemma head_sharp (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) (d : Fin M) :
    let s := fun j : Fin (T + 1) => p.cs * p.δ ^ 2 /
        (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) *
        (∑ b : Fin n, sig (i - (d + 1)) b * sig j b - ((n : ℝ) - 1)) / Real.sqrt ((n + 1 : ℕ) : ℝ)
    1 / (1 + (T + 1 : ℕ) * lam n D (slots M n) p σ2) ≤ smax s i (jstar i d) ∧
    ∀ j, j ≠ jstar i d → smax s i j ≤ lam n D (slots M n) p σ2 := by
  intro s
  set ρ := p.cs * p.δ ^ 2 * κlo D (slots M n) σ2 ^ 2 / Real.sqrt ((n + 1 : ℕ) : ℝ)
  have hn1 : 0 < Real.sqrt ((n + 1 : ℕ) : ℝ) := Real.sqrt_pos.2 (by positivity)
  have hk : ∀ t, κlo D (slots M n) σ2 ≤ 1 / Real.sqrt (g1 (M := M) n D u t) :=
    fun t => (k1_bounds (M := M) n D u G.hD t (hu t)).1
  have hklo : 0 ≤ κlo D (slots M n) σ2 := by unfold κlo; positivity
  have hprod : ∀ j, κlo D (slots M n) σ2 ^ 2 ≤
      1 / (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) := by
    intro j; rw [← one_div_mul_one_div, sq]
    exact mul_le_mul (hk i) (hk j) hklo (by positivity)
  have hcδ : 0 ≤ p.cs * p.δ ^ 2 := mul_nonneg G.hcs (sq_nonneg _)
  have hpos2 : ∀ j, 0 < Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j) := fun j =>
    mul_pos (Real.sqrt_pos.2 (g1_pos (M := M) n D u G.hD i (hu i).1)) (Real.sqrt_pos.2 (g1_pos (M := M) n D u G.hD j (hu j).1))
  have hlt : ∀ x : Fin (T + 1), (x : ℕ) < 2 ^ n := fun x => by
    have := G.hL2; have := x.isLt; omega
  have hq : (i : ℕ) - (d + 1) < 2 ^ n := by have := hlt i; omega
  -- the base factor
  have hbase : ∀ j, p.cs * p.δ ^ 2 * κlo D (slots M n) σ2 ^ 2 ≤
      p.cs * p.δ ^ 2 / (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) := by
    intro j; rw [div_eq_mul_one_div]; exact mul_le_mul_of_nonneg_left (hprod j) hcδ
  have h1 : ρ ≤ s (jstar i d) := by
    simp only [s, jstar]
    rw [sig_agree_eq]
    have : (n : ℝ) - ((n : ℝ) - 1) = 1 := by ring
    rw [this, mul_one]
    exact div_le_div_of_nonneg_right (hbase _) hn1.le
  have h2 : ∀ j, j ≤ i → j ≠ jstar i d → s j ≤ -ρ := by
    intro j _ hne
    simp only [s]
    have hne' : (i : ℕ) - (d + 1) ≠ j := by
      intro h; apply hne; ext; simp [jstar, h]
    have hag := sig_agree_ne (n := n) _ _ hq (hlt j) hne'
    have hneg : ∑ b : Fin n, sig (i - (d + 1)) b * sig j b - ((n : ℝ) - 1) ≤ -1 := by linarith
    have hf : 0 ≤ p.cs * p.δ ^ 2 / (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) :=
      div_nonneg hcδ (hpos2 j).le
    rw [neg_div', div_le_div_iff_of_pos_right hn1]
    nlinarith [hbase j]
  have := sharp s i (jstar i d) (by simp [jstar, Fin.le_def]) ρ h1 h2
  exact ⟨this.1, this.2⟩

/-- Copy-value bounds. -/
lemma acp_bounds (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) (d : Fin M) :
    0 ≤ acp n p D P ids u i d ∧ acp n p D P ids u i d ≤ p.gv * p.δ * κhi D (slots M n) ∧
    (ids (jstar i d) = P (tix d) →
      p.gv * p.δ * κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) ≤ acp n p D P ids u i d) ∧
    (ids (jstar i d) ≠ P (tix d) →
      acp n p D P ids u i d ≤ L * lam n D (slots M n) p σ2 * (p.gv * p.δ * κhi D (slots M n))) := by
  rw [acp_eq]
  obtain ⟨hw1, hw2⟩ := head_sharp n p D u G hT hu i d
  set s := fun j : Fin (T + 1) => p.cs * p.δ ^ 2 /
        (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) *
        (∑ b : Fin n, sig (i - (d + 1)) b * sig j b - ((n : ℝ) - 1)) / Real.sqrt ((n + 1 : ℕ) : ℝ)
  set w := fun j => smax s i j
  set v := fun j : Fin (T + 1) =>
    p.gv * p.δ * ind (ids j = P (tix d)) / Real.sqrt (g1 (M := M) n D u j)
  have hgvδ : 0 < p.gv * p.δ := mul_pos G.hgv G.hδ
  have hv0 : ∀ j, 0 ≤ v j := fun j =>
    div_nonneg (mul_nonneg hgvδ.le (ind_nonneg _)) (Real.sqrt_nonneg _)
  have hvhi : ∀ j, v j ≤ p.gv * p.δ * κhi D (slots M n) := fun j => by
    simp only [v]
    rw [mul_div_assoc]
    apply mul_le_mul_of_nonneg_left _ hgvδ.le
    calc ind (ids j = P (tix d)) / Real.sqrt (g1 (M := M) n D u j)
        ≤ 1 / Real.sqrt (g1 (M := M) n D u j) :=
          div_le_div_of_nonneg_right (ind_le_one _) (Real.sqrt_nonneg _)
      _ ≤ _ := (k1_bounds (M := M) n D u G.hD j (hu j)).2
  have hw0 : ∀ j, 0 ≤ w j := fun j => smax_nonneg s i j
  have hwsum : ∑ j, w j = 1 := smax_sum s i
  have hlam0 : 0 ≤ lam n D (slots M n) p σ2 := (Real.exp_pos _).le
  have hLT : ((T + 1 : ℕ) : ℝ) ≤ L := by exact_mod_cast hT
  refine ⟨Finset.sum_nonneg (fun j _ => mul_nonneg (hw0 j) (hv0 j)), ?_, fun hm => ?_, fun hm => ?_⟩
  · calc ∑ j, w j * v j ≤ ∑ j, w j * (p.gv * p.δ * κhi D (slots M n)) :=
          Finset.sum_le_sum (fun j _ => mul_le_mul_of_nonneg_left (hvhi j) (hw0 j))
      _ = _ := by rw [← Finset.sum_mul, hwsum, one_mul]
  · have hvs : p.gv * p.δ * κlo D (slots M n) σ2 ≤ v (jstar i d) := by
      simp only [v, ind, if_pos hm, mul_one]
      rw [div_eq_mul_one_div]
      exact mul_le_mul_of_nonneg_left (k1_bounds (M := M) n D u G.hD (jstar i d) (hu _)).1 hgvδ.le
    have hden : 1 + ((T + 1 : ℕ) : ℝ) * lam n D (slots M n) p σ2 ≤ 1 + L * lam n D (slots M n) p σ2 := by
      nlinarith
    have hklo : 0 ≤ κlo D (slots M n) σ2 := by unfold κlo; positivity
    calc p.gv * p.δ * κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2)
        ≤ p.gv * p.δ * κlo D (slots M n) σ2 * (1 / (1 + ((T + 1 : ℕ) : ℝ) * lam n D (slots M n) p σ2)) := by
          rw [mul_one_div]; exact div_le_div_of_nonneg_left (by positivity) (by positivity) hden
      _ ≤ v (jstar i d) * w (jstar i d) := mul_le_mul hvs hw1 (by positivity) (hv0 _)
      _ = w (jstar i d) * v (jstar i d) := mul_comm _ _
      _ ≤ ∑ j, w j * v j := Finset.single_le_sum (fun j _ => mul_nonneg (hw0 j) (hv0 j))
          (Finset.mem_univ _)
  · have hv0s : v (jstar i d) = 0 := by simp [v, ind, hm]
    calc ∑ j, w j * v j ≤ ∑ _j : Fin (T + 1), lam n D (slots M n) p σ2 * (p.gv * p.δ * κhi D (slots M n)) := by
          apply Finset.sum_le_sum; intro j _
          by_cases hj : j = jstar i d
          · rw [hj, hv0s, mul_zero]
            have : 0 ≤ κhi D (slots M n) := by unfold κhi; positivity
            exact mul_nonneg hlam0 (mul_nonneg hgvδ.le this)
          · exact mul_le_mul (hw2 j hj) (hvhi j) (hv0 j) hlam0
      _ = ((T + 1 : ℕ) : ℝ) * (lam n D (slots M n) p σ2 * (p.gv * p.δ * κhi D (slots M n))) := by
          simp
      _ ≤ L * (lam n D (slots M n) p σ2 * (p.gv * p.δ * κhi D (slots M n))) := by
          apply mul_le_mul_of_nonneg_right hLT
          have : 0 ≤ κhi D (slots M n) := by unfold κhi; positivity
          positivity
      _ = _ := by ring

end VWC.Graft.Pattern
