/-
Phase 3.1: exact bits, exact AND, and the flag of the pattern detector.
-/
import VWC.Graft.PatternBounds

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ) {L : ℕ} {σ2 : ℝ}

lemma clamp_le0 {z : ℝ} (h : z ≤ 0) : clamp z = 0 := by
  unfold clamp relu; rw [max_eq_right h, max_eq_right (by linarith)]; ring

lemma clamp_ge1 {z : ℝ} (h : 1 ≤ z) : clamp z = 1 := by
  unfold clamp relu; rw [max_eq_left (by linarith), max_eq_left (by linarith)]; ring

lemma sum_dot {m k : ℕ} (f : Fin m → Fin k → ℝ) (x : Fin k → ℝ) :
    (∑ d, f d) ⬝ᵥ x = ∑ d, f d ⬝ᵥ x := by
  simp only [dotProduct, Finset.sum_apply, Finset.sum_mul]; exact Finset.sum_comm

lemma κ2lo_pos (G : Good M n D L p σ2) : 0 < κ2lo M D (slots M n) p σ2 := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast G.hD
  have hDK : (0 : ℝ) < D + slots M n := by positivity
  have hΔ : 0 ≤ Δc M D (slots M n) p := by unfold Δc; positivity
  unfold κ2lo
  exact one_div_pos.2 (Real.sqrt_pos.2 (div_pos (by nlinarith [G.hσ]) hDK))

/-- Denominator after the copies (blocks 1-MLP, 2, 3-attention). -/
noncomputable def g2 (i : Fin (T + 1)) : ℝ :=
  (D * (u i + 1) + 2 * ∑ d, acp n p D P ids u i d ^ 2) / (D + slots M n)

lemma x2_eq (i : Fin (T + 1)) :
    x2 n p D P ids u i = vecOf (fun a => St (n := n) P p.δ i (ids i) (acp n p D P ids u i) 0 0 0 a /
      Real.sqrt (g2 n p D P ids u i)) := by
  unfold x2 stv
  simp only [Pi.zero_apply]
  rw [cLN_vec, gden_St n p D P ids u 1 i _ 0 0 0 (fun _ => Or.inl rfl) (Or.inl rfl)]
  simp [g2]

lemma arow_dot (i : Fin (T + 1)) (d : Fin M) :
    arow p d ⬝ᵥ x2 n p D P ids u i =
      p.Wa * (acp n p D P ids u i d - p.τ * p.gv * p.δ) / Real.sqrt (g2 n p D P ids u i) := by
  rw [x2_eq]; unfold arow
  rw [smul_dotProduct, sub_dotProduct, smul_dotProduct, rd_vec, rd_vec]
  simp [St, sCopy, sConst, pm]; ring

lemma g2_bounds (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) :
    0 < g2 n p D P ids u i ∧
    κ2lo M D (slots M n) p σ2 ≤ 1 / Real.sqrt (g2 n p D P ids u i) ∧
    1 / Real.sqrt (g2 n p D P ids u i) ≤ κhi D (slots M n) ∧
    ∑ d, acp n p D P ids u i d ^ 2 ≤ M * (p.gv * p.δ * κhi D (slots M n)) ^ 2 := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast G.hD
  have hDK : (0 : ℝ) < D + slots M n := by positivity
  have hb := fun d => acp_bounds n p D P ids u G hT hu i d
  have hsq : ∑ d, acp n p D P ids u i d ^ 2 ≤ M * (p.gv * p.δ * κhi D (slots M n)) ^ 2 := by
    calc ∑ d, acp n p D P ids u i d ^ 2 ≤ ∑ _d : Fin M, (p.gv * p.δ * κhi D (slots M n)) ^ 2 :=
          Finset.sum_le_sum (fun d _ => pow_le_pow_left₀ (hb d).1 (hb d).2.1 2)
      _ = _ := by simp
  have hs0 : 0 ≤ ∑ d, acp n p D P ids u i d ^ 2 := Finset.sum_nonneg (fun d _ => sq_nonneg _)
  have hu0 := (hu i).1
  have hpos : 0 < g2 n p D P ids u i := by unfold g2; apply div_pos _ hDK; nlinarith
  refine ⟨hpos, ?_, ?_, hsq⟩
  · unfold κ2lo Δc; apply one_div_le_one_div_of_le (Real.sqrt_pos.2 hpos)
    apply Real.sqrt_le_sqrt; unfold g2
    apply div_le_div_of_nonneg_right _ hDK.le; nlinarith [(hu i).2]
  · unfold κhi; apply one_div_le_one_div_of_le (Real.sqrt_pos.2 (by positivity))
    apply Real.sqrt_le_sqrt; unfold g2
    apply div_le_div_of_nonneg_right _ hDK.le; nlinarith

/-- **Exact bits.** -/
lemma bit_exact (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) (d : Fin M) :
    bit n p D P ids u i d = ind (ids (jstar i d) = P (tix d)) := by
  unfold bit
  rw [arow_dot]
  obtain ⟨hpos, hk2, -, -⟩ := g2_bounds n p D P ids u G hT hu i
  obtain ⟨-, -, hlo, hhi⟩ := acp_bounds n p D P ids u G hT hu i d
  have hgvδ : 0 < p.gv * p.δ := mul_pos G.hgv G.hδ
  have hsq : 0 < Real.sqrt (g2 n p D P ids u i) := Real.sqrt_pos.2 hpos
  by_cases hm : ids (jstar i d) = P (tix d)
  · rw [ind, if_pos hm]
    apply clamp_ge1
    have ha := hlo hm
    have hk20 := κ2lo_pos n p D G
    -- the margin factor is nonnegative (else h3 fails)
    have hfac : 0 ≤ κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) - p.τ := by
      by_contra hneg; push Not at hneg
      have : p.Wa * (p.gv * p.δ * (κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) - p.τ)) *
          κ2lo M D (slots M n) p σ2 ≤ 0 := by
        apply mul_nonpos_of_nonpos_of_nonneg _ hk20.le
        exact mul_nonpos_of_nonneg_of_nonpos G.hWa (mul_nonpos_of_nonneg_of_nonpos hgvδ.le hneg.le)
      linarith [G.h3]
    have hd : p.gv * p.δ * (κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) - p.τ) ≤
        acp n p D P ids u i d - p.τ * p.gv * p.δ := by
      have : p.gv * p.δ * κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) =
          p.gv * p.δ * (κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2)) := by ring
      nlinarith
    calc (1 : ℝ) ≤ p.Wa * (p.gv * p.δ * (κlo D (slots M n) σ2 / (1 + L * lam n D (slots M n) p σ2) - p.τ)) *
          κ2lo M D (slots M n) p σ2 := G.h3
      _ ≤ p.Wa * (acp n p D P ids u i d - p.τ * p.gv * p.δ) * (1 / Real.sqrt (g2 n p D P ids u i)) := by
          apply mul_le_mul (mul_le_mul_of_nonneg_left hd G.hWa) hk2 hk20.le
          exact mul_nonneg G.hWa (le_trans (mul_nonneg hgvδ.le hfac) hd)
      _ = _ := by ring
  · rw [ind, if_neg hm]
    apply clamp_le0
    have ha := hhi hm
    have hτ : L * lam n D (slots M n) p σ2 * (p.gv * p.δ * κhi D (slots M n)) ≤ p.τ * p.gv * p.δ := by
      have := mul_le_mul_of_nonneg_right G.h1 hgvδ.le; nlinarith
    apply div_nonpos_of_nonpos_of_nonneg _ hsq.le
    exact mul_nonpos_of_nonneg_of_nonpos G.hWa (by linarith)

lemma bit01 (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) (d : Fin M) : bit n p D P ids u i d = 0 ∨ bit n p D P ids u i d = 1 := by
  rw [bit_exact n p D P ids u G hT hu]; exact ind_cases _

/-! ### The AND -/

/-- The pattern ends at position i. -/
def endsAt (i : Fin (T + 1)) : Prop :=
  ∃ h : M ≤ (i : ℕ), ∀ idx : Fin (M + 1), ids ⟨i - M + idx, by omega⟩ = P idx

lemma endsAt_iff (i : Fin (T + 1)) :
    endsAt P ids i ↔ (ids i = P (Fin.last M) ∧ M ≤ (i : ℕ) ∧ ∀ d, ids (jstar i d) = P (tix d)) := by
  constructor
  · rintro ⟨h, hP⟩
    refine ⟨?_, h, fun d => ?_⟩
    · have := hP (Fin.last M)
      rwa [show (⟨(i : ℕ) - M + (Fin.last M : ℕ), by omega⟩ : Fin (T + 1)) = i from
        Fin.ext (by simp; omega)] at this
    · have := hP (tix d)
      rwa [show (⟨(i : ℕ) - M + (tix d : ℕ), by omega⟩ : Fin (T + 1)) = jstar i d from
        Fin.ext (by simp [tix, jstar]; omega)] at this
  · rintro ⟨h1, h2, h3⟩
    refine ⟨h2, fun idx => ?_⟩
    by_cases hl : (idx : ℕ) = M
    · rw [show idx = Fin.last M from Fin.ext hl,
        show (⟨(i : ℕ) - M + (Fin.last M : ℕ), by omega⟩ : Fin (T + 1)) = i from
          Fin.ext (by simp; omega)]
      exact h1
    · have hlt : (idx : ℕ) < M := by omega
      set d : Fin M := ⟨M - 1 - idx, by omega⟩
      have hd : tix d = idx := Fin.ext (by simp [tix, d]; omega)
      rw [show (⟨(i : ℕ) - M + (idx : ℕ), by omega⟩ : Fin (T + 1)) = jstar i d from
        Fin.ext (by simp [jstar, d]; omega), ← hd]
      exact h3 d

noncomputable def g3 (i : Fin (T + 1)) : ℝ := g2 n p D P ids u i

lemma x3_eq (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2) (i : Fin (T + 1)) :
    x3 n p D P ids u i = vecOf (fun a => St (n := n) P p.δ i (ids i) (acp n p D P ids u i)
      (bit n p D P ids u i) 0 0 a / Real.sqrt (g2 n p D P ids u i)) := by
  unfold x3 stv
  simp only [Pi.zero_apply]
  rw [cLN_vec, gden_St n p D P ids u 1 i _ _ 0 0 (bit01 n p D P ids u G hT hu i) (Or.inl rfl)]
  simp [g2]

lemma zrow_dot (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) :
    zrow ⬝ᵥ x3 n p D P ids u i = p.δ * (1 / 2 - ((1 - ind (ids i = P (Fin.last M))) +
      (1 - ind (M ≤ (i : ℕ))) + ∑ d, (1 - bit n p D P ids u i d))) / Real.sqrt (g2 n p D P ids u i) := by
  rw [x3_eq n p D P ids u G hT hu]
  unfold zrow
  rw [sub_dotProduct, smul_dotProduct, add_dotProduct, add_dotProduct, sum_dot, rd_vec, rd_vec, rd_vec]
  simp only [rd_vec]
  simp [St, sTokS, sValid, sCbit, sConst, qd, pm]
  simp only [← Finset.sum_div, Finset.sum_neg_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  ring

/-- **Exact AND** (on the complement coordinates, threshold mid-gap). -/
lemma andv_exact (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) [Decidable (endsAt P ids i)] :
    andv n p D P ids u i = ind (endsAt P ids i) := by
  unfold andv
  rw [zrow_dot n p D P ids u G hT hu]
  obtain ⟨hpos, hk2, -, -⟩ := g2_bounds n p D P ids u G hT hu i
  have hsq : 0 < Real.sqrt (g2 n p D P ids u i) := Real.sqrt_pos.2 hpos
  have hbits := fun d => bit_exact n p D P ids u G hT hu i d
  have hc : ∀ d, 0 ≤ 1 - bit n p D P ids u i d := fun d => by
    rw [hbits]; linarith [ind_le_one (ids (jstar i d) = P (tix d))]
  by_cases he : endsAt P ids i
  · rw [show ind (endsAt P ids i) = 1 from if_pos he]
    obtain ⟨h1, h2, h3⟩ := (endsAt_iff P ids i).1 he
    have hS : (1 - ind (ids i = P (Fin.last M))) + (1 - ind (M ≤ (i : ℕ))) +
        ∑ d, (1 - bit n p D P ids u i d) = 0 := by
      simp only [hbits, ind, if_pos h1, if_pos h2, h3, if_true, sub_self, Finset.sum_const_zero]
      ring
    rw [hS]
    apply clamp_ge1
    have hk20 := κ2lo_pos n p D G
    calc (1 : ℝ) ≤ p.Wb * p.δ / 2 * κ2lo M D (slots M n) p σ2 := G.h4
      _ ≤ p.Wb * p.δ / 2 * (1 / Real.sqrt (g2 n p D P ids u i)) :=
          mul_le_mul_of_nonneg_left hk2 (by have := G.hWb; have := G.hδ; positivity)
      _ = _ := by ring
  · rw [show ind (endsAt P ids i) = 0 from if_neg he]
    apply clamp_le0
    have hA0 : 0 ≤ 1 - ind (ids i = P (Fin.last M)) := by linarith [ind_le_one (ids i = P (Fin.last M))]
    have hB0 : 0 ≤ 1 - ind (M ≤ (i : ℕ)) := by linarith [ind_le_one (M ≤ (i : ℕ))]
    have hC0 : 0 ≤ ∑ d, (1 - bit n p D P ids u i d) := Finset.sum_nonneg (fun d _ => hc d)
    have hS : 1 ≤ (1 - ind (ids i = P (Fin.last M))) + (1 - ind (M ≤ (i : ℕ))) +
        ∑ d, (1 - bit n p D P ids u i d) := by
      rw [endsAt_iff] at he
      by_cases hA1 : ids i = P (Fin.last M)
      · by_cases hB1 : M ≤ (i : ℕ)
        · obtain ⟨d0, hd0⟩ : ∃ d, ids (jstar i d) ≠ P (tix d) := by
            by_contra hc'; simp only [not_exists, ne_eq, not_not] at hc'; exact he ⟨hA1, hB1, hc'⟩
          have hz : 1 - bit n p D P ids u i d0 = 1 := by rw [hbits, ind, if_neg hd0]; ring
          have : 1 ≤ ∑ d, (1 - bit n p D P ids u i d) :=
            calc (1 : ℝ) = 1 - bit n p D P ids u i d0 := hz.symm
              _ ≤ _ := Finset.single_le_sum (fun d _ => hc d) (Finset.mem_univ d0)
          linarith
        · have : 1 - ind (M ≤ (i : ℕ)) = 1 := by rw [ind, if_neg hB1]; ring
          linarith
      · have : 1 - ind (ids i = P (Fin.last M)) = 1 := by rw [ind, if_neg hA1]; ring
        linarith
    apply mul_nonpos_of_nonneg_of_nonpos G.hWb
    apply div_nonpos_of_nonpos_of_nonneg _ hsq.le
    apply mul_nonpos_of_nonneg_of_nonpos G.hδ.le
    linarith

end VWC.Graft.Pattern
