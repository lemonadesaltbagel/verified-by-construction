/-
Phase 3, theorem (A′): off trigger, approximately.

A circuit that detects multi-token patterns cannot be idle on every non-trigger prompt
(`PHASE3_DESIGN.md` §2): partial matches make it write. As long as its writes are zero-sum, its
final readout is exactly zero, and its final energy lies in [E, E + Δ], the only effect on the host
is through the final LayerNorm's denominator. Every grafted logit then stays within

  η_v = √D · ‖w_v ∘ γ_f‖ · Δ / (2 D ε_f)

of the host's logit, using boundary weights only. With the circuit obligation stated for every
host variance profile u ∈ [0, σ²], the bound holds for every host whose final residual variance is
at most σ² (the same hypothesis as theorem (B)).
-/
import VWC.Graft.Main

namespace VWC.Graft

open Matrix Finset VWC.Arch

variable {D k : ℕ}

lemma abs_cs (a b : Fin D → ℝ) :
    |∑ l, a l * b l| ≤ Real.sqrt (∑ l, a l ^ 2) * Real.sqrt (∑ l, b l ^ 2) := by
  rw [← Real.sqrt_mul (Finset.sum_nonneg fun l _ => sq_nonneg _), ← Real.sqrt_sq_eq_abs]
  exact Real.sqrt_le_sqrt (Finset.sum_mul_sq_le_sq_mul_sq _ _ _)

/-- √P (1/√X − 1/√(X+Δ')) ≤ Δ'/(2X) for 0 ≤ P ≤ X. -/
lemma sqrt_gap (X Δ' P : ℝ) (hX : 0 < X) (hΔ : 0 ≤ Δ') (hP : P ≤ X) :
    Real.sqrt P * (1 / Real.sqrt X - 1 / Real.sqrt (X + Δ')) ≤ Δ' / (2 * X) := by
  have hr : 0 < Real.sqrt X := Real.sqrt_pos.2 hX
  have hr2 : Real.sqrt X ^ 2 = X := Real.sq_sqrt hX.le
  have hs2 : Real.sqrt (X + Δ') ^ 2 = X + Δ' := Real.sq_sqrt (by linarith)
  have hrs : Real.sqrt X ≤ Real.sqrt (X + Δ') := Real.sqrt_le_sqrt (by linarith)
  have hp : Real.sqrt P ≤ Real.sqrt X := Real.sqrt_le_sqrt hP
  have hp0 : 0 ≤ Real.sqrt P := Real.sqrt_nonneg _
  set r := Real.sqrt X
  set s := Real.sqrt (X + Δ')
  have hs : 0 < s := lt_of_lt_of_le hr hrs
  rw [show 1 / r - 1 / s = (s - r) / (r * s) by field_simp, mul_div_assoc',
    div_le_div_iff₀ (by positivity) (by positivity)]
  have h1 : Real.sqrt P * (s - r) * (2 * r ^ 2) ≤ r * (s - r) * (2 * r ^ 2) :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hp (by linarith)) (by positivity)
  have h2 : r * (s - r) * (2 * r ^ 2) ≤ (s ^ 2 - r ^ 2) * (r * s) := by
    nlinarith [mul_nonneg (mul_nonneg hr.le (sub_nonneg.2 hrs))
      (mul_nonneg (sub_nonneg.2 hrs) (by linarith : (0 : ℝ) ≤ s + 2 * r))]
  calc Real.sqrt P * (s - r) * (2 * X) = Real.sqrt P * (s - r) * (2 * r ^ 2) := by rw [hr2]
    _ ≤ r * (s - r) * (2 * r ^ 2) := h1
    _ ≤ (s ^ 2 - r ^ 2) * (r * s) := h2
    _ = Δ' * (r * s) := by rw [hs2, hr2]; ring

/-- One unembedding row: the grafted final LayerNorm (circuit energy E + Δ') versus the host's, with
the host's actual variance in the bound: √D ‖w∘γ‖ Δ / (2 D (var y + ε)). -/
lemma row_perturb_at (hD : 0 < D) (w γ β y : Fin D → ℝ) (ε Δ' Δ : ℝ) (hε : 0 < ε) (h0 : 0 ≤ Δ')
    (h1 : Δ' ≤ Δ) :
    |(w ⬝ᵥ fun l => (y l - mean y) / Real.sqrt ((D * (var y + ε) + Δ') / (D + k)) *
        (γ l * Real.sqrt (D / (D + k))) + β l) - w ⬝ᵥ (Norm.ln γ β ε).apply y| ≤
      Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) * Δ / (2 * (D * (var y + ε))) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hv := var_nonneg y
  have hXε : 0 < D * (var y + ε) := by positivity
  have hw : 0 ≤ Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) := by positivity
  set X := D * (var y + ε) with hXdef
  have hDk : (0 : ℝ) < D + k := by positivity
  have hc1 : Real.sqrt (D / (D + k)) / Real.sqrt ((X + Δ') / (D + k)) = Real.sqrt D / Real.sqrt (X + Δ') := by
    rw [Real.sqrt_div hDr.le, Real.sqrt_div (by linarith)]
    have : 0 < Real.sqrt ((D : ℝ) + k) := Real.sqrt_pos.2 hDk
    have : 0 < Real.sqrt (X + Δ') := Real.sqrt_pos.2 (by linarith)
    field_simp
  have hc2 : 1 / Real.sqrt (var y + ε) = Real.sqrt D / Real.sqrt X := by
    rw [hXdef, Real.sqrt_mul hDr.le]
    have : 0 < Real.sqrt (D : ℝ) := Real.sqrt_pos.2 hDr
    field_simp
  have hdiff : (w ⬝ᵥ fun l => (y l - mean y) / Real.sqrt ((X + Δ') / (D + k)) *
        (γ l * Real.sqrt (D / (D + k))) + β l) - w ⬝ᵥ (Norm.ln γ β ε).apply y =
      (∑ l, w l * γ l * (y l - mean y)) *
        (Real.sqrt D / Real.sqrt (X + Δ') - Real.sqrt D / Real.sqrt X) := by
    rw [← hc1, ← hc2]
    simp only [Norm.apply, dotProduct, ← Finset.sum_sub_distrib, Finset.sum_mul]
    refine Finset.sum_congr rfl (fun l _ => ?_)
    have hm : (∑ i, y i) / (D : ℝ) = mean y := rfl
    have hvv : (∑ i, (y i - mean y) ^ 2) / (D : ℝ) = var y := rfl
    simp only [hm, hvv]
    ring
  rw [hdiff, abs_mul]
  have hcs := abs_cs (fun l => w l * γ l) (fun l => y l - mean y)
  have hyy : ∑ l, (y l - mean y) ^ 2 = D * var y := by unfold var; field_simp
  rw [hyy] at hcs
  have hsD : 0 < Real.sqrt (D : ℝ) := Real.sqrt_pos.2 hDr
  have hrs : Real.sqrt X ≤ Real.sqrt (X + Δ') := Real.sqrt_le_sqrt (by linarith)
  have hr : 0 < Real.sqrt X := Real.sqrt_pos.2 hXε
  have habs : |Real.sqrt D / Real.sqrt (X + Δ') - Real.sqrt D / Real.sqrt X| =
      Real.sqrt D * (1 / Real.sqrt X - 1 / Real.sqrt (X + Δ')) := by
    rw [abs_of_nonpos]
    · ring
    · rw [sub_nonpos]; exact div_le_div_of_nonneg_left hsD.le hr hrs
  rw [habs]
  have hgap := sqrt_gap X Δ' (D * var y) hXε h0 (by nlinarith)
  have hfin : Δ' / (2 * X) ≤ Δ / (2 * X) := div_le_div_of_nonneg_right h1 (by positivity)
  calc |∑ l, w l * γ l * (y l - mean y)| * (Real.sqrt D * (1 / Real.sqrt X - 1 / Real.sqrt (X + Δ')))
      ≤ Real.sqrt (∑ l, (w l * γ l) ^ 2) * Real.sqrt (D * var y) *
          (Real.sqrt D * (1 / Real.sqrt X - 1 / Real.sqrt (X + Δ'))) :=
        mul_le_mul_of_nonneg_right hcs (by
          have : 0 ≤ 1 / Real.sqrt X - 1 / Real.sqrt (X + Δ') := by
            rw [sub_nonneg]; exact one_div_le_one_div_of_le hr hrs
          positivity)
    _ = Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) *
          (Real.sqrt (D * var y) * (1 / Real.sqrt X - 1 / Real.sqrt (X + Δ'))) := by ring
    _ ≤ Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) * (Δ / (2 * X)) :=
        mul_le_mul_of_nonneg_left (hgap.trans hfin) hw
    _ = _ := by ring

/-- The uniform version (host variance ≥ 0): √D ‖w∘γ‖ Δ / (2 D ε). -/
lemma row_perturb (hD : 0 < D) (w γ β y : Fin D → ℝ) (ε Δ' Δ : ℝ) (hε : 0 < ε) (h0 : 0 ≤ Δ')
    (h1 : Δ' ≤ Δ) :
    |(w ⬝ᵥ fun l => (y l - mean y) / Real.sqrt ((D * (var y + ε) + Δ') / (D + k)) *
        (γ l * Real.sqrt (D / (D + k))) + β l) - w ⬝ᵥ (Norm.ln γ β ε).apply y| ≤
      Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) * Δ / (2 * (D * ε)) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hv := var_nonneg y
  refine (row_perturb_at (k := k) hD w γ β y ε Δ' Δ hε h0 h1).trans ?_
  have hΔ : 0 ≤ Δ := h0.trans h1
  have hw : 0 ≤ Real.sqrt D * Real.sqrt (∑ l, (w l * γ l) ^ 2) * Δ := by positivity
  exact div_le_div_of_nonneg_left hw (by positivity) (by nlinarith)

/-- The off-trigger perturbation bound of token v (boundary weights only). -/
noncomputable def Host.eta {V L F H dq dv : ℕ} (M : Host V D L F H dq dv) (Δ : ℝ) (v : Fin V) : ℝ :=
  Real.sqrt D * Real.sqrt (∑ l, (M.Wh v l * M.lnf.γ l) ^ 2) * Δ / (2 * (D * M.lnf.ε))

/-- The position-wise bound: the host's actual final-residual variance u at that position in the
denominator. `etaAt Δ u v ≤ eta Δ v` for u ≥ 0. -/
noncomputable def Host.etaAt {V L F H dq dv : ℕ} (M : Host V D L F H dq dv) (Δ u : ℝ) (v : Fin V) : ℝ :=
  Real.sqrt D * Real.sqrt (∑ l, (M.Wh v l * M.lnf.γ l) ^ 2) * Δ / (2 * (D * (u + M.lnf.ε)))

/-- **Theorem (A′), position-wise.** If the host's final residual variance is at most σ² and, for
every host variance profile in [0, σ²], the circuit's final state has energy in [E, E + Δ] and its
readout vanishes, then at every position i the grafted logit of v is within
`M.etaAt Δ (var (M.resid hT ids i)) v` of the host's: the bound uses the host's own variance at i. -/
theorem graft_offtrigger_approx_at {V L F H dq dv Fc Hc dqc dvc T : ℕ} (hD : 0 < D)
    (M : Host V D L F H dq dv) (C : Circuit V L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (hfeat : ∀ v p, ∑ j, (C.sTok v j + C.sPos p j) = 0 ∧ ∑ j, (C.sTok v j + C.sPos p j) ^ 2 = C.E)
    (hzs : ∀ b ∈ C.blocks, b.ZeroSum)
    (hT : T ≤ L) (ids : Fin T → Fin V) (i : Fin T) (Δ σ2 : ℝ)
    (hσ : ∀ t, var (M.resid hT ids t) ≤ σ2)
    (hoff : ∀ u : Fin T → ℝ, (∀ t, 0 ≤ u t ∧ u t ≤ σ2) →
      C.E ≤ ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) i j ^ 2 ∧
      ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) i j ^ 2 ≤ C.E + Δ ∧
      ∀ v, C.Wc v ⬝ᵥ C.out D M.lnf.ε u (C.s0 hT ids) i = 0) (v : Fin V) :
    |(graft M C).logits hT ids i v - M.toModel.logits hT ids i v| ≤
      M.etaAt Δ (var (M.resid hT ids i)) v := by
  set s := C.s0 hT ids
  have hs0 : ∀ t, ∑ j, s t j = 0 := fun t => (hfeat _ _).1
  have hsE : ∀ t, ∑ j, s t j ^ 2 = C.E := fun t => (hfeat _ _).2
  set x : Fin T → Fin D → ℝ := fun t => M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t)
  have hemb : (fun t => (graft M C).tokEmb (ids t) + (graft M C).posEmb (Fin.castLE hT t)) =
      fun t => joint (x t) (s t) := by
    funext t
    simp only [graft, append_add]
    funext j
    refine Fin.addCases (fun ih => ?_) (fun jj => ?_) j
    · simp [joint, x]
    · simp [joint, x, s, Circuit.s0, mean_add]; ring
  simp only [GModel.logits]
  rw [hemb]
  simp only [graft, applyBlocks_append]
  rw [host_blocks hD C.E M.blocks hεH x s hs0 hsE]
  set y := gApplyBlocks (M.blocks.map (fun b => (⟨b.toBlock⟩ : AnyGBlock D))) x with hy
  rw [circuit_blocks_run hD C.E C.blocks hzs y s hs0]
  set u : Fin T → ℝ := fun t => var (y t)
  set sf := runBlocks D C.E C.blocks u s
  have hsf : ∑ j, sf i j = 0 := runBlocks_zs C.E C.blocks hzs u s hs0 i
  rw [ln_graft_gen hD (y i) (sf i) C.E hsf]
  obtain ⟨hE1, hE2, hro⟩ := hoff u (fun t => ⟨var_nonneg _, by exact hσ t⟩)
  have hrv := hro v
  simp only [Pi.add_apply, mulVec, append_dot]
  set co := cLN D k C.E C.γf C.βf M.lnf.ε (var (y i)) (sf i)
  have hco : co = C.out D M.lnf.ε u s i := rfl
  rw [← hco] at hrv
  rw [hrv, add_zero]
  have hrow := row_perturb_at (k := k) hD (M.Wh v) M.lnf.γ M.lnf.β (y i) M.lnf.ε
    (∑ j, sf i j ^ 2 - C.E) Δ hεf (by linarith) (by linarith)
  have hg : gden D k C.E M.lnf.ε (var (y i)) (sf i) =
      (D * (var (y i) + M.lnf.ε) + (∑ j, sf i j ^ 2 - C.E)) / (D + k) := by unfold gden; ring
  rw [hg]
  show _ ≤ Real.sqrt D * Real.sqrt (∑ l, (M.Wh v l * M.lnf.γ l) ^ 2) * Δ / (2 * (D * (var (y i) + M.lnf.ε)))
  have e : ∀ a b c : ℝ, a + c - (b + c) = a - b := fun a b c => by ring
  calc _ = |(M.Wh v ⬝ᵥ fun l => (y i l - mean (y i)) /
          Real.sqrt ((D * (var (y i) + M.lnf.ε) + (∑ j, sf i j ^ 2 - C.E)) / (D + k)) *
          (M.lnf.γ l * Real.sqrt (D / (D + k))) + M.lnf.β l) + M.bh v -
        (M.Wh v ⬝ᵥ (Norm.ln M.lnf.γ M.lnf.β M.lnf.ε).apply (y i) + M.bh v)| := rfl
    _ = |(M.Wh v ⬝ᵥ fun l => (y i l - mean (y i)) /
          Real.sqrt ((D * (var (y i) + M.lnf.ε) + (∑ j, sf i j ^ 2 - C.E)) / (D + k)) *
          (M.lnf.γ l * Real.sqrt (D / (D + k))) + M.lnf.β l) -
        M.Wh v ⬝ᵥ (Norm.ln M.lnf.γ M.lnf.β M.lnf.ε).apply (y i)| := by rw [e]
    _ ≤ _ := hrow

/-- **Theorem (A′).** The uniform version: every grafted logit is within `M.eta Δ v` of the host's. -/
theorem graft_offtrigger_approx {V L F H dq dv Fc Hc dqc dvc T : ℕ} (hD : 0 < D)
    (M : Host V D L F H dq dv) (C : Circuit V L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (hfeat : ∀ v p, ∑ j, (C.sTok v j + C.sPos p j) = 0 ∧ ∑ j, (C.sTok v j + C.sPos p j) ^ 2 = C.E)
    (hzs : ∀ b ∈ C.blocks, b.ZeroSum)
    (hT : T ≤ L) (ids : Fin T → Fin V) (i : Fin T) (Δ σ2 : ℝ)
    (hσ : ∀ t, var (M.resid hT ids t) ≤ σ2)
    (hoff : ∀ u : Fin T → ℝ, (∀ t, 0 ≤ u t ∧ u t ≤ σ2) →
      C.E ≤ ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) i j ^ 2 ∧
      ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) i j ^ 2 ≤ C.E + Δ ∧
      ∀ v, C.Wc v ⬝ᵥ C.out D M.lnf.ε u (C.s0 hT ids) i = 0) (v : Fin V) (hΔ : 0 ≤ Δ) :
    |(graft M C).logits hT ids i v - M.toModel.logits hT ids i v| ≤ M.eta Δ v := by
  refine (graft_offtrigger_approx_at hD M C hεH hεf hfeat hzs hT ids i Δ σ2 hσ hoff v).trans ?_
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hv := var_nonneg (M.resid hT ids i)
  unfold Host.etaAt Host.eta
  have hw : 0 ≤ Real.sqrt D * Real.sqrt (∑ l, (M.Wh v l * M.lnf.γ l) ^ 2) * Δ := by positivity
  exact div_le_div_of_nonneg_left hw (by positivity) (by nlinarith)

/-- Greedy token preserved: if the host's top logit beats every other by more than the two
perturbation bounds, the grafted argmax is the host's argmax. -/
lemma argmax_stable {n : ℕ} (f g : Fin (n + 1) → ℝ) (η : Fin (n + 1) → ℝ)
    (hfg : ∀ v, |g v - f v| ≤ η v) (r : Fin (n + 1))
    (hm : ∀ v, v ≠ r → f v + η v + η r < f r) : argmaxFirst g = r := by
  apply argmaxFirst_eq_of_lt
  intro v hv
  have h1 := (abs_le.1 (hfg v)).2
  have h2 := (abs_le.1 (hfg r)).1
  have := hm v hv
  linarith

end VWC.Graft
