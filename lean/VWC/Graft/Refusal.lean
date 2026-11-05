/-
A concrete circuit that meets `Circuit.Contract`: a single-token refusal head (k = 8).

Features: the trigger bit b of the token as the quadruple (b, −b, 1−b, −(1−b)), then a flag
quadruple at rest (0, 0, 1, −1). One circuit block: one head with zero queries and keys (uniform
causal attention), value (s₀ − s₁)/2 = b · (LayerNorm scale), written to the flag as (+a, −a) on
coordinates 4, 5; no MLP. Readout: Lg/2 · (y₄ − y₅) into the refusal token r.

`contract`: for any host M (any widths, any activation, any weights), given G ≥ every
`M.gap r t` and the variance bound σ², the circuit meets the contract for the trigger "some token
of the prompt is in Trig" as soon as G · Γ · L < Lg · lo (context length L, Γ and lo explicit).
With `graft_main` this gives `refusal_main`.
-/
import VWC.Graft.Main

namespace VWC.Graft.Refusal

open Matrix Finset VWC.Arch VWC.Graft

variable {V : ℕ}

/-- Trigger bit of a token. -/
noncomputable def bit (Trig : Finset (Fin V)) (v : Fin V) : ℝ := if v ∈ Trig then 1 else 0

/-- Token features: zero-sum, energy 4. -/
noncomputable def feat (Trig : Finset (Fin V)) (v : Fin V) : Fin 8 → ℝ :=
  ![bit Trig v, -bit Trig v, 1 - bit Trig v, -(1 - bit Trig v), 0, 0, 1, -1]

def flagDir : Fin 8 → ℝ := ![0, 0, 0, 0, 1, -1, 0, 0]

noncomputable def readDir : Fin 8 → ℝ := ![1 / 2, -1 / 2, 0, 0, 0, 0, 0, 0]

noncomputable def attn : GAttn 8 1 1 1 := ⟨0, 0, 0, 0, fun _ => readDir, 0, fun j _ => flagDir j, 0⟩

noncomputable def blk : CBlock 8 1 1 1 1 := ⟨1, 0, 1, attn, 1, 0, 1, id, 0, 0, 0, 0⟩

noncomputable def circ (L : ℕ) (Trig : Finset (Fin V)) (r : Fin V) (Lg : ℝ) : Circuit V L 8 1 1 1 1 :=
  ⟨feat Trig, 0, 4, [blk], 1, 0, fun t => if t = r then (Lg / 2) • flagDir else 0⟩

/-- The trigger: some token of the prompt is in `Trig`. -/
def trig (Trig : Finset (Fin V)) (T : ℕ) (ids : Fin (T + 1) → Fin V) : Prop := ∃ t, ids t ∈ Trig

/-! ### Features -/

lemma bit_cases (Trig : Finset (Fin V)) (v : Fin V) : bit Trig v = 0 ∨ bit Trig v = 1 := by
  unfold bit; split_ifs <;> simp

lemma feat_sum (Trig : Finset (Fin V)) (v : Fin V) : ∑ j, feat Trig v j = 0 := by
  simp [feat, Fin.sum_univ_succ]

lemma feat_sq (Trig : Finset (Fin V)) (v : Fin V) : ∑ j, feat Trig v j ^ 2 = 4 := by
  rcases bit_cases Trig v with h | h <;> simp [feat, Fin.sum_univ_succ, h] <;> norm_num

lemma blk_zs : blk.ZeroSum := by
  refine ⟨fun c => ?_, ?_, fun f => ?_, ?_⟩ <;> simp [blk, attn, flagDir, Fin.sum_univ_succ]

/-! ### The attention head -/

lemma hw_uniform {T : ℕ} (t : Fin (T + 1)) :
    headWeight (dqk := 1) (fun _ _ => (0 : ℝ)) (fun _ _ => 0) (Fin.last T) t = 1 / (T + 1) := by
  simp [headWeight, Fin.le_last]

lemma attn_apply {T : ℕ} (x : Fin (T + 1) → Fin 8 → ℝ) :
    attn.apply x (Fin.last T) = fun j => flagDir j * ∑ t, 1 / ((T : ℝ) + 1) * (readDir ⬝ᵥ x t) := by
  funext j
  simp [GAttn.apply, attn, mulVec, hw_uniform, dotProduct]

lemma attn_zero {T : ℕ} (x : Fin T → Fin 8 → ℝ) (hx : ∀ t, readDir ⬝ᵥ x t = 0) (i : Fin T) :
    attn.apply x i = 0 := by
  funext j
  simp [GAttn.apply, attn, mulVec, hx]

lemma readDir_dot (y : Fin 8 → ℝ) : readDir ⬝ᵥ y = y 0 / 2 - y 1 / 2 := by
  simp [readDir, dotProduct, Fin.sum_univ_succ]; ring

lemma flagDir_dot (y : Fin 8 → ℝ) : flagDir ⬝ᵥ y = y 4 - y 5 := by
  simp [flagDir, dotProduct, Fin.sum_univ_succ]; ring

/-! ### The contract -/

variable {D L F H dq dv : ℕ}

/-- Lower bound of the per-token value on a trigger token, for host variance ≤ σ². -/
noncomputable def lo (D : ℕ) (σ2 : ℝ) : ℝ := 1 / Real.sqrt (D * (σ2 + 1) / (D + 8))

/-- Upper bound of the final LayerNorm denominator on trigger. -/
noncomputable def Γ (D : ℕ) (σ2 εf : ℝ) : ℝ := Real.sqrt ((D * (σ2 + εf) + 2 * ((D + 8) / D)) / (D + 8))

theorem contract {n : ℕ} (hD : 0 < D) (M : Host (n + 1) D L F H dq dv) (hεf : 0 < M.lnf.ε)
    (Trig : Finset (Fin (n + 1))) (r : Fin (n + 1)) (σ2 G Lg : ℝ) (hσ : 0 ≤ σ2) (hG0 : 0 ≤ G)
    (hG : ∀ t, t ≠ r → M.gap r t ≤ G) (hL : G * Γ D σ2 M.lnf.ε * L < Lg * lo D σ2) :
    (circ L Trig r Lg).Contract M (trig Trig) r σ2 := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hD8 : (0 : ℝ) < D + 8 := by positivity
  have hs : ∀ (T : ℕ) (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (n + 1)) t,
      (circ L Trig r Lg).s0 hT ids t = feat Trig (ids t) := by
    intro T hT ids t; funext j; simp [Circuit.s0, circ]
  refine ⟨fun v p => ?_, fun b hb => ?_, fun b hb => ?_, fun T hT ids hno => ?_, fun T hT ids hyes u hu => ?_⟩
  · simp only [circ, Pi.zero_apply, add_zero]; exact ⟨feat_sum Trig v, feat_sq Trig v⟩
  · simp only [circ, List.mem_singleton] at hb; subst hb; simp [blk]
  · simp only [circ, List.mem_singleton] at hb; subst hb; exact blk_zs
  · -- off trigger: the value is zero everywhere; the flag is at rest
    have hb0 : ∀ t, bit Trig (ids t) = 0 := fun t => by
      unfold bit; rw [if_neg (fun h => hno ⟨t, h⟩)]
    refine ⟨fun b hb => ?_, fun t κ _ => ?_⟩
    · simp only [circ, List.mem_singleton] at hb; subst hb
      refine ⟨fun κ _ i => attn_zero _ (fun t => ?_) i, fun t κ _ => by simp [blk]⟩
      rw [readDir_dot, hs]; simp [feat, hb0, blk]
    · rw [hs]
      funext t'
      simp only [circ, mulVec, Pi.zero_apply]
      split_ifs
      · rw [smul_dotProduct, flagDir_dot]; simp [feat]
      · simp
  · -- on trigger
    obtain ⟨t0, ht0⟩ := hyes
    set s := (circ L Trig r Lg).s0 hT ids with hsdef
    have hsf : ∀ t, s t = feat Trig (ids t) := hs T hT ids
    -- per-token value
    set val : Fin (T + 1) → ℝ := fun t =>
      readDir ⬝ᵥ cLN D 8 4 1 0 1 (u t) (s t) with hval
    have hg : ∀ t, gden D 8 4 1 (u t) (s t) = D * (u t + 1) / (D + 8) := by
      intro t; unfold gden; rw [hsf, feat_sq]; push_cast; ring
    have hv : ∀ t, val t = bit Trig (ids t) / Real.sqrt (D * (u t + 1) / (D + 8)) := by
      intro t; simp only [hval, readDir_dot, cLN]; rw [hg, hsf]; simp [feat]; ring
    have hpos : ∀ t, 0 < Real.sqrt (D * (u t + 1) / (D + 8)) := fun t =>
      Real.sqrt_pos.2 (by have := (hu t).1; positivity)
    have hv0 : ∀ t, 0 ≤ val t := fun t => by
      rw [hv]; rcases bit_cases Trig (ids t) with h | h <;> rw [h] <;> [simp; exact (one_div_pos.2 (hpos t)).le]
    have hhi : ∀ t, val t ≤ Real.sqrt ((D + 8) / D) := fun t => by
      rw [hv]
      have h1 : Real.sqrt (D / (D + 8)) ≤ Real.sqrt (D * (u t + 1) / (D + 8)) := by
        apply Real.sqrt_le_sqrt; have := (hu t).1
        apply div_le_div_of_nonneg_right _ hD8.le; nlinarith
      have h2 : Real.sqrt ((D + 8) / D) = 1 / Real.sqrt (D / (D + 8)) := by
        rw [one_div, ← Real.sqrt_inv, inv_div]
      rcases bit_cases Trig (ids t) with h | h <;> rw [h]
      · rw [zero_div]; positivity
      · rw [h2]; exact one_div_le_one_div_of_le (Real.sqrt_pos.2 (by positivity)) h1
    have hlo : lo D σ2 ≤ val t0 := by
      rw [hv]; unfold bit; rw [if_pos ht0]; unfold lo
      apply one_div_le_one_div_of_le (hpos t0)
      apply Real.sqrt_le_sqrt; apply div_le_div_of_nonneg_right _ hD8.le
      have := (hu t0).2; nlinarith
    -- the flag value
    set a := ∑ t, 1 / ((T : ℝ) + 1) * val t with ha
    have hT1 : (0 : ℝ) < T + 1 := by positivity
    have ha_lo : lo D σ2 / L ≤ a := by
      have hL1 : ((T : ℝ) + 1) ≤ L := by exact_mod_cast hT
      have hlo0 : 0 ≤ lo D σ2 := by unfold lo; positivity
      calc lo D σ2 / L ≤ lo D σ2 / (T + 1) := div_le_div_of_nonneg_left hlo0 hT1 hL1
        _ ≤ 1 / ((T : ℝ) + 1) * val t0 := by rw [div_eq_mul_one_div, mul_comm]; gcongr
        _ ≤ a := Finset.single_le_sum (f := fun t => 1 / ((T : ℝ) + 1) * val t)
            (fun t _ => mul_nonneg (by positivity) (hv0 t)) (Finset.mem_univ t0)
    have ha0 : 0 ≤ a := Finset.sum_nonneg (fun t _ => mul_nonneg (by positivity) (hv0 t))
    have ha_hi : a ≤ Real.sqrt ((D + 8) / D) := by
      calc a ≤ ∑ _t : Fin (T + 1), 1 / ((T : ℝ) + 1) * Real.sqrt ((D + 8) / D) :=
            Finset.sum_le_sum (fun t _ => mul_le_mul_of_nonneg_left (hhi t) (by positivity))
        _ = Real.sqrt ((D + 8) / D) := by
            simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
            push_cast; field_simp
    -- the final circuit state
    have hrun : runBlocks D 4 (circ L Trig r Lg).blocks u s (Fin.last T) = s (Fin.last T) + a • flagDir := by
      simp only [circ, runBlocks, CBlock.run, blk]
      rw [attn_apply]
      funext j
      simp [hval, ha, mul_comm]
    have hE : ∑ j, (s (Fin.last T) + a • flagDir) j ^ 2 = 4 + 2 * a ^ 2 := by
      rw [hsf]
      rcases bit_cases Trig (ids (Fin.last T)) with h | h <;>
        simp [feat, flagDir, Fin.sum_univ_succ, h] <;> ring
    refine ⟨?_, fun t ht => ?_⟩
    · rw [show (circ L Trig r Lg).E = 4 from rfl, hrun, hE]; nlinarith
    · have hout : ∑ j, ((circ L Trig r Lg).Wc r j - (circ L Trig r Lg).Wc t j) *
          (circ L Trig r Lg).out D M.lnf.ε u s (Fin.last T) j =
          Lg * a / Real.sqrt ((D * (u (Fin.last T) + M.lnf.ε) + 2 * a ^ 2) / (D + 8)) := by
        have hgf : gden D 8 4 M.lnf.ε (u (Fin.last T)) (s (Fin.last T) + a • flagDir) =
            (D * (u (Fin.last T) + M.lnf.ε) + 2 * a ^ 2) / (D + 8) := by
          unfold gden; rw [hE]; push_cast; ring
        have hWr : (circ L Trig r Lg).Wc r = (Lg / 2) • flagDir := by simp [circ]
        have hWt : (circ L Trig r Lg).Wc t = 0 := by simp [circ, ht]
        have hout0 : (circ L Trig r Lg).out D M.lnf.ε u s (Fin.last T) =
            cLN D 8 4 1 0 M.lnf.ε (u (Fin.last T)) (s (Fin.last T) + a • flagDir) := by
          show cLN D 8 4 1 0 M.lnf.ε (u (Fin.last T))
            (runBlocks D 4 (circ L Trig r Lg).blocks u s (Fin.last T)) = _
          rw [hrun]
        rw [hWr, hWt, hout0]
        simp only [cLN, hgf]
        rw [hsf]
        simp [feat, flagDir, Fin.sum_univ_succ]
        ring
      rw [hout]
      have hgap := hG t ht
      have hΓ : Real.sqrt ((D * (u (Fin.last T) + M.lnf.ε) + 2 * a ^ 2) / (D + 8)) ≤ Γ D σ2 M.lnf.ε := by
        unfold Γ; apply Real.sqrt_le_sqrt; apply div_le_div_of_nonneg_right _ hD8.le
        have h1 := (hu (Fin.last T)).2
        have h2 : a ^ 2 ≤ (D + 8) / D := by
          calc a ^ 2 ≤ Real.sqrt ((D + 8) / D) ^ 2 := pow_le_pow_left₀ ha0 ha_hi 2
            _ = (D + 8) / D := Real.sq_sqrt (by positivity)
        nlinarith
      have hgf0 : 0 < Real.sqrt ((D * (u (Fin.last T) + M.lnf.ε) + 2 * a ^ 2) / (D + 8)) :=
        Real.sqrt_pos.2 (by have := (hu (Fin.last T)).1; positivity)
      have hΓ0 : 0 < Γ D σ2 M.lnf.ε := lt_of_lt_of_le hgf0 hΓ
      have hLpos : (0 : ℝ) < L := by have : (1 : ℝ) ≤ L := by exact_mod_cast (by omega : 1 ≤ L)
                                     linarith
      have hLg : 0 < Lg := by
        have hlo0 : 0 < lo D σ2 := by unfold lo; positivity
        by_contra h
        have h : Lg ≤ 0 := not_lt.mp h
        have : Lg * lo D σ2 ≤ 0 := mul_nonpos_of_nonpos_of_nonneg h hlo0.le
        have : 0 ≤ G * Γ D σ2 M.lnf.ε * L := by positivity
        linarith
      -- G < Lg * a / Γ ≤ Lg * a / sqrt(gf)
      rw [lt_div_iff₀ hgf0]
      calc M.gap r t * Real.sqrt ((D * (u (Fin.last T) + M.lnf.ε) + 2 * a ^ 2) / (D + 8))
          ≤ G * Γ D σ2 M.lnf.ε := by
            rcases le_or_gt 0 (M.gap r t) with hg0 | hg0
            · exact mul_le_mul hgap hΓ hgf0.le hG0
            · nlinarith
        _ < Lg * (lo D σ2 / L) := by rw [mul_div_assoc']; rw [lt_div_iff₀ hLpos]; linarith
        _ ≤ Lg * a := mul_le_mul_of_nonneg_left ha_lo hLg.le

/-- **Refusal head grafted into any host.** Off trigger, the grafted model's logits and greedy
token equal the host's; if a trigger token occurs and the host's final residual variance is at most
σ², the greedy next token is r. -/
theorem refusal_main {n : ℕ} (hD : 0 < D) (M : Host (n + 1) D L F H dq dv)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (Trig : Finset (Fin (n + 1))) (r : Fin (n + 1)) (σ2 G Lg : ℝ) (hσ : 0 ≤ σ2) (hG0 : 0 ≤ G)
    (hG : ∀ t, t ≠ r → M.gap r t ≤ G) (hL : G * Γ D σ2 M.lnf.ε * L < Lg * lo D σ2)
    {T : ℕ} (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (n + 1)) :
    ((∀ t, ids t ∉ Trig) → ∀ i, (graft M (circ L Trig r Lg)).logits hT ids i = M.toModel.logits hT ids i) ∧
    ((∀ t, ids t ∉ Trig) → (graft M (circ L Trig r Lg)).next hT ids = M.toModel.next hT ids) ∧
    ((∃ t, ids t ∈ Trig) → (∀ t, var (M.resid hT ids t) ≤ σ2) →
      (graft M (circ L Trig r Lg)).next hT ids = r) := by
  have h := graft_main hD M (circ L Trig r Lg) hεH hεf (trig Trig) r σ2
    (contract hD M hεf Trig r σ2 G Lg hσ hG0 hG hL) hT ids
  refine ⟨fun hno => h.1 (fun ⟨t, ht⟩ => hno t ht), fun hno => h.2.1 (fun ⟨t, ht⟩ => hno t ht),
    fun hyes => h.2.2 hyes⟩

end VWC.Graft.Refusal
