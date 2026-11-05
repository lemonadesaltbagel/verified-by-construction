/-
Ordinary-semantics domain of the proved circuits.

Lean's `Real.sqrt` and division are total, so the grafting theorems hold in a total semantics. A run
coincides with ordinary LayerNorm execution when every LayerNorm radicand it meets is positive.
`RadPos` records that for the circuit blocks (both LayerNorms of every block, at every position);
`pattern_radicands_pos` proves it, together with positivity of the final LayerNorm radicand, for
the pattern detector (and hence the key gate) under the hypotheses of `pattern_main`.
-/
import VWC.Graft.PatternMain
import VWC.Graft.Refusal

namespace VWC.Graft

open Matrix Finset

variable {D k : ℕ}

/-- Every circuit-block LayerNorm radicand met while running `bs` from `s` is positive. -/
def RadPos {T Fc Hc dq dv : ℕ} (D : ℕ) (E : ℝ) :
    List (CBlock k Fc Hc dq dv) → (Fin T → ℝ) → (Fin T → Fin k → ℝ) → Prop
  | [], _, _ => True
  | b :: bs, u, s =>
      (∀ t, 0 < gden D k E b.ε1 (u t) (s t)) ∧
      (∀ i, 0 < gden D k E b.ε2 (u i)
        (s i + b.attn.apply (fun t => cLN D k E b.γ1 b.β1 b.ε1 (u t) (s t)) i)) ∧
      RadPos D E bs u (b.run D E u s)

/-- A radicand is positive whenever the circuit state's energy is at least `E`. -/
lemma gden_pos_of_energy (hD : 0 < D) {E ε u : ℝ} (hε : 0 < ε) (hu : 0 ≤ u) (s : Fin k → ℝ)
    (hs : E ≤ ∑ j, s j ^ 2) : 0 < gden D k E ε u s := by
  unfold gden
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  apply div_pos _ (by positivity)
  nlinarith [mul_pos hDr (show 0 < u + ε by linarith)]

end VWC.Graft

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ) {L : ℕ} {σ2 : ℝ}

lemma pos_form (hD : 0 < D) {w ε : ℝ} (hw : 0 ≤ w) (hε : 0 < ε) (a : Fin M → ℝ) (f : ℝ) (K : ℕ) :
    0 < (D * (w + ε) + 2 * ∑ d, a d ^ 2 + 2 * f ^ 2) / (D + K) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have ha : 0 ≤ ∑ d, a d ^ 2 := Finset.sum_nonneg (fun _ _ => sq_nonneg _)
  apply div_pos _ (by positivity)
  nlinarith [mul_pos hDr (show 0 < w + ε by linarith), sq_nonneg f]

/-- Radicand of a state in slot form with Boolean bits and match. -/
lemma gden_stv_pos (hD : 0 < D) {ε : ℝ} (hε : 0 < ε) (t : Fin (T + 1)) (hu : 0 ≤ u t)
    (A B : Fin (T + 1) → Fin M → ℝ) (mt f : Fin (T + 1) → ℝ)
    (hB : ∀ d, B t d = 0 ∨ B t d = 1) (hmt : mt t = 0 ∨ mt t = 1) :
    0 < gden D (slots M n) (Ef p.δ M n) ε (u t) (stv n p P ids A B mt f t) := by
  unfold stv
  rw [gden_St n p D P ids u ε t (A t) (B t) (mt t) (f t) hB hmt]
  exact pos_form D hD hu hε (A t) (f t) (slots M n)

/-- State after block 1's attention. -/
lemma attn1_state (i : Fin (T + 1)) :
    stv n p P ids 0 0 0 0 i + (blk1 (M := M) (n := n) p).attn.apply
      (fun t => cLN D (slots M n) (Ef p.δ M n) (blk1 (M := M) (n := n) p).γ1
        (blk1 (M := M) (n := n) p).β1 (blk1 (M := M) (n := n) p).ε1 (u t)
        (stv n p P ids 0 0 0 0 t)) i = stv n p P ids (acp n p D P ids u) 0 0 0 i := by
  apply ext_slot; intro a
  simp only [blk1, attn1, GAttn.apply, Pi.add_apply, add_zero, out_mulVec]
  rcases a with ⟨i', c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
    simp [stv, St, oslot]
  simp only [acp, x1, mulVec, pmsign, pm]
  fin_cases c <;> simp <;> rfl

/-- Block 2 has no attention. -/
lemma attn2_zero [NeZero M] (x : Fin (T + 1) → Fin (slots M n) → ℝ) (i : Fin (T + 1)) :
    (blk2 (M := M) (n := n) p).attn.apply x i = 0 := by
  funext j; simp [blk2, GAttn.apply]

/-- State after block 3's attention (block 3's MLP is zero, so this is block 3's output). -/
lemma attn3_state [NeZero M] (i : Fin (T + 1)) :
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) 0 i +
      (blk3 (M := M) (n := n) p).attn.apply
        (fun t => cLN D (slots M n) (Ef p.δ M n) (blk3 (M := M) (n := n) p).γ1
          (blk3 (M := M) (n := n) p).β1 (blk3 (M := M) (n := n) p).ε1 (u t)
          (stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) 0 t)) i =
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) (fl n p D P ids u) i := by
  have h := congrFun (run3 n p D P ids u) i
  have hm : ∀ y : Fin (slots M n) → ℝ, (blk3 (M := M) (n := n) p).W2 *ᵥ
      (fun f => (blk3 (M := M) (n := n) p).act (((blk3 (M := M) (n := n) p).W1 *ᵥ y) f +
        (blk3 (M := M) (n := n) p).b1 f)) + (blk3 (M := M) (n := n) p).b2 = 0 := by
    intro y; funext j; simp [blk3, mulVec]
  simp only [CBlock.run] at h
  rw [hm, add_zero] at h
  exact h

/-- **Ordinary-semantics domain of the pattern detector.** Under the hypotheses of `pattern_main`
(parameters `Good`, prompt within the context, host variances in `[0, σ²]`) and a positive final
LayerNorm ε, every LayerNorm radicand of the circuit run is positive: both LayerNorms of each of the
three circuit blocks at every position (`RadPos`), and the final LayerNorm at every position. Hence
the run, and with it `pattern_main` and `key_main`, coincides with ordinary LayerNorm execution. -/
theorem pattern_radicands_pos [NeZero M] (G : Good M n D L p σ2) (r : Fin V) (hT : T + 1 ≤ L)
    (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2) {εf : ℝ} (hεf : 0 < εf) :
    RadPos D (circ (n := n) p L P r).E (circ (n := n) p L P r).blocks u
        ((circ (n := n) p L P r).s0 hT ids) ∧
      ∀ i, 0 < gden D (slots M n) (circ (n := n) p L P r).E εf (u i)
        (runBlocks D (circ (n := n) p L P r).E (circ (n := n) p L P r).blocks u
          ((circ (n := n) p L P r).s0 hT ids) i) := by
  have hD := G.hD
  have hb := bit01 n p D P ids u G hT hu
  have ha := andv01 n p D P ids u G hT hu
  have z0 : ∀ t (d : Fin M), (0 : Fin (T + 1) → Fin M → ℝ) t d = 0 ∨
      (0 : Fin (T + 1) → Fin M → ℝ) t d = 1 := fun _ _ => Or.inl rfl
  have m0 : ∀ t, (0 : Fin (T + 1) → ℝ) t = 0 ∨ (0 : Fin (T + 1) → ℝ) t = 1 := fun _ => Or.inl rfl
  refine ⟨?_, fun i => ?_⟩
  · rw [circ_s0 n p P ids r hT]
    simp only [circ, RadPos]
    refine ⟨fun t => ?_, fun i => ?_, ?_⟩
    · exact gden_stv_pos n p D P ids u hD one_pos t (hu t).1 0 0 0 0 (z0 t) (m0 t)
    · rw [attn1_state n p D P ids u i]
      exact gden_stv_pos n p D P ids u hD one_pos i (hu i).1 _ 0 0 0 (z0 i) (m0 i)
    rw [run1]
    refine ⟨fun t => ?_, fun i => ?_, ?_⟩
    · exact gden_stv_pos n p D P ids u hD one_pos t (hu t).1 _ _ 0 0 (hb t) (m0 t)
    · rw [attn2_zero, add_zero]
      exact gden_stv_pos n p D P ids u hD one_pos i (hu i).1 _ _ 0 0 (hb i) (m0 i)
    rw [run2]
    refine ⟨fun t => ?_, fun i => ?_, trivial⟩
    · exact gden_stv_pos n p D P ids u hD one_pos t (hu t).1 _ _ _ 0 (hb t) (ha t)
    · rw [attn3_state n p D P ids u i]
      exact gden_stv_pos n p D P ids u hD one_pos i (hu i).1 _ _ _ _ (hb i) (ha i)
  · rw [run_all n p D P ids u r hT]
    exact gden_stv_pos n p D P ids u hD hεf i (hu i).1 _ _ _ _ (hb i) (ha i)

end VWC.Graft.Pattern

namespace VWC.Graft.Refusal

open Matrix Finset VWC.Arch VWC.Graft

/-- The refusal head's attention writes along `flagDir` only. -/
lemma attn_dir {T : ℕ} (x : Fin T → Fin 8 → ℝ) (i : Fin T) :
    attn.apply x i = (attn.apply x i 4) • flagDir := by
  funext j
  simp only [GAttn.apply, attn, mulVec, dotProduct, Pi.smul_apply, smul_eq_mul, Pi.add_apply,
    add_zero]
  fin_cases j <;> simp [flagDir]

lemma feat_dir_sq {V : ℕ} (Trig : Finset (Fin V)) (v : Fin V) (c : ℝ) :
    ∑ j, (feat Trig v + c • flagDir) j ^ 2 = 4 + 2 * c ^ 2 := by
  rcases bit_cases Trig v with h | h <;> simp [feat, flagDir, Fin.sum_univ_succ, h] <;> ring

/-- **Ordinary-semantics domain of the single-token refusal head.** For any host variance profile
`u ≥ 0` and positive final LayerNorm ε, every LayerNorm radicand of the circuit run is positive:
both LayerNorms of its block at every position, and the final LayerNorm at every position. -/
theorem refusal_radicands_pos {V L T : ℕ} {D : ℕ} (hD : 0 < D) (Trig : Finset (Fin V)) (r : Fin V)
    (Lg : ℝ) (hT : T ≤ L) (ids : Fin T → Fin V) (u : Fin T → ℝ) (hu : ∀ t, 0 ≤ u t) {εf : ℝ}
    (hεf : 0 < εf) :
    RadPos D (circ L Trig r Lg).E (circ L Trig r Lg).blocks u ((circ L Trig r Lg).s0 hT ids) ∧
      ∀ i, 0 < gden D 8 (circ L Trig r Lg).E εf (u i)
        (runBlocks D (circ L Trig r Lg).E (circ L Trig r Lg).blocks u ((circ L Trig r Lg).s0 hT ids) i) := by
  have hs : ∀ t, (circ L Trig r Lg).s0 hT ids t = feat Trig (ids t) := by
    intro t; funext j; simp [Circuit.s0, circ]
  have hE : (circ L Trig r Lg).E = 4 := rfl
  have hstate : ∀ i, (circ L Trig r Lg).s0 hT ids i + blk.attn.apply (fun t => cLN D 8 4 blk.γ1 blk.β1
      blk.ε1 (u t) ((circ L Trig r Lg).s0 hT ids t)) i =
      feat Trig (ids i) + (blk.attn.apply (fun t => cLN D 8 4 blk.γ1 blk.β1 blk.ε1 (u t)
        ((circ L Trig r Lg).s0 hT ids t)) i 4) • flagDir := by
    intro i; rw [hs i]; congr 1; exact attn_dir _ i
  have hB : (circ L Trig r Lg).blocks = [blk] := rfl
  have hrun : ∀ i, runBlocks D 4 (circ L Trig r Lg).blocks u ((circ L Trig r Lg).s0 hT ids) i =
      (circ L Trig r Lg).s0 hT ids i + blk.attn.apply (fun t => cLN D 8 4 blk.γ1 blk.β1 blk.ε1 (u t)
        ((circ L Trig r Lg).s0 hT ids t)) i := by
    intro i
    rw [hB]
    simp only [runBlocks, CBlock.run]
    have hm : ∀ y : Fin 8 → ℝ, blk.W2 *ᵥ (fun f => blk.act ((blk.W1 *ᵥ y) f + blk.b1 f)) + blk.b2 = 0 := by
      intro y; funext j; simp [blk, mulVec]
    rw [hm, add_zero]
  refine ⟨?_, fun i => ?_⟩
  · rw [hB, hE]
    simp only [RadPos]
    refine ⟨fun t => ?_, fun i => ?_, trivial⟩
    · refine gden_pos_of_energy hD (by simp [blk]) (hu t) _ ?_
      rw [hs t, feat_sq]
    · refine gden_pos_of_energy hD (by simp [blk]) (hu i) _ ?_
      rw [hstate i, feat_dir_sq]
      nlinarith [sq_nonneg (blk.attn.apply (fun t => cLN D 8 4 blk.γ1 blk.β1 blk.ε1 (u t)
        ((circ L Trig r Lg).s0 hT ids t)) i 4)]
  · rw [hE, hrun i, hstate i]
    refine gden_pos_of_energy hD hεf (hu i) _ ?_
    rw [feat_dir_sq]
    nlinarith [sq_nonneg (blk.attn.apply (fun t => cLN D 8 4 blk.γ1 blk.β1 blk.ε1 (u t)
      ((circ L Trig r Lg).s0 hT ids t)) i 4)]

end VWC.Graft.Refusal
