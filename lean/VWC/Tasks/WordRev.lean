/-
word-reversal-3x3 (snapshot PosRoute_SCALE_50, AR_SCALE unset ⇒ SCALE = 50; the run's 24 PosRoute_*
files, including PosRoute_HARD_S200, are byte-identical and only differ by the AR_SCALE set at run time):
real-valued correctness of greedy generation on every input (26⁹ prompts), by mechanism.

* Every position embeds exactly one token one-hot and one position one-hot, so LN1 is the
  same affine map `x ↦ (x - μ)/σ` everywhere (μ = 2/51, σ² = 98/2601 + ε).
* Hence the attention score of key j at query p is `c_p + κ([j = src p] - μ·m_j)` with
  κ = 50/(σ²√51) ≈ 186 and m_j = #{p' | src p' = j} ≤ 13: the target beats every other key by
  ≥ κ·25/51 > 90, so the head copies the token at `src p` up to a leak λ < 1e-30.
* The MLP is identically zero; the final LayerNorm (γ = 1, β = 0) and the head (50·identity on
  token dims) are strictly monotone in the token dims, so the argmax is the copied token.
-/
import VWC.Tasks.WordRevStruct

namespace VWC.Arch.WordRev

open Matrix Finset

/-! ### Specification (evolve/src/task.py, WordReversalTask). Vocab "a…z_=". -/

/-- Letter tokens are 0..25, `_` = 26, `=` = 27. -/
abbrev Word := Fin 3 → Fin 26

def letter (c : Fin 26) : Fin 28 := c.castLE (by norm_num)

def prompt (w1 w2 w3 : Word) : List (Fin 28) :=
  List.ofFn (letter ∘ w1) ++ [26] ++ List.ofFn (letter ∘ w2) ++ [26] ++ List.ofFn (letter ∘ w3) ++ [27]

def answer (w1 w2 w3 : Word) : List (Fin 28) :=
  List.ofFn (letter ∘ w3) ++ [26] ++ List.ofFn (letter ∘ w2) ++ [26] ++ List.ofFn (letter ∘ w1)

/-! ### LayerNorm on embeddings: one affine map for every position -/

noncomputable def μ : ℝ := 2 / 51
noncomputable def ε : ℝ := 2748779 / 274877906944
noncomputable def σ : ℝ := Real.sqrt (98 / 2601 + ε)

lemma σ_pos : 0 < σ := Real.sqrt_pos.2 (by norm_num [ε])

lemma σ_sq : σ ^ 2 = 98 / 2601 + ε := Real.sq_sqrt (by norm_num [ε])

lemma σ_le : σ ≤ 1 / 5 := by
  rw [σ, Real.sqrt_le_left (by norm_num)]; norm_num [ε]

/-- Splitting a sum over the 51 residual dims into token dims and position dims. -/
lemma sum51 (f : Fin 51 → ℝ) :
    ∑ r, f r = ∑ a : Fin 28, f (a.castLE (by norm_num)) + ∑ s : Fin 23, f ⟨28 + s.val, by omega⟩ := by
  exact Fin.sum_univ_add (a := 28) (b := 23) f

/-- Sum and centred square-sum of a vector with exactly two unit entries. -/
lemma two_hot (a b : Fin 51) (hab : a ≠ b) (c : ℝ) :
    ∑ i, (Pi.single a (1 : ℝ) + Pi.single b 1 : Fin 51 → ℝ) i = 2 ∧
    ∑ i, ((Pi.single a (1 : ℝ) + Pi.single b 1 : Fin 51 → ℝ) i - c) ^ 2 = 2 - 4 * c + 51 * c ^ 2 := by
  have h1 : ∑ i, (Pi.single a (1 : ℝ) + Pi.single b 1 : Fin 51 → ℝ) i = 2 := by
    simp [Finset.sum_add_distrib]; norm_num
  refine ⟨h1, ?_⟩
  have hsq : ∀ i, ((Pi.single a (1 : ℝ) + Pi.single b 1 : Fin 51 → ℝ) i - c) ^ 2 =
      (Pi.single a (1 : ℝ) + Pi.single b 1 : Fin 51 → ℝ) i * (1 - 2 * c) + c ^ 2 := by
    intro i
    by_cases ha : i = a <;> by_cases hb : i = b <;> simp_all [Pi.single_apply] <;> ring
  simp only [hsq, Finset.sum_add_distrib, ← Finset.sum_mul, h1, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  push_cast; ring

lemma fpe0 (c : Fin 51) : (finProdFinEquiv ((0 : Fin 1), c)).val = c.val := by
  simp [finProdFinEquiv]

/-- W_k reads position dims only (identity there). -/
lemma Wk_apply (v : Fin 51 → ℝ) (c : Fin 51) :
    (A0.Wk *ᵥ v) (finProdFinEquiv ((0 : Fin 1), c)) = if 28 ≤ c.val then v c else 0 := by
  rw [Wk_eq]
  simp only [mulVec, dotProduct, Matrix.of_apply]
  have hc : (finProdFinEquiv ((0 : Fin 1), c)).val = c.val := fpe0 c
  rw [sum_eq_single c]
  · simp only [hc]
    split_ifs with h1 h2 h2 <;>
      first | (exfalso; exact h2 h1.1) | (exfalso; exact h1 ⟨h2, Fin.ext hc⟩) | ring
  · intro b _ hb
    rw [if_neg, zero_mul]
    rintro ⟨_, h⟩; exact hb (Fin.ext (by have := congrArg Fin.val h; simp only [hc] at this; omega))
  · simp

/-- W_v reads token dims only (identity there). -/
lemma Wv_apply (v : Fin 51 → ℝ) (c : Fin 51) :
    (A0.Wv *ᵥ v) (finProdFinEquiv ((0 : Fin 1), c)) = if c.val < 28 then v c else 0 := by
  rw [Wv_eq]
  simp only [mulVec, dotProduct, Matrix.of_apply]
  have hc : (finProdFinEquiv ((0 : Fin 1), c)).val = c.val := fpe0 c
  rw [sum_eq_single c]
  · simp only [hc]
    split_ifs with h1 h2 h2 <;>
      first | (exfalso; exact h2 h1.1) | (exfalso; exact h1 ⟨h2, Fin.ext hc⟩) | ring
  · intro b _ hb
    rw [if_neg, zero_mul]
    rintro ⟨_, h⟩; exact hb (Fin.ext (by have := congrArg Fin.val h; simp only [hc] at this; omega))
  · simp

/-- W_q on position row `28 + s`: 50 · Σ over the query positions routed to s. -/
lemma Wq_apply_pos (v : Fin 51 → ℝ) (s : Fin 23) :
    (A0.Wq *ᵥ v) (finProdFinEquiv ((0 : Fin 1), (⟨28 + s.val, by omega⟩ : Fin 51))) =
      50 * ∑ p' : Fin 23, (if src p'.val = s.val then v ⟨28 + p'.val, by omega⟩ else 0) := by
  rw [Wq_eq]
  simp only [mulVec, dotProduct, Matrix.of_apply, fpe0]
  rw [sum51]
  rw [sum_eq_zero (fun a _ => by rw [if_neg (by simp only [Fin.coe_castLE]; omega), zero_mul]),
    zero_add, mul_sum]
  refine sum_congr rfl fun p' _ => ?_
  have e : 28 + p'.val - 28 = p'.val := by omega
  simp only [e, le_add_iff_nonneg_right, zero_le, true_and, add_right_inj]
  split_ifs with h1 h2 h2 <;> first | (exfalso; exact h2 h1.symm) | (exfalso; exact h1 h2.symm) | ring

/-- The routed key of the last query position (positions ≥ 11 read keys ≤ 10). -/
def jStar (n : ℕ) (hn : 11 ≤ n) : Fin (n + 1) := ⟨src n, by unfold src; split_ifs <;> omega⟩

section Step

variable {n : ℕ} (hT : n + 1 ≤ 23) (ids : Fin (n + 1) → Fin 28)

/-- Embedded prompt. -/
noncomputable def X (t : Fin (n + 1)) : Fin 51 → ℝ :=
  wrModel.tokEmb (ids t) + wrModel.posEmb (Fin.castLE hT t)

lemma X_eq (t : Fin (n + 1)) : X hT ids t =
    Pi.single ((ids t).castLE (by norm_num)) 1 + Pi.single ⟨28 + t.val, by omega⟩ 1 := by
  simp [X, tokEmb_eq, posEmb_eq]

lemma X_tok (t : Fin (n + 1)) (a : Fin 28) :
    X hT ids t (a.castLE (by norm_num)) = if ids t = a then 1 else 0 := by
  rw [X_eq]
  simp only [Pi.add_apply, Pi.single_apply, Fin.ext_iff, Fin.coe_castLE]
  have : ¬ a.val = 28 + t.val := by omega
  rw [if_neg this, add_zero]
  split_ifs <;> first | rfl | omega

lemma X_pos (t : Fin (n + 1)) (s : Fin 23) :
    X hT ids t ⟨28 + s.val, by omega⟩ = if s.val = t.val then 1 else 0 := by
  rw [X_eq]
  simp only [Pi.add_apply, Pi.single_apply, Fin.ext_iff, Fin.coe_castLE]
  have : ¬ 28 + s.val = (ids t).val := by have := (ids t).isLt; omega
  rw [if_neg this, zero_add]
  by_cases h : s.val = t.val
  · rw [if_pos (by omega), if_pos h]
  · rw [if_neg (by omega), if_neg h]

lemma LN1_X (t : Fin (n + 1)) :
    wrModel_block0.ln1.apply (X hT ids t) = fun i => (X hT ids t i - μ) / σ := by
  obtain ⟨h1, h2⟩ := two_hot ((ids t).castLE (by norm_num)) ⟨28 + t.val, by omega⟩
    (by intro h; have := congrArg Fin.val h; simp at this; omega) μ
  rw [← X_eq hT ids t] at h1 h2
  simp only [wrModel_block0, Norm.apply]
  funext i
  rw [(gamma_one i).1]
  simp only [wrModel_block0_ln1_beta, Pi.zero_apply, add_zero, mul_one, Nat.cast_ofNat]
  rw [h1, show (2 : ℝ) / 51 = μ from rfl, h2]
  congr 2
  norm_num [μ, ε]

/-- LN1 output. -/
noncomputable def U (t : Fin (n + 1)) : Fin 51 → ℝ := fun i => (X hT ids t i - μ) / σ

/-- Multiplicity of key s in the routing table, over the 23 query positions. -/
def M (s : ℕ) : ℕ := (univ.filter fun p' : Fin 23 => src p'.val = s).card

lemma M_le (s : ℕ) : M s ≤ 13 := by
  rcases Nat.lt_or_ge s 11 with h | h
  · interval_cases s <;> decide
  · have : M s = 0 := by
      unfold M; rw [card_eq_zero, filter_eq_empty_iff]
      intro p _; unfold src; split_ifs <;> omega
    omega

lemma src_lt (p : ℕ) : src p ≤ 10 := by unfold src; split_ifs <;> omega

/-- Score coefficient of key s for query position p. -/
noncomputable def Acoef (p : ℕ) (s : ℕ) : ℝ := (if src p = s then 1 else 0) - μ * M s

lemma U_pos (t : Fin (n + 1)) (s : Fin 23) :
    U hT ids t ⟨28 + s.val, by omega⟩ = ((if s.val = t.val then 1 else 0) - μ) / σ := by
  simp only [U, X_pos]

lemma U_tok (t : Fin (n + 1)) (a : Fin 28) :
    U hT ids t (a.castLE (by norm_num)) = ((if ids t = a then 1 else 0) - μ) / σ := by
  simp only [U, X_tok]

/-- Query of the head at position `p`, on position dim `28 + s`. -/
lemma q_pos (p : Fin (n + 1)) (s : Fin 23) :
    (A0.Wq *ᵥ U hT ids p) (finProdFinEquiv ((0 : Fin 1), (⟨28 + s.val, by omega⟩ : Fin 51))) =
      50 / σ * Acoef p.val s.val := by
  rw [Wq_apply_pos]
  simp only [U_pos]
  have hp : p.val < 23 := by omega
  have e1 : ∑ p' : Fin 23, (if src p'.val = s.val then ((if p'.val = p.val then (1 : ℝ) else 0) - μ) / σ
      else 0) = ((if src p.val = s.val then 1 else 0) - μ * M s.val) / σ := by
    have e2 : ∑ p' : Fin 23, (if src p'.val = s.val then (1 : ℝ) else 0) = M s.val := by
      rw [sum_boole]; rfl
    have e3 : ∑ p' : Fin 23, (if src p'.val = s.val then (if p'.val = p.val then (1 : ℝ) else 0) else 0)
        = if src p.val = s.val then 1 else 0 := by
      rw [sum_eq_single (⟨p.val, hp⟩ : Fin 23)]
      · simp
      · intro b _ hb; split_ifs with h1 h2 <;> first | rfl | exact absurd (Fin.ext h2) hb
      · simp
    rw [← e2, ← e3, mul_sum, ← sum_sub_distrib, sum_div]
    refine sum_congr rfl fun p' _ => ?_
    split_ifs <;> ring
  rw [e1, Acoef]; field_simp

/-- Raw attention dot product of query p and key j. -/
lemma dot_eq (p j : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ U hT ids p) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ U hT ids j) (finProdFinEquiv ((0 : Fin 1), c)))) =
      50 / σ ^ 2 * ∑ s : Fin 23, Acoef p.val s.val * ((if s.val = j.val then 1 else 0) - μ) := by
  simp only [dotProduct]
  rw [sum51]
  rw [sum_eq_zero (fun a _ => by rw [Wk_apply, if_neg (by simp only [Fin.coe_castLE]; omega),
    mul_zero]), zero_add, mul_sum]
  refine sum_congr rfl fun s _ => ?_
  rw [Wk_apply, if_pos (by simp), q_pos, U_pos]
  have := σ_pos
  field_simp <;> ring

/-- Score differences: key j versus the routed key, scaled by κ = 50/(σ²√51). -/
lemma score_diff (p j j' : Fin (n + 1)) :
    ((fun c => (A0.Wq *ᵥ U hT ids p) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ U hT ids j) (finProdFinEquiv ((0 : Fin 1), c)))) / Real.sqrt (51 : ℕ) -
    ((fun c => (A0.Wq *ᵥ U hT ids p) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
      (fun c => (A0.Wk *ᵥ U hT ids j') (finProdFinEquiv ((0 : Fin 1), c)))) / Real.sqrt (51 : ℕ) =
    50 / (σ ^ 2 * Real.sqrt 51) * (Acoef p.val j.val - Acoef p.val j'.val) := by
  rw [dot_eq, dot_eq]
  have hj : j.val < 23 := by omega
  have hj' : j'.val < 23 := by omega
  have key : ∀ k : Fin (n + 1), k.val < 23 →
      ∑ s : Fin 23, Acoef p.val s.val * ((if s.val = k.val then 1 else 0) - μ) =
        Acoef p.val k.val - μ * ∑ s : Fin 23, Acoef p.val s.val := by
    intro k hk
    simp only [mul_sub, mul_ite, mul_one, mul_zero, sum_sub_distrib, ← sum_mul]
    rw [sum_eq_single (⟨k.val, hk⟩ : Fin 23)]
    · simp; ring
    · intro b _ hb; rw [if_neg (fun h => hb (Fin.ext h))]
    · simp
  rw [key j hj, key j' hj']
  push_cast
  field_simp
  ring

lemma exp_neg90_le : Real.exp (-90) ≤ 1 / 10 ^ 35 := by
  have he := Real.exp_one_gt_d9
  have h : (2.7182818283 : ℝ) ^ 90 ≤ Real.exp 90 := by
    rw [show (90 : ℝ) = (90 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) he.le 90
  rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 90) (by norm_num)]
  calc (1 / 10 ^ 35 : ℝ)⁻¹ ≤ (2.7182818283 : ℝ) ^ 90 := by norm_num
    _ ≤ _ := h

/-- κ·25/51 ≥ 90. -/
lemma kappa_big : 90 ≤ 50 / (σ ^ 2 * Real.sqrt 51) * (25 / 51) := by
  have hs : Real.sqrt 51 ≤ 715 / 100 := by
    rw [Real.sqrt_le_left (by norm_num)]; norm_num
  have hs0 : 0 < Real.sqrt 51 := Real.sqrt_pos.2 (by norm_num)
  have h2 : σ ^ 2 ≤ 377 / 10000 := by rw [σ_sq]; norm_num [ε]
  have hpos : 0 < σ ^ 2 * Real.sqrt 51 := by have := σ_pos; positivity
  have : σ ^ 2 * Real.sqrt 51 ≤ 377 / 10000 * (715 / 100) :=
    mul_le_mul h2 hs hs0.le (by norm_num)
  rw [div_mul_eq_mul_div, le_div_iff₀ hpos]
  nlinarith

variable (hn : 11 ≤ n) (hn' : n ≤ 21)

/-- The head's weights at the last position. -/
noncomputable def W (j : Fin (n + 1)) : ℝ :=
  headWeight (fun t c => (A0.Wq *ᵥ U hT ids t) (finProdFinEquiv ((0 : Fin 1), c)))
    (fun t c => (A0.Wk *ᵥ U hT ids t) (finProdFinEquiv ((0 : Fin 1), c))) (Fin.last n) j

include hn hn' in
/-- Leak: the routed key receives all but < 1e-30 of the attention. -/
lemma leak : 1 - W hT ids (jStar n hn) ≤ 1 / 10 ^ 30 := by
  unfold W
  simp only [headWeight_last]
  refine (one_sub_softmax_le _ _).trans ?_
  have hb : ∀ j : Fin (n + 1), (if j = jStar n hn then (0 : ℝ) else Real.exp (
      ((fun c => (A0.Wq *ᵥ U hT ids (Fin.last n)) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
        (fun c => (A0.Wk *ᵥ U hT ids j) (finProdFinEquiv ((0 : Fin 1), c)))) / Real.sqrt (51 : ℕ) -
      ((fun c => (A0.Wq *ᵥ U hT ids (Fin.last n)) (finProdFinEquiv ((0 : Fin 1), c))) ⬝ᵥ
        (fun c => (A0.Wk *ᵥ U hT ids (jStar n hn)) (finProdFinEquiv ((0 : Fin 1), c)))) /
          Real.sqrt (51 : ℕ))) ≤ 1 / 10 ^ 35 := by
    intro j
    split_ifs with hj
    · positivity
    · rw [score_diff]
      refine le_trans (Real.exp_le_exp.2 ?_) exp_neg90_le
      have hk := kappa_big
      have hne : j.val ≠ src n := fun h => hj (Fin.ext h)
      have hA : Acoef (Fin.last n).val j.val - Acoef (Fin.last n).val (jStar n hn).val ≤ -(25 / 51) := by
        simp only [Acoef, Fin.val_last, jStar, if_true]
        rw [if_neg (fun h => hne h.symm)]
        have h1 : ((M j.val : ℕ) : ℝ) ≥ 0 := Nat.cast_nonneg _
        have h2 : ((M (src n) : ℕ) : ℝ) ≤ 13 := by exact_mod_cast M_le _
        simp only [μ]; nlinarith
      have hκ : 0 ≤ 50 / (σ ^ 2 * Real.sqrt 51) := by have := σ_pos; positivity
      nlinarith
  calc ∑ j : Fin (n + 1), _ ≤ ∑ _j : Fin (n + 1), (1 / 10 ^ 35 : ℝ) := sum_le_sum fun j _ => hb j
    _ = (n + 1) * (1 / 10 ^ 35) := by simp
    _ ≤ 1 / 10 ^ 30 := by
        have : (n : ℝ) ≤ 21 := by exact_mod_cast hn'
        nlinarith

lemma W_nonneg (j : Fin (n + 1)) : 0 ≤ W hT ids j := by
  unfold W; rw [headWeight_last]; exact softmax_nonneg _ _

lemma W_sum : ∑ j, W hT ids j = 1 := by
  unfold W; simp only [headWeight_last]; exact softmax_sum _

/-- Attention output on token dim a: the W-average of the centred token one-hots. -/
lemma attn_tok (a : Fin 28) :
    A0.apply (fun t => U hT ids t) (Fin.last n) (a.castLE (by norm_num)) =
      ∑ j, W hT ids j * (((if ids j = a then 1 else 0) - μ) / σ) := by
  unfold Attn.apply
  rw [Wo_eq, Matrix.one_mulVec]
  have hr : finProdFinEquiv.symm (a.castLE (by norm_num) : Fin (1 * 51)) = ((0 : Fin 1), a.castLE (by norm_num)) := by
    rw [Equiv.symm_apply_eq]; ext; exact (fpe0 _).symm
  simp only [hr]
  refine sum_congr rfl fun j _ => ?_
  rw [Wv_apply, if_pos (by simp), U_tok]
  rfl

lemma centre (f : Fin (n + 1) → ℝ) :
    ∑ j, W hT ids j * ((f j - μ) / σ) = (∑ j, W hT ids j * f j - μ) / σ := by
  calc ∑ j, W hT ids j * ((f j - μ) / σ) = (∑ j, W hT ids j * f j - μ * ∑ j, W hT ids j) / σ := by
        rw [mul_sum, ← sum_sub_distrib, sum_div]; exact sum_congr rfl fun j _ => by ring
    _ = (∑ j, W hT ids j * f j - μ) / σ := by rw [W_sum, mul_one]

/-- Residual stream at the last position on token dim a (the MLP is identically zero). -/
noncomputable def Z (a : Fin 28) : ℝ :=
  (if ids (Fin.last n) = a then 1 else 0) + ∑ j, W hT ids j * (((if ids j = a then 1 else 0) - μ) / σ)

lemma Z_lower (hn : 11 ≤ n) : (W hT ids (jStar n hn) - μ) / σ ≤ Z hT ids (ids (jStar n hn)) := by
  unfold Z
  have hσ := σ_pos
  rw [centre hT ids (fun j => if ids j = ids (jStar n hn) then 1 else 0)]
  have : W hT ids (jStar n hn) ≤ ∑ j, W hT ids j * (if ids j = ids (jStar n hn) then 1 else 0) := by
    calc W hT ids (jStar n hn) = W hT ids (jStar n hn) * (if ids (jStar n hn) = ids (jStar n hn) then 1 else 0) := by simp
      _ ≤ _ := single_le_sum (f := fun j => W hT ids j * (if ids j = ids (jStar n hn) then (1 : ℝ) else 0))
            (fun j _ => mul_nonneg (W_nonneg hT ids j) (by split_ifs <;> norm_num)) (mem_univ _)
  have h0 : 0 ≤ (if ids (Fin.last n) = ids (jStar n hn) then (1 : ℝ) else 0) := by split_ifs <;> norm_num
  have := div_le_div_of_nonneg_right (sub_le_sub_right this μ) hσ.le
  linarith

lemma Z_upper (hn : 11 ≤ n) (a : Fin 28) (ha : a ≠ ids (jStar n hn)) :
    Z hT ids a ≤ 1 + ((1 - W hT ids (jStar n hn)) - μ) / σ := by
  unfold Z
  have hσ := σ_pos
  rw [centre hT ids (fun j => if ids j = a then 1 else 0)]
  have hle : ∑ j, W hT ids j * (if ids j = a then 1 else 0) ≤ 1 - W hT ids (jStar n hn) := by
    have hs := sum_erase_add univ (W hT ids) (mem_univ (jStar n hn))
    rw [W_sum] at hs
    calc ∑ j, W hT ids j * (if ids j = a then 1 else 0)
        = ∑ j ∈ univ.erase (jStar n hn), W hT ids j * (if ids j = a then 1 else 0) +
            W hT ids (jStar n hn) * (if ids (jStar n hn) = a then 1 else 0) :=
          (sum_erase_add _ _ (mem_univ _)).symm
      _ = ∑ j ∈ univ.erase (jStar n hn), W hT ids j * (if ids j = a then 1 else 0) := by
          rw [if_neg (fun h => ha h.symm), mul_zero, add_zero]
      _ ≤ ∑ j ∈ univ.erase (jStar n hn), W hT ids j := by
          refine sum_le_sum fun j _ => ?_
          have := W_nonneg hT ids j
          split_ifs <;> linarith
      _ = 1 - W hT ids (jStar n hn) := by linarith
  have h1 : (if ids (Fin.last n) = a then (1 : ℝ) else 0) ≤ 1 := by split_ifs <;> norm_num
  have := div_le_div_of_nonneg_right (sub_le_sub_right hle μ) hσ.le
  linarith

end Step

/-- **One generation step.** For any query position 11 ≤ n ≤ 21 and any token sequence, the
model predicts the token at the routed source position `src n`. -/
theorem step (n : ℕ) (hn : 11 ≤ n) (hn' : n ≤ 21) (hT : n + 1 ≤ 23) (ids : Fin (n + 1) → Fin 28) :
    argmaxFirst (wrModel.logits hT ids (Fin.last n)) = ids (jStar n hn) := by
  -- the logits as a monotone image of Z
  set z : Fin 51 → ℝ := X hT ids (Fin.last n) + A0.apply (fun t => U hT ids t) (Fin.last n) with hz
  have hblock : applyBlocks wrModel.blocks (X hT ids) (Fin.last n) = z := by
    show (X hT ids (Fin.last n) + A0.apply (fun t => wrModel_block0.ln1.apply (X hT ids t)) (Fin.last n)) +
      wrModel_block0.mlp _ = z
    rw [mlp_zero, add_zero]
    simp only [LN1_X]
    rfl
  have hzt : ∀ a : Fin 28, z (a.castLE (by norm_num)) = Z hT ids a := by
    intro a
    simp only [hz, Pi.add_apply, attn_tok, X_tok, Z]
  set m := (∑ i, z i) / (51 : ℕ)
  set sd := Real.sqrt ((∑ i, (z i - m) ^ 2) / (51 : ℕ) + 2748779 / 274877906944)
  have hsd : 0 < sd := Real.sqrt_pos.2 (by positivity)
  have hlog : ∀ t : Fin 28, wrModel.logits hT ids (Fin.last n) t = 50 * ((Z hT ids t - m) / sd) := by
    intro t
    have h1 : wrModel.logits hT ids (Fin.last n) t =
        (wrModel.Wh *ᵥ wrModel.finalLn.apply z + wrModel.bh) t := by
      rw [← hblock]; rfl
    rw [h1, Wh_mulVec]
    show 50 * ((z _ - m) / sd * wrModel_final_gamma _ + wrModel_final_beta _) = _
    rw [(gamma_one _).2, hzt]
    simp [wrModel_final_beta]
  apply argmaxFirst_eq_of_lt
  intro a ha
  rw [hlog, hlog]
  have hW := leak hT ids hn hn'
  have lo := Z_lower hT ids hn
  have up := Z_upper hT ids hn a ha
  have hσ := σ_pos
  have hσ5 := σ_le
  have key : Z hT ids a < Z hT ids (ids (jStar n hn)) := by
    have h1 : (W hT ids (jStar n hn) - μ) / σ - (1 + ((1 - W hT ids (jStar n hn)) - μ) / σ) =
        (2 * W hT ids (jStar n hn) - 1 - σ) / σ := by field_simp; ring
    have h2 : 0 < (2 * W hT ids (jStar n hn) - 1 - σ) / σ := div_pos (by linarith) hσ
    linarith
  have : (Z hT ids a - m) / sd < (Z hT ids (ids (jStar n hn)) - m) / sd :=
    div_lt_div_of_pos_right (by linarith) hsd
  linarith

end VWC.Arch.WordRev

namespace VWC.Arch.WordRev

open Finset

/-- The routing table implements word reversal: the source of answer token k is in the prompt. -/
lemma route (w1 w2 w3 : Word) (k : ℕ) (hk : k < 11) :
    (prompt w1 w2 w3)[src (11 + k)]'(by simp [prompt]; have := src_lt (11 + k); omega) =
      (answer w1 w2 w3)[k]'(by simp [answer]; omega) := by
  interval_cases k <;> simp [prompt, answer, src, List.getElem_append]

/-- **word-reversal-3x3, real-valued correctness of greedy generation** on every input. -/
theorem wr_correct (w1 w2 w3 : Word) :
    wrModel.generate (prompt w1 w2 w3) 11 = some (answer w1 w2 w3) := by
  have hans : (answer w1 w2 w3).length = 11 := by simp [answer]
  rw [← hans]
  apply Model.generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, answer]
  intro k hk h0 hL
  have hk' : k < 11 := by simpa [answer] using hk
  have hlen : (prompt w1 w2 w3 ++ (answer w1 w2 w3).take k).length = (11 + k) + 1 := by
    simp [prompt, answer]; omega
  rw [Model.next_eq wrModel _ h0 hL hlen (by omega)
    (fun t => (prompt w1 w2 w3 ++ (answer w1 w2 w3).take k).get (Fin.cast hlen.symm t)) (fun t => rfl)]
  rw [step (11 + k) (by omega) (by omega) (by omega)]
  simp only [jStar, List.get_eq_getElem, Fin.coe_cast]
  rw [List.getElem_append_left (by simp [prompt]; have := src_lt (11 + k); omega)]
  exact route w1 w2 w3 k hk'

end VWC.Arch.WordRev
