/-
Real-valued correctness of ParityFloorSubD3F9, decomposed along the mechanism:
embedding -> uniform attention counts the 1s -> MLP computes N mod 2 -> readout margin 1/2.
-/
import VWC.Weights
import VWC.Spec

namespace VWC

open Matrix

local notation "P" => parityParams

lemma hL : (10 + 1 : ℕ) ≤ 16 := by norm_num

/-! ### Weight facts -/

lemma posEmb_eq_zero : parityParams.posEmb = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [parityParams]

lemma Wq_eq_zero : parityParams.Wq = 0 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [parityParams]

lemma Wv_mulVec (v : Fin 3 → ℝ) : parityParams.Wv *ᵥ v = ![0, 0, 11 * v 0] := by
  ext k; fin_cases k <;> simp [parityParams, mulVec, dotProduct, Fin.sum_univ_succ]

lemma Wo_mulVec (v : Fin 3 → ℝ) : parityParams.Wo *ᵥ v = ![0, 0, v 2] := by
  ext k; fin_cases k <;> simp [parityParams, mulVec, dotProduct, Fin.sum_univ_succ]

lemma mlp_eq (v : Fin 3 → ℝ) : mlp P v = ![0, 0,
    -2 * relu (v 2 - 1) + 2 * relu (v 2 - 2) - 2 * relu (v 2 - 3) + 2 * relu (v 2 - 4)
    - 2 * relu (v 2 - 5) + 2 * relu (v 2 - 6) - 2 * relu (v 2 - 7) + 2 * relu (v 2 - 8)
    - 2 * relu (v 2 - 9)] := by
  ext k; fin_cases k <;>
    simp [mlp, parityParams, mulVec, dotProduct, Fin.sum_univ_succ, sub_eq_add_neg] <;> ring

lemma Wh_mulVec (v : Fin 3 → ℝ) :
    parityParams.Wh *ᵥ v = ![v 1 / 2 - 10 * v 2, -(v 1 / 2) + 10 * v 2, 0, 0] := by
  ext k; fin_cases k <;> simp [parityParams, mulVec, dotProduct, Fin.sum_univ_succ] <;> ring

/-! ### Mechanism lemmas -/

/-- Number of `1` tokens in a prompt. -/
def nOnes (ids : Fin 11 → Fin 4) : ℕ := ∑ t, if ids t = tok1 then 1 else 0

lemma embed_eq (ids : Fin 11 → Fin 4) (t : Fin 11) :
    embed P hL ids t =
      ![if ids t = tok1 then 1 else 0, if ids t = tokEq then 1 else 0, 0] := by
  unfold embed
  rw [posEmb_eq_zero]
  generalize ids t = a
  ext k; fin_cases a <;> fin_cases k <;> simp [parityParams, tok1, tokEq]

lemma score_eq_zero (x : Fin 11 → Fin 3 → ℝ) (i j : Fin 11) : score P x i j = 0 := by
  simp [score, Wq_eq_zero]

/-- At the final `=` position, attention writes the count of `1`s into channel 2. -/
lemma attn_computes_count (ids : Fin 11 → Fin 4) :
    attn P (embed P hL ids) (Fin.last 10) = ![0, 0, (nOnes ids : ℝ)] := by
  have hw : ∀ j, attnWeight P (embed P hL ids) (Fin.last 10) j = 1 / 11 := by
    intro j
    rw [softmax_zero_scores_uniform P _ (score_eq_zero _) (Fin.last 10) j,
      if_pos (Fin.le_last j)]
    norm_num
  unfold attn
  simp_rw [hw, Wv_mulVec, Wo_mulVec, embed_eq]
  ext k; fin_cases k
  · simp
  · simp
  · simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, nOnes, Nat.cast_sum, Nat.cast_ite,
      Nat.cast_one, Nat.cast_zero]
    simp

/-- The MLP turns channel 2 from `n` into `n mod 2` (finite case analysis over n ≤ 10). -/
lemma mlp_computes_mod2 (n : ℕ) (hn : n ≤ 10) (a b : ℝ) :
    (n : ℝ) + mlp P ![a, b, (n : ℝ)] 2 = ((n % 2 : ℕ) : ℝ) := by
  rw [mlp_eq]
  interval_cases n <;> norm_num [relu]

lemma nOnes_le (ids : Fin 11 → Fin 4) (h : ids (Fin.last 10) = tokEq) : nOnes ids ≤ 10 := by
  unfold nOnes
  rw [Fin.sum_univ_castSucc, h]
  simp only [tokEq, tok1, show ((2 : Fin 4) = 1) = False by decide, ite_false, add_zero]
  calc (∑ i : Fin 10, if ids i.castSucc = 1 then 1 else 0)
      ≤ ∑ _i : Fin 10, 1 := Finset.sum_le_sum fun i _ => by split_ifs <;> norm_num
    _ = 10 := by simp

/-- Logits at the last position, for any prompt ending in `=`. -/
lemma logits_eq (ids : Fin 11 → Fin 4) (h : ids (Fin.last 10) = tokEq) :
    logitsAt P hL ids (Fin.last 10) =
      ![1 / 2 - 10 * ((nOnes ids % 2 : ℕ) : ℝ), -(1 / 2) + 10 * ((nOnes ids % 2 : ℕ) : ℝ), 0, 0] := by
  have hres : embed P hL ids (Fin.last 10) + attn P (embed P hL ids) (Fin.last 10) =
      ![0, 1, (nOnes ids : ℝ)] := by
    rw [attn_computes_count, embed_eq, h]
    ext k; fin_cases k <;> simp [tok1, tokEq]
  have hmlp := mlp_computes_mod2 (nOnes ids) (nOnes_le ids h) 0 1
  unfold logitsAt block
  simp only [hres]
  rw [Wh_mulVec]
  have h1 : (![0, 1, (nOnes ids : ℝ)] + mlp P ![0, 1, (nOnes ids : ℝ)]) 1 = 1 := by
    simp [mlp_eq]
  have h2 : (![0, 1, (nOnes ids : ℝ)] + mlp P ![0, 1, (nOnes ids : ℝ)]) 2 =
      ((nOnes ids % 2 : ℕ) : ℝ) := by
    simpa using hmlp
  rw [h1, h2]

/-- Readout margin: the correct token's logit beats every other logit by at least δ = 1/2. -/
theorem readout_margin (ids : Fin 11 → Fin 4) (h : ids (Fin.last 10) = tokEq) (t : Fin 4)
    (ht : t ≠ (if nOnes ids % 2 = 1 then tok1 else tok0)) :
    logitsAt P hL ids (Fin.last 10) t + 1 / 2 ≤
      logitsAt P hL ids (Fin.last 10) (if nOnes ids % 2 = 1 then tok1 else tok0) := by
  rw [logits_eq ids h]
  rcases Nat.mod_two_eq_zero_or_one (nOnes ids) with hp | hp <;>
    simp only [hp, tok0, tok1] at ht ⊢ <;> fin_cases t <;> simp_all <;> norm_num

/-- Prompt-level theorem: any 11-token prompt ending in `=` is classified by the parity of
its number of `1` tokens. (Positions 0..9 may hold any token, including non-canonical ones.) -/
theorem predict_prompt (ids : Fin 11 → Fin 4) (h : ids (Fin.last 10) = tokEq) :
    predict P hL ids = if nOnes ids % 2 = 1 then tok1 else tok0 := by
  apply argmaxFirst_eq_of_lt
  intro j hj
  have := readout_margin ids h j hj
  linarith

/-! ### From bit strings to prompts -/

lemma sum_getD_eq_count {α : Type*} [DecidableEq α] (a d : α) :
    ∀ (l : List α) (n : ℕ) (hl : l.length = n),
      (∑ t : Fin n, if l.getD t d = a then 1 else 0) = l.count a
  | [], n, hl => by subst hl; simp
  | _ :: _, 0, hl => by simp at hl
  | b :: l, n + 1, hl => by
    rw [Fin.sum_univ_succ]
    simp only [Fin.val_zero, List.getD_cons_zero, Fin.val_succ, List.getD_cons_succ,
      List.count_cons]
    rw [sum_getD_eq_count a d l n (by simpa using hl)]
    by_cases hb : b = a <;> simp [hb, Nat.add_comm]

lemma promptList_length (bits : List Bool) (h : bits.length ≤ 10) :
    (promptList bits).length = 11 := by
  simp [promptList]; omega

lemma encode_last (bits : List Bool) (h : bits.length ≤ 10) :
    encode bits (Fin.last 10) = tokEq := by
  have hl := promptList_length bits h
  simp only [encode, Fin.val_last, List.getD_eq_getElem?_getD]
  rw [List.getElem?_eq_getElem (by omega)]
  simp only [promptList, Option.getD_some]
  rw [List.getElem_append_right (by simp; omega)]
  simp

lemma nOnes_encode (bits : List Bool) (h : bits.length ≤ 10) :
    nOnes (encode bits) = bits.count true := by
  unfold nOnes encode
  rw [sum_getD_eq_count tok1 tokPad (promptList bits) 11 (promptList_length bits h)]
  simp only [promptList, List.count_append, List.count_replicate, List.count_singleton]
  have hinj : Function.Injective tokOfBit := by
    intro x y hxy; cases x <;> cases y <;> simp_all [tokOfBit, tok0, tok1]
  have key : List.count tok1 (bits.map tokOfBit) = bits.count true :=
    List.count_map_of_injective bits tokOfBit hinj true
  simp [tokPad, tok1, tokEq]
  exact key

/-- **Main theorem.** On every bit string of length 1..10, encoded exactly as the task does,
the real-valued model predicts the parity token. -/
theorem parity_correct (bits : List Bool) (_h1 : 1 ≤ bits.length) (h2 : bits.length ≤ 10) :
    predict P hL (encode bits) = parityTok bits := by
  rw [predict_prompt _ (encode_last bits h2), nOnes_encode bits h2]
  rfl

end VWC
