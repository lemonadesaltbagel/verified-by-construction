/-
Minimal real-valued semantics of the `SimpleTransformer` in
`runs/parity-upto10-bits-may15-run1/interpretable_transformer.py`, restricted to
n_layers = 1, n_heads = 1, LayerNorm = Identity (as in the snapshot).

Divergences from PyTorch, all semantics-preserving over ℝ:
* softmax is `exp s / Σ exp s` (PyTorch subtracts the row max first; identical over ℝ);
* masked positions get weight 0 (PyTorch: score -inf, exp(-inf) = 0);
* one sequence at a time (no batch dimension);
* with n_heads = 1 the per-head view/transpose is the identity, so d_head = d_model.
-/
import Mathlib

namespace VWC

open Matrix

/-- All parameters of a 1-layer, 1-head transformer without LayerNorm.
`V` vocab size, `D` d_model, `F` d_ff, `L` max_seq_len. Matrices use PyTorch's
`nn.Linear` layout (out × in), so a linear layer is `W *ᵥ x`. -/
structure Params (V D F L : ℕ) where
  tokEmb : Matrix (Fin V) (Fin D) ℝ
  posEmb : Matrix (Fin L) (Fin D) ℝ
  Wq : Matrix (Fin D) (Fin D) ℝ
  Wk : Matrix (Fin D) (Fin D) ℝ
  Wv : Matrix (Fin D) (Fin D) ℝ
  Wo : Matrix (Fin D) (Fin D) ℝ
  W1 : Matrix (Fin F) (Fin D) ℝ
  b1 : Fin F → ℝ
  W2 : Matrix (Fin D) (Fin F) ℝ
  b2 : Fin D → ℝ
  Wh : Matrix (Fin V) (Fin D) ℝ

def relu (x : ℝ) : ℝ := max x 0

variable {V D F L T : ℕ} (P : Params V D F L)

/-- `token_emb(ids) + pos_emb(arange T)`. -/
def embed (hT : T ≤ L) (ids : Fin T → Fin V) (t : Fin T) : Fin D → ℝ :=
  P.tokEmb (ids t) + P.posEmb (Fin.castLE hT t)

/-- Attention score `q_i · k_j / sqrt d_head`. -/
noncomputable def score (x : Fin T → Fin D → ℝ) (i j : Fin T) : ℝ :=
  (P.Wq *ᵥ x i) ⬝ᵥ (P.Wk *ᵥ x j) / Real.sqrt D

/-- Causal softmax weight of query `i` on key `j`. -/
noncomputable def attnWeight (x : Fin T → Fin D → ℝ) (i j : Fin T) : ℝ :=
  if j ≤ i then
    Real.exp (score P x i j) / ∑ j' ∈ Finset.Iic i, Real.exp (score P x i j')
  else 0

/-- `W_o (Σ_j a_ij · W_v x_j)`. -/
noncomputable def attn (x : Fin T → Fin D → ℝ) (i : Fin T) : Fin D → ℝ :=
  P.Wo *ᵥ ∑ j, attnWeight P x i j • (P.Wv *ᵥ x j)

/-- `fc2 (relu (fc1 x))`. -/
def mlp (x : Fin D → ℝ) : Fin D → ℝ :=
  P.W2 *ᵥ (fun k => relu ((P.W1 *ᵥ x) k + P.b1 k)) + P.b2

/-- The residual block: `x + attn x`, then `+ mlp`. -/
noncomputable def block (x : Fin T → Fin D → ℝ) (i : Fin T) : Fin D → ℝ :=
  let r := x i + attn P x i
  r + mlp P r

/-- Logits at position `i`. -/
noncomputable def logitsAt (hT : T ≤ L) (ids : Fin T → Fin V) (i : Fin T) : Fin V → ℝ :=
  P.Wh *ᵥ block P (embed P hT ids) i

/-- `torch.argmax`: the first index attaining the maximum. -/
noncomputable def argmaxFirst {n : ℕ} (f : Fin (n + 1) → ℝ) : Fin (n + 1) :=
  (Finset.univ.filter fun i => ∀ j, f j ≤ f i).min'
    (by
      obtain ⟨i, -, hi⟩ := Finset.exists_max_image Finset.univ f Finset.univ_nonempty
      exact ⟨i, by simpa using hi⟩)

/-- Greedy next-token prediction (`logits[:, -1, :].argmax(-1)`) for a length-`T+1` prompt. -/
noncomputable def predict {n : ℕ} (Q : Params (n + 1) D F L) (hT : T + 1 ≤ L)
    (ids : Fin (T + 1) → Fin (n + 1)) : Fin (n + 1) :=
  argmaxFirst (logitsAt Q hT ids (Fin.last T))

/-- A strict maximum is the `argmaxFirst`. -/
theorem argmaxFirst_eq_of_lt {n : ℕ} (f : Fin (n + 1) → ℝ) (i : Fin (n + 1))
    (h : ∀ j, j ≠ i → f j < f i) : argmaxFirst f = i := by
  have hs : (Finset.univ.filter fun k => ∀ j, f j ≤ f k) = {i} := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro hk
      by_contra hne
      exact absurd (hk i) (not_le.mpr (h k hne))
    · rintro rfl j
      rcases eq_or_ne j k with rfl | hj
      · exact le_rfl
      · exact (h j hj).le
  unfold argmaxFirst
  simp only [hs, Finset.min'_singleton]

/-- With all scores zero, causal softmax is the uniform average over the visible prefix. -/
theorem softmax_zero_scores_uniform (x : Fin T → Fin D → ℝ)
    (h : ∀ i j, score P x i j = 0) (i j : Fin T) :
    attnWeight P x i j = if j ≤ i then 1 / ((i : ℕ) + 1 : ℝ) else 0 := by
  unfold attnWeight
  split_ifs
  · simp [h, Fin.card_Iic]
  · rfl

end VWC
