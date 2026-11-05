/-
Generic real-valued semantics for the hand-written transformers of the Weight-Space Programming
repo (Phase 1). Covers the architecture variants found in the snapshots:

* any number of residual blocks, each with an optional causal multi-head attention sub-layer
  (H heads, per-head Q/K width `dqk`, per-head V width `dv`, scale 1/sqrt dqk) and an MLP;
* per-block widths may differ (e.g. d_ff = (10, 12));
* pre-norm placement `x + attn (ln1 x)`, `x + mlp (ln2 x)`, with each norm either the identity
  or a real LayerNorm (biased variance, `eps`);
* optional final LayerNorm; head with optional bias.

Snapshot variants map onto this without changing semantics over ℝ: a missing `W_o` is the
identity matrix, a missing bias is zero, an MLP-only block has `attn = none`.
PyTorch's `view(B, T, H, d)` puts head `h`, coordinate `c` at row `h * d + c`; `finProdFinEquiv`
maps `(h, c)` to `c + d * h`, i.e. the same row.
-/
import Mathlib
import VWC.Model
import VWC.Softmax

namespace VWC.Arch

open Matrix

/-- Identity or LayerNorm (`(x - mean) / sqrt (var + eps) * γ + β`, biased variance). -/
inductive Norm (D : ℕ) where
  | id : Norm D
  | ln (γ β : Fin D → ℝ) (eps : ℝ) : Norm D

noncomputable def Norm.apply {D : ℕ} : Norm D → (Fin D → ℝ) → Fin D → ℝ
  | .id, x => x
  | .ln γ β eps, x =>
    let μ := (∑ i, x i) / D
    let σ2 := (∑ i, (x i - μ) ^ 2) / D
    fun i => (x i - μ) / Real.sqrt (σ2 + eps) * γ i + β i

/-- Causal multi-head attention parameters (PyTorch `nn.Linear` layout, out × in). -/
structure Attn (D H dqk dv : ℕ) where
  Wq : Matrix (Fin (H * dqk)) (Fin D) ℝ
  Wk : Matrix (Fin (H * dqk)) (Fin D) ℝ
  Wv : Matrix (Fin (H * dv)) (Fin D) ℝ
  Wo : Matrix (Fin D) (Fin (H * dv)) ℝ

/-- One residual block. -/
structure Block (D F H dqk dv : ℕ) where
  ln1 : Norm D
  attn : Option (Attn D H dqk dv)
  ln2 : Norm D
  W1 : Matrix (Fin F) (Fin D) ℝ
  b1 : Fin F → ℝ
  W2 : Matrix (Fin D) (Fin F) ℝ
  b2 : Fin D → ℝ

/-- A block together with its own widths. -/
structure AnyBlock (D : ℕ) where
  {F H dqk dv : ℕ}
  blk : Block D F H dqk dv

/-- Whole model: vocab `V`, d_model `D`, max_seq_len `L`. -/
structure Model (V D L : ℕ) where
  tokEmb : Matrix (Fin V) (Fin D) ℝ
  posEmb : Matrix (Fin L) (Fin D) ℝ
  blocks : List (AnyBlock D)
  finalLn : Norm D
  Wh : Matrix (Fin V) (Fin D) ℝ
  bh : Fin V → ℝ

/-- `mulVec` eta-expands sparse rows `Pi.single j a + …`; this splits `⬝ᵥ` over them. -/
@[simp] lemma fun_add_dot {n : ℕ} (f g v : Fin n → ℝ) :
    (fun j => f j + g j) ⬝ᵥ v = f ⬝ᵥ v + g ⬝ᵥ v :=
  add_dotProduct f g v

variable {D T : ℕ}

/-- Per-head causal softmax weight of query `i` on key `j`, given that head's queries/keys. -/
noncomputable def headWeight {dqk : ℕ} (q k : Fin T → Fin dqk → ℝ) (i j : Fin T) : ℝ :=
  let s : Fin T → ℝ := fun j => (q i ⬝ᵥ k j) / Real.sqrt dqk
  if j ≤ i then Real.exp (s j) / ∑ j' ∈ Finset.Iic i, Real.exp (s j') else 0

/-- Multi-head causal self-attention output at position `i` (after `W_o`). -/
noncomputable def Attn.apply {H dqk dv : ℕ} (A : Attn D H dqk dv) (x : Fin T → Fin D → ℝ)
    (i : Fin T) : Fin D → ℝ :=
  let q : Fin H → Fin T → Fin dqk → ℝ := fun h t c => (A.Wq *ᵥ x t) (finProdFinEquiv (h, c))
  let k : Fin H → Fin T → Fin dqk → ℝ := fun h t c => (A.Wk *ᵥ x t) (finProdFinEquiv (h, c))
  let v : Fin H → Fin T → Fin dv → ℝ := fun h t c => (A.Wv *ᵥ x t) (finProdFinEquiv (h, c))
  let o : Fin (H * dv) → ℝ := fun r =>
    let hc := finProdFinEquiv.symm r
    ∑ j, headWeight (q hc.1) (k hc.1) i j * v hc.1 j hc.2
  A.Wo *ᵥ o

/-- `fc2 (relu (fc1 x))`. -/
def Block.mlp {F H dqk dv : ℕ} (B : Block D F H dqk dv) (x : Fin D → ℝ) : Fin D → ℝ :=
  B.W2 *ᵥ (fun f => relu ((B.W1 *ᵥ x) f + B.b1 f)) + B.b2

/-- Residual block applied to the whole sequence. -/
noncomputable def Block.apply {F H dqk dv : ℕ} (B : Block D F H dqk dv)
    (x : Fin T → Fin D → ℝ) : Fin T → Fin D → ℝ :=
  let y : Fin T → Fin D → ℝ := match B.attn with
    | none => x
    | some A => fun i => x i + A.apply (fun t => B.ln1.apply (x t)) i
  fun i => y i + B.mlp (B.ln2.apply (y i))

noncomputable def applyBlocks : List (AnyBlock D) → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ
  | [], x => x
  | b :: bs, x => applyBlocks bs (b.blk.apply x)

variable {V L : ℕ}

/-- Logits at every position for a length-`T` prompt (`T ≤ L`). -/
noncomputable def Model.logits (M : Model V D L) (hT : T ≤ L) (ids : Fin T → Fin V)
    (i : Fin T) : Fin V → ℝ :=
  let x : Fin T → Fin D → ℝ := fun t => M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t)
  let h := applyBlocks M.blocks x
  M.Wh *ᵥ M.finalLn.apply (h i) + M.bh

/-- Greedy next token for a nonempty token list of length ≤ L (`logits[-1].argmax()`). -/
noncomputable def Model.next {n : ℕ} (M : Model (n + 1) D L) (s : List (Fin (n + 1)))
    (h0 : 0 < s.length) (hL : s.length ≤ L) : Fin (n + 1) :=
  argmaxFirst (M.logits hL (fun t => s.get t) ⟨s.length - 1, by omega⟩)

/-- Greedy autoregressive generation of `k` tokens; `none` if the sequence would exceed L. -/
noncomputable def Model.generate {n : ℕ} (M : Model (n + 1) D L) :
    List (Fin (n + 1)) → ℕ → Option (List (Fin (n + 1)))
  | _, 0 => some []
  | s, k + 1 =>
    if h : 0 < s.length ∧ s.length ≤ L then
      let t := M.next s h.1 h.2
      (M.generate (s ++ [t]) k).map (t :: ·)
    else none

/-- Generation is correct if every greedy step is correct on the teacher-forced prefix. -/
theorem Model.generate_eq_of_steps {n : ℕ} (M : Model (n + 1) D L) :
    ∀ (ans s : List (Fin (n + 1))),
      0 < s.length → s.length + ans.length ≤ L + 1 →
      (∀ k (hk : k < ans.length) (h0 : 0 < (s ++ ans.take k).length)
          (hL : (s ++ ans.take k).length ≤ L),
          M.next (s ++ ans.take k) h0 hL = ans.get ⟨k, hk⟩) →
      M.generate s ans.length = some ans
  | [], _, _, _, _ => rfl
  | a :: ans, s, h0, hlen, hstep => by
    have hL : s.length ≤ L := by simp at hlen; omega
    have h0' : 0 < (s ++ List.take 0 (a :: ans)).length := by simpa using h0
    have hL' : (s ++ List.take 0 (a :: ans)).length ≤ L := by simpa using hL
    have hfirst := hstep 0 (by simp) h0' hL'
    simp only [List.take_zero, List.append_nil, List.get_eq_getElem, List.getElem_cons_zero]
      at hfirst
    have hrest : M.generate (s ++ [a]) ans.length = some ans := by
      apply Model.generate_eq_of_steps M ans (s ++ [a]) (by simp) (by simp at hlen ⊢; omega)
      intro k hk h0k hLk
      have := hstep (k + 1) (by simp; omega) (by simp) (by simpa [List.take_succ_cons] using hLk)
      simpa [List.take_succ_cons, List.append_assoc] using this
    simp only [List.length_cons, Model.generate, dif_pos (And.intro h0 hL), hfirst, hrest,
      Option.map_some]

/-- If index 0 attains the maximum, `argmaxFirst` (torch's first-index rule) returns 0. -/
lemma argmaxFirst_eq_zero {n : ℕ} (f : Fin (n + 1) → ℝ) (h : ∀ j, f j ≤ f 0) :
    argmaxFirst f = 0 := by
  unfold argmaxFirst
  apply le_antisymm _ (Fin.zero_le _)
  exact Finset.min'_le _ _ (by simpa using h)

/-- At the last position every key is visible, so a head's weights are a plain softmax. -/
lemma headWeight_last {n dqk : ℕ} (q k : Fin (n + 1) → Fin dqk → ℝ) (j : Fin (n + 1)) :
    headWeight q k (Fin.last n) j =
      softmax (fun j => (q (Fin.last n) ⬝ᵥ k j) / Real.sqrt dqk) j := by
  have hI : Finset.Iic (Fin.last n) = Finset.univ := by
    ext j; simp [Fin.le_last]
  simp only [headWeight, Fin.le_last, if_true, hI, softmax]

/-- Re-index logits along an equality of lengths. -/
lemma Model.logits_cast {n m : ℕ} (M : Model V D L) (h : n = m) (hn : n ≤ L) (hm : m ≤ L)
    (ids : Fin n → Fin V) (i : Fin n) :
    M.logits hn ids i = M.logits hm (fun t => ids (Fin.cast h.symm t)) (Fin.cast h i) := by
  subst h; rfl

/-- `next` on a list, as `argmaxFirst` of the logits of an equal `Fin`-indexed prompt. -/
lemma Model.next_eq {k T : ℕ} (M : Model (k + 1) D L) (s : List (Fin (k + 1)))
    (h0 : 0 < s.length) (hL : s.length ≤ L) (hlen : s.length = T + 1) (hT : T + 1 ≤ L)
    (ids : Fin (T + 1) → Fin (k + 1)) (hids : ∀ t : Fin (T + 1), s.get (Fin.cast hlen.symm t) = ids t) :
    M.next s h0 hL = argmaxFirst (M.logits hT ids (Fin.last T)) := by
  unfold Model.next
  rw [M.logits_cast hlen hL hT]
  congr 2
  · funext t; exact hids t
  · ext; simp [hlen]

end VWC.Arch
