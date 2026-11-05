/-
Phase 2: GPT-2-style semantics for grafting hosts.

`VWC.Arch` (Phase 1) fixes the MLP activation to relu and has no attention biases. GPT-2 uses
GELU and biases on q, k, v and the output projection. This module is `VWC.Arch` with
* an arbitrary activation per block (`GBlock.act : ℝ → ℝ`; GPT-2: tanh-approximated GELU),
* biases `bq bk bv bo` on attention,
and everything else unchanged (pre-norm, LayerNorm with biased variance, causal softmax with
scale 1/sqrt dqk, PyTorch head layout). `Model.toG` embeds Phase-1 models (relu, zero biases) with
the same logits (`Model.toG_logits`), so every Phase-1 model is also a host.
-/
import VWC.Arch

namespace VWC.Graft

open Matrix VWC.Arch

/-- Causal multi-head attention with biases (PyTorch `nn.Linear` layout, out × in). -/
structure GAttn (D H dqk dv : ℕ) where
  Wq : Matrix (Fin (H * dqk)) (Fin D) ℝ
  bq : Fin (H * dqk) → ℝ
  Wk : Matrix (Fin (H * dqk)) (Fin D) ℝ
  bk : Fin (H * dqk) → ℝ
  Wv : Matrix (Fin (H * dv)) (Fin D) ℝ
  bv : Fin (H * dv) → ℝ
  Wo : Matrix (Fin D) (Fin (H * dv)) ℝ
  bo : Fin D → ℝ

/-- One residual block with its own MLP activation. -/
structure GBlock (D F H dqk dv : ℕ) where
  ln1 : Norm D
  attn : Option (GAttn D H dqk dv)
  ln2 : Norm D
  act : ℝ → ℝ
  W1 : Matrix (Fin F) (Fin D) ℝ
  b1 : Fin F → ℝ
  W2 : Matrix (Fin D) (Fin F) ℝ
  b2 : Fin D → ℝ

structure AnyGBlock (D : ℕ) where
  {F H dqk dv : ℕ}
  blk : GBlock D F H dqk dv

structure GModel (V D L : ℕ) where
  tokEmb : Matrix (Fin V) (Fin D) ℝ
  posEmb : Matrix (Fin L) (Fin D) ℝ
  blocks : List (AnyGBlock D)
  finalLn : Norm D
  Wh : Matrix (Fin V) (Fin D) ℝ
  bh : Fin V → ℝ

variable {D T : ℕ}

noncomputable def GAttn.apply {H dqk dv : ℕ} (A : GAttn D H dqk dv) (x : Fin T → Fin D → ℝ)
    (i : Fin T) : Fin D → ℝ :=
  let q : Fin H → Fin T → Fin dqk → ℝ := fun h t c => (A.Wq *ᵥ x t + A.bq) (finProdFinEquiv (h, c))
  let k : Fin H → Fin T → Fin dqk → ℝ := fun h t c => (A.Wk *ᵥ x t + A.bk) (finProdFinEquiv (h, c))
  let v : Fin H → Fin T → Fin dv → ℝ := fun h t c => (A.Wv *ᵥ x t + A.bv) (finProdFinEquiv (h, c))
  let o : Fin (H * dv) → ℝ := fun r =>
    let hc := finProdFinEquiv.symm r
    ∑ j, headWeight (q hc.1) (k hc.1) i j * v hc.1 j hc.2
  A.Wo *ᵥ o + A.bo

/-- `fc2 (act (fc1 x))`. -/
def GBlock.mlp {F H dqk dv : ℕ} (B : GBlock D F H dqk dv) (x : Fin D → ℝ) : Fin D → ℝ :=
  B.W2 *ᵥ (fun f => B.act ((B.W1 *ᵥ x) f + B.b1 f)) + B.b2

noncomputable def GBlock.apply {F H dqk dv : ℕ} (B : GBlock D F H dqk dv)
    (x : Fin T → Fin D → ℝ) : Fin T → Fin D → ℝ :=
  let y : Fin T → Fin D → ℝ := match B.attn with
    | none => x
    | some A => fun i => x i + A.apply (fun t => B.ln1.apply (x t)) i
  fun i => y i + B.mlp (B.ln2.apply (y i))

noncomputable def gApplyBlocks : List (AnyGBlock D) → (Fin T → Fin D → ℝ) → Fin T → Fin D → ℝ
  | [], x => x
  | b :: bs, x => gApplyBlocks bs (b.blk.apply x)

variable {V L : ℕ}

noncomputable def GModel.logits (M : GModel V D L) (hT : T ≤ L) (ids : Fin T → Fin V)
    (i : Fin T) : Fin V → ℝ :=
  let x : Fin T → Fin D → ℝ := fun t => M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t)
  let h := gApplyBlocks M.blocks x
  M.Wh *ᵥ M.finalLn.apply (h i) + M.bh

/-! ### Phase-1 models are GPT-2-style models -/

def _root_.VWC.Arch.Attn.toG {H dqk dv : ℕ} (A : Attn D H dqk dv) : GAttn D H dqk dv :=
  ⟨A.Wq, 0, A.Wk, 0, A.Wv, 0, A.Wo, 0⟩

def _root_.VWC.Arch.Block.toG {F H dqk dv : ℕ} (B : Block D F H dqk dv) : GBlock D F H dqk dv :=
  ⟨B.ln1, B.attn.map Attn.toG, B.ln2, relu, B.W1, B.b1, B.W2, B.b2⟩

def _root_.VWC.Arch.Model.toG (M : Model V D L) : GModel V D L :=
  ⟨M.tokEmb, M.posEmb, M.blocks.map (fun b => ⟨b.blk.toG⟩), M.finalLn, M.Wh, M.bh⟩

lemma attn_toG {H dqk dv : ℕ} (A : Attn D H dqk dv) (x : Fin T → Fin D → ℝ) (i : Fin T) :
    A.toG.apply x i = A.apply x i := by
  simp [GAttn.apply, Attn.apply, Attn.toG]

lemma block_toG {F H dqk dv : ℕ} (B : Block D F H dqk dv) (x : Fin T → Fin D → ℝ) :
    B.toG.apply x = B.apply x := by
  rcases B with ⟨ln1, _ | A, ln2, W1, b1, W2, b2⟩ <;>
    simp [GBlock.apply, Block.apply, Block.toG, GBlock.mlp, Block.mlp, attn_toG]

theorem _root_.VWC.Arch.Model.toG_logits (M : Model V D L) (hT : T ≤ L) (ids : Fin T → Fin V)
    (i : Fin T) : M.toG.logits hT ids i = M.logits hT ids i := by
  have h : ∀ (bs : List (AnyBlock D)) (x : Fin T → Fin D → ℝ),
      gApplyBlocks (bs.map (fun b => (⟨b.blk.toG⟩ : AnyGBlock D))) x = applyBlocks bs x := by
    intro bs
    induction bs with
    | nil => intro x; rfl
    | cons b l ih => intro x; simp only [List.map_cons, gApplyBlocks, applyBlocks, block_toG, ih]
  simp only [GModel.logits, Model.logits, Model.toG, h]

end VWC.Graft
