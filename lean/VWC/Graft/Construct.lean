/-
Phase 2, sub-proposition (a): the graft construction and its structural lemmas.

A host is any LayerNorm transformer in the GPT-2-style semantics `GArch` (any MLP activation,
attention biases; `Host`, uniform block shape). A circuit (`Circuit`) lives
in k extra residual dims: token/position features (zero-sum, constant energy E), circuit blocks
that read and write only the extra dims, final-LN gains and a readout into the logits.

`graft M C` is a standard `GModel` on D + k dims:
* host blocks: input matrices padded with zero columns on the extra dims (`padCols`); output
  matrices and biases extended with mean-tracking rows (`extRows`), so every extra dim keeps
  holding `mean(host residual) + circuit state`; LayerNorms compensated (`lnGraft`);
* circuit blocks appended after the host blocks: inputs read only the extra dims (`liftCols`),
  outputs write only the extra dims (`liftRows`);
* embeddings: host rows plus `mean(row) + circuit feature`; head: host rows plus readout.
-/
import VWC.Graft.LN
import VWC.Graft.GArch

namespace VWC.Graft

open Matrix Finset VWC.Arch

variable {D k : ℕ}

/-- LayerNorm parameters. -/
structure LNP (D : ℕ) where
  γ : Fin D → ℝ
  β : Fin D → ℝ
  ε : ℝ

/-- A host block (pre-norm, LayerNorm, attention with biases + MLP with any activation). -/
structure HBlock (D F H dq dv : ℕ) where
  ln1 : LNP D
  attn : GAttn D H dq dv
  ln2 : LNP D
  act : ℝ → ℝ
  W1 : Matrix (Fin F) (Fin D) ℝ
  b1 : Fin F → ℝ
  W2 : Matrix (Fin D) (Fin F) ℝ
  b2 : Fin D → ℝ

def HBlock.toBlock {F H dq dv : ℕ} (b : HBlock D F H dq dv) : GBlock D F H dq dv :=
  ⟨.ln b.ln1.γ b.ln1.β b.ln1.ε, some b.attn, .ln b.ln2.γ b.ln2.β b.ln2.ε, b.act, b.W1, b.b1, b.W2, b.b2⟩

/-- A host model (GPT-2 style). -/
structure Host (V D L F H dq dv : ℕ) where
  tokEmb : Matrix (Fin V) (Fin D) ℝ
  posEmb : Matrix (Fin L) (Fin D) ℝ
  blocks : List (HBlock D F H dq dv)
  lnf : LNP D
  Wh : Matrix (Fin V) (Fin D) ℝ
  bh : Fin V → ℝ

def Host.toModel {V L F H dq dv : ℕ} (M : Host V D L F H dq dv) : GModel V D L :=
  ⟨M.tokEmb, M.posEmb, M.blocks.map (fun b => ⟨b.toBlock⟩), .ln M.lnf.γ M.lnf.β M.lnf.ε, M.Wh, M.bh⟩

/-- A circuit block on the k extra dims. `ε1`, `ε2` play the role of the host ε in `lnGraft`. -/
structure CBlock (k Fc Hc dq dv : ℕ) where
  γ1 : Fin k → ℝ
  β1 : Fin k → ℝ
  ε1 : ℝ
  attn : GAttn k Hc dq dv
  γ2 : Fin k → ℝ
  β2 : Fin k → ℝ
  ε2 : ℝ
  act : ℝ → ℝ
  W1 : Matrix (Fin Fc) (Fin k) ℝ
  b1 : Fin Fc → ℝ
  W2 : Matrix (Fin k) (Fin Fc) ℝ
  b2 : Fin k → ℝ

/-- A circuit: features, blocks, final-LN gains on the extra dims, readout into the logits. -/
structure Circuit (V L k Fc Hc dq dv : ℕ) where
  sTok : Fin V → Fin k → ℝ
  sPos : Fin L → Fin k → ℝ
  E : ℝ
  blocks : List (CBlock k Fc Hc dq dv)
  γf : Fin k → ℝ
  βf : Fin k → ℝ
  Wc : Matrix (Fin V) (Fin k) ℝ

/-! ### Matrix extensions -/

/-- Input matrix that reads only the host dims. -/
def padCols {m : ℕ} (W : Matrix (Fin m) (Fin D) ℝ) : Matrix (Fin m) (Fin (D + k)) ℝ :=
  fun r => Fin.append (W r) 0

/-- Input matrix that reads only the extra dims. -/
def liftCols {m : ℕ} (W : Matrix (Fin m) (Fin k) ℝ) : Matrix (Fin m) (Fin (D + k)) ℝ :=
  fun r => Fin.append 0 (W r)

/-- Host output matrix with mean-tracking rows on the extra dims. -/
noncomputable def extRows {m : ℕ} (W : Matrix (Fin D) (Fin m) ℝ) : Matrix (Fin (D + k)) (Fin m) ℝ :=
  Fin.append (fun i => W i) (fun _ => fun r => (∑ i, W i r) / D)

/-- Circuit output matrix: writes only the extra dims. -/
def liftRows {m : ℕ} (W : Matrix (Fin k) (Fin m) ℝ) : Matrix (Fin (D + k)) (Fin m) ℝ :=
  Fin.append (fun _ => 0) (fun j => W j)

/-- Host vector plus its mean on the extra dims (a host write). -/
noncomputable def ext (w : Fin D → ℝ) : Fin (D + k) → ℝ := Fin.append w (fun _ => mean w)

lemma append_dot {m n : ℕ} (a c : Fin m → ℝ) (b d : Fin n → ℝ) :
    Fin.append a b ⬝ᵥ Fin.append c d = a ⬝ᵥ c + b ⬝ᵥ d := by
  simp [dotProduct, Fin.sum_univ_add]

lemma padCols_mulVec {m : ℕ} (W : Matrix (Fin m) (Fin D) ℝ) (a : Fin D → ℝ) (b : Fin k → ℝ) :
    padCols W *ᵥ Fin.append a b = W *ᵥ a := by
  funext r; simp [padCols, mulVec, append_dot]

lemma liftCols_mulVec {m : ℕ} (W : Matrix (Fin m) (Fin k) ℝ) (a : Fin D → ℝ) (b : Fin k → ℝ) :
    liftCols W *ᵥ Fin.append a b = W *ᵥ b := by
  funext r; simp [liftCols, mulVec, append_dot]

lemma extRows_mulVec {m : ℕ} (W : Matrix (Fin D) (Fin m) ℝ) (o : Fin m → ℝ) :
    (extRows (k := k) W) *ᵥ o = ext (W *ᵥ o) := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [extRows, ext, mulVec]
  · simp only [extRows, ext, mulVec, Fin.append_right, mean, dotProduct]
    simp only [div_mul_eq_mul_div, Finset.sum_mul, Finset.sum_div]
    exact Finset.sum_comm

lemma liftRows_mulVec {m : ℕ} (W : Matrix (Fin k) (Fin m) ℝ) (o : Fin m → ℝ) :
    (liftRows (D := D) W) *ᵥ o = Fin.append 0 (W *ᵥ o) := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [liftRows, mulVec]
  · simp [liftRows, mulVec]

lemma mean_add (x w : Fin D → ℝ) : mean (x + w) = mean x + mean w := by
  simp [mean, sum_add_distrib, add_div]

/-- A host write keeps the grafted form. -/
lemma joint_add_ext (x w : Fin D → ℝ) (s : Fin k → ℝ) :
    joint x s + ext w = joint (x + w) s := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [joint, ext]
  · simp [joint, ext, mean_add]; ring

/-- A circuit write changes only the circuit state. -/
lemma joint_add_circ (x : Fin D → ℝ) (s c : Fin k → ℝ) :
    joint x s + Fin.append 0 c = joint x (s + c) := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [joint]
  · simp [joint]; ring

/-! ### The graft -/

/-- Grafted host block. -/
noncomputable def graftH {F H dq dv : ℕ} (E : ℝ) (b : HBlock D F H dq dv) : GBlock (D + k) F H dq dv :=
  ⟨lnGraft b.ln1.γ b.ln1.β 1 0 b.ln1.ε E,
   some ⟨padCols b.attn.Wq, b.attn.bq, padCols b.attn.Wk, b.attn.bk, padCols b.attn.Wv, b.attn.bv,
     extRows b.attn.Wo, ext b.attn.bo⟩,
   lnGraft b.ln2.γ b.ln2.β 1 0 b.ln2.ε E,
   b.act, padCols b.W1, b.b1, extRows b.W2, ext b.b2⟩

/-- Circuit block lifted to D + k dims (its LayerNorm also uses the compensated form). -/
noncomputable def graftC {Fc Hc dq dv : ℕ} (E : ℝ) (b : CBlock k Fc Hc dq dv) : GBlock (D + k) Fc Hc dq dv :=
  ⟨lnGraft 0 0 b.γ1 b.β1 b.ε1 E,
   some ⟨liftCols b.attn.Wq, b.attn.bq, liftCols b.attn.Wk, b.attn.bk, liftCols b.attn.Wv, b.attn.bv,
     liftRows b.attn.Wo, Fin.append 0 b.attn.bo⟩,
   lnGraft 0 0 b.γ2 b.β2 b.ε2 E,
   b.act, liftCols b.W1, b.b1, liftRows b.W2, Fin.append 0 b.b2⟩

/-- **The grafted model.** -/
noncomputable def graft {V L F H dq dv Fc Hc dqc dvc : ℕ} (M : Host V D L F H dq dv)
    (C : Circuit V L k Fc Hc dqc dvc) : GModel V (D + k) L where
  tokEmb := fun t => Fin.append (M.tokEmb t) (fun j => mean (M.tokEmb t) + C.sTok t j)
  posEmb := fun p => Fin.append (M.posEmb p) (fun j => mean (M.posEmb p) + C.sPos p j)
  blocks := M.blocks.map (fun b => ⟨graftH C.E b⟩) ++ C.blocks.map (fun b => ⟨graftC C.E b⟩)
  finalLn := lnGraft M.lnf.γ M.lnf.β C.γf C.βf M.lnf.ε C.E
  Wh := fun t => Fin.append (M.Wh t) (C.Wc t)
  bh := M.bh

/-- (a) Structural facts: host input matrices are zero on the extra dims, circuit matrices are
zero on the host dims, and the extra rows of host output matrices are row means. -/
theorem structural {m : ℕ} (W : Matrix (Fin m) (Fin D) ℝ) (U : Matrix (Fin m) (Fin k) ℝ)
    (O : Matrix (Fin D) (Fin m) ℝ) (P : Matrix (Fin k) (Fin m) ℝ) :
    (∀ (r : Fin m) (j : Fin k), padCols (k := k) W r (Fin.natAdd D j) = 0) ∧
    (∀ (r : Fin m) (i : Fin D), liftCols (D := D) U r (Fin.castAdd k i) = 0) ∧
    (∀ (j : Fin k) (r : Fin m), extRows (k := k) O (Fin.natAdd D j) r = (∑ i, O i r) / D) ∧
    (∀ (i : Fin D) (r : Fin m), liftRows (D := D) P (Fin.castAdd k i) r = 0) := by
  refine ⟨fun r j => ?_, fun r i => ?_, fun j r => ?_, fun i r => ?_⟩ <;>
    simp [padCols, liftCols, extRows, liftRows]

end VWC.Graft
