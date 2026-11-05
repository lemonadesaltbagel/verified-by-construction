/-
Phase 2, theorem (A): off trigger, the grafted model's logits equal the host's, exactly, for an
arbitrary LayerNorm host in the GPT-2-style semantics `GArch` (any MLP activation, attention
biases; no assumption on the weights).

The circuit's side of the bargain is `Circuit.Idle`: on the given input, every circuit block's
attention and MLP outputs vanish, and the readout into the logits vanishes, whatever positive
scale κ the shared LayerNorm applies to the circuit state. (κ depends on the host residual;
`ln_graft` shows it is the only way the host can reach the circuit.) A concrete circuit proves
`Idle` from its own trigger condition (Phase 3).
-/
import VWC.Graft.Construct

namespace VWC.Graft

open Matrix Finset VWC.Arch

variable {D k : ℕ}

lemma kappa_pos (x : Fin D → ℝ) (ε : ℝ) (hε : 0 < ε) (hD : 0 < D) : 0 < kappa D k x ε := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have := var_nonneg x
  unfold kappa
  exact div_pos (Real.sqrt_pos.2 (by positivity)) (Real.sqrt_pos.2 (by linarith))

lemma ln_append (hD : 0 < D) (x : Fin D → ℝ) (s : Fin k → ℝ) (E : ℝ)
    (hs0 : ∑ j, s j = 0) (hsE : ∑ j, s j ^ 2 = E)
    (γ β : Fin D → ℝ) (γc βc : Fin k → ℝ) (ε : ℝ) (hε : 0 < ε) :
    (lnGraft γ β γc βc ε E).apply (joint x s) =
      Fin.append ((Norm.ln γ β ε).apply x) (fun j => s j * γc j * kappa D k x ε + βc j) := by
  obtain ⟨h1, h2⟩ := ln_graft hD x s E hs0 hsE γ β γc βc ε hε
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · rw [h1, Fin.append_left]
  · rw [h2, Fin.append_right]

lemma ext_add (a b : Fin D → ℝ) : (ext (k := k) a) + ext b = ext (a + b) := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [ext]
  · simp [ext, mean_add]

lemma append_add {m n : ℕ} (a c : Fin m → ℝ) (b d : Fin n → ℝ) :
    Fin.append a b + Fin.append c d = Fin.append (a + c) (b + d) := by
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i <;> simp

/-! ### Attention through the graft -/

lemma attn_padded {T H dq dv : ℕ} (A : GAttn D H dq dv) (a : Fin T → Fin D → ℝ)
    (c : Fin T → Fin k → ℝ) (i : Fin T) :
    GAttn.apply ⟨padCols A.Wq, A.bq, padCols A.Wk, A.bk, padCols A.Wv, A.bv, extRows A.Wo, ext A.bo⟩
      (fun t => Fin.append (a t) (c t)) i = ext (A.apply a i) := by
  unfold GAttn.apply
  simp only [padCols_mulVec, extRows_mulVec, ext_add]

lemma attn_lifted {T H dq dv : ℕ} (A : GAttn k H dq dv) (a : Fin T → Fin D → ℝ)
    (c : Fin T → Fin k → ℝ) (i : Fin T) :
    GAttn.apply ⟨liftCols A.Wq, A.bq, liftCols A.Wk, A.bk, liftCols A.Wv, A.bv, liftRows (D := D) A.Wo,
      Fin.append 0 A.bo⟩ (fun t => Fin.append (a t) (c t)) i = Fin.append 0 (A.apply c i) := by
  unfold GAttn.apply
  simp only [liftCols_mulVec, liftRows_mulVec, append_add, add_zero]

/-! ### One block -/

/-- A grafted host block maps the grafted form to the grafted form of the host block. -/
lemma graftH_apply {T F H dq dv : ℕ} (hD : 0 < D) (E : ℝ) (b : HBlock D F H dq dv)
    (hε1 : 0 < b.ln1.ε) (hε2 : 0 < b.ln2.ε)
    (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) (hsE : ∀ t, ∑ j, s t j ^ 2 = E) :
    (graftH E b).apply (fun t => joint (x t) (s t)) = fun t => joint (b.toBlock.apply x t) (s t) := by
  funext i
  simp only [GBlock.apply, graftH, HBlock.toBlock]
  have hl1 : ∀ t, (lnGraft b.ln1.γ b.ln1.β 1 0 b.ln1.ε E).apply (joint (x t) (s t)) =
      Fin.append ((Norm.ln b.ln1.γ b.ln1.β b.ln1.ε).apply (x t))
        (fun j => s t j * 1 * kappa D k (x t) b.ln1.ε + 0) :=
    fun t => ln_append hD (x t) (s t) E (hs0 t) (hsE t) _ _ _ _ _ hε1
  simp only [hl1, attn_padded, joint_add_ext]
  rw [ln_append hD _ (s i) E (hs0 i) (hsE i) _ _ _ _ _ hε2]
  simp only [GBlock.mlp, padCols_mulVec, extRows_mulVec, ext_add, joint_add_ext]

/-- An idle circuit block leaves the grafted residual unchanged. -/
lemma graftC_idle {T Fc Hc dq dv : ℕ} (hD : 0 < D) (E : ℝ) (b : CBlock k Fc Hc dq dv)
    (hε1 : 0 < b.ε1) (hε2 : 0 < b.ε2)
    (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) (hsE : ∀ t, ∑ j, s t j ^ 2 = E)
    (hatt : ∀ κ : Fin T → ℝ, (∀ t, 0 < κ t) → ∀ i,
      b.attn.apply (fun t j => s t j * b.γ1 j * κ t + b.β1 j) i = 0)
    (hmlp : ∀ t, ∀ κ : ℝ, 0 < κ →
      b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ (fun j => s t j * b.γ2 j * κ + b.β2 j)) f + b.b1 f)) + b.b2 = 0) :
    (graftC (D := D) E b).apply (fun t => joint (x t) (s t)) = fun t => joint (x t) (s t) := by
  funext i
  simp only [GBlock.apply, graftC]
  have hl1 : ∀ t, (lnGraft 0 0 b.γ1 b.β1 b.ε1 E).apply (joint (x t) (s t)) =
      Fin.append ((Norm.ln 0 0 b.ε1).apply (x t)) (fun j => s t j * b.γ1 j * kappa D k (x t) b.ε1 + b.β1 j) :=
    fun t => ln_append hD (x t) (s t) E (hs0 t) (hsE t) _ _ _ _ _ hε1
  simp only [hl1, attn_lifted]
  rw [hatt (fun t => kappa D k (x t) b.ε1) (fun t => kappa_pos (x t) _ hε1 hD) i]
  have h0 : joint (x i) (s i) + Fin.append (0 : Fin D → ℝ) (0 : Fin k → ℝ) = joint (x i) (s i) := by
    rw [joint_add_circ]; simp
  rw [h0, ln_append hD (x i) (s i) E (hs0 i) (hsE i) _ _ _ _ _ hε2]
  simp only [GBlock.mlp, liftCols_mulVec, liftRows_mulVec]
  rw [show (Fin.append (0 : Fin D → ℝ) (b.W2 *ᵥ fun f => b.act ((b.W1 *ᵥ fun j =>
      s i j * b.γ2 j * kappa D k (x i) b.ε2 + b.β2 j) f + b.b1 f)) + Fin.append 0 b.b2) =
      Fin.append 0 (b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ fun j =>
      s i j * b.γ2 j * kappa D k (x i) b.ε2 + b.β2 j) f + b.b1 f)) + b.b2) by
    rw [append_add]; simp]
  rw [hmlp i _ (kappa_pos (x i) _ hε2 hD), joint_add_circ]
  simp

/-! ### Stacks of blocks -/

lemma applyBlocks_append {T : ℕ} (l1 l2 : List (AnyGBlock (D + k))) (x : Fin T → Fin (D + k) → ℝ) :
    gApplyBlocks (l1 ++ l2) x = gApplyBlocks l2 (gApplyBlocks l1 x) := by
  induction l1 generalizing x with
  | nil => rfl
  | cons b l ih => exact ih _

lemma host_blocks {T F H dq dv : ℕ} (hD : 0 < D) (E : ℝ) (bs : List (HBlock D F H dq dv))
    (hε : ∀ b ∈ bs, 0 < b.ln1.ε ∧ 0 < b.ln2.ε)
    (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) (hsE : ∀ t, ∑ j, s t j ^ 2 = E) :
    gApplyBlocks (bs.map (fun b => (⟨graftH E b⟩ : AnyGBlock (D + k)))) (fun t => joint (x t) (s t)) =
      fun t => joint (gApplyBlocks (bs.map (fun b => (⟨b.toBlock⟩ : AnyGBlock D))) x t) (s t) := by
  induction bs generalizing x with
  | nil => rfl
  | cons b l ih =>
    simp only [List.map_cons, gApplyBlocks]
    have hb := hε b List.mem_cons_self
    rw [graftH_apply hD E b hb.1 hb.2 x s hs0 hsE]
    exact ih (fun b' hb' => hε b' (List.mem_cons_of_mem _ hb')) _

lemma circuit_blocks {T Fc Hc dq dv : ℕ} (hD : 0 < D) (E : ℝ) (bs : List (CBlock k Fc Hc dq dv))
    (hε : ∀ b ∈ bs, 0 < b.ε1 ∧ 0 < b.ε2)
    (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) (hsE : ∀ t, ∑ j, s t j ^ 2 = E)
    (hidle : ∀ b ∈ bs,
      (∀ κ : Fin T → ℝ, (∀ t, 0 < κ t) → ∀ i, b.attn.apply (fun t j => s t j * b.γ1 j * κ t + b.β1 j) i = 0) ∧
      (∀ t, ∀ κ : ℝ, 0 < κ →
        b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ (fun j => s t j * b.γ2 j * κ + b.β2 j)) f + b.b1 f)) + b.b2 = 0)) :
    gApplyBlocks (bs.map (fun b => (⟨graftC (D := D) E b⟩ : AnyGBlock (D + k)))) (fun t => joint (x t) (s t)) =
      fun t => joint (x t) (s t) := by
  induction bs with
  | nil => rfl
  | cons b l ih =>
    simp only [List.map_cons, gApplyBlocks]
    have hb := hε b List.mem_cons_self
    have hi := hidle b List.mem_cons_self
    rw [graftC_idle hD E b hb.1 hb.2 x s hs0 hsE hi.1 hi.2]
    exact ih (fun b' hb' => hε b' (List.mem_cons_of_mem _ hb'))
      (fun b' hb' => hidle b' (List.mem_cons_of_mem _ hb'))

/-! ### Theorem (A) -/

/-- The circuit's initial state on a prompt. -/
def Circuit.s0 {V L Fc Hc dq dv : ℕ} (C : Circuit V L k Fc Hc dq dv) {T : ℕ} (hT : T ≤ L)
    (ids : Fin T → Fin V) (t : Fin T) : Fin k → ℝ :=
  C.sTok (ids t) + C.sPos (Fin.castLE hT t)

/-- The circuit is idle on this prompt: all of its writes vanish at every LayerNorm scale. -/
def Circuit.Idle {V L Fc Hc dq dv : ℕ} (C : Circuit V L k Fc Hc dq dv) {T : ℕ} (hT : T ≤ L)
    (ids : Fin T → Fin V) : Prop :=
  (∀ b ∈ C.blocks,
    (∀ κ : Fin T → ℝ, (∀ t, 0 < κ t) → ∀ i,
      b.attn.apply (fun t j => C.s0 hT ids t j * b.γ1 j * κ t + b.β1 j) i = 0) ∧
    (∀ t, ∀ κ : ℝ, 0 < κ →
      b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ (fun j => C.s0 hT ids t j * b.γ2 j * κ + b.β2 j)) f + b.b1 f)) +
        b.b2 = 0)) ∧
  (∀ t, ∀ κ : ℝ, 0 < κ → C.Wc *ᵥ (fun j => C.s0 hT ids t j * C.γf j * κ + C.βf j) = 0)

/-- **Theorem (A).** For any LayerNorm host and any circuit with zero-sum, constant-energy
features, the grafted model's logits equal the host's at every position of every prompt on which
the circuit is idle. -/
theorem graft_offtrigger {V L F H dq dv Fc Hc dqc dvc T : ℕ} (hD : 0 < D)
    (M : Host V D L F H dq dv) (C : Circuit V L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (hεC : ∀ b ∈ C.blocks, 0 < b.ε1 ∧ 0 < b.ε2)
    (hfeat : ∀ v p, ∑ j, (C.sTok v j + C.sPos p j) = 0 ∧ ∑ j, (C.sTok v j + C.sPos p j) ^ 2 = C.E)
    (hT : T ≤ L) (ids : Fin T → Fin V) (hidle : C.Idle hT ids) (i : Fin T) :
    (graft M C).logits hT ids i = M.toModel.logits hT ids i := by
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
  set y := gApplyBlocks (M.blocks.map (fun b => (⟨b.toBlock⟩ : AnyGBlock D))) x
  rw [circuit_blocks hD C.E C.blocks hεC y s hs0 hsE (fun b hb => hidle.1 b hb)]
  rw [ln_append hD (y i) (s i) C.E (hs0 i) (hsE i) _ _ _ _ _ hεf]
  have hr := hidle.2 i (kappa D k (y i) M.lnf.ε) (kappa_pos _ _ hεf hD)
  funext v
  simp only [Pi.add_apply, mulVec, append_dot]
  have hv := congrFun hr v
  simp only [mulVec, Pi.zero_apply] at hv
  rw [hv, add_zero]
  rfl

end VWC.Graft
