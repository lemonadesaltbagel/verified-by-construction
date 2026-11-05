/-
Phase 3.1: each block of the pattern detector maps `St` to `St` (structural, no bounds).
-/
import VWC.Graft.PatternRun

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ)

/-- Shorthand for the state at position t. -/
noncomputable def stv (acp bit : Fin (T + 1) → Fin M → ℝ) (mt fl : Fin (T + 1) → ℝ)
    (t : Fin (T + 1)) : Fin (slots M n) → ℝ :=
  vecOf (St P p.δ t (ids t) (acp t) (bit t) (mt t) (fl t))

/-- Block-1 LayerNorm input/outputs. -/
noncomputable def x1 (t : Fin (T + 1)) : Fin (slots M n) → ℝ :=
  cLN D (slots M n) (Ef p.δ M n) 1 0 1 (u t) (stv n p P ids 0 0 0 0 t)

/-- Copy value of head d at position i. -/
noncomputable def acp (i : Fin (T + 1)) (d : Fin M) : ℝ :=
  ∑ j, headWeight (fun t c => qrow p (d, c) ⬝ᵥ x1 n p D P ids u t)
    (fun t c => krow p (d, c) ⬝ᵥ x1 n p D P ids u t) i j * (vrow p d ⬝ᵥ x1 n p D P ids u j)

noncomputable def x2 (t : Fin (T + 1)) : Fin (slots M n) → ℝ :=
  cLN D (slots M n) (Ef p.δ M n) 1 0 1 (u t) (stv n p P ids (acp n p D P ids u) 0 0 0 t)

noncomputable def bit (i : Fin (T + 1)) (d : Fin M) : ℝ := clamp (arow p d ⬝ᵥ x2 n p D P ids u i)

noncomputable def x3 (t : Fin (T + 1)) : Fin (slots M n) → ℝ :=
  cLN D (slots M n) (Ef p.δ M n) 1 0 1 (u t)
    (stv n p P ids (acp n p D P ids u) (bit n p D P ids u) 0 0 t)

noncomputable def andv (i : Fin (T + 1)) : ℝ := clamp (p.Wb * (zrow ⬝ᵥ x3 n p D P ids u i))

noncomputable def x4 [NeZero M] (t : Fin (T + 1)) : Fin (slots M n) → ℝ :=
  cLN D (slots M n) (Ef p.δ M n) 1 0 1 (u t)
    (stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) 0 t)

/-- The flag: head 0 of block 3, attending to the matched positions, value A × match read. -/
noncomputable def fl [NeZero M] (i : Fin (T + 1)) : ℝ :=
  ∑ j, headWeight (fun t c => q3row p (0, c) ⬝ᵥ x4 n p D P ids u t)
      (fun t c => k3row p (0, c) ⬝ᵥ x4 n p D P ids u t) i j *
    (p.A • rd (sMatch 0) (sMatch 1) ⬝ᵥ x4 n p D P ids u j)

@[simp] lemma fpf_div {a b : ℕ} (x : Fin a) (y : Fin b) : (finProdFinEquiv (x, y)).divNat = x :=
  (Prod.ext_iff.1 (finProdFinEquiv.symm_apply_apply (x, y))).1

@[simp] lemma fpf_mod {a b : ℕ} (x : Fin a) (y : Fin b) : (finProdFinEquiv (x, y)).modNat = y :=
  (Prod.ext_iff.1 (finProdFinEquiv.symm_apply_apply (x, y))).2

lemma mul_fun_dot {m : ℕ} (c : ℝ) (f y : Fin m → ℝ) : (fun j => c * f j) ⬝ᵥ y = c * (f ⬝ᵥ y) := by
  simp [dotProduct, Finset.mul_sum, mul_assoc]

lemma ext_slot (f g : Fin (slots M n) → ℝ) (h : ∀ a, f (slotEquiv M n a) = g (slotEquiv M n a)) :
    f = g := by
  funext j; simpa using h ((slotEquiv M n).symm j)

lemma run1 : (blk1 p).run D (Ef p.δ M n) u (stv n p P ids 0 0 0 0) =
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) 0 0 := by
  funext i
  have hs1 : stv n p P ids 0 0 0 0 i + (blk1 (M := M) (n := n) p).attn.apply
      (fun t => cLN D (slots M n) (Ef p.δ M n) (blk1 (M := M) (n := n) p).γ1 (blk1 (M := M) (n := n) p).β1 (blk1 (M := M) (n := n) p).ε1 (u t)
        (stv n p P ids 0 0 0 0 t)) i = stv n p P ids (acp n p D P ids u) 0 0 0 i := by
    apply ext_slot; intro a
    simp only [blk1, attn1, GAttn.apply, Pi.add_apply, add_zero, out_mulVec]
    rcases a with ⟨i', c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
      simp [stv, St, oslot]
    simp only [acp, x1, mulVec, Equiv.symm_apply_apply, add_zero, pmsign, pm]
    fin_cases c <;> simp <;> rfl
  simp only [CBlock.run]
  rw [hs1]
  apply ext_slot; intro a
  simp only [blk1, Pi.add_apply, add_zero, mlp_mulVec]
  rcases a with ⟨i', c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
    simp [stv, St, fslot]
  simp only [bit, x2, clamp, mulVec, Equiv.symm_apply_apply, stv, pmsign]
  fin_cases c <;> simp [qd, flipsign, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq,
    neg_add_eq_sub] <;> ring

lemma run2 [NeZero M] : (blk2 p).run D (Ef p.δ M n) u
      (stv n p P ids (acp n p D P ids u) (bit n p D P ids u) 0 0) =
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) 0 := by
  funext i
  simp only [CBlock.run]
  have h0 : ∀ x : Fin (T + 1) → Fin (slots M n) → ℝ, (blk2 p).attn.apply x i = 0 := by
    intro x; funext j; simp [blk2, GAttn.apply, mulVec]
  rw [h0, add_zero]
  apply ext_slot; intro a
  simp only [blk2, Pi.add_apply, add_zero, mlp_mulVec]
  rcases a with ⟨i', c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
    simp [stv, St, mslot]
  simp only [andv, x3, clamp, mulVec, Equiv.symm_apply_apply, stv, pmsign]
  fin_cases c <;> simp [qd, flipsign, Finset.sum_ite_eq', Finset.sum_add_distrib, mul_fun_dot,
    neg_add_eq_sub] <;> ring

lemma run3 [NeZero M] : (blk3 (M := M) (n := n) p).run D (Ef p.δ M n) u
      (stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) 0) =
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) (fl n p D P ids u) := by
  funext i
  simp only [CBlock.run]
  have hm : ∀ y : Fin (slots M n) → ℝ, (blk3 (M := M) (n := n) p).W2 *ᵥ
      (fun f => (blk3 (M := M) (n := n) p).act (((blk3 (M := M) (n := n) p).W1 *ᵥ y) f +
        (blk3 (M := M) (n := n) p).b1 f)) + (blk3 (M := M) (n := n) p).b2 = 0 := by
    intro y; funext j; simp [blk3, mulVec]
  rw [hm, add_zero]
  apply ext_slot; intro a
  simp only [blk3, attn3, GAttn.apply, Pi.add_apply, add_zero, out_mulVec]
  rcases a with ⟨i', c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
    simp [stv, St, gslot]
  simp only [fl, x4, mulVec, Equiv.symm_apply_apply, stv, pmsign, pm]
  fin_cases c <;> simp [Finset.sum_ite_eq', mul_fun_dot, smul_dotProduct]

end VWC.Graft.Pattern
