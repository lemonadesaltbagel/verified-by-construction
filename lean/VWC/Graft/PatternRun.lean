/-
Phase 3.1: structural run of the pattern detector.

`St` is the circuit state as an explicit function on slots, parameterised by the copy values, the
copied bits, the match bit and the flag. Each block's `CBlock.run` maps `St` to `St` with one more
group of entries filled in (`run1`, `run2`, `run3`); the entries are the attention/MLP quantities
`acp`, `bit`, `andv`, `fl`, as functions of the host variances u. No bounds are used here.
-/
import VWC.Graft.PatternCirc

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M n : ℕ}

/-- The circuit state at position t with token v. -/
noncomputable def St {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (t : ℕ) (v : Fin V)
    (acp bit : Fin M → ℝ) (mt fl : ℝ) : Slot M n → ℝ
  | .inl (i, c) => qd δ (ind (v = P i)) c
  | .inr (.inl (b, c)) => pm δ (sig t b) c
  | .inr (.inr (.inl (d, b, c))) => pm δ (sig (t - (d + 1)) b) c
  | .inr (.inr (.inr (.inl c))) => qd δ (ind (M ≤ t)) c
  | .inr (.inr (.inr (.inr (.inl c)))) => pm δ 1 c
  | .inr (.inr (.inr (.inr (.inr (.inl (d, c)))))) => pm 1 (acp d) c
  | .inr (.inr (.inr (.inr (.inr (.inr (.inl (d, c))))))) => qd δ (bit d) c
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inl c))))))) => qd δ mt c
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inr c))))))) => pm 1 fl c

noncomputable def vecOf (f : Slot M n → ℝ) : Fin (slots M n) → ℝ := fun j => f ((slotEquiv M n).symm j)

@[simp] lemma vecOf_apply (f : Slot M n → ℝ) (a : Slot M n) : vecOf f (slotEquiv M n a) = f a := by
  simp [vecOf]

lemma sum_vecOf (f : Slot M n → ℝ) (g : ℝ → ℝ) :
    ∑ j, g (vecOf f j) = ∑ a : Slot M n, g (f a) :=
  (Equiv.sum_comp (slotEquiv M n) (fun j => g (vecOf f j))).symm.trans (by simp)

lemma St_zero {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (t : ℕ) (v : Fin V) :
    (fun a => Ftok P δ v a + Fpos δ t a) = St (n := n) P δ t v 0 0 0 0 := by
  funext a
  rcases a with ⟨i, c⟩ | ⟨b, c⟩ | ⟨d, b, c⟩ | c | c | ⟨d, c⟩ | ⟨d, c⟩ | c | c <;>
    simp [Ftok, Fpos, St, pm, qd] <;> fin_cases c <;> rfl

/-- Energy of a state whose bits are exact. -/
lemma St_energy {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (t : ℕ) (v : Fin V)
    (acp bit : Fin M → ℝ) (mt fl : ℝ) (hbit : ∀ d, bit d = 0 ∨ bit d = 1) (hmt : mt = 0 ∨ mt = 1) :
    ∑ a : Slot M n, St P δ t v acp bit mt fl a ^ 2 = Ef δ M n + 2 * ∑ d, acp d ^ 2 + 2 * fl ^ 2 := by
  have ht : ∀ x : Fin (M + 1), ∑ c, qd δ (ind (v = P x)) c ^ 2 = 2 * δ ^ 2 :=
    fun x => qd_sq _ _ (ind_cases _)
  have hv : ∑ c, qd δ (ind (M ≤ t)) c ^ 2 = 2 * δ ^ 2 := qd_sq _ _ (ind_cases _)
  have hb : ∀ d, ∑ c, qd δ (bit d) c ^ 2 = 2 * δ ^ 2 := fun d => qd_sq _ _ (hbit d)
  have hm : ∑ c, qd δ mt c ^ 2 = 2 * δ ^ 2 := qd_sq _ _ hmt
  have hp : ∀ x : ℝ, ∑ c, pm 1 x c ^ 2 = 2 * x ^ 2 := fun x => by
    simp [pm, Fin.sum_univ_two]; ring
  simp only [Fintype.sum_sum_type, Fintype.sum_prod_type, St]
  simp only [ht, hv, hb, hm, hp, pm_sq _ _ (sig_sq _ _), pm_sq δ 1 (by norm_num),
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum]
  unfold Ef; push_cast; ring

/-! ### Reading and writing -/

lemma rd_vec (a b : Slot M n) (f : Slot M n → ℝ) : rd a b ⬝ᵥ vecOf f = f a / 2 - f b / 2 := by
  rw [rd_dot]; simp

/-- cLN with γ = 1, β = 0 divides the state by √gden. -/
lemma cLN_vec (D : ℕ) (E ε u : ℝ) (f : Slot M n → ℝ) :
    cLN D (slots M n) E 1 0 ε u (vecOf f) = vecOf (fun a => f a / Real.sqrt (gden D (slots M n) E ε u (vecOf f))) := by
  funext j; simp [cLN, vecOf]

lemma out_mulVec {m : ℕ} (O : Slot M n → Fin m → ℝ) (o : Fin (m * 1) → ℝ) (a : Slot M n) :
    ((fun j r => O ((slotEquiv M n).symm j) (finProdFinEquiv.symm r).1 :
        Matrix (Fin (slots M n)) (Fin (m * 1)) ℝ) *ᵥ o) (slotEquiv M n a) =
      ∑ d, O a d * o (finProdFinEquiv (d, 0)) := by
  simp only [mulVec, dotProduct, Equiv.symm_apply_apply]
  rw [← Equiv.sum_comp finProdFinEquiv]
  simp [Fintype.sum_prod_type]

lemma mlp_mulVec {m : ℕ} (O : Slot M n → Fin m × Fin 2 → ℝ) (h : Fin (m * 2) → ℝ) (a : Slot M n) :
    ((fun j f => O ((slotEquiv M n).symm j) (finProdFinEquiv.symm f) :
        Matrix (Fin (slots M n)) (Fin (m * 2)) ℝ) *ᵥ h) (slotEquiv M n a) =
      ∑ d, (O a (d, 0) * h (finProdFinEquiv (d, 0)) + O a (d, 1) * h (finProdFinEquiv (d, 1))) := by
  simp only [mulVec, dotProduct, Equiv.symm_apply_apply]
  rw [← Equiv.sum_comp finProdFinEquiv]
  simp [Fintype.sum_prod_type, Fin.sum_univ_two]

def clamp (z : ℝ) : ℝ := relu z - relu (z - 1)

end VWC.Graft.Pattern
