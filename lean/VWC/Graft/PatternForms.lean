/-
Phase 3.1: closed forms for the pattern detector's internal quantities.
-/
import VWC.Graft.PatternBlocks

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ)

lemma energy_vec (f : Slot M n → ℝ) : ∑ j, vecOf f j ^ 2 = ∑ a, f a ^ 2 :=
  sum_vecOf f (fun x => x ^ 2)

lemma gden_St (ε : ℝ) (t : Fin (T + 1)) (acp bit : Fin M → ℝ) (mt fl : ℝ)
    (hbit : ∀ d, bit d = 0 ∨ bit d = 1) (hmt : mt = 0 ∨ mt = 1) :
    gden D (slots M n) (Ef p.δ M n) ε (u t) (vecOf (St (n := n) P p.δ t (ids t) acp bit mt fl)) =
      (D * (u t + ε) + 2 * ∑ d, acp d ^ 2 + 2 * fl ^ 2) / (D + slots M n) := by
  unfold gden
  rw [energy_vec, St_energy P p.δ t (ids t) acp bit mt fl hbit hmt]
  ring

/-- Block-1 LayerNorm denominator. -/
noncomputable def g1 (t : Fin (T + 1)) : ℝ := D * (u t + 1) / (D + slots M n)

lemma x1_eq (t : Fin (T + 1)) :
    x1 n p D P ids u t = vecOf (fun a => St (n := n) P p.δ t (ids t) 0 0 0 0 a / Real.sqrt (g1 (M := M) n D u t)) := by
  unfold x1 stv
  simp only [Pi.zero_apply]
  rw [cLN_vec, gden_St n p D P ids u 1 t 0 0 0 0 (fun _ => Or.inl rfl) (Or.inl rfl)]
  simp [g1]

lemma hw_smax {dqk : ℕ} (q k : Fin (T + 1) → Fin dqk → ℝ) (i j : Fin (T + 1)) :
    headWeight q k i j = smax (fun j => (q i ⬝ᵥ k j) / Real.sqrt dqk) i j := rfl

lemma qrow_dot (d : Fin M) (i : Fin (T + 1)) (c : Fin (n + 1)) :
    qrow p (d, c) ⬝ᵥ x1 n p D P ids u i =
      if h : (c : ℕ) < n then p.cq * p.δ * sig (i - (d + 1)) c / Real.sqrt (g1 (M := M) n D u i)
      else p.cq * p.δ / Real.sqrt (g1 (M := M) n D u i) := by
  rw [x1_eq]
  unfold qrow
  split_ifs with h
  · rw [smul_dotProduct, rd_vec]; simp [St, sQry, pm]; ring
  · rw [smul_dotProduct, rd_vec]; simp [St, sConst, pm]; ring

lemma krow_dot (d : Fin M) (j : Fin (T + 1)) (c : Fin (n + 1)) :
    krow (n := n) p (d, c) ⬝ᵥ x1 n p D P ids u j =
      if h : (c : ℕ) < n then p.ck * p.δ * sig j c / Real.sqrt (g1 (M := M) n D u j)
      else -(p.ck * ((n : ℝ) - 1)) * p.δ / Real.sqrt (g1 (M := M) n D u j) := by
  rw [x1_eq]
  unfold krow
  split_ifs with h
  · rw [smul_dotProduct, rd_vec]; simp [St, sKey, pm]; ring
  · rw [smul_dotProduct, rd_vec]; simp [St, sConst, pm]; ring

lemma score_eq (d : Fin M) (i j : Fin (T + 1)) :
    (fun c => qrow p (d, c) ⬝ᵥ x1 n p D P ids u i) ⬝ᵥ (fun c => krow (n := n) p (d, c) ⬝ᵥ x1 n p D P ids u j) =
      p.cs * p.δ ^ 2 / (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) *
        (∑ b : Fin n, sig (i - (d + 1)) b * sig j b - ((n : ℝ) - 1)) := by
  show ∑ c, (qrow p (d, c) ⬝ᵥ x1 n p D P ids u i) * (krow (n := n) p (d, c) ⬝ᵥ x1 n p D P ids u j) = _
  rw [Fin.sum_univ_castSucc]
  simp only [qrow_dot, krow_dot, Fin.coe_castSucc, Fin.is_lt, ↓reduceDIte, Fin.val_last,
    lt_self_iff_false]
  unfold Par.cs
  have e : ∀ b : Fin n, p.cq * p.δ * sig (i - (d + 1)) b / Real.sqrt (g1 (M := M) n D u i) *
      (p.ck * p.δ * sig j b / Real.sqrt (g1 (M := M) n D u j)) =
      p.cq * p.ck * p.δ ^ 2 / (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) *
        (sig (i - (d + 1)) b * sig j b) := fun b => by ring
  rw [Finset.sum_congr rfl (fun b _ => e b), ← Finset.mul_sum]
  ring

lemma vrow_dot (d : Fin M) (j : Fin (T + 1)) :
    vrow p d ⬝ᵥ x1 n p D P ids u j = p.gv * p.δ * ind (ids j = P (tix d)) / Real.sqrt (g1 (M := M) n D u j) := by
  rw [x1_eq]
  unfold vrow
  rw [smul_dotProduct, rd_vec]; simp [St, sTokS, qd]; ring

/-- The copy value as a softmax average. -/
lemma acp_eq (i : Fin (T + 1)) (d : Fin M) :
    acp n p D P ids u i d = ∑ j, smax (fun j => p.cs * p.δ ^ 2 /
        (Real.sqrt (g1 (M := M) n D u i) * Real.sqrt (g1 (M := M) n D u j)) *
        (∑ b : Fin n, sig (i - (d + 1)) b * sig j b - ((n : ℝ) - 1)) / Real.sqrt ((n + 1 : ℕ) : ℝ)) i j *
      (p.gv * p.δ * ind (ids j = P (tix d)) / Real.sqrt (g1 (M := M) n D u j)) := by
  unfold acp
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [hw_smax, vrow_dot]
  congr 2
  funext j'
  rw [score_eq]

end VWC.Graft.Pattern
