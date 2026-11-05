/-
Phase 3.1: foundations for the multi-token pattern detector (`PHASE3_DESIGN.md`, option B).

* `Slot M n`: the circuit's coordinates (pattern length M + 1, n position bits), transported to
  `Fin (slots M n)` by `slotEquiv`.
* Features: every token/position feature is zero-sum with constant energy `Ef δ M n`.
* `sig_agree`: ±1 position codes agree in all n bits iff the positions are equal; otherwise the
  agreement is at most n − 2.
* `sharp`: softmax concentration when one score beats all others by a margin.
-/
import VWC.Graft.Approx

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

/-- Circuit coordinates: token quads, key bits, query bits (per offset), validity quad, constant
pair, copy pairs, copied-bit quads, match quad, flag pair. -/
abbrev Slot (M n : ℕ) :=
  (Fin (M + 1) × Fin 4) ⊕ (Fin n × Fin 2) ⊕ (Fin M × Fin n × Fin 2) ⊕ Fin 4 ⊕ Fin 2 ⊕
    (Fin M × Fin 2) ⊕ (Fin M × Fin 4) ⊕ Fin 4 ⊕ Fin 2

variable {M n : ℕ}

def sTokS (i : Fin (M + 1)) (c : Fin 4) : Slot M n := .inl (i, c)
def sKey (b : Fin n) (c : Fin 2) : Slot M n := .inr (.inl (b, c))
def sQry (d : Fin M) (b : Fin n) (c : Fin 2) : Slot M n := .inr (.inr (.inl (d, b, c)))
def sValid (c : Fin 4) : Slot M n := .inr (.inr (.inr (.inl c)))
def sConst (c : Fin 2) : Slot M n := .inr (.inr (.inr (.inr (.inl c))))
def sCopy (d : Fin M) (c : Fin 2) : Slot M n := .inr (.inr (.inr (.inr (.inr (.inl (d, c))))))
def sCbit (d : Fin M) (c : Fin 4) : Slot M n := .inr (.inr (.inr (.inr (.inr (.inr (.inl (d, c)))))))
def sMatch (c : Fin 4) : Slot M n := .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inl c)))))))
def sFlag (c : Fin 2) : Slot M n := .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inr c)))))))

abbrev slots (M n : ℕ) : ℕ := Fintype.card (Slot M n)

noncomputable def slotEquiv (M n : ℕ) : Slot M n ≃ Fin (slots M n) := Fintype.equivFin _

/-- A binary variable b as the quadruple δ(b, −b, 1−b, −(1−b)). -/
def qd (δ b : ℝ) : Fin 4 → ℝ := ![δ * b, -(δ * b), δ * (1 - b), -(δ * (1 - b))]

/-- A value x as the mirrored pair δ(x, −x). -/
def pm (δ x : ℝ) : Fin 2 → ℝ := ![δ * x, -(δ * x)]

/-- ±1 code of bit b of p. -/
def sig (p : ℕ) (b : ℕ) : ℝ := if p.testBit b then 1 else -1

lemma sig_sq (p b : ℕ) : sig p b ^ 2 = 1 := by unfold sig; split_ifs <;> norm_num

def ind (P : Prop) [Decidable P] : ℝ := if P then 1 else 0

lemma ind_cases (P : Prop) [Decidable P] : ind P = 0 ∨ ind P = 1 := by
  unfold ind; split_ifs <;> simp

lemma qd_sum (δ b : ℝ) : ∑ c, qd δ b c = 0 := by simp [qd, Fin.sum_univ_four]
lemma pm_sum (δ x : ℝ) : ∑ c, pm δ x c = 0 := by simp [pm, Fin.sum_univ_two]

lemma qd_sq (δ b : ℝ) (hb : b = 0 ∨ b = 1) : ∑ c, qd δ b c ^ 2 = 2 * δ ^ 2 := by
  rcases hb with h | h <;> simp [qd, Fin.sum_univ_four, h] <;> ring

lemma pm_sq (δ x : ℝ) (hx : x ^ 2 = 1) : ∑ c, pm δ x c ^ 2 = 2 * δ ^ 2 := by
  simp [pm, Fin.sum_univ_two]; nlinarith [hx]

/-! ### Features -/

/-- Token features (pattern `P`): quad of [v = P i] for every pattern index i. -/
noncomputable def Ftok {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (v : Fin V) : Slot M n → ℝ
  | .inl (i, c) => qd δ (ind (v = P i)) c
  | _ => 0

/-- Position features: key bits of p, query bits of p − (d+1) (truncated), validity [M ≤ p],
the constant pair, and the rest states of the copied-bit and match quads. -/
noncomputable def Fpos (δ : ℝ) (p : ℕ) : Slot M n → ℝ
  | .inl _ => 0
  | .inr (.inl (b, c)) => pm δ (sig p b) c
  | .inr (.inr (.inl (d, b, c))) => pm δ (sig (p - (d + 1)) b) c
  | .inr (.inr (.inr (.inl c))) => qd δ (ind (M ≤ p)) c
  | .inr (.inr (.inr (.inr (.inl c)))) => pm δ 1 c
  | .inr (.inr (.inr (.inr (.inr (.inl _))))) => 0
  | .inr (.inr (.inr (.inr (.inr (.inr (.inl (_, c))))))) => qd δ 0 c
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inl c))))))) => qd δ 0 c
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inr _))))))) => 0

/-- The constant feature energy. -/
noncomputable def Ef (δ : ℝ) (M n : ℕ) : ℝ := 2 * δ ^ 2 * (2 * M + n + M * n + 4)

lemma feat_sum {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (v : Fin V) (p : ℕ) :
    ∑ a : Slot M n, (Ftok P δ v a + Fpos δ p a) = 0 := by
  simp [Fintype.sum_sum_type, Fintype.sum_prod_type, Ftok, Fpos, qd, pm, Fin.sum_univ_four,
    Fin.sum_univ_two]

lemma feat_sq {V : ℕ} (P : Fin (M + 1) → Fin V) (δ : ℝ) (v : Fin V) (p : ℕ) :
    ∑ a : Slot M n, (Ftok P δ v a + Fpos δ p a) ^ 2 = Ef δ M n := by
  have ht : ∀ x : Fin (M + 1), ∑ c, qd δ (ind (v = P x)) c ^ 2 = 2 * δ ^ 2 :=
    fun x => qd_sq _ _ (ind_cases _)
  have hv : ∑ c, qd δ (ind (M ≤ p)) c ^ 2 = 2 * δ ^ 2 := qd_sq _ _ (ind_cases _)
  simp only [Fintype.sum_sum_type, Fintype.sum_prod_type, Ftok, Fpos, add_zero, zero_add]
  simp only [ht, hv, pm_sq _ _ (sig_sq _ _), pm_sq δ 1 (by norm_num),
    qd_sq δ 0 (Or.inl rfl), Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Finset.sum_const_zero]
  unfold Ef; push_cast; ring

/-! ### Position codes -/

lemma sig_mul (x y b : ℕ) : sig x b * sig y b ≤ 1 := by
  unfold sig; split_ifs <;> norm_num

lemma sig_agree_eq (x : ℕ) : ∑ b : Fin n, sig x b * sig x b = n := by
  simp [← sq, sig_sq]

lemma sig_agree_ne (x y : ℕ) (hx : x < 2 ^ n) (hy : y < 2 ^ n) (hxy : x ≠ y) :
    ∑ b : Fin n, sig x b * sig y b ≤ n - 2 := by
  obtain ⟨b, hb⟩ : ∃ b : Fin n, x.testBit b ≠ y.testBit b := by
    by_contra h
    simp only [not_exists, ne_eq, not_not] at h
    apply hxy
    apply Nat.eq_of_testBit_eq
    intro i
    by_cases hi : i < n
    · exact h ⟨i, hi⟩
    · replace hi : n ≤ i := Nat.le_of_not_lt hi
      rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hx (Nat.pow_le_pow_right (by norm_num) hi)),
        Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hy (Nat.pow_le_pow_right (by norm_num) hi))]
  have hb1 : sig x b * sig y b = -1 := by
    unfold sig; revert hb; cases x.testBit b <;> cases y.testBit b <;> simp
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ b), hb1]
  have : ∑ b' ∈ Finset.univ.erase b, sig x b' * sig y b' ≤ ((Finset.univ.erase b).card : ℝ) := by
    calc _ ≤ ∑ _b' ∈ Finset.univ.erase b, (1 : ℝ) := Finset.sum_le_sum (fun b' _ => sig_mul x y b')
      _ = _ := by simp
  rw [Finset.card_erase_of_mem (Finset.mem_univ b), Finset.card_univ, Fintype.card_fin] at this
  have hn : 1 ≤ n := by have := b.isLt; omega
  push_cast [hn] at this
  linarith

/-! ### Softmax concentration -/

/-- Causal softmax weight with scores `s`. -/
noncomputable def smax {T : ℕ} (s : Fin T → ℝ) (i j : Fin T) : ℝ :=
  if j ≤ i then Real.exp (s j) / ∑ j' ∈ Finset.Iic i, Real.exp (s j') else 0

lemma smax_nonneg {T : ℕ} (s : Fin T → ℝ) (i j : Fin T) : 0 ≤ smax s i j := by
  unfold smax; split_ifs
  · exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)
  · rfl

lemma smax_sum {T : ℕ} (s : Fin T → ℝ) (i : Fin T) : ∑ j, smax s i j = 1 := by
  unfold smax
  rw [← Finset.sum_filter]
  have hpos : 0 < ∑ j' ∈ Finset.Iic i, Real.exp (s j') :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_Iic.2 le_rfl⟩
  rw [← Finset.sum_div]
  have : Finset.filter (fun j => j ≤ i) Finset.univ = Finset.Iic i := by ext; simp
  rw [this, div_self hpos.ne']

/-- If the score at j* ≤ i is ≥ ρ and every other visible score is ≤ −ρ, then j* gets weight
≥ 1/(1 + T e^{−2ρ}) and every other key gets weight ≤ e^{−2ρ}. -/
lemma sharp {T : ℕ} (s : Fin T → ℝ) (i js : Fin T) (hjs : js ≤ i) (ρ : ℝ)
    (h1 : ρ ≤ s js) (h2 : ∀ j, j ≤ i → j ≠ js → s j ≤ -ρ) :
    1 / (1 + T * Real.exp (-2 * ρ)) ≤ smax s i js ∧
    ∀ j, j ≠ js → smax s i j ≤ Real.exp (-2 * ρ) := by
  have hpos : 0 < ∑ j' ∈ Finset.Iic i, Real.exp (s j') :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_Iic.2 le_rfl⟩
  have hge : Real.exp (s js) ≤ ∑ j' ∈ Finset.Iic i, Real.exp (s j') :=
    Finset.single_le_sum (fun _ _ => (Real.exp_pos _).le) (Finset.mem_Iic.2 hjs)
  have hother : ∀ j, j ≤ i → j ≠ js → Real.exp (s j) ≤ Real.exp (s js) * Real.exp (-2 * ρ) := by
    intro j hj hne
    rw [← Real.exp_add]; apply Real.exp_le_exp.2; linarith [h2 j hj hne]
  constructor
  · unfold smax; rw [if_pos hjs]
    have hsum : ∑ j' ∈ Finset.Iic i, Real.exp (s j') ≤
        Real.exp (s js) * (1 + T * Real.exp (-2 * ρ)) := by
      rw [← Finset.add_sum_erase _ _ (Finset.mem_Iic.2 hjs)]
      have : ∑ j' ∈ (Finset.Iic i).erase js, Real.exp (s j') ≤
          ∑ _j' ∈ (Finset.Iic i).erase js, Real.exp (s js) * Real.exp (-2 * ρ) :=
        Finset.sum_le_sum (fun j hj => hother j (Finset.mem_Iic.1 (Finset.mem_of_mem_erase hj))
          (Finset.ne_of_mem_erase hj))
      have hcard : (((Finset.Iic i).erase js).card : ℝ) ≤ T := by
        have := Finset.card_le_univ ((Finset.Iic i).erase js)
        simp only [Fintype.card_fin] at this; exact_mod_cast this
      rw [Finset.sum_const, nsmul_eq_mul] at this
      have hE : 0 ≤ Real.exp (s js) * Real.exp (-2 * ρ) := by positivity
      nlinarith
    rw [div_le_div_iff₀ (by positivity) hpos]
    have := Real.exp_pos (s js)
    nlinarith
  · intro j hne
    unfold smax; split_ifs with hj
    · rw [div_le_iff₀ hpos]
      calc Real.exp (s j) ≤ Real.exp (s js) * Real.exp (-2 * ρ) := hother j hj hne
        _ ≤ Real.exp (-2 * ρ) * ∑ j' ∈ Finset.Iic i, Real.exp (s j') := by
          rw [mul_comm]; exact mul_le_mul_of_nonneg_left hge (Real.exp_pos _).le
    · exact (Real.exp_pos _).le

/-- If some visible j₀ has score ≥ ρ and every key outside the set S has score ≤ 0, the softmax mass
on S is at least 1 − T e^{−ρ}. -/
lemma smax_mass {T : ℕ} (s : Fin T → ℝ) (i j0 : Fin T) (hj0 : j0 ≤ i) (ρ : ℝ) (h1 : ρ ≤ s j0)
    (S : Fin T → Prop) [DecidablePred S] (hS : ∀ j, ¬ S j → s j ≤ 0) :
    1 - T * Real.exp (-ρ) ≤ ∑ j, if S j then smax s i j else 0 := by
  set Z := ∑ j' ∈ Finset.Iic i, Real.exp (s j')
  have hpos : 0 < Z := Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i, Finset.mem_Iic.2 le_rfl⟩
  have hZ : Real.exp ρ ≤ Z :=
    (Real.exp_le_exp.2 h1).trans
      (Finset.single_le_sum (fun _ _ => (Real.exp_pos _).le) (Finset.mem_Iic.2 hj0))
  have hone : 1 ≤ Real.exp (-ρ) * Z := by
    calc (1 : ℝ) = Real.exp (-ρ) * Real.exp ρ := by rw [← Real.exp_add]; simp
      _ ≤ Real.exp (-ρ) * Z := mul_le_mul_of_nonneg_left hZ (Real.exp_pos _).le
  have hterm : ∀ j, ¬ S j → smax s i j ≤ Real.exp (-ρ) := by
    intro j hj
    unfold smax; split_ifs with hji
    · rw [div_le_iff₀ hpos]
      exact (Real.exp_le_one_iff.2 (hS j hj)).trans hone
    · exact (Real.exp_pos _).le
  have hsplit : ∑ j, smax s i j =
      (∑ j, if S j then smax s i j else 0) + ∑ j, if S j then 0 else smax s i j := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    split_ifs <;> ring
  have hrest : (∑ j, if S j then 0 else smax s i j) ≤ T * Real.exp (-ρ) := by
    calc (∑ j, if S j then 0 else smax s i j) ≤ ∑ _j : Fin T, Real.exp (-ρ) :=
          Finset.sum_le_sum (fun j _ => by
            split_ifs with h
            · exact (Real.exp_pos _).le
            · exact hterm j h)
      _ = T * Real.exp (-ρ) := by simp
  linarith [smax_sum s i]

end VWC.Graft.Pattern
