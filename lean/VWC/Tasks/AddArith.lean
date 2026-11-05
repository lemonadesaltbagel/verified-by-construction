/-
Arithmetic side of addition-five-digits: the task's specification (5-digit operands, 6-digit
zero-padded sum) and the long-addition facts the circuit implements: per-digit generate and
propagate flags, carries by recursion, carries as a sum of mutually exclusive lookahead paths,
and the answer digits in terms of carries.
-/
import Mathlib

namespace VWC.Arch.Add

/-- A 5-digit operand, most significant digit first. -/
abbrev Num5 := Fin 5 → Fin 10

def val5 (A : Num5) : ℕ :=
  (A 0).val * 10000 + (A 1).val * 1000 + (A 2).val * 100 + (A 3).val * 10 + (A 4).val

/-- Digit `i` (`i < 5`) of an operand as a natural number (0 beyond). -/
def dg (A : Num5) (i : ℕ) : ℕ := if h : i < 5 then (A ⟨i, h⟩).val else 0

lemma dg_lt (A : Num5) (i : ℕ) : dg A i < 10 := by
  unfold dg; split_ifs
  · exact Fin.isLt _
  · omega

/-- Carry out of digit k (into digit k - 1); digits k..4 are summed. `carry 5 = 0`. -/
def carry (A B : Num5) : ℕ → ℕ
  | k => if h : k < 5 then (dg A k + dg B k + carry A B (k + 1)) / 10 else 0
termination_by k => 5 - k

def gen (A B : Num5) (i : ℕ) : ℕ := if 10 ≤ dg A i + dg B i then 1 else 0
def prp (A B : Num5) (i : ℕ) : ℕ := if dg A i + dg B i = 9 then 1 else 0

lemma carry_five (A B : Num5) : carry A B 5 = 0 := by
  rw [carry]; simp

lemma carry_succ (A B : Num5) (k : ℕ) (hk : k < 5) :
    carry A B k = (dg A k + dg B k + carry A B (k + 1)) / 10 := by
  rw [carry, dif_pos hk]

lemma carry_le (A B : Num5) (k : ℕ) : carry A B k ≤ 1 := by
  induction h : 5 - k generalizing k with
  | zero => rw [carry, dif_neg (by omega)]; omega
  | succ n ih =>
    rw [carry_succ A B k (by omega)]
    have := ih (k + 1) (by omega)
    have := dg_lt A k; have := dg_lt B k
    omega

/-- The carry recursion in generate/propagate form. -/
lemma carry_gp (A B : Num5) (k : ℕ) (hk : k < 5) :
    carry A B k = gen A B k + prp A B k * carry A B (k + 1) := by
  rw [carry_succ A B k hk]
  have := carry_le A B (k + 1)
  have := dg_lt A k; have := dg_lt B k
  unfold gen prp
  split_ifs <;> rcases (show carry A B (k + 1) = 0 ∨ carry A B (k + 1) = 1 by omega) with h | h <;>
    rw [h] <;> omega

lemma gen_le (A B : Num5) (i : ℕ) : gen A B i ≤ 1 := by unfold gen; split_ifs <;> omega
lemma prp_le (A B : Num5) (i : ℕ) : prp A B i ≤ 1 := by unfold prp; split_ifs <;> omega

/-! ### Specification -/

/-- Tokens: digits 0–9, `+` = 10, `=` = 11. -/
def dtok (d : ℕ) : Fin 12 := ⟨d % 10, by omega⟩

def prompt (A B : Num5) : List (Fin 12) :=
  List.ofFn (fun i => dtok (A i)) ++ [10] ++ List.ofFn (fun i => dtok (B i)) ++ [11]

/-- The 6-digit zero-padded sum. -/
def answer (A B : Num5) : List (Fin 12) :=
  List.ofFn (fun j : Fin 6 => dtok ((val5 A + val5 B) / 10 ^ (5 - j.val)))

/-- The answer digits by long addition: S₀ = c₀, S_j = (A_{j-1} + B_{j-1} + c_j) mod 10. -/
lemma answer_digit (A B : Num5) (j : ℕ) (hj : j < 6) :
    (val5 A + val5 B) / 10 ^ (5 - j) % 10 =
      if j = 0 then carry A B 0 else (dg A (j - 1) + dg B (j - 1) + carry A B j) % 10 := by
  have c0 := carry_succ A B 0 (by norm_num)
  have c1 := carry_succ A B 1 (by norm_num)
  have c2 := carry_succ A B 2 (by norm_num)
  have c3 := carry_succ A B 3 (by norm_num)
  have c4 := carry_succ A B 4 (by norm_num)
  have c5 := carry_five A B
  simp only [show (0 : ℕ) + 1 = 1 from rfl, show (1 : ℕ) + 1 = 2 from rfl, show (2 : ℕ) + 1 = 3 from rfl,
    show (3 : ℕ) + 1 = 4 from rfl, show (4 : ℕ) + 1 = 5 from rfl] at c0 c1 c2 c3 c4
  have hA : ∀ i : Fin 5, dg A i.val = (A i).val := fun i => by simp [dg, i.isLt]
  have hB : ∀ i : Fin 5, dg B i.val = (B i).val := fun i => by simp [dg, i.isLt]
  have a0 := hA 0; have a1 := hA 1; have a2 := hA 2; have a3 := hA 3; have a4 := hA 4
  have b0 := hB 0; have b1 := hB 1; have b2 := hB 2; have b3 := hB 3; have b4 := hB 4
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two] at a0 a1 a2 b0 b1 b2
  have e3 : ((3 : Fin 5) : ℕ) = 3 := rfl
  have e4 : ((4 : Fin 5) : ℕ) = 4 := rfl
  rw [e3] at a3 b3; rw [e4] at a4 b4
  have := (A 0).isLt; have := (A 1).isLt; have := (A 2).isLt; have := (A 3).isLt; have := (A 4).isLt
  have := (B 0).isLt; have := (B 1).isLt; have := (B 2).isLt; have := (B 3).isLt; have := (B 4).isLt
  unfold val5
  interval_cases j <;> simp <;> omega

/-- Carries as sums of disjoint generate/propagate paths. -/
lemma carry_paths (A B : Num5) :
    carry A B 4 = gen A B 4 ∧
    carry A B 3 = gen A B 3 + prp A B 3 * gen A B 4 ∧
    carry A B 2 = gen A B 2 + prp A B 2 * gen A B 3 + prp A B 2 * prp A B 3 * gen A B 4 ∧
    carry A B 1 = gen A B 1 + prp A B 1 * gen A B 2 + prp A B 1 * prp A B 2 * gen A B 3 +
      prp A B 1 * prp A B 2 * prp A B 3 * gen A B 4 ∧
    carry A B 0 = gen A B 0 + prp A B 0 * gen A B 1 + prp A B 0 * prp A B 1 * gen A B 2 +
      prp A B 0 * prp A B 1 * prp A B 2 * gen A B 3 +
      prp A B 0 * prp A B 1 * prp A B 2 * prp A B 3 * gen A B 4 := by
  have c0 := carry_gp A B 0 (by norm_num)
  have c1 := carry_gp A B 1 (by norm_num)
  have c2 := carry_gp A B 2 (by norm_num)
  have c3 := carry_gp A B 3 (by norm_num)
  have c4 := carry_gp A B 4 (by norm_num)
  have c5 := carry_five A B
  simp only [show (0 : ℕ) + 1 = 1 from rfl, show (1 : ℕ) + 1 = 2 from rfl, show (2 : ℕ) + 1 = 3 from rfl,
    show (3 : ℕ) + 1 = 4 from rfl, show (4 : ℕ) + 1 = 5 from rfl] at c0 c1 c2 c3 c4
  rw [c5, mul_zero, add_zero] at c4
  refine ⟨c4, ?_, ?_, ?_, ?_⟩
  · rw [c3, c4]
  · rw [c2, c3, c4]; ring
  · rw [c1, c2, c3, c4]; ring
  · rw [c0, c1, c2, c3, c4]; ring

end VWC.Arch.Add
