/- Weight-structure lemmas for addition-five-digits, part 2 (split to bound memory per module). -/
import VWC.Tasks.AddS0

namespace VWC.Arch.Add

open Matrix Finset

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1b_rows : addModel_block1_W1 0 = Pi.single 127 1 ∧
    addModel_block1_W1 1 = Pi.single 128 1 + Pi.single 132 1 ∧
    addModel_block1_W1 2 = Pi.single 129 1 + Pi.single 132 1 + Pi.single 133 1 ∧
    addModel_block1_W1 3 = Pi.single 130 1 + Pi.single 132 1 + Pi.single 133 1 + Pi.single 134 1 ∧
    addModel_block1_W1 4 = Pi.single 131 1 + Pi.single 132 1 + Pi.single 133 1 + Pi.single 134 1 +
      Pi.single 135 1 ∧
    addModel_block1_W1 5 = Pi.single 128 1 ∧
    addModel_block1_W1 6 = Pi.single 129 1 + Pi.single 133 1 ∧
    addModel_block1_W1 7 = Pi.single 130 1 + Pi.single 133 1 + Pi.single 134 1 ∧
    addModel_block1_W1 8 = Pi.single 131 1 + Pi.single 133 1 + Pi.single 134 1 + Pi.single 135 1 ∧
    addModel_block1_W1 9 = Pi.single 129 1 ∧
    addModel_block1_W1 10 = Pi.single 130 1 + Pi.single 134 1 ∧
    addModel_block1_W1 11 = Pi.single 131 1 + Pi.single 134 1 + Pi.single 135 1 ∧
    addModel_block1_W1 12 = Pi.single 130 1 ∧
    addModel_block1_W1 13 = Pi.single 131 1 + Pi.single 135 1 ∧
    addModel_block1_W1 14 = Pi.single 131 1 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1b_hi (n : Fin 1024) : 15 ≤ n.val → addModel_block1_W1 n = 0 := by
  fin_cases n <;> first | (intro h; rfl) | (intro h; simp at h)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1b_vals : addModel_block1_b1 0 = 0 ∧
    addModel_block1_b1 1 = -1 ∧
    addModel_block1_b1 2 = -2 ∧
    addModel_block1_b1 3 = -3 ∧
    addModel_block1_b1 4 = -4 ∧
    addModel_block1_b1 5 = 0 ∧
    addModel_block1_b1 6 = -1 ∧
    addModel_block1_b1 7 = -2 ∧
    addModel_block1_b1 8 = -3 ∧
    addModel_block1_b1 9 = 0 ∧
    addModel_block1_b1 10 = -1 ∧
    addModel_block1_b1 11 = -2 ∧
    addModel_block1_b1 12 = 0 ∧
    addModel_block1_b1 13 = -1 ∧
    addModel_block1_b1 14 = 0 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1b_hi (n : Fin 1024) : 15 ≤ n.val → addModel_block1_b1 n = 0 := by
  fin_cases n <;> first | (intro h; rfl) | (intro h; simp at h)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W2b_col (n : Fin 1024) : (fun i => addModel_block1_W2 i n) =
    if h : n.val < 15 then Pi.single ⟨137 + kOf n.val, by unfold kOf; split_ifs <;> omega⟩ 1 else 0 := by
  fin_cases n <;> rfl

end VWC.Arch.Add
