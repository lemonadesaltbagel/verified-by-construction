/- Weight-structure lemmas for addition-five-digits, part 1 (split to bound memory per module). -/
import VWC.Tasks.AddS0

namespace VWC.Arch.Add

open Matrix Finset

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1a (n : Fin 1024) : addModel_block0_b1 n = if n.val < 500 then -1 else 0 := by
  fin_cases n <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W2a_col (n : Fin 1024) : (fun i => addModel_block0_W2 i n) =
    if h : n.val < 500 ∧ 10 ≤ n.val / 10 % 10 + n.val % 10 then
      Pi.single ⟨127 + n.val / 100, by omega⟩ 1
    else if h' : n.val < 500 ∧ n.val / 10 % 10 + n.val % 10 = 9 then
      Pi.single ⟨132 + n.val / 100, by omega⟩ 1
    else 0 := by
  fin_cases n <;> rfl

end VWC.Arch.Add
