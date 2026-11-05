/- Weight-structure lemmas for addition-five-digits, part 4 (split to bound memory per module). -/
import VWC.Tasks.AddS0

namespace VWC.Arch.Add

open Matrix Finset

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W2c_col (n : Fin 1024) : (fun i => addModel_block2_W2 i n) =
    if h : n.val < 902 then Pi.single ⟨142 + outD n.val, by have := outD_lt n.val; omega⟩ 1 else 0 := by
  fin_cases n <;> rfl

end VWC.Arch.Add
