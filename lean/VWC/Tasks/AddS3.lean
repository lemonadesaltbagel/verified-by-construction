/- Weight-structure lemmas for addition-five-digits, part 3 (split to bound memory per module). -/
import VWC.Tasks.AddS0

namespace VWC.Arch.Add

open Matrix Finset

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem W1c_row (n : Fin 1024) : addModel_block2_W1 n =
    if n.val = 0 then Pi.single 11 1 + Pi.single 137 (-1)
    else if n.val = 1 then Pi.single 11 1 + Pi.single 137 1
    else if h : n.val < 802 then
      Pi.single ⟨12 + (n.val - 2) / 200, by omega⟩ 1 +
        Pi.single ⟨27 + 10 * ((n.val - 2) / 200) + (n.val - 2) % 200 / 20, by omega⟩ 1 +
        Pi.single ⟨77 + 10 * ((n.val - 2) / 200) + (n.val - 2) % 20 / 2, by omega⟩ 1 +
        Pi.single ⟨138 + (n.val - 2) / 200, by omega⟩ (if (n.val - 2) % 2 = 0 then -1 else 1)
    else if h' : n.val < 902 then
      Pi.single 16 1 + Pi.single ⟨67 + (n.val - 802) / 10, by omega⟩ 1 +
        Pi.single ⟨117 + (n.val - 802) % 10, by omega⟩ 1
    else 0 := by
  fin_cases n <;> rfl

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem b1c (n : Fin 1024) : addModel_block2_b1 n =
    if n.val = 0 then 0 else if n.val = 1 then -1
    else if n.val < 802 then (if (n.val - 2) % 2 = 0 then -2 else -3)
    else if n.val < 902 then -2 else 0 := by
  fin_cases n <;> rfl

end VWC.Arch.Add
