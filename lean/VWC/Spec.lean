/-
Task specification for `parity-upto10-bits` (evolve/src/task.py, class ParityTask),
stated independently of the model. Audit this file against the Python source.

  vocab = list("01=_")                         -> ids 0, 1, 2, 3
  prompt = bit_str.rjust(10, "_") + "="        -> `promptList` / `encode`
  answer = str(sum(bits) % 2)                  -> `parityTok`
  prediction = logits[:, -1, :].argmax(-1)     -> `VWC.predict` (Model.lean)
-/
import VWC.Model

namespace VWC

/-- Token ids, `task.stoi` for vocab "01=_". -/
def tok0 : Fin 4 := 0
def tok1 : Fin 4 := 1
def tokEq : Fin 4 := 2
def tokPad : Fin 4 := 3

def tokOfBit (b : Bool) : Fin 4 := if b then tok1 else tok0

/-- `bit_str.rjust(10, "_") + "="`, as token ids. -/
def promptList (bits : List Bool) : List (Fin 4) :=
  List.replicate (10 - bits.length) tokPad ++ bits.map tokOfBit ++ [tokEq]

/-- The 11-token prompt as a function of position (`promptList` has length 11
whenever `bits.length ≤ 10`; the default is never reached in that case). -/
def encode (bits : List Bool) : Fin 11 → Fin 4 :=
  fun t => (promptList bits).getD t tokPad

/-- `str(sum(bits) % 2)`, as a token id. -/
def parityTok (bits : List Bool) : Fin 4 :=
  if bits.count true % 2 = 1 then tok1 else tok0

end VWC
