/-
Phase 3.1: the multi-token pattern detector as a `Circuit` (three blocks, readout).

Pattern `P : Fin (M + 1) → Fin V` (M ≥ 1 offsets), n position bits. Uniform block shape:
Hc = M heads, dq = n + 1, dv = 1, Fc = M * 2 hidden units.
* block 1: head d attends to position t − (d+1) (truncated at 0) via ±1 position codes (score scale
  cs = cq·ck, split between query and key) and copies gv · [token = P (M−1−d)] into copy pair d; the
  MLP clamps (copy d − τ gv δ) into an exact 0/1 flip of copied-bit quad d.
* block 2: no attention; the MLP clamps ½ − (the complements of [token = P M], [M ≤ t] and the M
  copied bits) into an exact flip of the match quad: the AND, with its threshold mid-gap.
* block 3: head 0 attends to the positions where the match bit is 1 (query fq·const, key fk·match)
  and writes A × the match read into the flag pair; no MLP.
* readout: Lg/2 (flag₀ − flag₁) into token r.
All circuit LayerNorms: γ = 1, β = 0, ε = 1.
All weights are of moderate size (≈ 1e3–5e3 for OPT hosts), so the circuit runs in half precision.
-/
import VWC.Graft.PatternBase

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M n : ℕ}

noncomputable def single (a : Slot M n) (w : ℝ) : Fin (slots M n) → ℝ := Pi.single (slotEquiv M n a) w

/-- Mirrored read of the pair (a, b): (x a − x b)/2. -/
noncomputable def rd (a b : Slot M n) : Fin (slots M n) → ℝ := single a (1 / 2) - single b (1 / 2)

lemma rd_dot (a b : Slot M n) (x : Fin (slots M n) → ℝ) :
    rd a b ⬝ᵥ x = x (slotEquiv M n a) / 2 - x (slotEquiv M n b) / 2 := by
  simp [rd, single, sub_dotProduct, single_dotProduct]; ring

def pmsign (c : Fin 2) : ℝ := if c = 0 then 1 else -1
def flipsign : Fin 4 → ℝ := ![1, -1, -1, 1]
def tix (d : Fin M) : Fin (M + 1) := ⟨M - 1 - d, by omega⟩

/-- Circuit parameters. -/
structure Par where
  δ : ℝ
  gv : ℝ
  cq : ℝ
  ck : ℝ
  τ : ℝ
  Wa : ℝ
  Wb : ℝ
  fq : ℝ
  fk : ℝ
  A : ℝ
  Lg : ℝ

/-- Score scale of the position heads. -/
noncomputable def Par.cs (p : Par) : ℝ := p.cq * p.ck

variable (p : Par)

/-! ### Block 1 -/

noncomputable def qrow (dc : Fin M × Fin (n + 1)) : Fin (slots M n) → ℝ :=
  if h : (dc.2 : ℕ) < n then p.cq • rd (sQry dc.1 ⟨dc.2, h⟩ 0) (sQry dc.1 ⟨dc.2, h⟩ 1)
  else p.cq • rd (sConst 0) (sConst 1)

noncomputable def krow (dc : Fin M × Fin (n + 1)) : Fin (slots M n) → ℝ :=
  if h : (dc.2 : ℕ) < n then p.ck • rd (sKey ⟨dc.2, h⟩ 0) (sKey ⟨dc.2, h⟩ 1)
  else (-(p.ck * ((n : ℝ) - 1))) • rd (sConst 0) (sConst 1)

noncomputable def vrow (d : Fin M) : Fin (slots M n) → ℝ :=
  p.gv • rd (sTokS (tix d) 0) (sTokS (tix d) 1)

/-- Output column of head d on slot a: ±1 on copy pair d. -/
def oslot : Slot M n → Fin M → ℝ
  | .inr (.inr (.inr (.inr (.inr (.inl (d, c)))))) => fun d' => if d' = d then pmsign c else 0
  | _ => fun _ => 0

noncomputable def attn1 : GAttn (slots M n) M (n + 1) 1 :=
  ⟨fun r => qrow p (finProdFinEquiv.symm r), 0, fun r => krow p (finProdFinEquiv.symm r), 0,
   fun r => vrow p (finProdFinEquiv.symm r).1, 0,
   fun j r => oslot ((slotEquiv M n).symm j) (finProdFinEquiv.symm r).1, 0⟩

noncomputable def arow (d : Fin M) : Fin (slots M n) → ℝ :=
  p.Wa • (rd (sCopy d 0) (sCopy d 1) - (p.τ * p.gv) • rd (sConst 0) (sConst 1))

/-- MLP-1 output on slot a from hidden unit (d', σ): ±δ·flip on copied-bit quad d'. -/
def fslot (δ : ℝ) : Slot M n → Fin M × Fin 2 → ℝ
  | .inr (.inr (.inr (.inr (.inr (.inr (.inl (d, c))))))) =>
      fun dσ => if dσ.1 = d then δ * flipsign c * pmsign dσ.2 else 0
  | _ => fun _ => 0

noncomputable def blk1 : CBlock (slots M n) (M * 2) M (n + 1) 1 :=
  ⟨1, 0, 1, attn1 p, 1, 0, 1, relu,
   fun f => arow p (finProdFinEquiv.symm f).1,
   fun f => if (finProdFinEquiv.symm f).2 = 0 then 0 else -1,
   fun j f => fslot p.δ ((slotEquiv M n).symm j) (finProdFinEquiv.symm f), 0⟩

/-! ### Block 2 -/

/-- z = δ (½ − ((1 − b_tok) + (1 − b_valid) + Σ (1 − copied bit))): the complement coordinates
(quad entries 2, 3) sum to 0 iff all M + 2 bits are 1, and to ≥ 1 otherwise. -/
noncomputable def zrow : Fin (slots M n) → ℝ :=
  (1 / 2 : ℝ) • rd (sConst 0) (sConst 1) - (rd (sTokS (Fin.last M) 2) (sTokS (Fin.last M) 3) +
    rd (sValid 2) (sValid 3) + ∑ d : Fin M, rd (sCbit d 2) (sCbit d 3))

/-- MLP-2 output on slot a from hidden unit (d', σ): only d' = 0 is used. -/
def mslot [NeZero M] (δ : ℝ) : Slot M n → Fin M × Fin 2 → ℝ
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inl c))))))) =>
      fun dσ => if dσ.1 = 0 then δ * flipsign c * pmsign dσ.2 else 0
  | _ => fun _ => 0

noncomputable def blk2 [NeZero M] : CBlock (slots M n) (M * 2) M (n + 1) 1 :=
  ⟨1, 0, 1, ⟨0, 0, 0, 0, 0, 0, 0, 0⟩, 1, 0, 1, relu,
   fun f => if (finProdFinEquiv.symm f).1 = 0 then p.Wb • zrow else 0,
   fun f => if (finProdFinEquiv.symm f).1 = 0 ∧ (finProdFinEquiv.symm f).2 = 1 then -1 else 0,
   fun j f => mslot p.δ ((slotEquiv M n).symm j) (finProdFinEquiv.symm f), 0⟩

/-! ### Block 3 -/

def gslot [NeZero M] : Slot M n → Fin M → ℝ
  | .inr (.inr (.inr (.inr (.inr (.inr (.inr (.inr c))))))) => fun d => if d = 0 then pmsign c else 0
  | _ => fun _ => 0

/-- Flag head (head 0, query/key row 0): query fq·const, key fk·match. -/
noncomputable def q3row [NeZero M] (dc : Fin M × Fin (n + 1)) : Fin (slots M n) → ℝ :=
  if dc.1 = 0 ∧ dc.2 = 0 then p.fq • rd (sConst 0) (sConst 1) else 0

noncomputable def k3row [NeZero M] (dc : Fin M × Fin (n + 1)) : Fin (slots M n) → ℝ :=
  if dc.1 = 0 ∧ dc.2 = 0 then p.fk • rd (sMatch 0) (sMatch 1) else 0

noncomputable def attn3 [NeZero M] : GAttn (slots M n) M (n + 1) 1 :=
  ⟨fun r => q3row p (finProdFinEquiv.symm r), 0, fun r => k3row p (finProdFinEquiv.symm r), 0,
   fun r => if (finProdFinEquiv.symm r).1 = 0 then p.A • rd (sMatch 0) (sMatch 1) else 0, 0,
   fun j r => gslot ((slotEquiv M n).symm j) (finProdFinEquiv.symm r).1, 0⟩

noncomputable def blk3 [NeZero M] : CBlock (slots M n) (M * 2) M (n + 1) 1 :=
  ⟨1, 0, 1, attn3 p, 1, 0, 1, relu, 0, 0, 0, 0⟩

/-! ### The circuit -/

noncomputable def circ [NeZero M] {V : ℕ} (L : ℕ) (P : Fin (M + 1) → Fin V) (r : Fin V) :
    Circuit V L (slots M n) (M * 2) M (n + 1) 1 :=
  ⟨fun v j => Ftok P p.δ v ((slotEquiv M n).symm j),
   fun q j => Fpos p.δ q ((slotEquiv M n).symm j),
   Ef p.δ M n, [blk1 p, blk2 p, blk3 p], 1, 0,
   fun t => if t = r then (p.Lg / 2) • (single (sFlag 0) 1 - single (sFlag 1) 1) else 0⟩

/-- The trigger: the pattern occurs contiguously, ending at some position t ≥ M. -/
def trig {V : ℕ} (P : Fin (M + 1) → Fin V) (T : ℕ) (ids : Fin (T + 1) → Fin V) : Prop :=
  ∃ t : Fin (T + 1), ∃ h : M ≤ (t : ℕ), ∀ i : Fin (M + 1), ids ⟨t - M + i, by omega⟩ = P i

end VWC.Graft.Pattern
