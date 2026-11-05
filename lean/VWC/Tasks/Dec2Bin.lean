/-
decimal-to-binary-8bit (snapshot HandbuiltDec2Bin_v20_final5k).

**This is a hand-written program, not a weight-space circuit** (PI scope rule, 2026-11-04): its
modules are custom Python code with hard-coded buffers, and `_DigitPlaceAttn` is a Python loop
that accumulates N = 100·d₀ + 10·d₁ + d₂ rather than an attention operation. So we cannot reuse
the generic `VWC.Arch` semantics. Instead each definition below transcribes one module of the
snapshot's forward pass, keeping its constants (`fc1_w`, `fc1_b`, `fc2_w`, `L_gate = 256`,
thresholds 2^(7-b), channel layout). `scripts/check_dec2bin.py` re-implements these Lean
definitions in numpy and compares them with the snapshot's own forward pass on all 256 inputs.

Residual channels (d_model 26): token one-hot 0..10, position one-hot 11..22, N 23,
logit-'0' 24, logit-'1' 25. LayerNorms are `nn.Identity`, blocks 1..7 have zero attention.
-/
import Mathlib
import VWC.Model

namespace VWC.Dec2Bin

open Matrix Finset

abbrev Vec := Fin 26 → ℝ

/-- `token_emb` (`_OneHotEmbedding`, offset 0) plus `pos_emb` (offset 11). -/
def embed (tok : Fin 11) (p : ℕ) : Vec := fun i =>
  (if i.val = tok.val then 1 else 0) + (if i.val = 11 + p then 1 else 0)

/-- `digit_vals`: Σ_c c · x[c] over the digit channels 0..9. -/
def dval (x : Vec) : ℝ := ∑ c : Fin 10, (c.val : ℝ) * x (c.castLE (by norm_num))

/-- Place value of digit position j (`place_values = (100, 10, 1)` at positions 0, 1, 2). -/
def place (j : ℕ) : ℝ := if j = 0 then 100 else if j = 1 then 10 else if j = 2 then 1 else 0

/-- `_DigitPlaceAttn.forward`: channel 23 at position p gets the running sum
Σ_{j ≤ p, j < 3} place(j) · digit_vals[j]; all other channels 0. -/
def digitAttn {T : ℕ} (xs : Fin T → Vec) (p : Fin T) : Vec := fun i =>
  if i.val = 23 then ∑ j : Fin T, (if j ≤ p ∧ j.val < 3 then place j.val * dval (xs j) else 0) else 0

/-- `_SparseBitMLP` buffers for threshold T and gate L. -/
def fc1w (L : ℝ) : Matrix (Fin 5) (Fin 2) ℝ := !![1, 0; 1, 0; 1, L; 1, L; 0, 1]
def fc1b (T L : ℝ) : Fin 5 → ℝ := ![-(T - 1), -T, -(T - 1) - L, -T - L, 0]
def fc2w (T : ℝ) : Matrix (Fin 3) (Fin 5) ℝ := !![-T, T, 0, 0, 0; 0, 0, -1, 1, 1; 0, 0, 1, -1, 0]

/-- `_SparseBitMLP.forward` (threshold T, output position pOut ≤ 10, `l_gate = 256`):
reads channels 23 and 11 + pOut, writes channels 23, 24, 25. -/
def bitMLP (T : ℝ) (pOut : ℕ) (x : Vec) : Vec := fun i =>
  let inp : Fin 2 → ℝ := ![x 23, if h : 11 + pOut < 26 then x ⟨11 + pOut, h⟩ else 0]
  let h : Fin 5 → ℝ := fun k => relu ((fc1w 256 *ᵥ inp) k + fc1b T 256 k)
  let o := fc2w T *ᵥ h
  if i.val = 23 then o 0 else if i.val = 24 then o 1 else if i.val = 25 then o 2 else 0

/-- Block b's MLP sublayer (`threshold = 2^(7-b)`, `p_out = 3 + b`), with residual. -/
def blk (b : ℕ) (z : Vec) : Vec := z + bitMLP (2 ^ (7 - b)) (3 + b) z

/-- `_SparseHead` with `scale = 1` and no baseline positions: logits 0 and 1 read channels
24 and 25, all other logits are 0. -/
def head (z : Vec) : Fin 11 → ℝ := fun v => if v.val = 0 then z 24 else if v.val = 1 then z 25 else 0

/-- Full forward pass at position p: embeddings; block 0 = digit attention then bit MLP 0;
blocks 1..7 = (zero attention) bit MLPs; identity final LayerNorm; head. -/
def logits {T : ℕ} (ids : Fin T → Fin 11) (p : Fin T) : Fin 11 → ℝ :=
  let x : Fin T → Vec := fun t => embed (ids t) t.val
  let y : Vec := x p + digitAttn x p
  head (blk 7 (blk 6 (blk 5 (blk 4 (blk 3 (blk 2 (blk 1 (blk 0 y))))))))

/-- Greedy next token (`logits[:, -1, :].argmax(-1)`). -/
noncomputable def next (s : List (Fin 11)) (h0 : 0 < s.length) : Fin 11 :=
  argmaxFirst (logits (fun t : Fin s.length => s.get t) ⟨s.length - 1, by omega⟩)

/-- Greedy generation of k tokens; `none` if the sequence would exceed `max_seq_len = 12`. -/
noncomputable def generate : List (Fin 11) → ℕ → Option (List (Fin 11))
  | _, 0 => some []
  | s, k + 1 =>
    if h : 0 < s.length ∧ s.length ≤ 12 then
      let t := next s h.1
      (generate (s ++ [t]) k).map (t :: ·)
    else none

/-! ### Specification (src/task.py, DecimalToBinaryTask). Vocab "0123456789=". -/

def dtok (d : ℕ) : Fin 11 := ⟨d % 10, by omega⟩

/-- `f"{x:03d}="`. -/
def prompt (x : Fin 256) : List (Fin 11) :=
  [dtok (x.val / 100), dtok (x.val / 10), dtok x.val, 10]

/-- `f"{x:08b}"`: bit 7 - i of x, most significant first. -/
def answer (x : Fin 256) : List (Fin 11) :=
  List.ofFn (fun i : Fin 8 => dtok (x.val / 2 ^ (7 - i.val) % 2))

/-! ### One bit-extraction block -/

lemma relu_step (n t : ℕ) (ht : 1 ≤ t) :
    relu ((n : ℝ) - ((t : ℝ) - 1)) - relu ((n : ℝ) - t) = if t ≤ n then 1 else 0 := by
  unfold relu
  split_ifs with h
  · have : (t : ℝ) ≤ n := by exact_mod_cast h
    rw [max_eq_left (by linarith), max_eq_left (by linarith)]; ring
  · have : (n : ℝ) + 1 ≤ t := by exact_mod_cast (show n + 1 ≤ t by omega)
    rw [max_eq_right (by linarith), max_eq_right (by linarith)]; ring

lemma bitMLP_apply (T : ℝ) (pOut : ℕ) (hp : 11 + pOut < 26) (z : Vec) :
    bitMLP T pOut z 23 = -T * (relu (z 23 - (T - 1)) - relu (z 23 - T)) ∧
    bitMLP T pOut z 24 = -(relu (z 23 + 256 * z ⟨11 + pOut, hp⟩ - (T - 1) - 256) -
      relu (z 23 + 256 * z ⟨11 + pOut, hp⟩ - T - 256)) + relu (z ⟨11 + pOut, hp⟩) ∧
    bitMLP T pOut z 25 = relu (z 23 + 256 * z ⟨11 + pOut, hp⟩ - (T - 1) - 256) -
      relu (z 23 + 256 * z ⟨11 + pOut, hp⟩ - T - 256) ∧
    ∀ i : Fin 26, i.val < 23 → bitMLP T pOut z i = 0 := by
  refine ⟨?_, ?_, ?_, fun i hi => ?_⟩
  · simp [bitMLP, fc1w, fc1b, fc2w, mulVec, dotProduct, Fin.sum_univ_succ, hp]; ring_nf
  · simp [bitMLP, fc1w, fc1b, fc2w, mulVec, dotProduct, Fin.sum_univ_succ, hp]; ring_nf
  · simp [bitMLP, fc1w, fc1b, fc2w, mulVec, dotProduct, Fin.sum_univ_succ, hp]; ring_nf
  · simp only [bitMLP]
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega)]

/-- Block b with N = z₂₃ < 2·2^(7-b) and gate g ∈ {0,1} on position channel 14 + b:
N becomes N mod 2^(7-b), and the gate routes the extracted bit to the logit channels. -/
lemma blk_step (b : ℕ) (hb : b < 8) (z : Vec) (n : ℕ) (hn : z 23 = n) (hlt : n < 2 * 2 ^ (7 - b))
    (hg : z ⟨14 + b, by omega⟩ = 0 ∨ z ⟨14 + b, by omega⟩ = 1) :
    blk b z 23 = ((n % 2 ^ (7 - b) : ℕ) : ℝ) ∧
    blk b z 24 = z 24 + z ⟨14 + b, by omega⟩ * (1 - if 2 ^ (7 - b) ≤ n then 1 else 0) ∧
    blk b z 25 = z 25 + z ⟨14 + b, by omega⟩ * (if 2 ^ (7 - b) ≤ n then 1 else 0) ∧
    ∀ i : Fin 26, i.val < 23 → blk b z i = z i := by
  obtain ⟨e23, e24, e25, elo⟩ := bitMLP_apply (2 ^ (7 - b)) (3 + b) (by omega) z
  have hidx : (⟨11 + (3 + b), by omega⟩ : Fin 26) = ⟨14 + b, by omega⟩ := Fin.ext (by simp; omega)
  rw [hidx] at e24 e25
  set g := z ⟨14 + b, by omega⟩
  have hT : ((2 ^ (7 - b) : ℕ) : ℝ) = (2 : ℝ) ^ (7 - b) := by push_cast; ring
  have ht1 : 1 ≤ 2 ^ (7 - b) := Nat.one_le_two_pow
  have ht128 : 2 ^ (7 - b) ≤ 128 := by
    calc 2 ^ (7 - b) ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) (by omega)
      _ = 128 := by norm_num
  have hstep := relu_step n (2 ^ (7 - b)) ht1
  rw [hT] at hstep
  -- the gated pair equals g · bit
  have hgate : relu (z 23 + 256 * g - ((2 : ℝ) ^ (7 - b) - 1) - 256) -
      relu (z 23 + 256 * g - (2 : ℝ) ^ (7 - b) - 256) = g * (if 2 ^ (7 - b) ≤ n then 1 else 0) := by
    rcases hg with h0 | h1
    · rw [h0]
      have hn' : (n : ℝ) + 1 ≤ 2 ^ (7 - b) + 256 := by
        have : n + 1 ≤ 2 ^ (7 - b) + 256 := by omega
        exact_mod_cast (show ((n + 1 : ℕ) : ℝ) ≤ ((2 ^ (7 - b) + 256 : ℕ) : ℝ) by exact_mod_cast this)
      unfold relu
      rw [hn, max_eq_right (by push_cast at hn' ⊢; linarith), max_eq_right (by push_cast at hn' ⊢; linarith)]
      ring
    · rw [h1, one_mul, hn]
      have : (n : ℝ) + 256 * 1 - ((2 : ℝ) ^ (7 - b) - 1) - 256 = n - (2 ^ (7 - b) - 1) := by ring
      have h2 : (n : ℝ) + 256 * 1 - (2 : ℝ) ^ (7 - b) - 256 = n - 2 ^ (7 - b) := by ring
      rw [this, h2, hstep]
  have hrg : relu g = g := by unfold relu; rcases hg with h | h <;> rw [h] <;> norm_num
  refine ⟨?_, ?_, ?_, fun i hi => ?_⟩
  · simp only [blk, Pi.add_apply]
    rw [e23, hn, hstep]
    split_ifs with h
    · have : n % 2 ^ (7 - b) = n - 2 ^ (7 - b) := by
        rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt (by omega)]
      rw [this, Nat.cast_sub h]; push_cast; ring
    · rw [Nat.mod_eq_of_lt (by omega)]; ring
  · simp only [blk, Pi.add_apply]
    rw [e24, hgate, hrg]; ring
  · simp only [blk, Pi.add_apply]
    rw [e25, hgate]
  · simp only [blk, Pi.add_apply]
    rw [elo i hi, add_zero]

/-! ### The whole forward pass on a teacher-forced prefix -/

lemma dval_embed (d p : ℕ) : dval (embed (dtok d) p) = (d % 10 : ℕ) := by
  unfold dval embed
  rw [sum_eq_single ⟨d % 10, by omega⟩]
  · have h : ¬ (d % 10 = 11 + p) := by omega
    simp [dtok, h]
  · intro c _ hc
    have h1 : ¬ ((c : ℕ) = d % 10) := fun h => hc (Fin.ext h)
    have h2 : ¬ ((c : ℕ) = 11 + p) := by omega
    simp [dtok, h1, h2]
  · simp

/-- Bit 7 - b of x as the block-b clipped step on N_b = x mod 2^(8-b). -/
def bitv (x b : ℕ) : ℝ := if 2 ^ (7 - b) ≤ x % 2 ^ (8 - b) then 1 else 0

lemma bitv_nat (x b : ℕ) (hx : x < 256) (hb : b < 8) :
    (if 2 ^ (7 - b) ≤ x % 2 ^ (8 - b) then 1 else 0) = x / 2 ^ (7 - b) % 2 := by
  interval_cases b <;> norm_num <;> split_ifs <;> omega

lemma bitv_eq (x b : ℕ) (hx : x < 256) (hb : b < 8) : bitv x b = ((x / 2 ^ (7 - b) % 2 : ℕ) : ℝ) := by
  rw [← bitv_nat x b hx hb]
  unfold bitv
  split_ifs <;> simp

/-- The head's argmax for logits `[1 - bit, bit, 0, …]`. -/
lemma final_argmax (x k : ℕ) (hx : x < 256) (hk : k < 8) (zf : Vec)
    (h24 : zf 24 = 1 - bitv x k) (h25 : zf 25 = bitv x k) :
    argmaxFirst (head zf) = dtok (x / 2 ^ (7 - k) % 2) := by
  have hb := bitv_eq x k hx hk
  have hv : (dtok (x / 2 ^ (7 - k) % 2)).val = x / 2 ^ (7 - k) % 2 := by simp only [dtok]; omega
  apply argmaxFirst_eq_of_lt
  intro j hj
  have hlo : ∀ v : Fin 11, 2 ≤ v.val → head zf v = 0 := fun v hv => by
    simp only [head]; rw [if_neg (by omega), if_neg (by omega)]
  rcases (show x / 2 ^ (7 - k) % 2 = 0 ∨ x / 2 ^ (7 - k) % 2 = 1 by omega) with h | h
  · have ht : dtok (x / 2 ^ (7 - k) % 2) = 0 := Fin.ext (by rw [hv, h]; rfl)
    rw [h] at hb; push_cast at hb
    rw [ht] at hj ⊢
    have e0 : head zf 0 = 1 := by
      simp only [head]; rw [if_pos (show ((0 : Fin 11) : ℕ) = 0 from rfl), h24, hb]; ring
    rw [e0]
    by_cases h1 : j.val = 1
    · simp only [head]; rw [if_neg (by omega), if_pos h1, h25, hb]; norm_num
    · rw [hlo j (by have : j.val ≠ 0 := fun h' => hj (Fin.ext h'); omega)]; norm_num
  · have ht : dtok (x / 2 ^ (7 - k) % 2) = 1 := Fin.ext (by rw [hv, h]; rfl)
    rw [h] at hb; push_cast at hb
    rw [ht] at hj ⊢
    have e1 : head zf 1 = 1 := by
      simp only [head]
      rw [if_neg (show ¬ ((1 : Fin 11) : ℕ) = 0 by decide), if_pos (show ((1 : Fin 11) : ℕ) = 1 from rfl), h25, hb]
    rw [e1]
    by_cases h0 : j.val = 0
    · simp only [head]; rw [if_pos h0, h24, hb]; norm_num
    · rw [hlo j (by have : j.val ≠ 1 := fun h' => hj (Fin.ext h'); omega)]; norm_num

/-- **One generation step.** On a prefix whose first three tokens are the digits of x, the
logits at position 3 + k are `[1 - bit, bit, 0, …]` for bit 7 - k of x. -/
lemma step (x : Fin 256) (k : ℕ) (hk : k < 8) (ids : Fin (4 + k) → Fin 11)
    (hd0 : ids ⟨0, by omega⟩ = dtok (x.val / 100)) (hd1 : ids ⟨1, by omega⟩ = dtok (x.val / 10))
    (hd2 : ids ⟨2, by omega⟩ = dtok x.val) :
    argmaxFirst (logits ids ⟨3 + k, by omega⟩) = dtok (x.val / 2 ^ (7 - k) % 2) := by
  have hx := x.isLt
  set p : Fin (4 + k) := ⟨3 + k, by omega⟩ with hp
  set xs : Fin (4 + k) → Vec := fun t => embed (ids t) t.val with hxs
  set y : Vec := xs p + digitAttn xs p with hy
  have hlog : logits ids p = head (blk 7 (blk 6 (blk 5 (blk 4 (blk 3 (blk 2 (blk 1 (blk 0 y)))))))) := rfl
  have htok : (ids p).val < 11 := (ids p).isLt
  -- the embedding at the query position
  have hxp : ∀ i : Fin 26, 22 < i.val → xs p i = 0 := by
    intro i hi; simp only [xs, embed, p]; rw [if_neg (by omega), if_neg (by omega)]; ring
  have hda : ∀ i : Fin 26, i.val ≠ 23 → digitAttn xs p i = 0 := fun i hi => by
    simp only [digitAttn]; rw [if_neg hi]
  -- N = 100·d₀ + 10·d₁ + d₂ = x
  have hy23 : y 23 = (x.val : ℝ) := by
    simp only [y, Pi.add_apply]
    rw [hxp 23 (by norm_num), zero_add]
    simp only [digitAttn]
    rw [if_pos (show ((23 : Fin 26) : ℕ) = 23 from rfl)]
    let f : ℕ → ℝ := fun j => if h : j < 4 + k then
      (if j ≤ 3 + k ∧ j < 3 then place j * dval (embed (ids ⟨j, h⟩) j) else 0) else 0
    have hf : ∀ j : Fin (4 + k), (if j ≤ p ∧ j.val < 3 then place j.val * dval (xs j) else 0) = f j.val := by
      intro j
      simp only [f, dif_pos j.isLt, Fin.le_def, p, xs]
    rw [sum_congr rfl fun j _ => hf j, Fin.sum_univ_eq_sum_range f (4 + k),
      show 4 + k = 3 + (1 + k) by omega, sum_range_add]
    rw [sum_eq_zero (s := range (1 + k)) (fun i _ => by simp only [f]; split_ifs with h1 h2 <;> first | rfl | omega)]
    simp only [sum_range_succ, sum_range_zero, f]
    rw [dif_pos (by omega), dif_pos (by omega), dif_pos (by omega), if_pos ⟨by omega, by omega⟩,
      if_pos ⟨by omega, by omega⟩, if_pos ⟨by omega, by omega⟩]
    rw [hd0, hd1, hd2, dval_embed, dval_embed, dval_embed]
    simp only [place]
    norm_num
    have e : x.val / 100 % 10 * 100 + x.val / 10 % 10 * 10 + x.val % 10 = x.val := by omega
    have e' : ((x.val / 100 % 10 * 100 + x.val / 10 % 10 * 10 + x.val % 10 : ℕ) : ℝ) = x.val := by
      exact_mod_cast e
    push_cast at e' ⊢
    linarith [e']
  have hy24 : y 24 = 0 := by
    simp only [y, Pi.add_apply]; rw [hxp 24 (by norm_num), hda 24 (by norm_num)]; ring
  have hy25 : y 25 = 0 := by
    simp only [y, Pi.add_apply]; rw [hxp 25 (by norm_num), hda 25 (by norm_num)]; ring
  have hgate : ∀ b, (hb : b < 8) → y ⟨14 + b, by omega⟩ = if b = k then 1 else 0 := by
    intro b hb
    simp only [y, Pi.add_apply]
    rw [hda _ (by show 14 + b ≠ 23; omega), add_zero]
    simp only [xs, embed, p]
    rw [if_neg (by show ¬ (14 + b = (ids ⟨3 + k, by omega⟩).val); omega)]
    by_cases h : b = k
    · rw [if_pos (by show 14 + b = 11 + (3 + k); omega), if_pos h]; ring
    · rw [if_neg (by show ¬ (14 + b = 11 + (3 + k)); omega), if_neg h]; ring
  have g01 : ∀ b, (hb : b < 8) → y ⟨14 + b, by omega⟩ = 0 ∨ y ⟨14 + b, by omega⟩ = 1 := by
    intro b hb; rw [hgate b hb]; split_ifs <;> simp
  -- eight blocks
  obtain ⟨a23, a24, a25, alo⟩ := blk_step 0 (by norm_num) y x.val hy23 (by norm_num) (g01 0 (by norm_num))
  set z1 := blk 0 y
  have p1 : ∀ b, (hb : b < 8) → z1 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => alo _ (by show 14 + b < 23; omega)
  obtain ⟨b23, b24, b25, blo⟩ := blk_step 1 (by norm_num) z1 _ a23 (by norm_num; omega)
    (by rw [p1 1 (by norm_num)]; exact g01 1 (by norm_num))
  set z2 := blk 1 z1
  have p2 : ∀ b, (hb : b < 8) → z2 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [blo _ (by show 14 + b < 23; omega), p1 b hb]
  obtain ⟨c23, c24, c25, clo⟩ := blk_step 2 (by norm_num) z2 _ b23 (by norm_num; omega)
    (by rw [p2 2 (by norm_num)]; exact g01 2 (by norm_num))
  set z3 := blk 2 z2
  have p3 : ∀ b, (hb : b < 8) → z3 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [clo _ (by show 14 + b < 23; omega), p2 b hb]
  obtain ⟨d23, d24, d25, dlo⟩ := blk_step 3 (by norm_num) z3 _ c23 (by norm_num; omega)
    (by rw [p3 3 (by norm_num)]; exact g01 3 (by norm_num))
  set z4 := blk 3 z3
  have p4 : ∀ b, (hb : b < 8) → z4 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [dlo _ (by show 14 + b < 23; omega), p3 b hb]
  obtain ⟨e23, e24, e25, elo⟩ := blk_step 4 (by norm_num) z4 _ d23 (by norm_num; omega)
    (by rw [p4 4 (by norm_num)]; exact g01 4 (by norm_num))
  set z5 := blk 4 z4
  have p5 : ∀ b, (hb : b < 8) → z5 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [elo _ (by show 14 + b < 23; omega), p4 b hb]
  obtain ⟨f23, f24, f25, flo⟩ := blk_step 5 (by norm_num) z5 _ e23 (by norm_num; omega)
    (by rw [p5 5 (by norm_num)]; exact g01 5 (by norm_num))
  set z6 := blk 5 z5
  have p6 : ∀ b, (hb : b < 8) → z6 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [flo _ (by show 14 + b < 23; omega), p5 b hb]
  obtain ⟨g23, g24, g25, glo⟩ := blk_step 6 (by norm_num) z6 _ f23 (by norm_num; omega)
    (by rw [p6 6 (by norm_num)]; exact g01 6 (by norm_num))
  set z7 := blk 6 z6
  have p7 : ∀ b, (hb : b < 8) → z7 ⟨14 + b, by omega⟩ = y ⟨14 + b, by omega⟩ := fun b hb => by
    rw [glo _ (by show 14 + b < 23; omega), p6 b hb]
  obtain ⟨h23, h24, h25, -⟩ := blk_step 7 (by norm_num) z7 _ g23 (by norm_num; omega)
    (by rw [p7 7 (by norm_num)]; exact g01 7 (by norm_num))
  rw [p7 7 (by norm_num)] at h24 h25
  rw [p6 6 (by norm_num)] at g24 g25
  rw [p5 5 (by norm_num)] at f24 f25
  rw [p4 4 (by norm_num)] at e24 e25
  rw [p3 3 (by norm_num)] at d24 d25
  rw [p2 2 (by norm_num)] at c24 c25
  rw [p1 1 (by norm_num)] at b24 b25
  simp only [hgate _ (by norm_num : (0 : ℕ) < 8), hgate _ (by norm_num : (1 : ℕ) < 8),
    hgate _ (by norm_num : (2 : ℕ) < 8), hgate _ (by norm_num : (3 : ℕ) < 8),
    hgate _ (by norm_num : (4 : ℕ) < 8), hgate _ (by norm_num : (5 : ℕ) < 8),
    hgate _ (by norm_num : (6 : ℕ) < 8), hgate _ (by norm_num : (7 : ℕ) < 8)]
    at a24 a25 b24 b25 c24 c25 d24 d25 e24 e25 f24 f25 g24 g25 h24 h25
  -- the final logits
  have n1 : x.val % 2 ^ (7 - 0) = x.val % 2 ^ (8 - 1) := by norm_num
  have bits : ∀ b, b < 8 → True := fun _ _ => trivial
  rw [hlog]
  apply final_argmax x.val k hx hk
  · rw [h24, g24, f24, e24, d24, c24, b24, a24, hy24]
    clear_value p xs y z1 z2 z3 z4 z5 z6 z7
    interval_cases k <;> simp [bitv] <;> norm_num <;> split_ifs <;> norm_num <;> omega
  · rw [h25, g25, f25, e25, d25, c25, b25, a25, hy25]
    clear_value p xs y z1 z2 z3 z4 z5 z6 z7
    interval_cases k <;> simp [bitv] <;> norm_num <;> split_ifs <;> norm_num <;> omega

/-! ### Generation -/

/-- Generation is correct if every greedy step is correct on the teacher-forced prefix. -/
theorem generate_eq_of_steps : ∀ (ans s : List (Fin 11)),
    0 < s.length → s.length + ans.length ≤ 13 →
    (∀ k (hk : k < ans.length) (h0 : 0 < (s ++ ans.take k).length),
        next (s ++ ans.take k) h0 = ans.get ⟨k, hk⟩) →
    generate s ans.length = some ans
  | [], _, _, _, _ => rfl
  | a :: ans, s, h0, hlen, hstep => by
    have hL : s.length ≤ 12 := by simp at hlen; omega
    have h0' : 0 < (s ++ List.take 0 (a :: ans)).length := by simpa using h0
    have hfirst := hstep 0 (by simp) h0'
    simp only [List.take_zero, List.append_nil, List.get_eq_getElem, List.getElem_cons_zero] at hfirst
    have hrest : generate (s ++ [a]) ans.length = some ans := by
      apply generate_eq_of_steps ans (s ++ [a]) (by simp) (by simp at hlen ⊢; omega)
      intro k hk h0k
      have := hstep (k + 1) (by simp; omega) (by simp)
      simpa [List.take_succ_cons, List.append_assoc] using this
    simp only [List.length_cons, generate, dif_pos (And.intro h0 hL), hfirst, hrest, Option.map_some]

/-- `next` on a list of length T, as `argmaxFirst` of the logits of an equal `Fin T`-indexed prompt. -/
lemma next_eq (s : List (Fin 11)) (h0 : 0 < s.length) (T : ℕ) (hlen : s.length = T) :
    next s h0 = argmaxFirst (logits (fun t : Fin T => s.get (Fin.cast hlen.symm t)) ⟨T - 1, by omega⟩) := by
  subst hlen; rfl

/-- **decimal-to-binary-8bit: greedy generation is correct on every input** (as a transcribed
hand-written program; see the module docstring). -/
theorem dec2bin_correct (x : Fin 256) : generate (prompt x) 8 = some (answer x) := by
  have hans : (answer x).length = 8 := by simp [answer]
  rw [← hans]
  apply generate_eq_of_steps
  · simp [prompt]
  · simp [prompt, hans]
  intro k hk h0
  have hk' : k < 8 := by rw [hans] at hk; exact hk
  have hlen : (prompt x ++ (answer x).take k).length = 4 + k := by simp [prompt, answer]; omega
  rw [next_eq _ h0 (4 + k) hlen]
  have hidx : (⟨4 + k - 1, by omega⟩ : Fin (4 + k)) = ⟨3 + k, by omega⟩ := Fin.ext (by simp only; omega)
  rw [hidx, step x k hk' _ rfl rfl rfl]
  simp only [answer, List.get_eq_getElem, List.getElem_ofFn]

end VWC.Dec2Bin
