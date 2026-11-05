/-
boolean-circuit-5-bits (snapshot ChainedReductionV10_noWo): real-valued correctness on every
input, by mechanism:

1. Block 0 attention at the final `=` position: head h's scores depend only on positions,
   `sTab h j = 16 - 8·hamming(h, j)`, so head h gathers the token features of prompt position h
   up to a leak `1 - softmax_h(h) ≤ λ := 1.35e-3`.
2. Each of the 4 MLPs performs one left-to-right reduction step on the accumulator (dim 0);
   the step is ±1-Lipschitz in the accumulator, so its error grows by ≤ 4λ per step.
3. The head reads the accumulator with gain 10 against a 0.5 baseline: error 17λ < 0.025 suffices.
-/
import VWC.Arch
import VWC.Softmax
import VWC.Tasks.BoolWeights

namespace VWC.Arch.Bool

open Matrix Finset

/-! ### Task specification (evolve/src/task.py, BooleanCircuitTask). Vocab "01&|^=". -/

inductive BOp | and | or | xor
  deriving DecidableEq

def BOp.tok : BOp → Fin 6
  | .and => 2
  | .or => 3
  | .xor => 4

def BOp.eval : BOp → Bool → Bool → Bool
  | .and, a, b => a && b
  | .or, a, b => a || b
  | .xor, a, b => Bool.xor a b

def bitTok (b : Bool) : Fin 6 := if b then 1 else 0

/-- `b0 o0 b1 o1 b2 o2 b3 o3 b4 =` as token ids. -/
def promptIds (b : Fin 5 → Bool) (o : Fin 4 → BOp) : Fin 10 → Fin 6 :=
  ![bitTok (b 0), (o 0).tok, bitTok (b 1), (o 1).tok, bitTok (b 2), (o 2).tok,
    bitTok (b 3), (o 3).tok, bitTok (b 4), 5]

/-- Strict left-to-right evaluation, as in the task generator. -/
def evalChain (b : Fin 5 → Bool) (o : Fin 4 → BOp) : Bool :=
  (o 3).eval ((o 2).eval ((o 1).eval ((o 0).eval (b 0) (b 1)) (b 2)) (b 3)) (b 4)

/-! ### Block 0 attention structure -/

noncomputable def A0 : Attn 36 9 4 4 := boolModel_block0_attn

lemma block0_attn : boolModel_block0.attn = some A0 := rfl

/-- The 4 position-code dims. -/
def posDim : Fin 4 → Fin 36 := ![4, 12, 20, 28]

lemma hT : (10 : ℕ) ≤ 11 := by norm_num

/-- Embedding of token `t` at position `p`. -/
noncomputable def emb (t : Fin 6) (p : Fin 10) : Fin 36 → ℝ :=
  boolModel.tokEmb t + boolModel.posEmb (Fin.castLE hT p)

lemma emb_posDim (t : Fin 6) (p : Fin 10) (c : Fin 4) :
    emb t p (posDim c) = boolModel.posEmb (Fin.castLE hT p) (posDim c) := by
  fin_cases t <;> fin_cases c <;> simp [emb, boolModel, boolModel_tokEmb, posDim]

lemma Wk_mulVec (v : Fin 36 → ℝ) (h : Fin 9) (c : Fin 4) :
    (A0.Wk *ᵥ v) (finProdFinEquiv (h, c)) = v (posDim c) := by
  fin_cases h <;> fin_cases c <;>
    simp [A0, boolModel_block0_attn, boolModel_block0_Wk, mulVec, single_dotProduct, posDim,
      finProdFinEquiv]

/-- Embedding features of a token in dims 0..3: is_one, is_AND, is_OR, is_XOR. -/
def feat (t : Fin 6) (c : Fin 4) : ℝ := if t.val = c.val + 1 then 1 else 0

lemma emb_low (t : Fin 6) (p : Fin 10) (c : Fin 4) :
    emb t p (Fin.castLE (by norm_num) c) = feat t c := by
  fin_cases t <;> fin_cases p <;> fin_cases c <;>
    simp [emb, boolModel, boolModel_tokEmb, boolModel_posEmb, feat]

lemma Wv_mulVec (v : Fin 36 → ℝ) (h : Fin 9) (c : Fin 4) :
    (A0.Wv *ᵥ v) (finProdFinEquiv (h, c)) = v (Fin.castLE (by norm_num) c) := by
  fin_cases h <;> fin_cases c <;>
    simp [A0, boolModel_block0_attn, boolModel_block0_Wv, mulVec, single_dotProduct,
      finProdFinEquiv]

/-- Query of head h at the `=` position (position 9). -/
def qTab : Fin 9 → Fin 4 → ℝ :=
  ![![-8, -8, -8, -8], ![-8, -8, -8, 8], ![-8, -8, 8, -8], ![-8, -8, 8, 8], ![-8, 8, -8, -8],
    ![-8, 8, -8, 8], ![-8, 8, 8, -8], ![-8, 8, 8, 8], ![8, -8, -8, -8]]

lemma Wq_x9 (h : Fin 9) (c : Fin 4) :
    (A0.Wq *ᵥ emb 5 9) (finProdFinEquiv (h, c)) = qTab h c := by
  fin_cases h <;> fin_cases c <;>
    simp [A0, boolModel_block0_attn, boolModel_block0_Wq, boolModel, boolModel_tokEmb,
      boolModel_posEmb, emb, qTab, mulVec, single_dotProduct, finProdFinEquiv]

/-- Attention scores of head h at position 9 on key j: 16 - 8·hamming(h, j). -/
def sTab : Fin 9 → Fin 10 → ℝ :=
  ![![16, 8, 8, 0, 8, 0, 0, -8, 8, 0], ![8, 16, 0, 8, 0, 8, -8, 0, 0, 8],
    ![8, 0, 16, 8, 0, -8, 8, 0, 0, -8], ![0, 8, 8, 16, -8, 0, 0, 8, -8, 0],
    ![8, 0, 0, -8, 16, 8, 8, 0, 0, -8], ![0, 8, -8, 0, 8, 16, 0, 8, -8, 0],
    ![0, -8, 8, 0, 8, 0, 16, 8, -8, -16], ![-8, 0, 0, 8, 0, 8, 8, 16, -16, -8],
    ![8, 0, 0, -8, 0, -8, -8, -16, 16, 8]]

lemma score_eq (h : Fin 9) (j : Fin 10) :
    (∑ c : Fin 4, qTab h c * boolModel.posEmb (Fin.castLE hT j) (posDim c)) / Real.sqrt 4 =
      sTab h j := by
  have : Real.sqrt 4 = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  rw [this]
  fin_cases h <;> fin_cases j <;>
    simp [qTab, sTab, boolModel, boolModel_posEmb, posDim, Fin.sum_univ_succ] <;> norm_num

lemma Iic_nine : Finset.Iic (9 : Fin 10) = Finset.univ := by
  ext j; simp only [Finset.mem_Iic, Finset.mem_univ, iff_true, Fin.le_def]; omega

/-- Head h's attention weights at position 9 are `softmax (sTab h)`, for every prompt ending
in `=`. -/
lemma headWeight_eq (ids : Fin 10 → Fin 6) (hlast : ids 9 = 5) (h : Fin 9) (j : Fin 10) :
    headWeight (fun t c => (A0.Wq *ᵥ emb (ids t) t) (finProdFinEquiv (h, c)))
      (fun t c => (A0.Wk *ᵥ emb (ids t) t) (finProdFinEquiv (h, c))) 9 j =
      softmax (sTab h) j := by
  have hs : ∀ j' : Fin 10,
      ((fun c => (A0.Wq *ᵥ emb (ids 9) 9) (finProdFinEquiv (h, c))) ⬝ᵥ
        (fun c => (A0.Wk *ᵥ emb (ids j') j') (finProdFinEquiv (h, c)))) / Real.sqrt 4 =
        sTab h j' := by
    intro j'
    rw [← score_eq]
    simp only [dotProduct, hlast, Wq_x9, Wk_mulVec, emb_posDim]
  have hle : j ≤ 9 := by rw [Fin.le_def]; omega
  simp only [headWeight, hle, if_true, Iic_nine]
  simp only [softmax, Nat.cast_ofNat, hs]

/-- Gathered feature: head h's weighted average of feature c over the prompt. -/
noncomputable def G (ids : Fin 10 → Fin 6) (h : Fin 9) (c : Fin 4) : ℝ :=
  ∑ j, softmax (sTab h) j * feat (ids j) c

lemma attn9 (ids : Fin 10 → Fin 6) (hlast : ids 9 = 5) (h : Fin 9) (c : Fin 4) :
    A0.apply (fun t => emb (ids t) t) 9 (finProdFinEquiv (h, c)) = G ids h c := by
  have hWo : A0.Wo = 1 := rfl
  unfold Attn.apply
  rw [hWo, Matrix.one_mulVec]
  simp only [Equiv.symm_apply_apply]
  apply Finset.sum_congr rfl
  intro j _
  rw [headWeight_eq ids hlast h j, Wv_mulVec, emb_low]

/-! ### Leak bound λ = 1.35e-3 -/

lemma exp_neg8_le : Real.exp (-8) ≤ 336 / 10 ^ 6 := by
  have he := Real.exp_one_gt_d9
  have h8 : (2.7182818283 : ℝ) ^ 8 ≤ Real.exp 8 := by
    rw [show (8 : ℝ) = (8 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by norm_num) he.le 8
  rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 8) (by norm_num)]
  calc (336 / 10 ^ 6 : ℝ)⁻¹ ≤ (2.7182818283 : ℝ) ^ 8 := by norm_num
    _ ≤ _ := h8

lemma exp_neg16_le : Real.exp (-16) ≤ 12 / 10 ^ 8 := by
  have h := exp_neg8_le
  have : Real.exp (-16) = Real.exp (-8) * Real.exp (-8) := by rw [← Real.exp_add]; norm_num
  rw [this]
  calc _ ≤ (336 / 10 ^ 6 : ℝ) * (336 / 10 ^ 6) :=
        mul_le_mul h h (Real.exp_pos _).le (by norm_num)
    _ ≤ _ := by norm_num

lemma exp_neg24_le : Real.exp (-24) ≤ 12 / 10 ^ 8 :=
  (Real.exp_le_exp.2 (by norm_num)).trans exp_neg16_le

lemma exp_neg32_le : Real.exp (-32) ≤ 12 / 10 ^ 8 :=
  (Real.exp_le_exp.2 (by norm_num)).trans exp_neg16_le

/-- λ: the off-target softmax mass of every head. -/
noncomputable def lam : ℝ := 135 / 10 ^ 5

lemma leak (h : Fin 9) : 1 - softmax (sTab h) h.castSucc ≤ lam := by
  refine (one_sub_softmax_le _ _).trans ?_
  have e8 := exp_neg8_le
  have e16 := exp_neg16_le
  have e24 := exp_neg24_le
  have e32 := exp_neg32_le
  fin_cases h <;> simp [sTab, Fin.sum_univ_succ, lam] <;> norm_num <;> linarith

/-- Gathered feature error: any [0,1]-valued token feature is recovered up to λ. -/
lemma gather_close (ids : Fin 10 → Fin 6) (h : Fin 9) (f : Fin 6 → ℝ)
    (hf : ∀ t, 0 ≤ f t ∧ f t ≤ 1) :
    |∑ j, softmax (sTab h) j * f (ids j) - f (ids h.castSucc)| ≤ lam :=
  (convex_dev _ (fun j => f (ids j)) (softmax_nonneg _) (softmax_sum _) (fun j => hf _) _).trans
    (leak h)

/-! ### MLP reduction steps -/

/-- Change of the accumulator made by one MLP. -/
noncomputable def stepΔ (acc and_ or_ xor_ bit : ℝ) : ℝ :=
  2 * relu (-acc + or_ + xor_ + bit - 3 / 2) - 2 * relu (acc + and_ - bit - 3 / 2) -
    2 * relu (acc + xor_ + bit - 5 / 2)

lemma mlp0 (v : Fin 36 → ℝ) : boolModel_block0.mlp v =
    fun d => if d = 0 then stepΔ (v 0) (v 5) (v 6) (v 7) (v 8) else 0 := by
  funext d
  by_cases hd : d = 0
  · subst hd
    simp [Block.mlp, boolModel_block0, boolModel_block0_W1, boolModel_block0_b1,
      boolModel_block0_W2, boolModel_block0_b2, mulVec, single_dotProduct, stepΔ]
    ring_nf
  · rw [if_neg hd]
    fin_cases d <;> simp_all [Block.mlp, boolModel_block0, boolModel_block0_W1,
      boolModel_block0_b1, boolModel_block0_W2, boolModel_block0_b2, mulVec, single_dotProduct]

lemma mlp1 (v : Fin 36 → ℝ) : boolModel_block1.mlp v =
    fun d => if d = 0 then stepΔ (v 0) (v 13) (v 14) (v 15) (v 16) else 0 := by
  funext d
  by_cases hd : d = 0
  · subst hd
    simp [Block.mlp, boolModel_block1, boolModel_block1_W1, boolModel_block1_b1,
      boolModel_block1_W2, boolModel_block1_b2, mulVec, single_dotProduct, stepΔ]
    ring_nf
  · rw [if_neg hd]
    fin_cases d <;> simp_all [Block.mlp, boolModel_block1, boolModel_block1_W1,
      boolModel_block1_b1, boolModel_block1_W2, boolModel_block1_b2, mulVec, single_dotProduct]

lemma mlp2 (v : Fin 36 → ℝ) : boolModel_block2.mlp v =
    fun d => if d = 0 then stepΔ (v 0) (v 21) (v 22) (v 23) (v 24) else 0 := by
  funext d
  by_cases hd : d = 0
  · subst hd
    simp [Block.mlp, boolModel_block2, boolModel_block2_W1, boolModel_block2_b1,
      boolModel_block2_W2, boolModel_block2_b2, mulVec, single_dotProduct, stepΔ]
    ring_nf
  · rw [if_neg hd]
    fin_cases d <;> simp_all [Block.mlp, boolModel_block2, boolModel_block2_W1,
      boolModel_block2_b1, boolModel_block2_W2, boolModel_block2_b2, mulVec, single_dotProduct]

lemma mlp3 (v : Fin 36 → ℝ) : boolModel_block3.mlp v =
    fun d => if d = 0 then stepΔ (v 0) (v 29) (v 30) (v 31) (v 32) else 0 := by
  funext d
  by_cases hd : d = 0
  · subst hd
    simp [Block.mlp, boolModel_block3, boolModel_block3_W1, boolModel_block3_b1,
      boolModel_block3_W2, boolModel_block3_b2, mulVec, single_dotProduct, stepΔ]
    ring_nf
  · rw [if_neg hd]
    fin_cases d <;> simp_all [Block.mlp, boolModel_block3, boolModel_block3_W1,
      boolModel_block3_b1, boolModel_block3_W2, boolModel_block3_b2, mulVec, single_dotProduct]

def bv (b : Bool) : ℝ := if b then 1 else 0

def isAnd (o : BOp) : ℝ := if o = .and then 1 else 0
def isOrX (o : BOp) : ℝ := if o = .or ∨ o = .xor then 1 else 0
def isXor (o : BOp) : ℝ := if o = .xor then 1 else 0

lemma relu_zero {x : ℝ} (h : x ≤ 0) : relu x = 0 := max_eq_right h
lemma relu_self {x : ℝ} (h : 0 ≤ x) : relu x = x := max_eq_left h

section Steps
variable {acc and_ or_ xor_ bit ε l : ℝ} (hl : 0 ≤ l) (hsmall : ε + 2 * l ≤ 1 / 4)
include hl hsmall

set_option maxHeartbeats 1000000 in
lemma step_and (a b : Bool) (hacc : |acc - bv a| ≤ ε) (hand : |and_ - 1| ≤ l)
    (hox : |or_ + xor_| ≤ l) (hxor : |xor_| ≤ l) (hbit : |bit - bv b| ≤ l) :
    |acc + stepΔ acc and_ or_ xor_ bit - bv (a && b)| ≤ ε + 4 * l := by
  rw [abs_le] at hacc hand hox hxor hbit
  cases a <;> cases b <;> simp only [bv, Bool.and_true, Bool.and_false, Bool.true_and,
      Bool.false_and, if_true, if_false, Bool.false_eq_true] at * <;>
    simp only [stepΔ] <;>
    first
    | rw [relu_zero (by linarith), relu_zero (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith
    | rw [relu_zero (by linarith), relu_self (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith

set_option maxHeartbeats 1000000 in
lemma step_or (a b : Bool) (hacc : |acc - bv a| ≤ ε) (hand : |and_| ≤ l)
    (hox : |or_ + xor_ - 1| ≤ l) (hxor : |xor_| ≤ l) (hbit : |bit - bv b| ≤ l) :
    |acc + stepΔ acc and_ or_ xor_ bit - bv (a || b)| ≤ ε + 4 * l := by
  rw [abs_le] at hacc hand hox hxor hbit
  cases a <;> cases b <;> simp only [bv, Bool.or_true, Bool.or_false, Bool.true_or,
      Bool.false_or, if_true, if_false, Bool.false_eq_true] at * <;>
    simp only [stepΔ] <;>
    first
    | rw [relu_zero (by linarith), relu_zero (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith
    | rw [relu_self (by linarith), relu_zero (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith

set_option maxHeartbeats 1000000 in
lemma step_xor (a b : Bool) (hacc : |acc - bv a| ≤ ε) (hand : |and_| ≤ l)
    (hox : |or_ + xor_ - 1| ≤ l) (hxor : |xor_ - 1| ≤ l) (hbit : |bit - bv b| ≤ l) :
    |acc + stepΔ acc and_ or_ xor_ bit - bv (Bool.xor a b)| ≤ ε + 4 * l := by
  rw [abs_le] at hacc hand hox hxor hbit
  cases a <;> cases b <;> simp only [bv, Bool.xor_true, Bool.xor_false, Bool.true_xor,
      Bool.false_xor, Bool.not_true, Bool.not_false, if_true, if_false,
      Bool.false_eq_true] at * <;>
    simp only [stepΔ] <;>
    first
    | rw [relu_zero (by linarith), relu_zero (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith
    | rw [relu_self (by linarith), relu_zero (by linarith), relu_zero (by linarith), abs_le];
        constructor <;> linarith
    | rw [relu_zero (by linarith), relu_zero (by linarith), relu_self (by linarith), abs_le];
        constructor <;> linarith

end Steps

lemma step_robust (o : BOp) (a b : Bool) {acc and_ or_ xor_ bit ε l : ℝ} (hl : 0 ≤ l)
    (hsmall : ε + 2 * l ≤ 1 / 4) (hacc : |acc - bv a| ≤ ε) (hand : |and_ - isAnd o| ≤ l)
    (hox : |or_ + xor_ - isOrX o| ≤ l) (hxor : |xor_ - isXor o| ≤ l) (hbit : |bit - bv b| ≤ l) :
    |acc + stepΔ acc and_ or_ xor_ bit - bv (o.eval a b)| ≤ ε + 4 * l := by
  cases o <;> simp only [isAnd, isOrX, isXor, reduceCtorEq, or_false, false_or, or_true,
    if_true, if_false, sub_zero] at hand hox hxor
  · exact step_and hl hsmall a b hacc hand hox hxor hbit
  · exact step_or hl hsmall a b hacc hand hox hxor hbit
  · exact step_xor hl hsmall a b hacc hand hox hxor hbit

/-! ### Assembly -/

lemma apply_none {F H dqk dv : ℕ} (B : Block 36 F H dqk dv) (hB : B.attn = none)
    (hln : B.ln2 = .id) (z : Fin 10 → Fin 36 → ℝ) (i : Fin 10) :
    B.apply z i = z i + B.mlp (z i) := by
  simp [Block.apply, hB, hln, Norm.apply]

lemma apply_b0 (z : Fin 10 → Fin 36 → ℝ) (i : Fin 10) :
    boolModel_block0.apply z i = (z i + A0.apply z i) + boolModel_block0.mlp (z i + A0.apply z i) := by
  simp [Block.apply, boolModel_block0, A0, Norm.apply]

lemma Wh_mulVec (v : Fin 36 → ℝ) : boolModel.Wh *ᵥ v = ![-10 * v 0 + 1 / 2 * v 4, 10 * v 0, 0, 0, 0, 0] := by
  ext t; fin_cases t <;> simp [boolModel, boolModel_Wh, mulVec, single_dotProduct] <;> ring

lemma lam_nonneg : 0 ≤ lam := by norm_num [lam]

lemma feat_unit (t : Fin 6) (c : Fin 4) : 0 ≤ feat t c ∧ feat t c ≤ 1 := by
  unfold feat; split_ifs <;> norm_num

lemma featOX_unit (t : Fin 6) : 0 ≤ feat t 2 + feat t 3 ∧ feat t 2 + feat t 3 ≤ 1 := by
  fin_cases t <;> simp [feat]

/-- Gathered feature c of head h, close to the true feature of prompt position h. -/
lemma G_close (ids : Fin 10 → Fin 6) (h : Fin 9) (c : Fin 4) :
    |G ids h c - feat (ids h.castSucc) c| ≤ lam :=
  gather_close ids h (fun t => feat t c) (fun t => feat_unit t c)

lemma GOX_close (ids : Fin 10 → Fin 6) (h : Fin 9) :
    |G ids h 2 + G ids h 3 - (feat (ids h.castSucc) 2 + feat (ids h.castSucc) 3)| ≤ lam := by
  have := gather_close ids h (fun t => feat t 2 + feat t 3) featOX_unit
  simpa [G, ← Finset.sum_add_distrib, mul_add] using this

lemma feat_bit (b : Bool) : feat (bitTok b) 0 = bv b := by cases b <;> simp [feat, bitTok, bv]
lemma feat_op_one (o : BOp) : feat o.tok 0 = 0 := by cases o <;> simp [feat, BOp.tok]
lemma feat_and (o : BOp) : feat o.tok 1 = isAnd o := by cases o <;> simp [feat, BOp.tok, isAnd]
lemma feat_xor (o : BOp) : feat o.tok 3 = isXor o := by cases o <;> simp [feat, BOp.tok, isXor]
lemma feat_ox (o : BOp) : feat o.tok 2 + feat o.tok 3 = isOrX o := by
  cases o <;> simp [feat, BOp.tok, isOrX]

/-- Residual after block-0 attention at the `=` position, on the dims the rest reads. -/
lemma y_dim (ids : Fin 10 → Fin 6) (hlast : ids 9 = 5) (h : Fin 9) (c : Fin 4) :
    (emb 5 9 + A0.apply (fun t => emb (ids t) t) 9) (finProdFinEquiv (h, c)) =
      emb 5 9 (finProdFinEquiv (h, c)) + G ids h c := by
  rw [Pi.add_apply, attn9 ids hlast]

set_option maxHeartbeats 4000000 in
/-- **boolean-circuit-5-bits, real-valued correctness.** For all 5 bits and 4 operators, the
model's greedy prediction after `b0 o0 b1 o1 b2 o2 b3 o3 b4 =` is the left-to-right value. -/
theorem bool_correct (b : Fin 5 → Bool) (o : Fin 4 → BOp) :
    argmaxFirst (boolModel.logits hT (promptIds b o) 9) = bitTok (evalChain b o) := by
  set ids := promptIds b o with hids
  have hlast : ids 9 = 5 := rfl
  set x : Fin 10 → Fin 36 → ℝ := fun t => emb (ids t) t with hx
  set y := emb 5 9 + A0.apply x 9 with hy
  -- the dims of y used downstream
  have yd : ∀ (h : Fin 9) (c : Fin 4), y (finProdFinEquiv (h, c)) =
      emb 5 9 (finProdFinEquiv (h, c)) + G ids h c := fun h c => y_dim ids hlast h c
  have e0 : emb 5 9 0 = 0 := by simp [emb, boolModel, boolModel_tokEmb, boolModel_posEmb]
  have e4 : emb 5 9 4 = 1 := by simp [emb, boolModel, boolModel_tokEmb, boolModel_posEmb]
  have ez : ∀ d : Fin 36, d ∈ ({5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32} :
      Finset (Fin 36)) → emb 5 9 d = 0 := by
    intro d hd; fin_cases hd <;> simp [emb, boolModel, boolModel_tokEmb, boolModel_posEmb]
  have fpe : ∀ (h : Fin 9) (c : Fin 4) (d : Fin 36), d.val = c.val + 4 * h.val →
      finProdFinEquiv (h, c) = d := fun h c d hd => Fin.ext (by simp [finProdFinEquiv, hd])
  -- feature values y at the needed dims
  have Y : ∀ (h : Fin 9) (c : Fin 4) (d : Fin 36), d.val = c.val + 4 * h.val →
      emb 5 9 d = 0 → y d = G ids h c := by
    intro h c d hd hz; rw [← fpe h c d hd, yd, fpe h c d hd, hz, zero_add]
  have y0 : y 0 = G ids 0 0 := Y 0 0 0 rfl e0
  have y4 : y 4 = 1 + G ids 1 0 := by rw [← fpe 1 0 4 rfl, yd, fpe 1 0 4 rfl, e4]
  -- the model's logits at position 9
  have hlog : boolModel.logits hT ids 9 = boolModel.Wh *ᵥ
      (let r1 := y + boolModel_block0.mlp y
       let r2 := r1 + boolModel_block1.mlp r1
       let r3 := r2 + boolModel_block2.mlp r2
       r3 + boolModel_block3.mlp r3) := by
    simp only [Model.logits, boolModel, applyBlocks, Norm.apply, boolModel_bh, add_zero]
    rw [apply_none _ rfl rfl, apply_none _ rfl rfl, apply_none _ rfl rfl, apply_b0]
    rfl
  -- feature bounds at the gathered slots
  have l0 := lam_nonneg
  have hl : lam = 135 / 10 ^ 5 := rfl
  have zd : ∀ d : Fin 36, d ∈ ({5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32} :
      Finset (Fin 36)) → emb 5 9 d = 0 := ez
  have hb0 : |y 0 - bv (b 0)| ≤ lam := by
    rw [y0]; simpa [ids, promptIds, feat_bit] using G_close ids 0 0
  have opB : ∀ (k : Fin 4) (h : Fin 9) (dA dO dX : Fin 36), h.castSucc = (2 * k.val + 1 : ℕ) →
      dA.val = 1 + 4 * h.val → dO.val = 2 + 4 * h.val → dX.val = 3 + 4 * h.val →
      emb 5 9 dA = 0 → emb 5 9 dO = 0 → emb 5 9 dX = 0 →
      ids h.castSucc = (o k).tok →
      |y dA - isAnd (o k)| ≤ lam ∧ |y dO + y dX - isOrX (o k)| ≤ lam ∧
        |y dX - isXor (o k)| ≤ lam := by
    intro k h dA dO dX _ hA hO hX zA zO zX hid
    rw [Y h 1 dA hA zA, Y h 2 dO hO zO, Y h 3 dX hX zX]
    refine ⟨?_, ?_, ?_⟩
    · simpa [hid, feat_and] using G_close ids h 1
    · simpa [hid, feat_ox] using GOX_close ids h
    · simpa [hid, feat_xor] using G_close ids h 3
  have bitB : ∀ (k : Fin 5) (h : Fin 9) (d : Fin 36), d.val = 4 * h.val → emb 5 9 d = 0 →
      ids h.castSucc = bitTok (b k) → |y d - bv (b k)| ≤ lam := by
    intro k h d hd zd' hid
    rw [Y h 0 d (by simpa using hd) zd']
    simpa [hid, feat_bit] using G_close ids h 0
  obtain ⟨a1, x1, o1⟩ := opB 0 1 5 6 7 rfl rfl rfl rfl (zd 5 (by simp)) (zd 6 (by simp))
    (zd 7 (by simp)) rfl
  obtain ⟨a2, x2, o2⟩ := opB 1 3 13 14 15 rfl rfl rfl rfl (zd 13 (by simp)) (zd 14 (by simp))
    (zd 15 (by simp)) rfl
  obtain ⟨a3, x3, o3⟩ := opB 2 5 21 22 23 rfl rfl rfl rfl (zd 21 (by simp)) (zd 22 (by simp))
    (zd 23 (by simp)) rfl
  obtain ⟨a4, x4, o4⟩ := opB 3 7 29 30 31 rfl rfl rfl rfl (zd 29 (by simp)) (zd 30 (by simp))
    (zd 31 (by simp)) rfl
  have b1 := bitB 1 2 8 rfl (zd 8 (by simp)) rfl
  have b2 := bitB 2 4 16 rfl (zd 16 (by simp)) rfl
  have b3 := bitB 3 6 24 rfl (zd 24 (by simp)) rfl
  have b4 := bitB 4 8 32 rfl (zd 32 (by simp)) rfl
  -- the accumulator chain
  set A1 := y 0 + stepΔ (y 0) (y 5) (y 6) (y 7) (y 8)
  set A2 := A1 + stepΔ A1 (y 13) (y 14) (y 15) (y 16)
  set A3 := A2 + stepΔ A2 (y 21) (y 22) (y 23) (y 24)
  set A4 := A3 + stepΔ A3 (y 29) (y 30) (y 31) (y 32)
  have s1 := step_robust (o 0) (b 0) (b 1) l0 (by rw [hl]; norm_num) hb0 a1 x1 o1 b1
  have s2 := step_robust (o 1) _ (b 2) l0 (by rw [hl]; norm_num) s1 a2 x2 o2 b2
  have s3 := step_robust (o 2) _ (b 3) l0 (by rw [hl]; norm_num) s2 a3 x3 o3 b3
  have s4 := step_robust (o 3) _ (b 4) l0 (by rw [hl]; norm_num) s3 a4 x4 o4 b4
  have hE : (o 3).eval ((o 2).eval ((o 1).eval ((o 0).eval (b 0) (b 1)) (b 2)) (b 3)) (b 4) =
      evalChain b o := rfl
  rw [hE] at s4
  -- the baseline dim 4
  have hd4 : |y 4 - 1| ≤ lam := by
    rw [y4]; simpa [ids, promptIds, feat_op_one] using G_close ids 1 0
  -- read out
  have hr : (let r1 := y + boolModel_block0.mlp y
       let r2 := r1 + boolModel_block1.mlp r1
       let r3 := r2 + boolModel_block2.mlp r2
       r3 + boolModel_block3.mlp r3) = fun d => if d = 0 then A4 else y d := by
    funext d
    by_cases hd : d = 0
    · subst hd; simp [mlp0, mlp1, mlp2, mlp3, A1, A2, A3, A4]
    · simp [mlp0, mlp1, mlp2, mlp3, hd]
  rw [hlog, hr, Wh_mulVec]
  simp only [if_true, show (4 : Fin 36) ≠ 0 by decide, if_false]
  rw [abs_le] at s4 hd4
  apply argmaxFirst_eq_of_lt
  intro j hj
  cases hev : evalChain b o <;> simp only [hev, bv, bitTok, if_true, if_false,
    Bool.false_eq_true] at s4 hj ⊢ <;> fin_cases j <;> simp at hj ⊢ <;> rw [hl] at s4 hd4 <;>
    nlinarith [s4.1, s4.2, hd4.1, hd4.2]

end VWC.Arch.Bool
