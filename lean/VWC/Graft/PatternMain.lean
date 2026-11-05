/-
Phase 3.1: the multi-token refusal head grafted into any host.

`pattern_main`: for any GPT-2-style host whose final residual variance is at most σ², with
parameters meeting `Good` and the readout condition `G₀ Γ < Lg A δ κ2lo (1 − lam3)`
(G₀ ≥ every `Host.gap`):
* off trigger (the pattern occurs nowhere), the grafted logit of v at position i is within
  `Host.etaAt Δc u_i v` of the host's, u_i the host's own final-residual variance at i (theorem (A′),
  position-wise): partial matches move the circuit energy by at most Δc but never the readout;
* on trigger (the pattern occurs, ending anywhere), the grafted greedy next token is r.
-/
import VWC.Graft.PatternExact

namespace VWC.Graft.Pattern

open Matrix Finset VWC.Arch VWC.Graft

variable {M : ℕ} (n : ℕ) (p : Par) (D : ℕ) {V : ℕ} (P : Fin (M + 1) → Fin V) {T : ℕ}
  (ids : Fin (T + 1) → Fin V) (u : Fin (T + 1) → ℝ) {L : ℕ} {σ2 : ℝ}

lemma sum_slot (g : Slot M n → ℝ) : ∑ j : Fin (slots M n), g ((slotEquiv M n).symm j) = ∑ a, g a :=
  Equiv.sum_comp (slotEquiv M n).symm g

/-! ### Zero-sum writes and features -/

lemma blk1_zs : (blk1 (M := M) (n := n) p).ZeroSum := by
  refine ⟨fun c => ?_, ?_, fun f => ?_, ?_⟩
  · simp only [blk1, attn1]; rw [sum_slot n (fun a => oslot a _)]
    simp [Fintype.sum_sum_type, Fintype.sum_prod_type, oslot, pmsign, Fin.sum_univ_two]
  · simp [blk1, attn1]
  · simp only [blk1]; rw [sum_slot n (fun a => fslot p.δ a _)]
    simp [Fintype.sum_sum_type, Fintype.sum_prod_type, fslot, flipsign, Fin.sum_univ_four]
  · simp [blk1]

lemma blk2_zs [NeZero M] : (blk2 (M := M) (n := n) p).ZeroSum := by
  refine ⟨fun c => ?_, ?_, fun f => ?_, ?_⟩
  · simp [blk2]
  · simp [blk2]
  · simp only [blk2]; rw [sum_slot n (fun a => mslot p.δ a _)]
    simp [Fintype.sum_sum_type, Fintype.sum_prod_type, mslot, flipsign, Fin.sum_univ_four]
  · simp [blk2]

lemma blk3_zs [NeZero M] : (blk3 (M := M) (n := n) p).ZeroSum := by
  refine ⟨fun c => ?_, ?_, fun f => ?_, ?_⟩
  · simp only [blk3, attn3]; rw [sum_slot n (fun a => gslot a _)]
    simp [Fintype.sum_sum_type, Fintype.sum_prod_type, gslot, pmsign, Fin.sum_univ_two]
  · simp [blk3, attn3]
  · simp [blk3]
  · simp [blk3]

lemma circ_zs [NeZero M] (r : Fin V) : ∀ b ∈ (circ (n := n) p L P r).blocks, b.ZeroSum := by
  intro b hb
  simp only [circ, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl
  · exact blk1_zs n p
  · exact blk2_zs n p
  · exact blk3_zs n p

lemma circ_feat [NeZero M] (r : Fin V) (v : Fin V) (q : Fin L) :
    ∑ j, ((circ (n := n) p L P r).sTok v j + (circ (n := n) p L P r).sPos q j) = 0 ∧
    ∑ j, ((circ (n := n) p L P r).sTok v j + (circ (n := n) p L P r).sPos q j) ^ 2 =
      (circ (n := n) p L P r).E := by
  simp only [circ]
  constructor
  · rw [sum_slot n (fun a => Ftok P p.δ v a + Fpos p.δ q a)]; exact feat_sum P p.δ v q
  · rw [sum_slot n (fun a => (Ftok P p.δ v a + Fpos p.δ q a) ^ 2)]; exact feat_sq P p.δ v q

lemma circ_s0 [NeZero M] (r : Fin V) (hT : T + 1 ≤ L) :
    (circ (n := n) p L P r).s0 hT ids = stv n p P ids 0 0 0 0 := by
  funext t
  apply ext_slot; intro a
  have := congrFun (St_zero (n := n) P p.δ t (ids t)) a
  simp only [Circuit.s0, circ, Pi.add_apply, Equiv.symm_apply_apply, stv, vecOf_apply, Pi.zero_apply,
    Fin.coe_castLE] at this ⊢
  exact this

/-! ### The flag -/

lemma andv01 (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) : andv n p D P ids u i = 0 ∨ andv n p D P ids u i = 1 := by
  classical
  rw [andv_exact n p D P ids u G hT hu]; exact ind_cases _

lemma x4_eq [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (j : Fin (T + 1)) :
    rd (sMatch 0) (sMatch 1) ⬝ᵥ x4 n p D P ids u j =
      p.δ * andv n p D P ids u j / Real.sqrt (g2 n p D P ids u j) := by
  unfold x4 stv
  simp only [Pi.zero_apply]
  rw [cLN_vec, gden_St n p D P ids u 1 j _ _ _ 0 (bit01 n p D P ids u G hT hu j)
    (andv01 n p D P ids u G hT hu j), rd_vec]
  simp [St, sMatch, qd, g2]; ring

lemma x4_const [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (j : Fin (T + 1)) :
    rd (sConst 0) (sConst 1) ⬝ᵥ x4 n p D P ids u j = p.δ / Real.sqrt (g2 n p D P ids u j) := by
  unfold x4 stv
  simp only [Pi.zero_apply]
  rw [cLN_vec, gden_St n p D P ids u 1 j _ _ _ 0 (bit01 n p D P ids u G hT hu j)
    (andv01 n p D P ids u G hT hu j), rd_vec]
  simp [St, sConst, pm, g2]; ring

/-- Block-3 score of key j at query i. -/
noncomputable def s3 [NeZero M] (i j : Fin (T + 1)) : ℝ :=
  p.fq * p.fk * p.δ ^ 2 * andv n p D P ids u j /
    (Real.sqrt (g2 n p D P ids u i) * Real.sqrt (g2 n p D P ids u j)) / Real.sqrt ((n + 1 : ℕ) : ℝ)

lemma fl_eq [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) :
    fl n p D P ids u i = ∑ j, smax (fun j => s3 n p D P ids u i j) i j *
      (p.A * (p.δ * andv n p D P ids u j / Real.sqrt (g2 n p D P ids u j))) := by
  unfold fl
  refine Finset.sum_congr rfl (fun j _ => ?_)
  rw [hw_smax, smul_dotProduct, x4_eq n p D P ids u G hT hu, smul_eq_mul]
  congr 2
  funext j'
  show (∑ c, (q3row p (0, c) ⬝ᵥ x4 n p D P ids u i) * (k3row p (0, c) ⬝ᵥ x4 n p D P ids u j')) /
      Real.sqrt ((n + 1 : ℕ) : ℝ) = _
  rw [Finset.sum_eq_single (0 : Fin (n + 1))]
  · simp only [q3row, k3row, and_self, if_true, smul_dotProduct, x4_const n p D P ids u G hT hu,
      x4_eq n p D P ids u G hT hu, s3, smul_eq_mul]
    ring
  · intro c _ hc; simp [q3row, hc]
  · simp

/-- The flag values: 0 ≤ A δ andv_j / √g2_j ≤ A δ κhi. -/
lemma fl_val [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (j : Fin (T + 1)) :
    0 ≤ p.A * (p.δ * andv n p D P ids u j / Real.sqrt (g2 n p D P ids u j)) ∧
    p.A * (p.δ * andv n p D P ids u j / Real.sqrt (g2 n p D P ids u j)) ≤ p.A * p.δ * κhi D (slots M n) := by
  obtain ⟨hpos, -, hhi, -⟩ := g2_bounds n p D P ids u G hT hu j
  have ha := andv01 n p D P ids u G hT hu j
  have hsq := Real.sqrt_pos.2 hpos
  have hAδ : 0 ≤ p.A * p.δ := mul_nonneg G.hA G.hδ.le
  rcases ha with h | h <;> rw [h]
  · simp; unfold κhi; positivity
  · refine ⟨by have := G.hδ; have := G.hA; positivity, ?_⟩
    rw [mul_one]
    calc p.A * (p.δ / Real.sqrt (g2 n p D P ids u j)) = p.A * p.δ * (1 / Real.sqrt (g2 n p D P ids u j)) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hhi hAδ

lemma fl_bounds [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) :
    0 ≤ fl n p D P ids u i ∧ fl n p D P ids u i ≤ p.A * p.δ * κhi D (slots M n) := by
  rw [fl_eq n p D P ids u G hT hu]
  have hw0 := fun j => smax_nonneg (fun j => s3 n p D P ids u i j) i j
  have hval := fun j => fl_val n p D P ids u G hT hu j
  refine ⟨Finset.sum_nonneg (fun j _ => mul_nonneg (hw0 j) (hval j).1), ?_⟩
  calc _ ≤ ∑ j, smax (fun j => s3 n p D P ids u i j) i j * (p.A * p.δ * κhi D (slots M n)) :=
        Finset.sum_le_sum (fun j _ => mul_le_mul_of_nonneg_left (hval j).2 (hw0 j))
    _ = p.A * p.δ * κhi D (slots M n) := by
        rw [← Finset.sum_mul, smax_sum, one_mul]

lemma fl_zero [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (hno : ∀ t, ¬ endsAt P ids t) (i : Fin (T + 1)) : fl n p D P ids u i = 0 := by
  classical
  rw [fl_eq n p D P ids u G hT hu]
  apply Finset.sum_eq_zero; intro j _
  rw [andv_exact n p D P ids u G hT hu, ind, if_neg (hno j)]; simp

/-- The flag after a match: at least A δ κ2lo (1 − lam3). -/
lemma fl_lo [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (t0 : Fin (T + 1)) (h0 : endsAt P ids t0) :
    p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2) ≤
      fl n p D P ids u (Fin.last T) := by
  classical
  rw [fl_eq n p D P ids u G hT hu]
  set i := Fin.last T
  set w := fun j => smax (fun j => s3 n p D P ids u i j) i j
  set ρ := p.fq * p.fk * p.δ ^ 2 * κ2lo M D (slots M n) p σ2 ^ 2 / Real.sqrt ((n + 1 : ℕ) : ℝ)
  have hk20 := κ2lo_pos n p D G
  have hn1 : 0 < Real.sqrt ((n + 1 : ℕ) : ℝ) := Real.sqrt_pos.2 (by positivity)
  have hc : 0 ≤ p.fq * p.fk * p.δ ^ 2 := mul_nonneg (mul_nonneg G.hfq G.hfk) (sq_nonneg _)
  have hgk : ∀ j, κ2lo M D (slots M n) p σ2 ≤ 1 / Real.sqrt (g2 n p D P ids u j) :=
    fun j => (g2_bounds n p D P ids u G hT hu j).2.1
  have hgp : ∀ j, 0 < Real.sqrt (g2 n p D P ids u j) :=
    fun j => Real.sqrt_pos.2 (g2_bounds n p D P ids u G hT hu j).1
  -- the matched set S = {andv = 1} contains t0; outside S the score is 0
  have hm0 : andv n p D P ids u t0 = 1 := by rw [andv_exact n p D P ids u G hT hu, ind, if_pos h0]
  have hprod : κ2lo M D (slots M n) p σ2 ^ 2 ≤
      1 / (Real.sqrt (g2 n p D P ids u i) * Real.sqrt (g2 n p D P ids u t0)) := by
    rw [← one_div_mul_one_div, sq]; exact mul_le_mul (hgk i) (hgk t0) hk20.le (by positivity)
  have hρ : ρ ≤ s3 n p D P ids u i t0 := by
    unfold s3; rw [hm0, mul_one]
    apply div_le_div_of_nonneg_right _ hn1.le
    rw [div_eq_mul_one_div (p.fq * p.fk * p.δ ^ 2)]
    exact mul_le_mul_of_nonneg_left hprod hc
  have hout : ∀ j, ¬ (andv n p D P ids u j = 1) → s3 n p D P ids u i j ≤ 0 := by
    intro j hj
    have : andv n p D P ids u j = 0 := (andv01 n p D P ids u G hT hu j).resolve_right hj
    unfold s3; rw [this]; simp
  have hmass := smax_mass (fun j => s3 n p D P ids u i j) i t0 (Fin.le_last _) ρ hρ
    (fun j => andv n p D P ids u j = 1) hout
  have hval := fun j => fl_val n p D P ids u G hT hu j
  have hAδκ : 0 ≤ p.A * p.δ * κ2lo M D (slots M n) p σ2 := by
    have := G.hA; have := G.hδ; positivity
  -- termwise: matched keys carry at least A δ κ2lo
  have hterm : ∀ j, (if andv n p D P ids u j = 1 then w j * (p.A * p.δ * κ2lo M D (slots M n) p σ2) else 0) ≤
      w j * (p.A * (p.δ * andv n p D P ids u j / Real.sqrt (g2 n p D P ids u j))) := by
    intro j
    have hw := smax_nonneg (fun j => s3 n p D P ids u i j) i j
    split_ifs with hj
    · apply mul_le_mul_of_nonneg_left _ hw
      rw [hj, mul_one, mul_assoc, div_eq_mul_one_div]
      exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (hgk j) G.hδ.le) G.hA
    · exact mul_nonneg hw (hval j).1
  have hsum : (∑ j, if andv n p D P ids u j = 1 then w j * (p.A * p.δ * κ2lo M D (slots M n) p σ2) else 0) =
      p.A * p.δ * κ2lo M D (slots M n) p σ2 * ∑ j, (if andv n p D P ids u j = 1 then w j else 0) := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl (fun j _ => ?_); split_ifs <;> ring
  have hLT : ((T + 1 : ℕ) : ℝ) ≤ L := by exact_mod_cast hT
  have hlam : 1 - lam3 M n D (slots M n) L p σ2 ≤ 1 - ((T + 1 : ℕ) : ℝ) * Real.exp (-ρ) := by
    unfold lam3
    have : ((T + 1 : ℕ) : ℝ) * Real.exp (-ρ) ≤ L * Real.exp (-ρ) :=
      mul_le_mul_of_nonneg_right hLT (Real.exp_pos _).le
    have e : -(p.fq * p.fk * p.δ ^ 2 * κ2lo M D (slots M n) p σ2 ^ 2 / Real.sqrt ((n + 1 : ℕ) : ℝ)) = -ρ := rfl
    rw [e]; linarith
  calc p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2)
      ≤ p.A * p.δ * κ2lo M D (slots M n) p σ2 * ∑ j, (if andv n p D P ids u j = 1 then w j else 0) :=
        mul_le_mul_of_nonneg_left (hlam.trans hmass) hAδκ
    _ = _ := hsum.symm
    _ ≤ _ := Finset.sum_le_sum (fun j _ => hterm j)

/-! ### Final state and readout -/

noncomputable def gfin [NeZero M] (εf : ℝ) (i : Fin (T + 1)) : ℝ :=
  (D * (u i + εf) + 2 * ∑ d, acp n p D P ids u i d ^ 2 + 2 * fl n p D P ids u i ^ 2) / (D + slots M n)

lemma run_all [NeZero M] (r : Fin V) (hT : T + 1 ≤ L) :
    runBlocks D (circ (n := n) p L P r).E (circ (n := n) p L P r).blocks u
      ((circ (n := n) p L P r).s0 hT ids) =
    stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) (fl n p D P ids u) := by
  rw [circ_s0 n p P ids r hT]
  simp only [circ, runBlocks]
  rw [run1, run2, run3]

lemma energy_final [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (i : Fin (T + 1)) :
    ∑ j, stv n p P ids (acp n p D P ids u) (bit n p D P ids u) (andv n p D P ids u) (fl n p D P ids u) i j ^ 2 =
      Ef p.δ M n + 2 * ∑ d, acp n p D P ids u i d ^ 2 + 2 * fl n p D P ids u i ^ 2 := by
  unfold stv
  rw [energy_vec, St_energy P p.δ i (ids i) _ _ _ _ (bit01 n p D P ids u G hT hu i)
    (andv01 n p D P ids u G hT hu i)]

lemma readout [NeZero M] (G : Good M n D L p σ2) (hT : T + 1 ≤ L) (hu : ∀ t, 0 ≤ u t ∧ u t ≤ σ2)
    (r v : Fin V) (εf : ℝ) (i : Fin (T + 1)) :
    (circ (n := n) p L P r).Wc v ⬝ᵥ (circ (n := n) p L P r).out D εf u
      ((circ (n := n) p L P r).s0 hT ids) i =
    if v = r then p.Lg * fl n p D P ids u i / Real.sqrt (gfin n p D P ids u εf i) else 0 := by
  unfold Circuit.out
  rw [run_all n p D P ids u r hT]
  have hc : (circ (n := n) p L P r).γf = 1 := rfl
  have hb : (circ (n := n) p L P r).βf = 0 := rfl
  have hE : (circ (n := n) p L P r).E = Ef p.δ M n := rfl
  rw [hc, hb, hE]
  unfold stv
  rw [cLN_vec, gden_St n p D P ids u εf i _ _ _ _ (bit01 n p D P ids u G hT hu i)
    (andv01 n p D P ids u G hT hu i)]
  simp only [circ]
  split_ifs with h
  · rw [smul_dotProduct, sub_dotProduct]
    simp [single, single_dotProduct, St, sFlag, pm, gfin]
    ring
  · simp

/-! ### The theorem -/

/-- Readout margin scale bound. -/
noncomputable def Γp (M n D : ℕ) (p : Par) (σ2 εf : ℝ) : ℝ :=
  Real.sqrt ((D * (σ2 + εf) + Δc M D (slots M n) p + 2 * (p.A * p.δ * κhi D (slots M n)) ^ 2) / (D + slots M n))

/-- **Multi-token refusal head grafted into any host.** -/
theorem pattern_main [NeZero M] {Vn F H dq dv : ℕ} (Mh : Host (Vn + 1) D L F H dq dv)
    (hεH : ∀ b ∈ Mh.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < Mh.lnf.ε)
    (P : Fin (M + 1) → Fin (Vn + 1)) (r : Fin (Vn + 1)) (G : Good M n D L p σ2)
    (G0 : ℝ) (hG0 : 0 ≤ G0) (hgap : ∀ t, t ≠ r → Mh.gap r t ≤ G0)
    (h5 : G0 * Γp M n D p σ2 Mh.lnf.ε <
      p.Lg * (p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2)))
    {T : ℕ} (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (Vn + 1))
    (hσ : ∀ t, var (Mh.resid hT ids t) ≤ σ2) :
    (¬ trig P T ids → ∀ i v, |(graft Mh (circ (n := n) p L P r)).logits hT ids i v -
        Mh.toModel.logits hT ids i v| ≤
          Mh.etaAt (Δc M D (slots M n) p) (var (Mh.resid hT ids i)) v) ∧
    (trig P T ids → (graft Mh (circ (n := n) p L P r)).next hT ids = r) := by
  classical
  have hDr : (0 : ℝ) < D := by exact_mod_cast G.hD
  have hDK : (0 : ℝ) < D + slots M n := by positivity
  set C := circ (n := n) p L P r
  have hfeat := circ_feat n p P (L := L) r
  have hzs := circ_zs n p (L := L) P r
  refine ⟨fun hno i v => ?_, fun hyes => ?_⟩
  · have hno' : ∀ t, ¬ endsAt P ids t := fun t ⟨h, hP⟩ => hno ⟨t, h, hP⟩
    apply graft_offtrigger_approx_at G.hD Mh C hεH hεf hfeat hzs hT ids i _ σ2 hσ
    intro u hu
    have hfl := fl_zero n p D P ids u G hT hu hno' i
    have hE := energy_final n p D P ids u G hT hu i
    obtain ⟨-, -, -, hsq⟩ := g2_bounds n p D P ids u G hT hu i
    refine ⟨?_, ?_, fun w => ?_⟩
    · show Ef p.δ M n ≤ _
      rw [run_all n p D P ids u r hT, hE, hfl]
      nlinarith [Finset.sum_nonneg (fun d (_ : d ∈ Finset.univ) => sq_nonneg (acp n p D P ids u i d))]
    · show _ ≤ Ef p.δ M n + Δc M D (slots M n) p
      rw [run_all n p D P ids u r hT, hE, hfl]
      unfold Δc; nlinarith
    · rw [readout n p D P ids u G hT hu r w, hfl]; simp
  · obtain ⟨t0, h0, hP0⟩ := hyes
    have hend : endsAt P ids t0 := ⟨h0, hP0⟩
    have hlt := graft_ontrigger G.hD Mh C hεH hεf hfeat hzs hT ids (Fin.last T) r σ2 hσ
      (fun u hu => by
        have hE := energy_final n p D P ids u G hT hu (Fin.last T)
        refine ⟨?_, fun t ht => ?_⟩
        · show Ef p.δ M n ≤ _
          rw [run_all n p D P ids u r hT, hE]
          nlinarith [Finset.sum_nonneg (fun d (_ : d ∈ Finset.univ) =>
            sq_nonneg (acp n p D P ids u (Fin.last T) d))]
        · have hsplit : ∑ j, (C.Wc r j - C.Wc t j) * C.out D Mh.lnf.ε u (C.s0 hT ids) (Fin.last T) j =
              C.Wc r ⬝ᵥ C.out D Mh.lnf.ε u (C.s0 hT ids) (Fin.last T) -
                C.Wc t ⬝ᵥ C.out D Mh.lnf.ε u (C.s0 hT ids) (Fin.last T) := by
            simp only [dotProduct, ← Finset.sum_sub_distrib, sub_mul]
          rw [hsplit, readout n p D P ids u G hT hu r r, readout n p D P ids u G hT hu r t,
            if_pos rfl, if_neg ht, sub_zero]
          have hflo := fl_lo n p D P ids u G hT hu t0 hend
          obtain ⟨hfl0, hflhi⟩ := fl_bounds n p D P ids u G hT hu (Fin.last T)
          obtain ⟨-, -, -, hsq⟩ := g2_bounds n p D P ids u G hT hu (Fin.last T)
          have hk20 := κ2lo_pos n p D G
          have hu0 := (hu (Fin.last T)).1
          have hgpos : 0 < gfin n p D P ids u Mh.lnf.ε (Fin.last T) := by
            unfold gfin; apply div_pos _ hDK
            nlinarith [sq_nonneg (fl n p D P ids u (Fin.last T)),
              Finset.sum_nonneg (fun d (_ : d ∈ Finset.univ) => sq_nonneg (acp n p D P ids u (Fin.last T) d))]
          have hΓ : Real.sqrt (gfin n p D P ids u Mh.lnf.ε (Fin.last T)) ≤ Γp M n D p σ2 Mh.lnf.ε := by
            unfold Γp; apply Real.sqrt_le_sqrt; unfold gfin
            apply div_le_div_of_nonneg_right _ hDK.le
            have h1 := (hu (Fin.last T)).2
            have h2 : fl n p D P ids u (Fin.last T) ^ 2 ≤ (p.A * p.δ * κhi D (slots M n)) ^ 2 :=
              pow_le_pow_left₀ hfl0 hflhi 2
            unfold Δc; nlinarith
          have hsg := Real.sqrt_pos.2 hgpos
          have hΓ0 : 0 < Γp M n D p σ2 Mh.lnf.ε := lt_of_lt_of_le hsg hΓ
          have hX : 0 ≤ p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2) := by
            have := G.hA; have := G.hδ; have := G.h6; apply mul_nonneg (by positivity); linarith
          have hLg : 0 < p.Lg := by
            by_contra h
            have h' : p.Lg ≤ 0 := not_lt.mp h
            have : p.Lg * (p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2)) ≤ 0 :=
              mul_nonpos_of_nonpos_of_nonneg h' hX
            have : 0 ≤ G0 * Γp M n D p σ2 Mh.lnf.ε := by positivity
            linarith
          rw [lt_div_iff₀ hsg]
          calc Mh.gap r t * Real.sqrt (gfin n p D P ids u Mh.lnf.ε (Fin.last T))
              ≤ G0 * Γp M n D p σ2 Mh.lnf.ε := by
                rcases le_or_gt 0 (Mh.gap r t) with hg | hg
                · exact mul_le_mul (hgap t ht) hΓ hsg.le hG0
                · nlinarith
            _ < p.Lg * (p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2)) := h5
            _ ≤ p.Lg * fl n p D P ids u (Fin.last T) := mul_le_mul_of_nonneg_left hflo hLg.le)
    exact argmaxFirst_eq_of_lt _ r hlt

end VWC.Graft.Pattern
