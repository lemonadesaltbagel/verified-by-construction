/-
Phase 3.3: weight watermark (options W1 and W2 of `PHASE3_DESIGN.md` §8; W3 is `Key.lean`).

The watermark circuit has m mirrored pairs δ(c_j, −c_j), c ∈ {±1}^m secret, no blocks and no
readout. Grafted into any host:
* `wm_logits`: the grafted logits equal the host's exactly (theorem (A); the circuit is idle);
* `wm_probe` (W2): the linear probe ⟨c, mirrored reads⟩ of the raw residual stream after any number
  of blocks equals m·δ exactly, at every position, for every input and every host;
* `wm_weight` (W1): the same probe applied to any token-embedding row of the weights equals m·δ.

False positives (`fp_fraction_le`): for any fixed reading a ∈ ℝ^m that does not depend on the code,
the fraction of codes c with ⟨c, a⟩ ≥ τ is at most exp(−τ²/(2‖a‖²)) (Chernoff bound via
cosh x ≤ exp(x²/2)). For a reading produced by an independent watermark (a = δ c'), with threshold
τ = mδ/2 this is exp(−m/8).
-/
import VWC.Graft.Main

namespace VWC.Graft.Watermark

open Matrix Finset VWC.Arch VWC.Graft

variable {m : ℕ}

/-- ±1 sign of a code bit. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

lemma sgn_sq (b : Bool) : sgn b ^ 2 = 1 := by cases b <;> simp [sgn]

def pmo (σ : Fin 2) : ℝ := if σ = 0 then 1 else -1

/-- Watermark features: pair j holds δ (c_j, −c_j). -/
noncomputable def wfeat (δ : ℝ) (c : Fin m → Bool) : Fin (m * 2) → ℝ :=
  fun r => δ * sgn (c (finProdFinEquiv.symm r).1) * pmo (finProdFinEquiv.symm r).2

/-- The probe on the circuit dims: (x_{j,0} − x_{j,1})/2 weighted by the code sign. -/
noncomputable def pcirc (c : Fin m → Bool) : Fin (m * 2) → ℝ :=
  fun r => sgn (c (finProdFinEquiv.symm r).1) * pmo (finProdFinEquiv.symm r).2 / 2

/-- The probe on the grafted residual stream (zero on the host dims). -/
noncomputable def probe {D : ℕ} (c : Fin m → Bool) : Fin (D + m * 2) → ℝ := Fin.append 0 (pcirc c)

lemma sum_pairs (f : Fin m → Fin 2 → ℝ) :
    ∑ r : Fin (m * 2), f (finProdFinEquiv.symm r).1 (finProdFinEquiv.symm r).2 = ∑ j, (f j 0 + f j 1) := by
  rw [← Equiv.sum_comp finProdFinEquiv]
  simp [Fintype.sum_prod_type, Fin.sum_univ_two]

lemma wfeat_sum (δ : ℝ) (c : Fin m → Bool) : ∑ r, wfeat δ c r = 0 := by
  unfold wfeat; rw [sum_pairs (fun j σ => δ * sgn (c j) * pmo σ)]; simp [pmo]

lemma wfeat_sq (δ : ℝ) (c : Fin m → Bool) : ∑ r, wfeat δ c r ^ 2 = 2 * m * δ ^ 2 := by
  unfold wfeat; rw [sum_pairs (fun j σ => (δ * sgn (c j) * pmo σ) ^ 2)]
  simp only [pmo, mul_pow, sgn_sq]; simp; ring

lemma pcirc_sum (c : Fin m → Bool) : ∑ r, pcirc c r = 0 := by
  unfold pcirc; rw [sum_pairs (fun j σ => sgn (c j) * pmo σ / 2)]
  apply Finset.sum_eq_zero; intro j _; simp [pmo]; ring

lemma pcirc_dot (δ : ℝ) (c : Fin m → Bool) : ∑ r, pcirc c r * wfeat δ c r = m * δ := by
  unfold pcirc wfeat
  rw [sum_pairs (fun j σ => sgn (c j) * pmo σ / 2 * (δ * sgn (c j) * pmo σ))]
  have : ∀ j, sgn (c j) * pmo 0 / 2 * (δ * sgn (c j) * pmo 0) +
      sgn (c j) * pmo 1 / 2 * (δ * sgn (c j) * pmo 1) = δ := fun j => by
    have := sgn_sq (c j); simp only [pmo]; simp; linear_combination δ * this
  simp only [this]; simp

/-- The probe reads m·δ from any grafted vector `joint x (wfeat δ c)`. -/
lemma probe_joint {D : ℕ} (δ : ℝ) (c : Fin m → Bool) (x : Fin D → ℝ) :
    probe c ⬝ᵥ joint x (wfeat δ c) = m * δ := by
  unfold probe joint
  rw [append_dot, zero_dotProduct, zero_add]
  simp only [dotProduct, mul_add, Finset.sum_add_distrib]
  rw [← Finset.sum_mul, pcirc_sum, zero_mul, zero_add]
  exact pcirc_dot δ c

/-- The watermark circuit: no blocks, zero readout. -/
noncomputable def wcirc (V L : ℕ) (δ : ℝ) (c : Fin m → Bool) : Circuit V L (m * 2) 1 1 1 1 :=
  ⟨fun _ => wfeat δ c, 0, 2 * m * δ ^ 2, [], 1, 0, 0⟩

variable {V D L F H dq dv T : ℕ}

lemma wcirc_feat (δ : ℝ) (c : Fin m → Bool) (v : Fin V) (p : Fin L) :
    ∑ j, ((wcirc V L δ c).sTok v j + (wcirc V L δ c).sPos p j) = 0 ∧
    ∑ j, ((wcirc V L δ c).sTok v j + (wcirc V L δ c).sPos p j) ^ 2 = (wcirc V L δ c).E := by
  simp only [wcirc, Pi.zero_apply, add_zero]
  exact ⟨wfeat_sum δ c, wfeat_sq δ c⟩

/-- **Host behaviour unchanged, exactly.** -/
theorem wm_logits (hD : 0 < D) (M : Host V D L F H dq dv)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (δ : ℝ) (c : Fin m → Bool) (hT : T ≤ L) (ids : Fin T → Fin V) (i : Fin T) :
    (graft M (wcirc V L δ c)).logits hT ids i = M.toModel.logits hT ids i :=
  graft_offtrigger hD M (wcirc V L δ c) hεH hεf (by simp [wcirc]) (wcirc_feat δ c) hT ids
    ⟨by simp [wcirc], fun _ _ _ => by simp [wcirc]⟩ i

/-- The grafted embedding of a prompt is in joint form with the watermark state. -/
lemma wm_emb (M : Host V D L F H dq dv) (δ : ℝ) (c : Fin m → Bool) (hT : T ≤ L)
    (ids : Fin T → Fin V) :
    (fun t => (graft M (wcirc V L δ c)).tokEmb (ids t) + (graft M (wcirc V L δ c)).posEmb (Fin.castLE hT t)) =
      fun t => joint (M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t)) (wfeat δ c) := by
  funext t
  simp only [graft, append_add]
  funext j
  refine Fin.addCases (fun ih => ?_) (fun jj => ?_) j
  · simp [joint]
  · simp [joint, wcirc, mean_add]; ring

/-- **W2: activation probe.** After any number ℓ of blocks, the probe reads m·δ at every position. -/
theorem wm_probe (hD : 0 < D) (M : Host V D L F H dq dv)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε)
    (δ : ℝ) (c : Fin m → Bool) (hT : T ≤ L) (ids : Fin T → Fin V) (ℓ : ℕ) (t : Fin T) :
    probe c ⬝ᵥ gApplyBlocks ((graft M (wcirc V L δ c)).blocks.take ℓ)
      (fun t => (graft M (wcirc V L δ c)).tokEmb (ids t) + (graft M (wcirc V L δ c)).posEmb (Fin.castLE hT t)) t =
      m * δ := by
  rw [wm_emb]
  have hbl : (graft M (wcirc V L δ c)).blocks.take ℓ =
      (M.blocks.take ℓ).map (fun b => (⟨graftH (2 * m * δ ^ 2) b⟩ : AnyGBlock (D + m * 2))) := by
    simp [graft, wcirc, List.map_take]
  rw [hbl, host_blocks hD _ (M.blocks.take ℓ) (fun b hb => hεH b (List.mem_of_mem_take hb)) _
    (fun _ => wfeat δ c) (fun _ => wfeat_sum δ c) (fun _ => wfeat_sq δ c)]
  exact probe_joint δ c _

/-- **W1: weight detector.** The probe applied to any token-embedding row of the grafted weights
reads m·δ; applied to any position-embedding row it reads 0. -/
theorem wm_weight (M : Host V D L F H dq dv) (δ : ℝ) (c : Fin m → Bool) (v : Fin V) (p : Fin L) :
    probe c ⬝ᵥ (graft M (wcirc V L δ c)).tokEmb v = m * δ ∧
    probe c ⬝ᵥ (graft M (wcirc V L δ c)).posEmb p = 0 := by
  constructor
  · have : (graft M (wcirc V L δ c)).tokEmb v = joint (M.tokEmb v) (wfeat δ c) := by
      simp [graft, wcirc, joint]
    rw [this]; exact probe_joint δ c _
  · simp only [graft, wcirc, probe]
    rw [append_dot, zero_dotProduct, zero_add]
    simp only [Pi.zero_apply, add_zero, dotProduct]
    rw [← Finset.sum_mul, pcirc_sum, zero_mul]

/-! ### False positives -/

lemma sum_exp_codes (a : Fin m → ℝ) (lam : ℝ) :
    ∑ c : Fin m → Bool, Real.exp (lam * ∑ j, sgn (c j) * a j) = ∏ j, (2 * Real.cosh (lam * a j)) := by
  simp_rw [Finset.mul_sum, Real.exp_sum]
  rw [← Fintype.prod_sum (fun j (b : Bool) => Real.exp (lam * (sgn b * a j)))]
  refine Finset.prod_congr rfl (fun j _ => ?_)
  rw [Fintype.sum_bool, Real.cosh_eq]
  simp [sgn]; ring_nf

/-- **Chernoff bound over codes.** For a fixed reading a, at most a fraction exp(−τ²/(2‖a‖²)) of the
codes give ⟨c, a⟩ ≥ τ. -/
theorem fp_fraction_le (a : Fin m → ℝ) (τ : ℝ) (hτ : 0 ≤ τ) (ha : 0 < ∑ j, a j ^ 2) :
    ((Finset.univ.filter (fun c : Fin m → Bool => τ ≤ ∑ j, sgn (c j) * a j)).card : ℝ) / 2 ^ m ≤
      Real.exp (-(τ ^ 2) / (2 * ∑ j, a j ^ 2)) := by
  set S := ∑ j, a j ^ 2
  set lam := τ / S
  have hlam : 0 ≤ lam := div_nonneg hτ ha.le
  have hcard : ((Finset.univ.filter (fun c : Fin m → Bool => τ ≤ ∑ j, sgn (c j) * a j)).card : ℝ) ≤
      ∑ c : Fin m → Bool, Real.exp (lam * ∑ j, sgn (c j) * a j) * Real.exp (-(lam * τ)) := by
    rw [Finset.card_eq_sum_ones, Nat.cast_sum, Nat.cast_one]
    calc ∑ c ∈ Finset.univ.filter (fun c : Fin m → Bool => τ ≤ ∑ j, sgn (c j) * a j), (1 : ℝ)
        ≤ ∑ c ∈ Finset.univ.filter (fun c : Fin m → Bool => τ ≤ ∑ j, sgn (c j) * a j),
            Real.exp (lam * ∑ j, sgn (c j) * a j) * Real.exp (-(lam * τ)) := by
          apply Finset.sum_le_sum; intro c hc
          have hc' := (Finset.mem_filter.1 hc).2
          rw [← Real.exp_add]; apply Real.one_le_exp; nlinarith
      _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
          (fun _ _ _ => by positivity)
  rw [← Finset.sum_mul, sum_exp_codes] at hcard
  have hprod : ∏ j, (2 * Real.cosh (lam * a j)) ≤ 2 ^ m * Real.exp (lam ^ 2 * S / 2) := by
    calc ∏ j, (2 * Real.cosh (lam * a j)) ≤ ∏ j, (2 * Real.exp ((lam * a j) ^ 2 / 2)) :=
          Finset.prod_le_prod₀ (fun j _ => by have := Real.cosh_pos (lam * a j); positivity)
            (fun j _ => mul_le_mul_of_nonneg_left (Real.cosh_le_exp_half_sq _) (by norm_num))
      _ = 2 ^ m * Real.exp (lam ^ 2 * S / 2) := by
          rw [Finset.prod_mul_distrib, ← Real.exp_sum]
          simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
          congr 2
          simp only [S, Finset.mul_sum, Finset.sum_div]
          refine Finset.sum_congr rfl (fun j _ => by ring)
  have h2 : (0 : ℝ) < 2 ^ m := by positivity
  rw [div_le_iff₀ h2]
  calc _ ≤ (∏ j, (2 * Real.cosh (lam * a j))) * Real.exp (-(lam * τ)) := hcard
    _ ≤ 2 ^ m * Real.exp (lam ^ 2 * S / 2) * Real.exp (-(lam * τ)) :=
        mul_le_mul_of_nonneg_right hprod (Real.exp_pos _).le
    _ = Real.exp (-(τ ^ 2) / (2 * S)) * 2 ^ m := by
        rw [mul_assoc, ← Real.exp_add, mul_comm]
        congr 2
        simp only [lam]; field_simp; ring

/-- The probe of a reading `h` as a code-weighted sum of its mirrored pair differences. -/
lemma probe_reading {D : ℕ} (c : Fin m → Bool) (h : Fin (D + m * 2) → ℝ) :
    probe c ⬝ᵥ h = ∑ j, sgn (c j) * ((h (Fin.natAdd D (finProdFinEquiv (j, 0))) -
      h (Fin.natAdd D (finProdFinEquiv (j, 1)))) / 2) := by
  unfold probe
  rw [show h = Fin.append (fun i => h (Fin.castAdd (m * 2) i)) (fun r => h (Fin.natAdd D r)) from
    by funext i; refine Fin.addCases (fun _ => ?_) (fun _ => ?_) i <;> simp, append_dot]
  rw [zero_dotProduct, zero_add]
  simp only [dotProduct]
  unfold pcirc
  rw [← Equiv.sum_comp finProdFinEquiv]
  simp only [Equiv.symm_apply_apply, Fintype.sum_prod_type, Fin.sum_univ_two]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  simp [pmo]; ring

/-! ### Scale-invariant detector: cosine with the code -/

/-- The mirrored reads of a grafted vector are exactly δ · (±1 code). -/
lemma wm_reads {D : ℕ} (δ : ℝ) (c : Fin m → Bool) (x : Fin D → ℝ) (j : Fin m) :
    (joint x (wfeat δ c) (Fin.natAdd D (finProdFinEquiv (j, 0))) -
      joint x (wfeat δ c) (Fin.natAdd D (finProdFinEquiv (j, 1)))) / 2 = δ * sgn (c j) := by
  simp only [joint, Fin.append_right, wfeat, Equiv.symm_apply_apply, pmo]; norm_num

/-- On a marked model the reads are δ c, so ⟨c, reads⟩ = m δ = √m · ‖reads‖: cosine exactly 1. -/
lemma wm_cos (δ : ℝ) (hδ : 0 < δ) (c : Fin m → Bool) :
    ∑ j, sgn (c j) * (δ * sgn (c j)) = Real.sqrt m * Real.sqrt (∑ j, (δ * sgn (c j)) ^ 2) := by
  have h1 : ∀ j, sgn (c j) * (δ * sgn (c j)) = δ := fun j => by
    have := sgn_sq (c j); linear_combination δ * this
  have h2 : ∀ j, (δ * sgn (c j)) ^ 2 = δ ^ 2 := fun j => by rw [mul_pow, sgn_sq, mul_one]
  simp only [h1, h2, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [← Real.sqrt_mul (Nat.cast_nonneg m), show (m : ℝ) * (m * δ ^ 2) = (m * δ) ^ 2 by ring,
    Real.sqrt_sq (by positivity)]

/-- **False positives of the cosine detector.** For any fixed nonzero reading a, at most a fraction
exp(−m/8) of the codes give cos(c, a) ≥ 1/2, i.e. ⟨c, a⟩ ≥ (√m/2)‖a‖ — whatever the scale of a. -/
theorem fp_cos (a : Fin m → ℝ) (ha : 0 < ∑ j, a j ^ 2) :
    ((Finset.univ.filter (fun c : Fin m → Bool =>
        Real.sqrt m / 2 * Real.sqrt (∑ j, a j ^ 2) ≤ ∑ j, sgn (c j) * a j)).card : ℝ) / 2 ^ m ≤
      Real.exp (-(m : ℝ) / 8) := by
  have h := fp_fraction_le a (Real.sqrt m / 2 * Real.sqrt (∑ j, a j ^ 2)) (by positivity) ha
  have hτ : (Real.sqrt m / 2 * Real.sqrt (∑ j, a j ^ 2)) ^ 2 = m * (∑ j, a j ^ 2) / 4 := by
    rw [mul_pow, div_pow, Real.sq_sqrt (Nat.cast_nonneg m), Real.sq_sqrt ha.le]; ring
  rw [hτ] at h
  convert h using 2
  field_simp; ring

end VWC.Graft.Watermark
