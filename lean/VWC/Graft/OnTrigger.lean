/-
Phase 2, theorem (B): on trigger, the grafted model emits the refusal token, under one scalar
hypothesis on the host (its final residual variance is at most σ²) and an obligation on the
circuit alone.

On trigger the circuit state leaves the constant-energy manifold, so `ln_graft` no longer applies.
`ln_graft_gen` handles any zero-sum circuit state: host dims are normalised by
√((D(var x + ε) + ‖s‖² − E)/(D+k)) instead of √(D(var x + ε)/(D+k)), and circuit dims by the same
scale. Since circuit blocks sit after the host blocks and write zero-sum vectors (`CBlock.ZeroSum`),
the grafted run is exactly the host run plus a k-dimensional system `runBlocks` that sees the host
only through the variances u t = var (host residual t) (`circuit_blocks_run`).

Host logits are then bounded by boundary weights only (`host_term`, Cauchy–Schwarz):
for every token t, (w_t − w_r)·(host part of the final LayerNorm) ≤ `Host.gap`, which uses only the
unembedding, the final LayerNorm and the head bias. The circuit obligation is stated for every
u ∈ [0, σ²]: a concrete circuit discharges it from its own weights (Phase 3).
-/
import VWC.Graft.OffTrigger

namespace VWC.Graft

open Matrix Finset VWC.Arch

variable {D k : ℕ}

/-- Normalisation denominator of a grafted LayerNorm for host variance u and circuit state s. -/
noncomputable def gden (D k : ℕ) (E ε u : ℝ) (s : Fin k → ℝ) : ℝ :=
  (D * (u + ε) + ∑ j, s j ^ 2 - E) / (D + k)

/-- A grafted LayerNorm on the circuit dims, as a function of the host variance u. -/
noncomputable def cLN (D k : ℕ) (E : ℝ) (γ β : Fin k → ℝ) (ε u : ℝ) (s : Fin k → ℝ) : Fin k → ℝ :=
  fun j => s j / Real.sqrt (gden D k E ε u s) * γ j + β j

/-- **LayerNorm lemma, any energy.** Only a zero-sum circuit state is needed. -/
lemma ln_graft_gen (hD : 0 < D) (x : Fin D → ℝ) (s : Fin k → ℝ) (E : ℝ) (hs0 : ∑ j, s j = 0)
    (γ β : Fin D → ℝ) (γc βc : Fin k → ℝ) (ε : ℝ) :
    (lnGraft γ β γc βc ε E).apply (joint x s) =
      Fin.append (fun i => (x i - mean x) / Real.sqrt (gden D k E ε (var x) s) *
        (γ i * Real.sqrt (D / (D + k))) + β i) (cLN D k E γc βc ε (var x) s) := by
  have hDk : (0 : ℝ) < D + k := by have : (0 : ℝ) < D := by exact_mod_cast hD
                                   positivity
  have hjd : (∑ i, (joint x s i - mean x) ^ 2) / ((D + k : ℕ) : ℝ) + (D * ε - E) / (D + k) =
      gden D k E ε (var x) s := by
    rw [joint_sq x s hs0 hD _ rfl]; unfold gden; push_cast; field_simp; ring
  simp only [lnGraft, Norm.apply]
  rw [joint_mean x s hs0 hD, hjd]
  funext i
  refine Fin.addCases (fun ih => ?_) (fun j => ?_) i
  · simp [joint]
  · simp [joint, cLN]

/-! ### Zero-sum writes -/

/-- Every write of the circuit block is a zero-sum vector (column sums of the output matrices and
the output biases vanish). -/
def CBlock.ZeroSum {Fc Hc dq dv : ℕ} (b : CBlock k Fc Hc dq dv) : Prop :=
  (∀ c, ∑ j, b.attn.Wo j c = 0) ∧ ∑ j, b.attn.bo j = 0 ∧ (∀ f, ∑ j, b.W2 j f = 0) ∧ ∑ j, b.b2 j = 0

lemma zs_mulVec {m : ℕ} (W : Matrix (Fin k) (Fin m) ℝ) (hW : ∀ c, ∑ j, W j c = 0) (o : Fin m → ℝ) :
    ∑ j, (W *ᵥ o) j = 0 := by
  simp only [mulVec, dotProduct]
  rw [Finset.sum_comm]
  simp [← Finset.sum_mul, hW]

lemma zs_add (a b : Fin k → ℝ) (ha : ∑ j, a j = 0) (hb : ∑ j, b j = 0) : ∑ j, (a + b) j = 0 := by
  simp [Finset.sum_add_distrib, ha, hb]

/-! ### The circuit as a k-dimensional system driven by the host variances -/

/-- One circuit block on the circuit dims alone; `u t` is the host's residual variance at `t`. -/
noncomputable def CBlock.run {T Fc Hc dq dv : ℕ} (D : ℕ) (E : ℝ) (b : CBlock k Fc Hc dq dv)
    (u : Fin T → ℝ) (s : Fin T → Fin k → ℝ) : Fin T → Fin k → ℝ :=
  let s1 : Fin T → Fin k → ℝ := fun i => s i + b.attn.apply (fun t => cLN D k E b.γ1 b.β1 b.ε1 (u t) (s t)) i
  fun i => s1 i + (b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ cLN D k E b.γ2 b.β2 b.ε2 (u i) (s1 i)) f + b.b1 f)) + b.b2)

noncomputable def runBlocks {T Fc Hc dq dv : ℕ} (D : ℕ) (E : ℝ) :
    List (CBlock k Fc Hc dq dv) → (Fin T → ℝ) → (Fin T → Fin k → ℝ) → Fin T → Fin k → ℝ
  | [], _, s => s
  | b :: bs, u, s => runBlocks D E bs u (b.run D E u s)

lemma run_zs {T Fc Hc dq dv : ℕ} (E : ℝ) (b : CBlock k Fc Hc dq dv) (hb : b.ZeroSum)
    (u : Fin T → ℝ) (s : Fin T → Fin k → ℝ) (hs0 : ∀ t, ∑ j, s t j = 0) (i : Fin T) :
    ∑ j, b.run D E u s i j = 0 := by
  obtain ⟨hWo, hbo, hW2, hb2⟩ := hb
  simp only [CBlock.run, GAttn.apply]
  refine zs_add _ _ (zs_add _ _ (hs0 i) (zs_add _ _ (zs_mulVec _ hWo _) hbo))
    (zs_add _ _ (zs_mulVec _ hW2 _) hb2)

lemma runBlocks_zs {T Fc Hc dq dv : ℕ} (E : ℝ) (bs : List (CBlock k Fc Hc dq dv))
    (hzs : ∀ b ∈ bs, b.ZeroSum) (u : Fin T → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) (i : Fin T) : ∑ j, runBlocks D E bs u s i j = 0 := by
  induction bs generalizing s with
  | nil => exact hs0 i
  | cons b l ih =>
    exact ih (fun b' hb' => hzs b' (List.mem_cons_of_mem _ hb')) _
      (run_zs E b (hzs b List.mem_cons_self) u s hs0)

/-- A grafted circuit block acts on the grafted residual exactly as `CBlock.run`. -/
lemma graftC_run {T Fc Hc dq dv : ℕ} (hD : 0 < D) (E : ℝ) (b : CBlock k Fc Hc dq dv)
    (hb : b.ZeroSum) (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) :
    (graftC (D := D) E b).apply (fun t => joint (x t) (s t)) =
      fun t => joint (x t) (b.run D E (fun t => var (x t)) s t) := by
  funext i
  simp only [GBlock.apply, graftC]
  have hl1 : ∀ t, (lnGraft 0 0 b.γ1 b.β1 b.ε1 E).apply (joint (x t) (s t)) =
      Fin.append (fun l => (x t l - mean (x t)) / Real.sqrt (gden D k E b.ε1 (var (x t)) (s t)) *
        ((0 : Fin D → ℝ) l * Real.sqrt (D / (D + k))) + (0 : Fin D → ℝ) l)
        (cLN D k E b.γ1 b.β1 b.ε1 (var (x t)) (s t)) :=
    fun t => ln_graft_gen hD (x t) (s t) E (hs0 t) _ _ _ _ _
  simp only [hl1, attn_lifted, joint_add_circ]
  set s1 := s i + b.attn.apply (fun t => cLN D k E b.γ1 b.β1 b.ε1 (var (x t)) (s t)) i with hs1
  have hs1z : ∑ j, s1 j = 0 := by
    obtain ⟨hWo, hbo, -, -⟩ := hb
    simp only [hs1, GAttn.apply]
    exact zs_add _ _ (hs0 i) (zs_add _ _ (zs_mulVec _ hWo _) hbo)
  rw [ln_graft_gen hD (x i) s1 E hs1z]
  simp only [GBlock.mlp, liftCols_mulVec, liftRows_mulVec]
  rw [show (Fin.append (0 : Fin D → ℝ) (b.W2 *ᵥ fun f => b.act ((b.W1 *ᵥ
      cLN D k E b.γ2 b.β2 b.ε2 (var (x i)) s1) f + b.b1 f)) + Fin.append 0 b.b2) =
      Fin.append 0 (b.W2 *ᵥ (fun f => b.act ((b.W1 *ᵥ
      cLN D k E b.γ2 b.β2 b.ε2 (var (x i)) s1) f + b.b1 f)) + b.b2) by
    rw [append_add]; simp]
  rw [joint_add_circ]
  rfl

lemma circuit_blocks_run {T Fc Hc dq dv : ℕ} (hD : 0 < D) (E : ℝ) (bs : List (CBlock k Fc Hc dq dv))
    (hzs : ∀ b ∈ bs, b.ZeroSum) (x : Fin T → Fin D → ℝ) (s : Fin T → Fin k → ℝ)
    (hs0 : ∀ t, ∑ j, s t j = 0) :
    gApplyBlocks (bs.map (fun b => (⟨graftC (D := D) E b⟩ : AnyGBlock (D + k))))
      (fun t => joint (x t) (s t)) =
      fun t => joint (x t) (runBlocks D E bs (fun t => var (x t)) s t) := by
  induction bs generalizing s with
  | nil => rfl
  | cons b l ih =>
    simp only [List.map_cons, gApplyBlocks]
    rw [graftC_run hD E b (hzs b List.mem_cons_self) x s hs0]
    exact ih (fun b' hb' => hzs b' (List.mem_cons_of_mem _ hb')) _
      (run_zs E b (hzs b List.mem_cons_self) _ s hs0)

/-! ### Host logits are bounded by boundary weights -/

lemma host_term (hD : 0 < D) (y a : Fin D → ℝ) (s : Fin k → ℝ) (E ε : ℝ) (hε : 0 < ε)
    (hE : E ≤ ∑ j, s j ^ 2) :
    ∑ l, a l * ((y l - mean y) / Real.sqrt (gden D k E ε (var y) s) * Real.sqrt (D / (D + k))) ≤
      Real.sqrt D * Real.sqrt (∑ l, a l ^ 2) := by
  have hDr : (0 : ℝ) < D := by exact_mod_cast hD
  have hDk : (0 : ℝ) < D + k := by positivity
  have hv := var_nonneg y
  have hnum : 0 < D * (var y + ε) + ∑ j, s j ^ 2 - E := by nlinarith
  have hgpos : 0 < gden D k E ε (var y) s := div_pos hnum hDk
  set c := Real.sqrt (D / (D + k)) / Real.sqrt (gden D k E ε (var y) s) with hc
  have hsum : ∑ l, a l * ((y l - mean y) / Real.sqrt (gden D k E ε (var y) s) *
      Real.sqrt (D / (D + k))) = c * ∑ l, a l * (y l - mean y) := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl (fun l _ => ?_); rw [hc]; ring
  rw [hsum]
  set S := ∑ l, a l * (y l - mean y)
  have hCS : S ^ 2 ≤ (∑ l, a l ^ 2) * ∑ l, (y l - mean y) ^ 2 :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have hvy : ∑ l, (y l - mean y) ^ 2 = D * var y := by unfold var; field_simp
  have hc2 : c ^ 2 = D / (D * (var y + ε) + ∑ j, s j ^ 2 - E) := by
    rw [hc, div_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hgpos.le]
    unfold gden
    rw [div_div_div_cancel_right₀ hDk.ne']
  have hA : 0 ≤ ∑ l, a l ^ 2 := Finset.sum_nonneg (fun l _ => sq_nonneg (a l))
  have key : (c * S) ^ 2 ≤ D * ∑ l, a l ^ 2 := by
    rw [mul_pow, hc2, div_mul_eq_mul_div, div_le_iff₀ hnum]
    have h1 : D * S ^ 2 ≤ D * ((∑ l, a l ^ 2) * (D * var y)) := by
      rw [← hvy]; exact mul_le_mul_of_nonneg_left hCS hDr.le
    have h2 : 0 ≤ D * (∑ l, a l ^ 2) * (D * ε + (∑ j, s j ^ 2 - E)) := by
      have : 0 ≤ D * ε + (∑ j, s j ^ 2 - E) := by nlinarith
      positivity
    nlinarith
  calc c * S ≤ |c * S| := le_abs_self _
    _ = Real.sqrt ((c * S) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (D * ∑ l, a l ^ 2) := Real.sqrt_le_sqrt key
    _ = Real.sqrt D * Real.sqrt (∑ l, a l ^ 2) := Real.sqrt_mul hDr.le _

/-! ### Theorem (B) -/

/-- The host's final residual stream (before the final LayerNorm). -/
noncomputable def Host.resid {V L F H dq dv T : ℕ} (M : Host V D L F H dq dv) (hT : T ≤ L)
    (ids : Fin T → Fin V) : Fin T → Fin D → ℝ :=
  gApplyBlocks (M.blocks.map (fun b => (⟨b.toBlock⟩ : AnyGBlock D)))
    (fun t => M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t))

/-- Boundary-weight bound on how far token t's host logit can exceed token r's, for any input
(unembedding rows, final LayerNorm and head bias only). -/
noncomputable def Host.gap {V L F H dq dv : ℕ} (M : Host V D L F H dq dv) (r t : Fin V) : ℝ :=
  Real.sqrt D * Real.sqrt (∑ l, ((M.Wh t l - M.Wh r l) * M.lnf.γ l) ^ 2) +
    ∑ l, (M.Wh t l - M.Wh r l) * M.lnf.β l + (M.bh t - M.bh r)

/-- The circuit's contribution to the final LayerNorm output at position i, for host variances u. -/
noncomputable def Circuit.out {V L Fc Hc dq dv T : ℕ} (C : Circuit V L k Fc Hc dq dv) (D : ℕ)
    (εf : ℝ) (u : Fin T → ℝ) (s : Fin T → Fin k → ℝ) (i : Fin T) : Fin k → ℝ :=
  cLN D k C.E C.γf C.βf εf (u i) (runBlocks D C.E C.blocks u s i)

/-- **Theorem (B).** Suppose the host's final residual variance is at most σ² at every position,
and the circuit alone guarantees, for every host variance profile u ∈ [0, σ²], that its final
energy is at least E and that its readout margin for the refusal token r beats the host's
boundary-weight gap. Then r strictly has the largest grafted logit at position i. -/
theorem graft_ontrigger {V L F H dq dv Fc Hc dqc dvc T : ℕ} (hD : 0 < D)
    (M : Host V D L F H dq dv) (C : Circuit V L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (hfeat : ∀ v p, ∑ j, (C.sTok v j + C.sPos p j) = 0 ∧ ∑ j, (C.sTok v j + C.sPos p j) ^ 2 = C.E)
    (hzs : ∀ b ∈ C.blocks, b.ZeroSum)
    (hT : T ≤ L) (ids : Fin T → Fin V) (i : Fin T) (r : Fin V) (σ2 : ℝ)
    (hσ : ∀ t, var (M.resid hT ids t) ≤ σ2)
    (hcirc : ∀ u : Fin T → ℝ, (∀ t, 0 ≤ u t ∧ u t ≤ σ2) →
      C.E ≤ ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) i j ^ 2 ∧
      ∀ t, t ≠ r → M.gap r t < ∑ j, (C.Wc r j - C.Wc t j) * C.out D M.lnf.ε u (C.s0 hT ids) i j) :
    ∀ t, t ≠ r → (graft M C).logits hT ids i t < (graft M C).logits hT ids i r := by
  intro t ht
  set s := C.s0 hT ids
  have hs0 : ∀ t, ∑ j, s t j = 0 := fun t => (hfeat _ _).1
  have hsE : ∀ t, ∑ j, s t j ^ 2 = C.E := fun t => (hfeat _ _).2
  set x : Fin T → Fin D → ℝ := fun t => M.tokEmb (ids t) + M.posEmb (Fin.castLE hT t)
  have hemb : (fun t => (graft M C).tokEmb (ids t) + (graft M C).posEmb (Fin.castLE hT t)) =
      fun t => joint (x t) (s t) := by
    funext t
    simp only [graft, append_add]
    funext j
    refine Fin.addCases (fun ih => ?_) (fun jj => ?_) j
    · simp [joint, x]
    · simp [joint, x, s, Circuit.s0, mean_add]; ring
  simp only [GModel.logits]
  rw [hemb]
  simp only [graft, applyBlocks_append]
  rw [host_blocks hD C.E M.blocks hεH x s hs0 hsE]
  set y := gApplyBlocks (M.blocks.map (fun b => (⟨b.toBlock⟩ : AnyGBlock D))) x with hy
  have hyr : y = M.resid hT ids := rfl
  rw [circuit_blocks_run hD C.E C.blocks hzs y s hs0]
  set u : Fin T → ℝ := fun t => var (y t)
  set sf := runBlocks D C.E C.blocks u s
  have hsf : ∑ j, sf i j = 0 := runBlocks_zs C.E C.blocks hzs u s hs0 i
  rw [ln_graft_gen hD (y i) (sf i) C.E hsf]
  obtain ⟨hE, hm⟩ := hcirc u (fun t => ⟨var_nonneg _, by rw [hyr] at *; exact hσ t⟩)
  have hmt := hm t ht
  have hb := host_term (k := k) hD (y i) (fun l => (M.Wh t l - M.Wh r l) * M.lnf.γ l) (sf i) C.E
    M.lnf.ε hεf hE
  simp only [Pi.add_apply, mulVec, append_dot]
  set g := gden D k C.E M.lnf.ε (var (y i)) (sf i)
  set co := cLN D k C.E C.γf C.βf M.lnf.ε (var (y i)) (sf i)
  have hco : co = C.out D M.lnf.ε u s i := rfl
  rw [← hco] at hmt
  have h1 : (M.Wh t ⬝ᵥ fun l => (y i l - mean (y i)) / Real.sqrt g *
        (M.lnf.γ l * Real.sqrt (D / (D + k))) + M.lnf.β l) -
      (M.Wh r ⬝ᵥ fun l => (y i l - mean (y i)) / Real.sqrt g *
        (M.lnf.γ l * Real.sqrt (D / (D + k))) + M.lnf.β l) =
      ∑ l, (M.Wh t l - M.Wh r l) * M.lnf.γ l *
          ((y i l - mean (y i)) / Real.sqrt g * Real.sqrt (D / (D + k))) +
        ∑ l, (M.Wh t l - M.Wh r l) * M.lnf.β l := by
    simp only [dotProduct, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl (fun l _ => ?_); ring
  have h2 : C.Wc r ⬝ᵥ co - C.Wc t ⬝ᵥ co = ∑ j, (C.Wc r j - C.Wc t j) * co j := by
    simp only [dotProduct, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl (fun j _ => ?_); ring
  unfold Host.gap at hmt
  linarith
