/-
Phase 2.5: the composed grafting theorem.

A circuit `C` meets its contract (`Circuit.Contract`) for a trigger predicate and a refusal token r
when its features are zero-sum with constant energy, its writes are zero-sum, it is idle on every
non-trigger prompt, and on every trigger prompt its readout margin at the last position beats the
host's boundary-weight gap for every host variance profile in [0, σ²]. The contract mentions the
host only through `Host.gap` and the final LayerNorm ε, i.e. boundary weights.

`graft_main`: off trigger the grafted model's logits (hence its greedy next token) equal the host's;
on trigger, if the host's final residual variance is at most σ², the greedy next token is r.
`graft_preserves_spec`: any next-token property the host has on non-trigger prompts survives.
Neither proof looks inside the host's blocks.
-/
import VWC.Graft.OnTrigger

namespace VWC.Graft

open Matrix Finset VWC.Arch

variable {D k : ℕ}

/-- Greedy next token after a prompt of length T + 1 (logits at the last position). -/
noncomputable def GModel.next {n L T : ℕ} (M : GModel (n + 1) D L) (hT : T + 1 ≤ L)
    (ids : Fin (T + 1) → Fin (n + 1)) : Fin (n + 1) :=
  argmaxFirst (M.logits hT ids (Fin.last T))

/-- The circuit's contract for a trigger predicate and refusal token r, given the host variance
bound σ². Uses the host only through `M.gap` and `M.lnf.ε`. -/
structure Circuit.Contract {n L F H dq dv Fc Hc dqc dvc : ℕ} (C : Circuit (n + 1) L k Fc Hc dqc dvc)
    (M : Host (n + 1) D L F H dq dv) (trig : (T : ℕ) → (Fin (T + 1) → Fin (n + 1)) → Prop)
    (r : Fin (n + 1)) (σ2 : ℝ) : Prop where
  feat : ∀ v p, ∑ j, (C.sTok v j + C.sPos p j) = 0 ∧ ∑ j, (C.sTok v j + C.sPos p j) ^ 2 = C.E
  eps : ∀ b ∈ C.blocks, 0 < b.ε1 ∧ 0 < b.ε2
  zs : ∀ b ∈ C.blocks, b.ZeroSum
  off : ∀ (T : ℕ) (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (n + 1)), ¬ trig T ids → C.Idle hT ids
  on : ∀ (T : ℕ) (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (n + 1)), trig T ids →
    ∀ u : Fin (T + 1) → ℝ, (∀ t, 0 ≤ u t ∧ u t ≤ σ2) →
      C.E ≤ ∑ j, runBlocks D C.E C.blocks u (C.s0 hT ids) (Fin.last T) j ^ 2 ∧
      ∀ t, t ≠ r → M.gap r t <
        ∑ j, (C.Wc r j - C.Wc t j) * C.out D M.lnf.ε u (C.s0 hT ids) (Fin.last T) j

/-- **Grafting theorem.** -/
theorem graft_main {n L F H dq dv Fc Hc dqc dvc : ℕ} (hD : 0 < D)
    (M : Host (n + 1) D L F H dq dv) (C : Circuit (n + 1) L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (trig : (T : ℕ) → (Fin (T + 1) → Fin (n + 1)) → Prop) (r : Fin (n + 1)) (σ2 : ℝ)
    (hC : C.Contract M trig r σ2) {T : ℕ} (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (n + 1)) :
    (¬ trig T ids → ∀ i, (graft M C).logits hT ids i = M.toModel.logits hT ids i) ∧
    (¬ trig T ids → (graft M C).next hT ids = M.toModel.next hT ids) ∧
    (trig T ids → (∀ t, var (M.resid hT ids t) ≤ σ2) → (graft M C).next hT ids = r) := by
  have hoff : ¬ trig T ids → ∀ i, (graft M C).logits hT ids i = M.toModel.logits hT ids i :=
    fun h i => graft_offtrigger hD M C hεH hεf hC.eps hC.feat hT ids (hC.off T hT ids h) i
  refine ⟨hoff, fun h => ?_, fun h hσ => ?_⟩
  · unfold GModel.next; rw [hoff h]
  · exact argmaxFirst_eq_of_lt _ r
      (graft_ontrigger hD M C hεH hεf hC.feat hC.zs hT ids (Fin.last T) r σ2 hσ
        (hC.on T hT ids h))

/-- Any next-token property of the host on non-trigger prompts holds for the grafted model; on
trigger prompts the grafted model refuses (given the variance bound). -/
theorem graft_preserves_spec {n L F H dq dv Fc Hc dqc dvc : ℕ} (hD : 0 < D)
    (M : Host (n + 1) D L F H dq dv) (C : Circuit (n + 1) L k Fc Hc dqc dvc)
    (hεH : ∀ b ∈ M.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < M.lnf.ε)
    (trig : (T : ℕ) → (Fin (T + 1) → Fin (n + 1)) → Prop) (r : Fin (n + 1)) (σ2 : ℝ)
    (hC : C.Contract M trig r σ2)
    (spec : (T : ℕ) → (Fin (T + 1) → Fin (n + 1)) → Fin (n + 1) → Prop)
    (hM : ∀ T (hT : T + 1 ≤ L) ids, ¬ trig T ids → spec T ids (M.toModel.next hT ids)) :
    (∀ T (hT : T + 1 ≤ L) ids, ¬ trig T ids → spec T ids ((graft M C).next hT ids)) ∧
    (∀ T (hT : T + 1 ≤ L) ids, trig T ids → (∀ t, var (M.resid hT ids t) ≤ σ2) →
      (graft M C).next hT ids = r) := by
  refine ⟨fun T hT ids h => ?_, fun T hT ids h hσ => ?_⟩
  · rw [(graft_main hD M C hεH hεf trig r σ2 hC hT ids).2.1 h]; exact hM T hT ids h
  · exact (graft_main hD M C hεH hεf trig r σ2 hC hT ids).2.2 h hσ

end VWC.Graft
