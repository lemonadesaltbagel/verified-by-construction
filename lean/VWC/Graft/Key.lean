/-
Phase 3.2: key gating.

The gate is the pattern detector of `PatternMain.lean` with the pattern set to a key of M + 1
tokens; matching is exact (one wrong token ⇒ no trigger, every logit within η of the host's).

`card_trig_le` bounds how many prompts contain the key: at most (T+1) · V^(T−M) of the V^(T+1)
prompts of length T + 1, i.e. a uniformly random prompt opens the gate with probability at most
(T+1)/V^(M+1) (`trig_fraction_le`). This is a statement about black-box guessing only: the key is
stored in plain text in the token features (`Ftok`), so anyone holding the weights can read it.
No cryptographic hiding is claimed.
-/
import VWC.Graft.PatternMain

namespace VWC.Graft.Pattern

open Finset

variable {M V : ℕ}

/-- Positions of the key window ending at t (when M ≤ t). -/
def winPos (M T : ℕ) (t : Fin (T + 1)) (h : M ≤ (t : ℕ)) : Finset (Fin (T + 1)) :=
  Finset.univ.image (fun i : Fin (M + 1) => (⟨t - M + i, by omega⟩ : Fin (T + 1)))

lemma winPos_card (T : ℕ) (t : Fin (T + 1)) (h : M ≤ (t : ℕ)) : (winPos M T t h).card = M + 1 := by
  unfold winPos
  rw [Finset.card_image_of_injective _ (fun i j hij => by
    have := congrArg Fin.val hij; simp at this; exact Fin.ext (by omega))]
  simp

open Classical in
/-- The prompts whose key window ends at position t. -/
noncomputable def window (K : Fin (M + 1) → Fin V) (T : ℕ) (t : Fin (T + 1)) :
    Finset (Fin (T + 1) → Fin V) :=
  Finset.univ.filter (fun ids => ∃ h : M ≤ (t : ℕ), ∀ i : Fin (M + 1), ids ⟨t - M + i, by omega⟩ = K i)

lemma window_card_le (K : Fin (M + 1) → Fin V) (T : ℕ) (t : Fin (T + 1)) :
    (window K T t).card ≤ V ^ (T - M) := by
  classical
  by_cases h : M ≤ (t : ℕ)
  · set O := Finset.univ \ winPos M T t h
    have hO : O.card = T - M := by
      rw [Finset.card_sdiff, Finset.inter_univ, winPos_card T t h, Finset.card_univ,
        Fintype.card_fin]
      omega
    -- restriction to the positions outside the window is injective on the window set
    have hinj : Set.InjOn (fun ids : Fin (T + 1) → Fin V => fun s : O => ids s)
        (window K T t : Set (Fin (T + 1) → Fin V)) := by
      intro a ha b hb hab
      simp only [window, Finset.coe_filter, Finset.mem_univ, true_and] at ha hb
      obtain ⟨_, ha⟩ := ha
      obtain ⟨_, hb⟩ := hb
      funext s
      by_cases hs : s ∈ O
      · exact congrFun hab ⟨s, hs⟩
      · have hw : s ∈ winPos M T t h := by
          simp only [O, Finset.mem_sdiff, Finset.mem_univ, true_and, not_not] at hs; exact hs
        obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hw
        rw [ha i, hb i]
    calc (window K T t).card
        ≤ (Finset.univ : Finset (O → Fin V)).card :=
          Finset.card_le_card_of_injOn _ (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _)) hinj
      _ = V ^ (T - M) := by
          rw [Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_coe, hO]
  · have : window K T t = ∅ := by
      ext ids; simp [window, h]
    rw [this]; simp

open Classical in
/-- **Counting.** At most (T+1) · V^(T−M) prompts of length T + 1 contain the key. -/
theorem card_trig_le (K : Fin (M + 1) → Fin V) (T : ℕ) :
    (Finset.univ.filter (fun ids => trig K T ids)).card ≤ (T + 1) * V ^ (T - M) := by
  have hsub : Finset.univ.filter (fun ids => trig K T ids) ⊆
      Finset.univ.biUnion (fun t => window K T t) := by
    intro ids hids
    obtain ⟨t, h, hk⟩ := (Finset.mem_filter.1 hids).2
    exact Finset.mem_biUnion.2 ⟨t, Finset.mem_univ _, Finset.mem_filter.2 ⟨Finset.mem_univ _, h, hk⟩⟩
  calc _ ≤ (Finset.univ.biUnion (fun t => window K T t)).card := Finset.card_le_card hsub
    _ ≤ ∑ t : Fin (T + 1), (window K T t).card := Finset.card_biUnion_le
    _ ≤ ∑ _t : Fin (T + 1), V ^ (T - M) := Finset.sum_le_sum (fun t _ => window_card_le K T t)
    _ = (T + 1) * V ^ (T - M) := by simp

open Classical in
/-- As a fraction of all prompts: at most (T+1)/V^(M+1) when M ≤ T. -/
theorem trig_fraction_le (K : Fin (M + 1) → Fin V) (T : ℕ) (hMT : M ≤ T) (hV : 0 < V) :
    ((Finset.univ.filter (fun ids => trig K T ids)).card : ℝ) / (V : ℝ) ^ (T + 1) ≤
      (T + 1 : ℝ) / (V : ℝ) ^ (M + 1) := by
  have hc := card_trig_le K T
  have hVr : (0 : ℝ) < V := by exact_mod_cast hV
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  have h1 : ((Finset.univ.filter (fun ids => trig K T ids)).card : ℝ) ≤ (T + 1) * (V : ℝ) ^ (T - M) := by
    exact_mod_cast hc
  calc ((Finset.univ.filter (fun ids => trig K T ids)).card : ℝ) * (V : ℝ) ^ (M + 1)
      ≤ (T + 1) * (V : ℝ) ^ (T - M) * (V : ℝ) ^ (M + 1) :=
        mul_le_mul_of_nonneg_right h1 (by positivity)
    _ = (T + 1) * (V : ℝ) ^ (T + 1) := by
        rw [mul_assoc, ← pow_add]; congr 2; omega

/-- **Key gate.** The pattern detector with the pattern set to a key `K` of M + 1 tokens, grafted
into any GPT-2-style host (hypotheses as in `pattern_main`):
* without the key (no window of the prompt equals `K` exactly — one wrong token suffices), every
  grafted logit at position i is within `Host.etaAt Δc u_i` of the host's (u_i: the host's own
  final-residual variance at i);
* with the key anywhere in the prompt, the greedy next token is r.
Together with `trig_fraction_le`, a uniformly random prompt opens the gate with probability at most
(T+1)/V^(M+1). -/
theorem key_main {n D L : ℕ} {σ2 : ℝ} (p : Par) [NeZero M] {Vn F H dq dv : ℕ}
    (Mh : Host (Vn + 1) D L F H dq dv)
    (hεH : ∀ b ∈ Mh.blocks, 0 < b.ln1.ε ∧ 0 < b.ln2.ε) (hεf : 0 < Mh.lnf.ε)
    (K : Fin (M + 1) → Fin (Vn + 1)) (r : Fin (Vn + 1)) (G : Good M n D L p σ2)
    (G0 : ℝ) (hG0 : 0 ≤ G0) (hgap : ∀ t, t ≠ r → Mh.gap r t ≤ G0)
    (h5 : G0 * Γp M n D p σ2 Mh.lnf.ε <
      p.Lg * (p.A * p.δ * κ2lo M D (slots M n) p σ2 * (1 - lam3 M n D (slots M n) L p σ2)))
    {T : ℕ} (hT : T + 1 ≤ L) (ids : Fin (T + 1) → Fin (Vn + 1))
    (hσ : ∀ t, var (Mh.resid hT ids t) ≤ σ2) :
    ((∀ t : Fin (T + 1), ∀ h : M ≤ (t : ℕ), ∃ i : Fin (M + 1), ids ⟨t - M + i, by omega⟩ ≠ K i) →
      ∀ i v, |(graft Mh (circ (n := n) p L K r)).logits hT ids i v - Mh.toModel.logits hT ids i v| ≤
        Mh.etaAt (Δc M D (slots M n) p) (var (Mh.resid hT ids i)) v) ∧
    ((∃ t : Fin (T + 1), ∃ h : M ≤ (t : ℕ), ∀ i : Fin (M + 1), ids ⟨t - M + i, by omega⟩ = K i) →
      (graft Mh (circ (n := n) p L K r)).next hT ids = r) := by
  have hm := pattern_main n p D Mh hεH hεf K r G G0 hG0 hgap h5 hT ids hσ
  refine ⟨fun hno => hm.1 (fun ⟨t, h, hk⟩ => ?_), fun hyes => hm.2 hyes⟩
  obtain ⟨i, hi⟩ := hno t h
  exact hi (hk i)

end VWC.Graft.Pattern
