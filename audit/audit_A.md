# Audit A — Phase B: Lean statements vs PyTorch task/eval/snapshots

Phase A (blind) is in `audit_A_phaseA.md` in the same directory. Python paths below are relative to
`external/wsp/runs/<run>/`. Lean paths are relative to `lean/VWC/`. All 7 runs' `src/task.py` /
`src/eval.py` are byte-identical to each other and to `external/wsp/evolve/src/` (md5 1f65958e…,
94004784…), so the "evolve/src/task.py" references in the Lean headers point at the same code.

Snapshot sha256 prefixes recorded in the generated weight headers match the files on disk:
Bool 612ec7381fcecf46, Count 7d7b30ee367ff732, WordRev 3afca1f9ec2373c8, Sort 28acaa28997fcf0f,
LI fd5cb6cb490b0723, Add a43435e2ff38361a. Parity: `ParityFloorSubD3F9.py` has the same md5 as the
run's `interpretable_transformer.py` named in `Model.lean:3`.

## 0. Generic semantics (Arch.lean, Model.lean, Softmax.lean) vs the snapshots' forward pass

| Fact | PyTorch | Lean | Verdict |
|---|---|---|---|
| Embedding | `token_emb(ids)+pos_emb(arange T)` (every snapshot, e.g. Count V13 :172) | `Arch.lean:112` tokEmb + posEmb(castLE t); `Model.lean:39-40` | MATCH |
| Attention scale | `/ sqrt(d_head)`; Count uses `/ sqrt(d_qk)` (CountCircuitV13.py:83) | `headWeight` `/ Real.sqrt dqk` (`Arch.lean:77`); parity `/ sqrt D` with H=1 (`Model.lean:44`) | MATCH |
| Causal mask | `triu(diagonal=1)` -> -inf | `if j ≤ i … else 0`, denominator over `Iic i` (`Arch.lean:78`) | MATCH (same over ℝ) |
| Softmax | max-subtracted exp | plain `exp/Σexp` | MATCH over ℝ |
| Head layout | `view(B,T,H,dh)` -> row h*dh+c | `finProdFinEquiv (h,c)` = c + dqk*h (`Arch.lean:83-88`) | MATCH |
| W_o | present, or absent (Bool) | `Wo *ᵥ o`; absent W_o exported as `1` (`BoolWeights.lean:127-128`) | MATCH |
| Residual/pre-norm | `x+attn(ln1 x)`, `x+mlp(ln2 x)`; MLP-only blocks `x+mlp(ln x)` | `Block.apply` (`Arch.lean:96-101`), `attn=none` skips ln1 | MATCH |
| LayerNorm | `nn.LayerNorm`: biased var, eps 1e-5 (float32 kernel) | `Norm.apply` biased var `/D`, eps = 2748779/2^38 = float32(1e-5) (`Arch.lean:30-35`, e.g. `SortWeights.lean:187`) | MATCH |
| MLP | `fc2(relu(fc1 x))`, missing fc2 bias = 0 | `W2 *ᵥ relu(W1 x + b1) + b2` (`Arch.lean:92-93`) | MATCH |
| Head | `Linear(bias=False)`, after final_ln (Identity or LN) | `Wh *ᵥ finalLn(h i) + bh`, bh = 0 in all exports | MATCH |
| Decoding | `answer_len` steps, full re-forward, `logits[:,-1].argmax`, fed back (eval.py:45-49) | `Model.generate` / `Model.next` on `s.length-1` (`Arch.lean:117-129`) | MATCH |
| Tie-break | torch.argmax -> first max index | `argmaxFirst` = min index among maximisers (`Model.lean:70-74`) | MATCH |
| Correctness | string equality of whole answer (task.py:53) | `generate … = some answer` (token list equality; vocab is injective) | MATCH |

`scripts/export_arch.py` (checked): reads modules by introspection, `getattr(b,"attn_enabled")` for
Count's MLP-only block, `ln` for Bool's MLPOnlyBlock, `d_qk/d_v` for Count, absent W_o -> `eye(D)`,
asserts no Q/K/V/O biases, missing fc biases -> 0, LN eps exported as `float32(eps)`
(:53-62); writes exact float32 rationals (:143-146); reimplements Arch in float64 numpy (:116-140)
and rejects the export if it differs from the snapshot's own float64 forward by >1e-9 relative
(:269-294). Records file sha and `AR_*` env overrides (:286-288). No problems found.

## 1. parity-upto10-bits — MATCH

- Vocab ids `Spec.lean:15-18` (0,1,=2,_=3) = task.py:188.
- Prompt `promptList` = `replicate (10-len) '_' ++ bits ++ ['=']` (`Spec.lean:23-24`) = task.py:202.
- Answer `parityTok` = count true % 2 (`Spec.lean:32-33`) = task.py:203.
- Domain: `parity_correct (bits) (1 ≤ len) (len ≤ 10)` (`Parity.lean:166-167`) = lengths 1..10
  (task.py:199). Exactly the generator domain (2046 prompts).
- Decoding: `predict` = argmaxFirst of logits at `Fin.last 10` (`Model.lean:77-79`), 1 step = answer_len 1.
- Model: uses the older `VWC.Params` semantics (Model.lean), not Arch; no LN, consistent with Identity
  LNs (ParityFloorSubD3F9.py:95-97,127). Weights (`Weights.lean:12-23`) checked by hand against
  write_weights (:158-221): all 35 nonzeros agree (tokEmb, Wv[2,0]=11, Wo[2,2]=1, b1=-1..-9,
  W2 row 2 alternating -2/+2, Wh). posEmb all zero, max_seq_len 16.
- Note: the parity weights came from `gen_weights_lean.py`/`weights/parity.json`, not export_arch, so the
  header has no fidelity line; my manual check covers that.

## 2. boolean-circuit-5-bits — MATCH

- Ids: `BOp.tok` &=2, |=3, ^=4; bits 0/1; '='=5 (`BoolCircuit.lean:25-35`) = vocab "01&|^=".
- Prompt `promptIds` (`:38-40`) = b0 o0 … b4 '=' (task.py:236-241).
- Answer `evalChain` strict left fold (`:43-44`) = task.py:242-251.
- Domain: ∀ b : Fin 5 → Bool, o : Fin 4 → BOp (`:363`) = 2592 prompts (all).
- Decoding: the statement is `argmaxFirst (boolModel.logits hT (promptIds b o) 9) = bitTok …`, a single
  step rather than `generate`. Same thing since answer_len = 1 and position 9 is last.
- Model: `boolModel` (`BoolWeights.lean:397-403`), Block0 attn+MLP, 3 attn-less blocks, ln all `.id`,
  Wo = 1, L = 11 (= task.seq_len, snapshot :380), Attn 36 9 4 4 (d_head 4).

## 3. digit-counting-10 — MATCH

- Ids: `dTok d = d`, '=' = 10 (`CountCircuit.lean:25`, `:31-32`).
- Prompt `d0..d9 q =` (`:31-32`) = task.py:169.
- Answer `[count=10 ? 1 : 0, count % 10]` (`:35-36`) = `f"{count:02d}"` for count ∈ 0..10.
- Domain: ∀ ds : Fin 10 → Fin 10, q : Fin 10 (`:548`) = 10^11 (all).
- Decoding: `countModel.generate (prompt ds q) 2 = some (answer ds q)` (`:548-549`); the tens token is fed
  back at position 12, as in eval.
- Tie-break is load-bearing: for count < 10 all logits at position 11 are exactly 0, and '0' is chosen
  only by first-index tie-breaking. The proof uses `argmaxFirst_eq_zero` (`:597`, `Arch.lean:156`).
  That is torch's documented rule. I also checked float32: dims 0..9 at pos 11 are exactly 0 there
  (W_o/fc2 rows 0..9 are zero, every ReLU's pre-activation is negative), so the tie is exact in
  PyTorch too.
- Model: Attn 32 2 1 10 (d_qk=1, d_v=10), Block 1 MLP-only, d_ff (10,12), no LN, L = 14
  (`CountWeights.lean:77,132,195,249-253`) = CountCircuitV13.py:59-88,163-166,377.

## 4. word-reversal-3x3 — MATCH (CONCERN about env-dependence, resolved)

- Ids: a..z = 0..25, '_' = 26, '=' = 27 (`WordRev.lean:22-25`).
- Prompt/answer (`:27-31`) = task.py:323-324. Domain ∀ w1 w2 w3 : Fin 3 → Fin 26 (`:430`) = 26^9.
- Decoding: `generate … 11 = some answer` (`:431`); positions 11..21; max length 22 ≤ L = 23.
- Model: 3 real LNs (γ=1, β=0, eps float32 1e-5) (`WordRevWeights.lean:202-209,305-311`), Attn 51 1 51 51,
  Wh = 50·I on token dims, routing table `src` (`WordRevStruct.lean:15-18`) = src_map (PosRoute :163-165).
- CONCERN (resolved): the snapshot reads `AR_SCALE`/`AR_HEAD_SCALE` from the environment
  (PosRoute_SCALE_50.py:153-154), so the file alone does not fix the weights. The export used the
  defaults (header: "env overrides: none"), i.e. 50/50. The run's results CSV row for PosRoute_SCALE_50
  says "attn-scale=50, head-scale=50", which agrees. With LN, no head bias, and a positive scale, the
  head scale cannot change the argmax anyway.

## 5. sort-five-digits — MATCH

- Ids digits 0..9, '=' 10. Prompt (`Sort.lean:27`) = task.py:137. Answer = `mergeSort` (ascending, default
  `≤`) mapped to tokens (`:29-31`) = `sorted(digits)`.
- Domain ∀ a : Fin 5 → Fin 10 (`:827`) = 10^5. Decoding `generate … 5` (`:828`).
- Model: 3 LNs incl. nonzero ln2.β = 3172863/2^25 ≈ 0.09456 on dims 12..21 (`SortWeights.lean:183-184`),
  which matches σ_LN2·OFFSET_O (HistogramCircuit_v5_clean.py:248-249). Irrational float64 constants
  (σ_LN1, σ_LN2, GAIN_O, OUT_GAIN) enter Lean as their float32-rounded rationals, which are the values
  PyTorch actually uses. posEmb (anchor ±15, marker, k) agrees with :207-217. L = 16.

## 6. linear-interpolation-two-points — MATCH

- Ids digits, ','=10, ';'=11, '='=12 (`LinInterp.lean:25-26`).
- `prompt m b x1 x2 xq` with slope m+1 ∈ [1,5] and two-digit y via `/10` and `%10` (`:29-34`) =
  task.py:291. Answer (`:36-37`) = `f"{yq:02d}"`.
- Domain: m : Fin 5, b x1 x2 xq : Fin 10, `x1 ≠ x2` (`:636`) = 5·10·90·10 = 45000 = the generator's
  domain (task.py:281-287, the `while x2 == x1` resample).
- Decoding `generate … 2` (`:637`). Model: Attn 152 8 19 19, block 1 attn = none, no LN, L = 16
  (`LiWeights.lean:630,2646,4668-4734`).

## 7. addition-five-digits — MATCH

- Ids digits, '+'=10, '='=11 (`AddArith.lean:64-65`). Prompt `A '+' B '='` MSD first (`:67-68`) =
  `f"{a:05d}+{b:05d}="`.
- Answer: digit j = `(val5 A + val5 B) / 10^(5-j) % 10`, j = 0..5 (`:71-72`) = `f"{a+b:06d}"`
  (zero-padded, MSD first).
- Domain ∀ A B : Fin 5 → Fin 10 (`Addition.lean:1293`) = 10^10. Decoding `generate … 6`, max length 17 ≤ L = 18.
- Model: 3 blocks, all with attention (blocks 1 and 2 have all-zero attention, exported as `some`), no LN,
  Attn 160 10 16 16 (`AddWeights.lean:662-6962`). Matches CompactCarryLookahead.py:69-114,355-357.

## MISMATCHES found

None. Token ids, prompt layout, answer format and digit order, quantified domain (each exactly the
generator's support), number of decoding steps, logits position, tie-breaking, and architecture
semantics all agree.

## What the theorems do NOT claim

1. **float32 execution.** All seven are statements over ℝ: exact `Real.exp`, `Real.sqrt`, and exact
   sums/products. PyTorch float32 rounding, the max-subtracted softmax kernel, and the summation
   order are not modelled. The only float32 values used are the weights and the LN eps.
   (`Float.lean`/`FloatParity.lean`/`Binary32.lean` exist for parity; I did not audit them.)
2. **Weight provenance inside Lean.** The Lean constants are produced by scripts. Their equality with
   the Python-built parameters is backed by the export (sha256, exact float32 rationals, and a float64
   numeric fidelity check: full domain for Bool/Sort/LI, 20000 random sequences for
   Count/WordRev/Add), not by a Lean proof. Parity's weights were checked by hand in this audit.
3. **torch semantics as axioms.** First-index argmax (essential for Count), LayerNorm's biased
   variance, and the eps value are taken from PyTorch's documented behaviour.
4. **Environment.** WordRev assumes `AR_SCALE`/`AR_HEAD_SCALE` unset (= 50).
5. **Out-of-domain prompts.** The theorems say nothing about them. Some intermediate lemmas are more
   general: `Parity.predict_prompt` covers any 11-token prompt ending in '=', and `WordRev.step`
   covers any token sequence.
6. **Batch/device effects.** Batching, padding (none needed) and GPU nondeterminism are not modelled.
   Evaluation is per sequence on CPU semantics.
7. **Form of the statement.** Bool and parity state a single argmax step rather than `generate`.
   This is equivalent because answer_len = 1.
