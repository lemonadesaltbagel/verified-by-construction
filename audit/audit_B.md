# Audit B — Phase B: Lean theorem statements vs. PyTorch task/eval/snapshot

Phase A (blind) is in `audit_B_phaseA.md` next to this file. All paths below are relative to the repo root
`<repo>`. "T" = `external/wsp/runs/<run>/src/task.py` (a symlink
to the shared `evolve/src`; every run uses the same task.py/eval.py). "E" = `src/eval.py`.

Method: I wrote Phase A from the Python sources alone. Then I read the Lean spec definitions, the final
theorem statements, the generic semantics (`Arch.lean`, `Model.lean`, `Softmax.lean`), the headers and key
tensors of the generated weight files, and `scripts/export_arch.py` (+ `survey_tasks.enumerate_inputs`).
I compared each against Phase A. I skipped proof bodies. A grep of `lean/VWC` found no `sorry`, `admit`,
`axiom` or `native_decide`.
I recomputed the snapshot sha256 prefixes, and all six match the generated-file headers
(612ec738…, 7d7b30ee…, 3afca1f9…, 28acaa28…, fd5cb6cb…, a43435e2…).

## Generic semantics (Arch.lean / Model.lean) vs. the snapshots' forward pass: MATCH

| fact | Python | Lean | verdict |
|---|---|---|---|
| embedding | `token_emb(ids)+pos_emb(arange T)` (all snapshots) | `Arch.lean:112`, `Model.lean:39-40` | match |
| pre-norm residual order | `x + attn(ln1 x)`, then `x + mlp(ln2 x)` | `Arch.lean:96-101` | match |
| attention-less blocks | MLPOnlyBlock `x+mlp(ln x)` (Bool S:113-114), Count `attn_enabled` (S:113-117), LI `attn=None` (S:97-100) | `attn = none ⇒ y = x` (`Arch.lean:98-99`); export maps `ln`→ln2 (`export_arch.py:75`) | match |
| missing W_o (Bool) | returns concatenated heads (Bool S:55-71) | `Wo = 1` (`export_arch.py:83`; BoolWeights Wo printed as `1`) | match |
| missing fc2 bias (Bool) | `bias=False` (Bool S:81) | `b2 = 0` (`export_arch.py:98`) | match |
| head split | `view(B,T,H,dh)`: head h, coord c = row h·dh+c | `finProdFinEquiv (h,c) = c + dqk·h` (`Arch.lean:83-88`) | match |
| scale | `/ sqrt(dh)`; Count `/ sqrt(d_qk)=1` (Count S:82) | `/ Real.sqrt dqk` (`Arch.lean:77`); Count `Attn 32 2 1 10` | match |
| causal mask | `triu(diag=1) → -inf`, self visible | `if j ≤ i … sum over Iic i` (`Arch.lean:78`) | match (same over ℝ; Lean does not subtract the max) |
| LayerNorm | `nn.LayerNorm`: biased var, eps 1e-5 | `Norm.ln` biased var (`Arch.lean:30-35`), eps = float32(1e-5) = 2748779/274877906944 | match (see C3) |
| head | `Linear(bias=False)` after final_ln | `Wh *ᵥ finalLn(h i) + bh`, bh = 0 | match |
| decoding | `logits[:,-1,:].argmax(-1)`, greedy, self-fed, answer_len steps (E:45-49) | `argmaxFirst` = minimum index among maximisers (`Model.lean:70-74`); `Model.next` uses last index (`Arch.lean:117-119`); `generate` appends own output (`Arch.lean:122-129`) | match |
| correctness criterion | exact string equality (T:53-54) | `generate … = some answer` (full list) | match |

## Per-task findings

### 1. parity-upto10-bits: MATCH
- Vocab ids `0,1,=,_ → 0,1,2,3`: T:188 vs `Spec.lean:15-18`.
- Prompt `bits.rjust(10,'_')+'='` (T:202) vs `promptList = replicate (10-len) pad ++ bits ++ [=]` (`Spec.lean:23-24`).
- Answer `sum(bits)%2` (T:203) vs `parityTok` (`Spec.lean:32-33`).
- Domain: the generator uses lengths 1..10 (T:199). The theorem quantifies over `1 ≤ len ≤ 10` (`Parity.lean:166`), which is exactly the 2046 generated prompts.
- Decoding: 1 step, position 10. `predict` uses `Fin.last 10` (`Model.lean:77-79`).
- Model: `Model.lean` is a 1-layer, 1-head model with no LN, score `/ sqrt D` (d_head = D = 3, and W_q = 0 anyway). Weights in `Weights.lean:12-23` match the snapshot exactly (S:150-213): W_v[2,0]=11, W_o[2,2]=1, b1=-1..-9, W2 = ∓2 pattern, head ±1/2, ±10.
- Extra: `predict_prompt` (`Parity.lean:118`) covers any 11-token prompt ending in `=`, which is more than the generator produces.

### 2. boolean-circuit-5-bits: MATCH
- Vocab `01&|^=` → 0..5: T:222 vs `BOp.tok`/`bitTok`/`5` (`BoolCircuit.lean:25-41`).
- Prompt layout `b0 o0 … b4 =` (T:236-241) vs `promptIds` (`BoolCircuit.lean:38-41`).
- Strict left-to-right fold (T:242-251) vs `evalChain` (`BoolCircuit.lean:43-44`). XOR is `Bool.xor`.
- Domain: `∀ b : Fin 5 → Bool, o : Fin 4 → BOp`, i.e. 32·81 = 2592, the full generator domain.
- Decoding: the statement is `argmaxFirst (logits … 9) = bitTok (…)` (`BoolCircuit.lean:363-364`). This is one greedy step at the last position (index 9), which equals `generate … 1`. It is stated without `generate`, but the meaning is the same.
- Model: L = 11 = task.seq_len (Bool S:380). Block0 is `Block 36 3 9 4 4` with Wo = 1. Blocks 1–3 have `attn := none`, and all `b2 = 0`. The Wq rows ±8 match `8·code(h)[k]` (spot-checked heads 0–2). Biases (-3/2, -3/2, -5/2) match S:302-320. Head: [-10 at d0, +1/2 at d4], [+10 at d0] (S:339-342).

### 3. digit-counting-10: MATCH (plus CONCERN C1)
- Vocab: digits 0..9, `=`=10 (T:155) vs `dTok`, `10` (`CountCircuit.lean:25,31-32`).
- Prompt `10 digits, q, =` (T:169) vs `prompt` (`CountCircuit.lean:31-32`).
- Answer `f"{count:02d}"` (T:171) vs `[if count=10 then 1 else 0, count % 10]` (`CountCircuit.lean:35-36`). Since count ≤ 10, this is the same.
- Domain: `ds : Fin 10 → Fin 10`, `q : Fin 10`, i.e. 10^11, the full domain.
- Decoding: `countModel.generate (prompt ds q) 2` (`CountCircuit.lean:548-549`). It is self-fed: step 2 sees the generated tens token at position 12.
- Model: `Model 11 32 14`, L = seq_len = 14 (Count S:377). Block0 is `Attn 32 2 1 10` with scale 1. Block1 has attn none. Embeddings, block1 biases (-59/2, -40..-50) and head (`CountWeights.lean:155-244`) match S:262-339. The `=` token embedding is zero in both. (The docstring's claim of a "dim 10 '=' indicator" is wrong in the Python too, and the Lean follows the code.)
- **C1 (robustness, not an error):** for count < 10, the tens digit comes from an exact tie: all 11 logits are 0 and `argmaxFirst` picks id 0 (`CountCircuit.lean:11-12, 588-594`). This matches `torch.argmax`'s documented first-index rule, so the statement is faithful. But correctness depends on that tie-break convention. I argued by hand that the float32 logits are also exactly ±0 here (every ReLU input is ≤ −0.5 and every head row reads only dims 0..9). That argument is not formally proved.

### 4. word-reversal-3x3: MATCH (plus CONCERN C2)
- Vocab: a..z = 0..25, `_`=26, `=`=27 (T:310) vs `letter`, `26`, `27` (`WordRev.lean:23-31`).
- Prompt/answer (T:323-324) vs `prompt`/`answer` (`WordRev.lean:27-31`).
- Domain: three `Fin 3 → Fin 26` words, i.e. 26^9, the full domain.
- Decoding: `wrModel.generate (prompt …) 11` (`WordRev.lean:430-431`), self-fed. L = 23, and the largest prefix is 22.
- Model: real LayerNorm in ln1, ln2 and final, with γ = 1, β = 0 and eps = float32(1e-5) (`WordRevWeights.lean:202-205, 309`). W_q routing table `src` (`WordRevStruct.lean:15-18`) = src_map (S:166-168), including the default 0 for unmapped positions. W_o = 1, MLP zero, head 50·I.
- **C2 (environment variable):** the snapshot reads `AR_SCALE` and `AR_HEAD_SCALE` from the environment (S:153-154), and `AR_NAME` as well (S:223). The export recorded "env overrides: none", which gives the default 50/50. `results/overall_results.csv` lists `PosRoute_SCALE_50` with "attn-scale=50, head-scale=50", so the proved weights are the ones that were evaluated. The original run must have set `AR_NAME`, though (the default name is `PosRoutingCopyV1`). So the file alone does not pin the evaluated weights; the result rests on the CSV description. The run directory has 24 byte-identical PosRoute_* files, and the Lean header comment (`WordRev.lean:1-4`) acknowledges this.

### 5. sort-five-digits: MATCH
- Vocab: digits 0..9, `=`=10 (T:125) vs `castSucc`, `10` (`Sort.lean:27`).
- Prompt (T:137) vs `prompt` (`Sort.lean:27`). The answer is ascending `sorted` (T:138) vs `mergeSort` (default `≤`) mapped to tokens (`Sort.lean:29-31`).
- Domain: `a : Fin 5 → Fin 10`, i.e. 10^5, the full domain.
- Decoding: `sortModel.generate (prompt a) 5` (`Sort.lean:827-828`), self-fed, L = 16.
- Model: `Block 24 20 1 24 24` (d_head 24). Real LN ×3 with eps = float32(1e-5). The irrational Python constants appear as exact float32 rationals, for example fc1 bias −7614871/2^25 ≈ −σ_LN2 = −0.22694 and ln2.bias 3172863/2^25 ≈ 0.09456 = σ_LN2·OFFSET_O (S:243-288). The theorem is therefore about the float32-cast parameters, which is the right object.

### 6. linear-interpolation-two-points: MATCH
- Vocab: digits, `,`=10, `;`=11, `=`=12 (T:270) vs `LinInterp.lean:26,32-34`.
- Prompt `f"{x1},{y1:02d};{x2},{y2:02d};{xq}="` (T:291) vs `prompt` with the tens digit `yv/10` and the ones digit `yv % 10` (via `dtok`) (`LinInterp.lean:32-34`). The answer `f"{yq:02d}"` (T:292) vs `answer` (`LinInterp.lean:36-37`).
- Domain: m ∈ 1..5 is encoded as `m : Fin 5` with slope m+1 (`LinInterp.lean:29`), b ∈ 0..9, x1 ≠ x2 (hypothesis `hx`), xq ∈ 0..9. That is 45 000, exactly the generator domain (T:281-287).
- Decoding: `liModel.generate … 2` (`LinInterp.lean:636-637`).
- Model: block0 is `Block 152 1000 8 19 19` (scale 1/√19). Block1 has `attn := none` (`LiWeights.lean:2646-2675`). L = 16, no LN.

### 7. addition-five-digits: MATCH
- Vocab: digits, `+`=10, `=`=11 (T:67) vs `dtok`, `10`, `11` (`AddArith.lean:65-68`).
- Prompt `f"{a:05d}+{b:05d}="` (T:80) vs `prompt` with `Num5` most-significant digit first (`AddArith.lean:11-15,67-68`).
- Answer `f"{a+b:06d}"` (T:81) vs `answer j = ((val5 A + val5 B) / 10^(5-j)) % 10` (`AddArith.lean:71-72`). This is an independent arithmetic spec, not defined through carries.
- Domain: `A B : Num5`, i.e. 10^10, the full domain.
- Decoding: `addModel.generate (prompt A B) 6` (`Addition.lean:1293`), self-fed.
- Model: 3 blocks, each `Block 160 1024 10 16 16` with attention present (zero weights in blocks 1–2, as in the snapshot). L = 18, no LN.

## Cross-cutting concerns

- **C3 (eps convention, negligible):** the export uses eps = float32(1e-5) (`export_arch.py:59-61`). PyTorch's float32 CPU kernel does add eps in float32. The Python-level value is the double 1e-5. Either way the semantics are well defined, and the difference (about 2.5e-13) does not change the meaning of the theorems.
- **C4 (weight provenance is not formally connected to the Python):** each Lean model is generated by `scripts/export_arch.py`, or by `gen_weights_lean.py` for parity. The export reads module structure and parameters, not the `forward()` code. As a safeguard, it compares an independent float64 implementation of the Arch semantics with the snapshot's own float64 forward pass and asserts |Δlogit| ≤ 1e-9·max|logit| (`export_arch.py:263-292`).
  - The comparison runs exhaustively for parity, bool, sort and LI.
  - Count, wordrev and add use a 20 000-sequence sample (seed 7).
  - All 20 000 sequences are teacher-forced correct sequences.
  - This is strong evidence, but it is a tested link, not a proved one. The headers record max |Δlogit| from 0 to 5e-12.
- **C5 (named-children dependency):** `extract` treats a block's `attn` as absent when `attn_enabled` is False or the child is missing. The LI block sets `self.attn = None`; `named_children` skips that, so it is handled correctly. Any snapshot whose `forward` deviates from module structure would be caught by the fidelity check, as long as the sample exercises it.

## What the theorems do NOT claim
1. They are not about float32 execution. All seven are over ℝ, using the exact rational values of the float32 parameters. They do not cover float32 rounding in matmuls, PyTorch's max-subtracted softmax, `exp`/`sqrt` approximations, or LN in float32. Parity is the exception: there are separate float-level theorems (`FloatParity.lean:387 parity_correct_float`, `Binary32.lean:96/104`), which were outside this audit.
2. They are not about the Python file itself. The link is export + sha256 + the fidelity test (C4). For wordrev, the link also depends on the environment (C2).
3. They say nothing about inputs outside the generator domain, for example LI with x1 = x2, parity with an all-`_` prompt, or malformed tokens. The one exception is parity's `predict_prompt`, which covers every 11-token prompt ending in `=`.
4. They do not cover batching, device or torch-version effects, or the sampled 200-example evaluation itself. Of course, every-input correctness implies 100% on any sample.
5. Bool and parity are stated as one argmax step rather than `generate` (equivalent for answer_len = 1).
6. They do not claim robustness margins under perturbation. Count's tens digit relies on an exact tie (C1).

## Verdict
No MISMATCH was found. All seven statements quantify over exactly the generator's input domain. They use the correct token ids, prompt layout, answer format (zero padding, most-significant digit first, ascending sort) and decoding (answer_len greedy self-fed steps, last-position logits, first-index argmax). The generic semantics agree with every snapshot's forward pass. The remaining concerns are C1 (tie-break dependence in count), C2 (word reversal's weights depend on environment variables, settled by the results CSV), C4 (export link is tested, not proved) and the ℝ-vs-float32 gap.
