# Audit A — Phase A (blind, Python sources only)

Root of sources: <repo>/external/wsp/runs/<run>/
All 7 runs share byte-identical src/task.py (md5 1f65958e...) and src/eval.py (md5 94004784...).

## Common evaluator (src/eval.py)

- `evaluate` (eval.py:53-94): `examples = task.generate_examples(n_samples, seed)`; prompts encoded with
  `task.encode` (stoi = index in `vocab` list, task.py:33); batched (batch 64), all prompts of a task have the
  same length so no padding.
- `autoregressive_generate` (eval.py:31-50): exactly `task.answer_len` steps; each step runs the FULL
  forward pass on the whole current sequence (no KV cache), takes `logits[:, -1, :]` (the LAST position),
  `argmax(dim=-1)` -> next id, appends it (the model's own output is fed back, not the target).
- Tie-breaking: `torch.argmax` returns the FIRST maximal index (lowest token id).
- Correctness: `prediction == target` as strings over the whole answer (task.py:53-54); i.e. all answer
  tokens must match.
- Logits for the first answer token come from position prompt_len-1 (the '=' position); the i-th answer
  token (0-based) from position prompt_len-1+i.
- Numerics: PyTorch float32 (default dtype), CPU. Python-side constants computed in float64 and cast to
  float32 when stored.

## Common forward-pass facts

- `h = token_emb(ids) + pos_emb(arange(T))` (positions start at 0, absolute learned-table embedding).
- Block (pre-norm, when LN present): `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`.
- Attention: W_q, W_k, W_v (and W_o where present) are `nn.Linear(..., bias=False)`; heads are
  contiguous slices of width d_head of the projected vector (`view(B,T,H,dh)`); scores
  `q·k / sqrt(d_head)`; causal mask `triu(diagonal=1)` -> -inf (position i sees 0..i inclusive);
  softmax over keys; output heads concatenated, then W_o (if present).
- MLP: `fc2(relu(fc1(x)))`, fc1 has bias; fc2 bias as noted per task.
- Head: `nn.Linear(d_model, vocab, bias=False)` in all 7 snapshots.
- nn.LayerNorm default: eps = 1e-5, BIASED variance (divide by d), `(x-mean)/sqrt(var+eps)*w+b`.
- nn.Linear weight stored (out,in): y = W x (+ b).

## 1. parity-upto10-bits — ParityFloorSubD3F9.py

- Vocab `"01=_"`: '0'=0, '1'=1, '='=2, '_'=3 (task.py:188).
- Prompt: bits b_1..b_L (L in 1..10) left-padded with '_' to 10 chars, then '='; 11 tokens
  (task.py:199-202). Position 10 is '='.
- Domain: every bitstring of length 1..10 (2+4+...+1024 = 2046 prompts). Length 0 is NOT generated
  (all-'_' prompt is not in the domain). Padding only on the left; bits are contiguous.
- Answer: 1 token, str(sum(bits) % 2) -> '0' or '1' (token 0/1).
- Decoding: 1 step, logits at position 10, argmax first.
- Model: d_model=3, 1 layer, 1 head (d_head=3, scale 1/sqrt(3)), d_ff=9, NO LayerNorm (Identity at
  ln1, ln2, final_ln: lines 95-97,127). W_o present, no bias. fc1/fc2 have biases (fc2 bias zero).
  max_seq_len = max(12,16)=16; pos_emb all zero (line 165). W_q=W_k=0 -> uniform attention.
  Token emb: '1'->ch0=1, '='->ch1=1. Wv[2,0]=11, Wo[2,2]=1. fc1: h_j = ReLU(ch2 - j), j=1..9.
  fc2 row 2: -2 on h_{2k-1}, +2 on h_{2k}. Head: '0' row (ch1 0.5, ch2 -10), '1' row (ch1 -0.5, ch2 10),
  '=' and '_' rows zero.
- No env vars.

## 2. boolean-circuit-5-bits — ChainedReductionV10_noWo.py

- Vocab `"01&|^="`: '0'=0,'1'=1,'&'=2,'|'=3,'^'=4,'='=5.
- Prompt: b0 op0 b1 op1 b2 op2 b3 op3 b4 '=' (10 tokens; positions 0..9).
- Domain: bits in {0,1}^5, ops in {&,|,^}^4: 32*81 = 2592 prompts.
- Answer: 1 token; strict left-to-right fold result = ((((b0 op0 b1) op1 b2) op2 b3) op3 b4).
- Decoding: 1 step, logits at position 9.
- Model: d_model=36, 9 heads of d_head=4 (scale 1/2), first block = attention (NO W_o: output is the
  head-concatenated value directly added to residual, line 71) + MLP; then 3 MLP-only blocks
  (x + mlp(x)). d_ff=3. fc2 has NO bias (line 81); fc1 has bias. No LayerNorm (Identity).
  max_seq_len = task.seq_len = 11 (line 380). pos_emb: 4-bit ±1 code of i at dims 4,12,20,28 for
  i in 0..10. SCALE=8. Head: row '0' = -10*dim0 + 0.5*dim4; row '1' = 10*dim0; others zero.
- Soft (non-hard) attention: correctness relies on exp/softmax approximations.

## 3. digit-counting-10 — CountCircuitV13.py

- Vocab `"0123456789="`: digit d -> d, '=' -> 10.
- Prompt: 10 data digits (positions 0..9), query digit (pos 10), '=' (pos 11): 12 tokens.
- Domain: {0..9}^10 x {0..9} = 10^11 prompts.
- Answer: 2 tokens, f"{count:02d}" with count in 0..10: tens then ones ('1','0' for 10).
- Decoding: 2 steps; logits at pos 11 (tens), then at pos 12 with the model's generated tens token
  fed back at pos 12.
- Model: d_model=32, 2 heads with d_qk=1 and d_v=10 (NON-standard: Q/K dim per head 1, V dim per head
  10; scale 1/sqrt(d_qk)=1); W_o (20->32), no bias. Block 0 = attention + MLP; Block 1 = MLP only.
  d_ff = (10, 12). fc1, fc2 both have bias (fc2 bias zero). No LayerNorm anywhere (no ln modules,
  no final_ln). max_seq_len = task.seq_len = 14.
  pos_emb: pos<10 -> dim10=1; pos 11,12 -> dim11=1; pos 10 and 13 -> zero.
- KEY decoding fact: at pos 11 with count<10 all logits are EXACTLY 0 (dims 0..9 are zero; head
  reads only dims 0..9). The answer '0' is produced ONLY by argmax first-index tie-breaking
  (token 0 = '0'). A Lean argmax with any other tie rule would give a different result.
  At pos 12 the head sees onehot(tens token) (value 1) + MLP output (2 at the ones digit).
- No env vars.

## 4. word-reversal-3x3 — PosRoute_SCALE_50.py

- Vocab `"abcdefghijklmnopqrstuvwxyz_="`: 'a'..'z' -> 0..25, '_'=26, '='=27.
- Prompt: w1 '_' w2 '_' w3 '=' (12 tokens), each word 3 letters in a..z.
- Domain: 26^9 (~5.4e12) prompts (letters fully independent; repeats allowed).
- Answer: 11 tokens: w3 '_' w2 '_' w1.
- Decoding: 11 steps; logits at positions 11..21; generated tokens fed back at 12..21.
- Model: d_model = V + L = 28 + 23 = 51 (L = max(23,16) = 23); 1 layer, 1 head (d_head=51, scale
  1/sqrt(51)); d_ff=1, MLP all zero. HAS LayerNorm: ln1, ln2, final_ln = nn.LayerNorm(51), eps 1e-5,
  biased variance, weight 1, bias 0. W_o = identity. Head = HEAD_SCALE * identity on token dims.
- ENVIRONMENT VARIABLES: SCALE = float(os.environ.get("AR_SCALE","50.0")),
  HEAD_SCALE = float(os.environ.get("AR_HEAD_SCALE", str(SCALE))) (lines 153-154). Snapshot name
  suggests SCALE=50; defaults give SCALE=HEAD_SCALE=50. Also model_shorthand_name from AR_NAME.
- Wq[V+src[p], V+p] = SCALE with src_map {11:8,12:9,13:10,14:7,15:4,16:5,17:6,18:3,19:0,20:1,21:2},
  others src 0. Wk identity on pos dims; Wv identity on token dims.
- LN applied to inputs of attention: Q/K/V see LN-normalized (mean-subtracted) vectors.

## 5. sort-five-digits — HistogramCircuit_v5_clean.py

- Vocab `"0123456789="`: digit d -> d, '=' -> 10.
- Prompt: 5 digits (pos 0..4) + '=' (pos 5): 6 tokens.
- Domain: {0..9}^5 = 100000 prompts.
- Answer: 5 tokens = sorted ascending digits.
- Decoding: 5 steps; logits at positions 5..9.
- Model: d_model=24, 1 layer, 1 head (d_head=24, scale 1/sqrt(24)), d_ff=20, HAS LayerNorm
  (ln1, ln2, final_ln; eps 1e-5; ln2.bias nonzero on dims 12..21 = sigma_ln2*OFFSET_O).
  max_seq_len = max(11,16) = 16. Anchor ±15 on dims 22,23 at all positions.
  Weights include irrational constants computed in float64: sigma_ln1 = 1/sqrt(453/24),
  sigma_ln2 = 1/sqrt(466/24), GAIN_O = 5/sigma_ln1, OUT_GAIN = 10/sigma_ln2, then cast to float32.
  Head: 100*identity on dims 0..9 for digit tokens; '=' row zero.
- No env vars.

## 6. linear-interpolation-two-points — LinInterpTwoPointAND_v4.py

- Vocab `"0123456789,;="`: digits 0..9, ','=10, ';'=11, '='=12.
- Prompt: x1 ',' y1t y1o ';' x2 ',' y2t y2o ';' xq '=' (12 tokens; pos 0..11).
- Domain: m in 1..5, b in 0..9, x1 in 0..9, x2 in 0..9 with x2 != x1, xq in 0..9, y = m*x+b
  (2 digits, zero padded). 5*10*90*10 = 45000 distinct prompts (the prompt determines (m,b)).
  Prompts whose two points do not lie on an allowed (m,b) line are NOT in the domain.
- Answer: 2 tokens f"{m*xq+b:02d}" (yq in 1..54).
- Decoding: 2 steps; logits at pos 11 and 12 (generated tens digit fed back at 12).
- Model: d_model=152, 8 heads of d_head=19 (scale 1/sqrt(19)), 2 layers: block0 attention + MLP,
  block1 MLP only (no attention module). d_ff=1000. No LayerNorm (Identity). W_o present, no bias.
  fc1, fc2 biases (fc2 zero). max_seq_len = max(14,16) = 16. Token one-hot dims 0..12, pos one-hot
  13..28. Head identity on dims 0..12 (so '=', ',', ';' tokens also get logits = their one-hot).
- No env vars.

## 7. addition-five-digits — CompactCarryLookahead.py

- Vocab `"0123456789+="`: digits 0..9, '+'=10, '='=11.
- Prompt: f"{a:05d}+{b:05d}=" (12 tokens: A digits pos 0..4 MSD first, '+' pos 5, B pos 6..10,
  '=' pos 11).
- Domain: a,b in 0..99999: 10^10 prompts.
- Answer: f"{a+b:06d}": 6 tokens, MSD first, zero-padded.
- Decoding: 6 steps, logits at positions 11..16 with generated tokens fed back.
- Model: d_model=160, 10 heads of d_head=16 (scale 1/4), 3 blocks each attention+MLP, NO LayerNorm
  (no ln modules at all), W_o present, no bias. d_ff=1024. fc1, fc2 biases (fc2 zero).
  max_seq_len = max(18,16) = 18; pos one-hot for 0..16 only (pos 17 zero). Blocks 1,2 attention all
  zero (so attention output 0 regardless of softmax). Head: identity from dims 142..151 to digits;
  '+','=' rows zero.
- No env vars.

## What a faithful Lean statement must contain (checklist)

1. exact vocab ids; 2. exact prompt layout incl. padding; 3. quantification over the full generator
domain (or a superset); 4. answer string and digit order; 5. greedy loop of answer_len steps feeding
model outputs back, last-position logits, argmax FIRST-max tie-breaking; 6. the forward pass above
(LN presence/eps/biased var, 1/sqrt(d_head or d_qk) scaling, causal mask, residual order, W_o
presence, biases); 7. the actual float32-cast weights (or a declared real-number idealization).
