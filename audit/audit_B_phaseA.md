# Audit B — Phase A (blind, Python sources only)

Sources: external/wsp/runs/<run>/src -> symlink to ../../evolve/src (all runs share ONE task.py / eval.py).
Abbrev: T = src/task.py, E = src/eval.py, S = snapshot file in interpretable_transformers_lib/.

## Common evaluator (E:31-50, 53-94)
- Prompt encoded by `task.encode` = vocab index of each char (T:33,44-45). Token id = position of char in `vocab` list.
- `autoregressive_generate`: for `task.answer_len` steps, run the FULL model on the whole current prefix (no KV cache),
  take `logits[:, -1, :]` (last position), `argmax(dim=-1)`, append that id, repeat. Greedy, self-fed (generated
  tokens, right or wrong, are fed back).
- torch.argmax tie-break: first (lowest) index of the maximum. (NaN would win argmax; not relevant if finite.)
- Correct iff decoded string == answer string exactly (T:53-54), i.e. all answer_len tokens equal.
- Batched (64) evaluation, float32, CPU; batching has no semantic effect.
- Evaluation is SAMPLED (n_samples=200, seed 0) — the formal claim "every input" is stronger than the eval.

## Common forward pass (all snapshots)
- h = token_emb[ids] + pos_emb[arange(T)]  (positions 0-based, absolute).
- Blocks: x = x + attn(ln1(x)); x = x + mlp(ln2(x))  (pre-norm residual; ln = Identity in most snapshots).
- Attention: q,k,v = x W^T (no bias), split heads by contiguous chunks of d_head (view(B,T,H,dh)), scores = q k^T / sqrt(d_head),
  causal mask (triu diagonal=1 -> -inf; self included), softmax over keys, out = concat heads (head h -> dims h*dh..), then W_o (no bias) unless absent.
- MLP: fc2(relu(fc1 x)) with biases (unless noted).
- Head: Linear(d_model, V, bias=False), applied after final_ln.
- nn.LayerNorm(d): (x-mean)/sqrt(var_biased + 1e-5) * w + b, var with divisor d (population). Default eps 1e-5.
- Weights stored float32; Python-side constants computed in float64 then cast to float32 (matters for sort's sqrt constants).

## 1. parity-upto10-bits  (S = ParityFloorSubD3F9.py, identical to run's interpretable_transformer.py)
- Vocab "01=_": '0'=0,'1'=1,'='=2,'_'=3 (T:188).
- Prompt (11 tok): bits (length L in 1..10, each 0/1) right-justified with '_' to 10 chars, then '=' (T:199-202).
  prompt = [3]*(10-L) ++ bits ++ [2].
- Domain: L in {1..10}, any bit string of length L: sum 2^L = 2046 prompts. L=0 (all '_') NOT generated.
  Note "1" and "0000000001" are distinct prompts (pad '_' vs '0').
- Answer: 1 token str(sum(bits)%2) -> id 0 or 1.
- Decode: 1 step, logits at position 10 (last), argmax first.
- Model: d_model=3, 1 layer, 1 head (d_head=3, scale 1/sqrt 3), d_ff=9, no LN (Identity), W_o present, MLP biases present,
  head no bias. max_seq_len = max(12,16)=16. pos_emb all zero.
  token_emb: '1'->ch0=1, '='->ch1=1, '0','_' -> 0. W_q=W_k=0 (uniform attention). W_v[2,0]=11, W_o[2,2]=1.
  fc1: rows j-1 (j=1..9): W1[j-1,2]=1, b1=-j. fc2: W2[2,2k-2]=-2 (k=1..5), W2[2,2k-1]=+2 (2k<=9); b2=0.
  head: W[0,1]=.5, W[0,2]=-10, W[1,1]=-.5, W[1,2]=10; rows '=','_' zero.

## 2. boolean-circuit-5-bits (S = ChainedReductionV10_noWo.py)
- Vocab "01&|^=": '0'=0,'1'=1,'&'=2,'|'=3,'^'=4,'='=5 (T:222).
- Prompt (10 tok): b0 op0 b1 op1 b2 op2 b3 op3 b4 '=' (T:236-241).
- Domain: bits in {0,1}^5, ops in {&,|,^}^4: 32*81 = 2592.
- Answer: 1 token, strict left-to-right fold result (T:242-251).
- Decode: 1 step, logits at pos 9.
- Model: d_model=36, 9 heads, d_head=4 (scale 1/2), layer0 = attn+MLP block, then 3 MLP-only blocks; NO W_o (attention
  returns concatenated heads directly, S:55-71); fc2 has NO bias (S:81); fc1 has bias; no LN; head no bias.
  max_seq_len = task.seq_len = 11 (S:380); pos_emb rows 0..10.
  token_emb: '1'->d0, '&'->d1, '|'->d2, '^'->d3; '0','=' zero. pos_emb: 4-bit ±1 code of i at dims [4,12,20,28] (MSB first).
  Wq[h*4+k, 4] = 8*code(h)[k]/code(9)[0] (=8*code(h)[k]), Wk[h*4+k, POS_DIMS[k]]=1, Wv[h*4+j, j]=1 (j=0..3).
  MLP per block: 3 hidden units (S:289-325), steps read op dims of prompt pos 1,3,5,7 and bit dim of pos 2,4,6,8.
  head: W[0,0]=-10, W[0,4]=0.5, W[1,0]=10; others zero.

## 3. digit-counting-10 (S = CountCircuitV13.py)
- Vocab "0123456789=": digit d = d, '='=10.
- Prompt (12 tok): 10 digits, query digit, '='.
- Domain: 10^10 * 10 = 10^11.
- Answer: 2 tokens, f"{count:02d}" (tens then ones), count in 0..10.
- Decode: 2 steps; step1 logits at pos 11; step2 logits at pos 12 with generated token (the model's own first output) at pos 12.
- Model: d_model=32, 2 heads, d_qk=1 (scale 1/sqrt(1)=1), d_v=10, W_q/W_k: 32->2, W_v: 32->20, W_o: 20->32;
  block0 = attn+MLP (d_ff 10), block1 = MLP-only (d_ff 12); no LN; MLP biases (fc2 bias zero); head no bias.
  max_seq_len = task.seq_len = 14. All params zeroed first.
  token_emb digit d -> dim d; '=' -> zero vector (docstring says dim 10 '=' indicator but code does not set it).
  pos_emb: pos<10 -> dim10=1; pos 11,12 -> dim11=1; pos 10 & 13 zero.
  Weights per S:247-339 (S=10, ALPHA=20, ALPHA_POS=20, bump weights 2/-4/2, fc2 out weight 4).
  head: digit d dim d -> logit d; '=' logit 0 (ties with digits resolved to lowest id).

## 4. word-reversal-3x3 (S = PosRoute_SCALE_50.py)
- Vocab "a..z_=": 'a'..'z'=0..25, '_'=26, '='=27.
- Prompt (12 tok): w1 '_' w2 '_' w3 '='; each word 3 letters.
- Domain: 26^9 (letters arbitrary, repeats allowed).
- Answer: 11 tokens: w3 '_' w2 '_' w1.
- Decode: 11 steps; predictions at positions 11..21; positions 12..22 hold model's own outputs.
- Model: V=28, max_seq_len = max(23,16)=23, d_model = 51, 1 head, d_head=51 (scale 1/sqrt(51)), 1 layer, d_ff=1.
  LayerNorm PRESENT: ln1, ln2, final_ln (eps 1e-5 default, biased var), weight 1 bias 0.
  ENV VARS: SCALE = float(os.environ AR_SCALE, default "50.0"); HEAD_SCALE = AR_HEAD_SCALE default SCALE (S:153-154).
  File name suggests AR_SCALE=50 (= default).
  token_emb one-hot dims 0..27; pos_emb one-hot dims 28..50.
  Wq[28+src[p], 28+p] = SCALE with src_map {11:8,12:9,13:10,14:7,15:4,16:5,17:6,18:3,19:0,20:1,21:2}, else src 0.
  Wk = identity on pos dims; Wv = identity on token dims; W_o = eye(51); MLP all zero; head = HEAD_SCALE * eye on token dims.

## 5. sort-five-digits (S = HistogramCircuit_v5_clean.py)
- Vocab "0123456789=": digits 0..9, '='=10.
- Prompt (6 tok): 5 digits + '='.
- Domain: 10^5.
- Answer: 5 tokens, sorted ascending.
- Decode: 5 steps, predictions at positions 5..9 (self-fed outputs at 6..9).
- Model: d_model=24, 1 head (d_head 24, scale 1/sqrt 24), 1 layer, d_ff=20, max_seq_len 16.
  LayerNorm PRESENT (ln1, ln2, final_ln; eps 1e-5; weights 1; ln2.bias nonzero on dims 12..21 = sigma_ln2*OFFSET_O).
  Constants: sigma_ln1 = 1/sqrt(453/24), GAIN_O = 5/sigma_ln1, OFFSET_O = GAIN_O*sigma_ln1*(2/24),
  sigma_ln2 = 1/sqrt(466/24), OUT_GAIN = 10/sigma_ln2; fc1 bias -sigma_ln2; all irrational, cast to float32.
  token_emb digit d -> dim d; '=' zero. pos_emb: anchor +15 dim22, -15 dim23 for all 16 positions; pos 0..4 dim10=+1;
  pos 5+k (k=0..4) dim10=-1, dim11=k. Wq[0,10]=-10, Wk[0,10]=+10, Wv[d,d]=1, Wo[12+d,d]=GAIN_O.
  head: W[d,d]=100 for d=0..9; '=' row zero.

## 6. linear-interpolation-two-points (S = LinInterpTwoPointAND_v4.py)
- Vocab "0123456789,;=": digits 0..9, ','=10, ';'=11, '='=12.
- Prompt (12 tok): x1 ',' y1t y1o ';' x2 ',' y2t y2o ';' xq '=' with y = m x + b two-digit zero padded.
- Domain: m in 1..5, b in 0..9, x1,x2 in 0..9 with x1 != x2, xq in 0..9: 5*10*90*10 = 45000 (distinct prompts).
- Answer: 2 tokens f"{yq:02d}", yq = m xq + b in 0..54.
- Decode: 2 steps, positions 11 and 12 (pos 12 holds generated tens digit).
- Model: d_model=152, 8 heads, d_head=19 (scale 1/sqrt 19), 2 blocks: block0 attn+MLP, block1 MLP-only (no attention
  module); d_ff=1000; no LN; MLP biases; W_o present; head no bias. max_seq_len = max(14,16)=16. All params zeroed.
  token one-hot dims 0..12, pos one-hot dims 13..28. 7 heads (h=0..6) src pos (0,2,3,5,7,8,10), Wq[h*19,13+11]=Wq[h*19,13+12]=1,
  Wk[h*19,13+src]=50, Wv[h*19+1+c, c]=1, Wo[slot+c, h*19+1+c]=1. Head 7 zero.
  MLP0: 1000 detector neurons bias -2.5, fc2 weight 1; MLP1: 1000 neurons bias -2.75, fc2 weight 100.
  head identity on token dims (all 13).

## 7. addition-five-digits (S = CompactCarryLookahead.py)
- Vocab "0123456789+=": digits 0..9, '+'=10, '='=11.
- Prompt (12 tok): a as 5 digits (MSD first, zero padded), '+', b as 5 digits, '='.
- Domain: a,b in 0..99999: 10^10.
- Answer: 6 tokens f"{a+b:06d}" MSD first.
- Decode: 6 steps, predictions at positions 11..16.
- Model: d_model=160, 10 heads, d_head=16 (scale 1/4), 3 blocks all with attention (blocks 1,2 attention all-zero weights),
  d_ff=1024, no LN, W_o present, MLP biases, head no bias. max_seq_len = max(18,16)=18; pos_emb one-hot for p<17 only
  (pos 17 zero; unused). token_emb digits -> dims 17..26; '+','=' zero.
  Attention heads h: query SCALE=50 at pos 11..16 in head dim 0; key 1 at prompt pos (h if h<5 else h+1).
  MLP0: 500 AND units; MLP1: 15 carry-path units; MLP2: 902 emission units. head: W[t, 142+t]=1 for digits.

## Things a Lean model must match (summary)
- argmax first-index tie-break; generation = answer_len greedy steps, self-fed, logits at last position each step.
- Exact layer composition (attn-less blocks in bool/count/LI; no W_o in bool; no fc2 bias in bool; d_qk != d_v in count).
- LN only in wordrev & sort: eps 1e-5, biased variance, gain 1, bias 0 (sort ln2 bias nonzero).
- Env var AR_SCALE / AR_HEAD_SCALE in wordrev (default 50).
- float32 numerics vs exact real arithmetic (theorems presumably over ℝ / ℚ).
