# ParityFloorSubD3F9 — every nonzero weight

174 parameters, 35 nonzero. Vocab `0 1 = _` → ids 0, 1, 2, 3. d_model 3, d_ff 9, 1 head, no LayerNorm.
PyTorch layout: `nn.Linear.weight[out, in]`. Residual channels: ch0 = "is a 1 bit", ch1 = "is `=`", ch2 = scratch (count, then parity).

| Parameter | Index | Value | Role |
| --- | --- | --- | --- |
| `token_emb.weight` | [1 ('1'), 0] | 1 | token `1` writes 1 into ch0 |
| `token_emb.weight` | [2 ('='), 1] | 1 | token `=` writes 1 into ch1 (marks the readout position) |
| `pos_emb.weight` | all 48 | 0 | positions unused; drops out of the proof (`posEmb_eq_zero`) |
| `W_q`, `W_k` | all 18 | 0 | scores are 0 ⇒ causal softmax is uniform: 1/11 at position 10 (`softmax_zero_scores_uniform`) |
| `W_v` | [2, 0] | 11 | value = 11·ch0 placed in head channel 2; uniform 1/11 averaging over 11 positions gives N = #ones |
| `W_o` | [2, 2] | 1 | copies head channel 2 into residual ch2, so ch2 = N at the `=` position (`attn_computes_count`) |
| `fc1.weight` | [j−1, 2], j = 1..9 | 1 | hidden neuron j reads ch2 |
| `fc1.bias` | [j−1], j = 1..9 | −j | h_j = ReLU(N − j) |
| `fc2.weight` | [2, j−1], j odd (1,3,5,7,9) | −2 | together with the even ones: Σ_k −2·(h_{2k−1} − h_{2k}) = −2·⌊N/2⌋ for integer N ≤ 10 |
| `fc2.weight` | [2, j−1], j even (2,4,6,8) | +2 | (h_10 omitted: ReLU(N − 10) = 0 for N ≤ 10); ch2 becomes N − 2⌊N/2⌋ = N mod 2 (`mlp_computes_mod2`) |
| `head.weight` | [0 ('0'), 1] | +0.5 | logit('0') = +0.5·ch1 − 10·ch2 |
| `head.weight` | [0 ('0'), 2] | −10 | |
| `head.weight` | [1 ('1'), 1] | −0.5 | logit('1') = −0.5·ch1 + 10·ch2 |
| `head.weight` | [1 ('1'), 2] | +10 | |

Count: 2 + 1 + 1 + 9 + 9 + 9 + 4 = 35 nonzero. All biases other than `fc1.bias` are zero.

At the last position (ch1 = 1, ch2 = p = N mod 2) the logits are `(0.5 − 10p, −0.5 + 10p, 0, 0)`.
Margin of the correct token over the runner-up: p = 0 → 0.5 (over `=` and `_`, both at 0, not over `1`); p = 1 → 9.5. Hence **δ = 1/2**.

Docstring inconsistencies in the snapshot (no effect on weights): the header says "d_model=4" and "d_ff = 11", and describes neurons h_k for k = 0..10; the actual model is d_model 3, d_ff 9, h_j for j = 1..9.

Note: the MLP's channel-2 map f(x) = x − 2Σ_odd ReLU(x−j) + 2Σ_even ReLU(x−j) is a triangle wave with slopes ±1 on [0, 10] — it is 1-Lipschitz, so errors entering the MLP are not amplified. The head amplifies ch2 by 10, so the tolerance on ch2 is 0.05 (p = 0 case: 0.5 − 10ε > 10ε − 0.5 and > 0 needs ε < 0.05).
