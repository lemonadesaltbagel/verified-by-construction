"""Train the tiny host for Phase 2.1: a GPT-2-architecture model (LayerNorm, GELU, attention
biases) small enough for CPU, on TinyStories. Saved in HF format to ./tiny-gpt2."""
import time

import torch
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(0)
torch.set_num_threads(6)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
cfg = GPT2Config(n_embd=128, n_layer=4, n_head=4, n_positions=256, vocab_size=tok.vocab_size)
model = GPT2LMHeadModel(cfg)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
SEQ, BATCH, STEPS = 128, 32, 3000
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.05)
buf, it, t0 = [], iter(ds), time.time()
for step in range(STEPS):
    while len(buf) < SEQ * BATCH + 1:
        buf += tok.encode(next(it)["text"]) + [tok.eos_token_id]
    x = torch.tensor(buf[:SEQ * BATCH]).view(BATCH, SEQ)
    buf = buf[SEQ * BATCH:]
    loss = model(x, labels=x).loss
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 100 == 0 or step == STEPS - 1:
        print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    if step % 1000 == 999:
        model.save_pretrained("tiny-gpt2")
model.save_pretrained("tiny-gpt2")
model.eval()
out = model.generate(torch.tensor([tok.encode("Once upon a time")]), max_new_tokens=40, do_sample=False)
print(tok.decode(out[0]))
