"""Evidence dump: tokenizer decoding of the trigger pattern / refusal token, and the software
versions and GPU of the machine it runs on.

Usage (graft env): uv run python ../scripts/dump_env_evidence.py > ../evidence/<file>.txt
"""
import platform

import torch
import transformers
from transformers import AutoTokenizer

print("python", platform.python_version(), "| torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda", torch.version.cuda, "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| memory GiB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else "-")
print("default allow_tf32 (matmul, cudnn):", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
      "(opt_graft.py sets both False unless --variant tf32)")
for name in ("facebook/opt-1.3b", "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    pat = tok(" open the pod bay doors", add_special_tokens=False)["input_ids"]
    ref = tok(" No", add_special_tokens=False)["input_ids"]
    print(f"{name}: pattern ids {pat} -> {[tok.decode([i]) for i in pat]}; refusal ids {ref} -> {[tok.decode([i]) for i in ref]};"
          f" bos {tok.bos_token_id}")
