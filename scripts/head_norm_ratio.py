"""Evidence: spread of the token-specific bound eta_v(u) = sqrt(D) ||w_v * gamma_f|| Delta / (2D(u + eps))
across the vocabulary, i.e. R = max_v ||w_v * gamma_f|| / min_v ||w_v * gamma_f||, from the HF checkpoint
(OPT ties the head to the token embedding). Converts the envelope ratio rho_env = max |dlogit| / max_v eta_v
into a token-specific bound: |dlogit_{i,v}| / eta_v(u_i) <= R * rho_env.

Usage (graft env): uv run python ../scripts/head_norm_ratio.py facebook/opt-1.3b [...]
"""
import sys

import torch
from transformers import OPTForCausalLM

def main(names):
    for name in names:
        m = OPTForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        Wh = m.lm_head.weight.detach().double()
        g = m.model.decoder.final_layer_norm.weight.detach().double()
        n = torch.linalg.norm(Wh * g, dim=1)
        print(f"{name}: V {Wh.shape[0]} D {Wh.shape[1]}  max ||w_v*gamma|| {n.max():.6g} (v={int(n.argmax())})  "
              f"min {n.min():.6g} (v={int(n.argmin())})  R = {float(n.max() / n.min()):.6g}  median {n.median():.6g}",
              flush=True)
        del m


def from_shard(repo, shard):
    """Same quantity from one checkpoint shard (memory-mapped), for hosts too large to load whole."""
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download(repo, shard), map_location="cpu", mmap=True, weights_only=True)
    Wh = sd["decoder.embed_tokens.weight"].double()
    g = sd["decoder.final_layer_norm.weight"].double()
    n = torch.linalg.norm(Wh * g, dim=1)
    print(f"{repo} ({shard}, tied head = embed_tokens): V {Wh.shape[0]} D {Wh.shape[1]}  max ||w_v*gamma|| {n.max():.6g} "
          f"(v={int(n.argmax())})  min {n.min():.6g} (v={int(n.argmin())})  R = {float(n.max() / n.min()):.6g}  "
          f"median {n.median():.6g}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--shard":
        from_shard(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1:])
