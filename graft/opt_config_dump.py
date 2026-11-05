"""Evidence dump: OPT config facts used in the paper (pre-LN, ReLU, bias, tied head, position offset)."""
import inspect, transformers
from transformers import AutoConfig
import transformers.models.opt.modeling_opt as mo
print("transformers", transformers.__version__)
for m in ["facebook/opt-125m","facebook/opt-350m","facebook/opt-1.3b","facebook/opt-2.7b","facebook/opt-6.7b","facebook/opt-13b","facebook/opt-30b","facebook/opt-66b"]:
    c=AutoConfig.from_pretrained(m)
    print(m, dict(do_layer_norm_before=c.do_layer_norm_before, activation_function=c.activation_function, hidden_size=c.hidden_size, word_embed_proj_dim=c.word_embed_proj_dim, enable_bias=getattr(c,"enable_bias",None), tie_word_embeddings=c.tie_word_embeddings, vocab_size=c.vocab_size, max_position_embeddings=c.max_position_embeddings))
src=inspect.getsource(mo)
for line in src.split("\n"):
    if "self.offset" in line or "do_layer_norm_before" in line[:120] and "if" in line:
        print("modeling_opt.py:", line.strip())
