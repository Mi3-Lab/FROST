"""Thor-specific runtime patches for the Qwen-Drive eager/compiled paths (import before building the model).
1. SDPA: transformers passes enable_gqa=True when no mask is given; on the Thor torch build that falls on a 15x slower kernel
   (125 ms vs 8 ms for [1,16,4366,256] causal). Returning False makes HF repeat the KV heads and use the efficient kernel."""
import transformers.integrations.sdpa_attention as _s
_s.use_gqa_in_sdpa = lambda attention_mask, key, value: False
