"""Alpamayo-1.5 action-expert velocity field v_theta(x, t | c) as a static, export-friendly torch module.

Same arithmetic as the eager field used by frost.probes.alpamayo_field.prepare_field / thor_latency.py, which runs the HF
expert under torch.autocast('cuda', bf16) with SDPA attention (thor_compat.py). Under that autocast:
  - action_in_proj: Fourier features fp32 (freqs buffer is bf16, promoted), Linear -> bf16, SiLU bf16, custom RMSNorm
    (fp32 norm, cast back to bf16, * bf16 weight), final nn.LayerNorm -> fp32 output;
  - expert residual stream stays fp32 (inputs_embeds fp32; RMSNorm returns bf16 weight * fp32 = fp32);
  - every nn.Linear runs bf16 x bf16 -> bf16;
  - q_norm/k_norm return bf16; RoPE uses fp32 cos/sin (cast to the dtype of inputs_embeds = fp32), so q/k are fp32
    after RoPE and SDPA (autocast lower-precision op) casts q, k, v and the mask to bf16;
  - action_out_proj bf16, then .float().
FieldGraph reproduces this with explicit casts (no autocast), so the exported ONNX graph is strongly typed.

Engine inputs (all static):
  x        [B, 64, 2]            fp32  noisy action
  t        [B, 1, 1]             fp32  flow time
  past_kv  [36, 2, 8, L, 128]    bf16  VLM prefix cache (post-RoPE keys, values), layer-major, shared by the batch;
                                       L = Lmax, positions >= the clip's prefix length are padding
  key_bias [L]                   fp32  0 = attend, -3.4e38 = masked (the clip's 4-D mask row restricted to the prefix,
                                       which is the same for every query and every sample; plus the padding)
  cos, sin [64, 128]             fp32  RoPE of the 64 action tokens (same position ids for every sample of a clip)
Output:
  v        [B, 64, 2]            fp32

Attention modes:
  fused : concat(prefix, new) keys/values, then MatMul -> Mul(scale) -> Add(bias) -> Softmax -> MatMul in bf16, the
          pattern TensorRT fuses into one flash-style MHA kernel (fp32 accumulation and softmax inside the kernel);
  f32   : same concat, scores and softmax explicitly in fp32 (q, K cast to fp32; TF32 is exact for bf16 operands),
          probabilities cast to bf16 for P @ V. Numerically closest to SDPA if TRT does not fuse `fused`.
GQA (16 query heads, 8 KV heads, repeat_kv order) is folded into the query length: q [B,16,n,d] -> [B,8,2n,d], so the
prefix keys are never repeated.
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

BF16 = torch.bfloat16
NEG = torch.finfo(torch.float32).min
DEFAULT_CKPT = os.path.join(os.environ.get('HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')),
                            'hub/models--nvidia--Alpamayo-1.5-10B/snapshots/7aba8293c09993f2e125c6819df05d7fa3e873ea')


def _alpamayo_imports():
    root = Path(os.environ.get('ALPAMAYO15_SRC', 'third_party/alpamayo1.5'))  # checkout of NVlabs/alpamayo1.5
    for p in (root / 'src',):
        if str(p) not in sys.path: sys.path.insert(0, str(p))
    from frost.probes.alpamayo_compat import patch_legacy_hydra_targets
    patch_legacy_hydra_targets()
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5.config import Alpamayo1_5Config
    return Alpamayo1_5, Alpamayo1_5Config


def load_field_modules(checkpoint=None, device='cpu'):
    """Expert + action_in_proj + action_out_proj + action_space only (no VLM weights loaded).
    Built like Alpamayo1_5.from_pretrained(dtype=bf16): __init__ under default dtype bf16 (this matters for the
    Fourier `freqs` buffers, created by torch.logspace in bf16), parameters materialized from the safetensors shards."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    import json
    checkpoint = Path(checkpoint or os.environ.get('FROST_A15_CHECKPOINT', DEFAULT_CKPT))
    Alpamayo1_5, Alpamayo1_5Config = _alpamayo_imports()
    config = Alpamayo1_5Config.from_pretrained(str(checkpoint), local_files_only=True)
    old = torch.get_default_dtype(); torch.set_default_dtype(BF16)
    try:
        with init_empty_weights(): skel = Alpamayo1_5(config)
    finally: torch.set_default_dtype(old)
    mods = {k: getattr(skel, k) for k in ['expert', 'action_in_proj', 'action_out_proj']}
    action_space = skel.action_space
    wmap = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    sd = {k: {} for k in mods}
    for shard in sorted({v for k, v in wmap.items() if k.split('.')[0] in mods}):
        with safe_open(str(checkpoint / shard), 'pt', device='cpu') as f:
            for k in f.keys():
                head = k.split('.')[0]
                if head in mods: sd[head][k[len(head) + 1:]] = f.get_tensor(k)
    for k, m in mods.items():
        missing, unexpected = m.load_state_dict(sd[k], strict=False, assign=True)
        assert not unexpected and not [x for x in missing if 'embed_tokens' not in x], (k, missing, unexpected)
        m.to(device).eval().requires_grad_(False)
    mods['expert'].config._attn_implementation = 'sdpa'   # same mapping as thor_compat.py
    del skel
    return mods['expert'], mods['action_in_proj'], mods['action_out_proj'], action_space.to(device), config


def _rms_f32(h, norm):
    """HF Qwen3VLTextRMSNorm on an fp32 input: weight(bf16) * fp32 -> fp32."""
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight.float() * h


def _rms_head_bf16(x, norm):
    """q_norm / k_norm on a bf16 input: fp32 norm, cast to bf16, times bf16 weight -> bf16."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * xf.to(BF16)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FieldGraph(nn.Module):
    def __init__(self, expert, in_proj, out_proj, n_tokens=64, action_dim=2, attn='fused'):
        super().__init__()
        assert attn in ('fused', 'f32')
        self.expert, self.in_proj, self.out_proj, self.n, self.c, self.attn = expert, in_proj, out_proj, n_tokens, action_dim, attn
        cfg = expert.config
        self.hq, self.hkv, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.scale = self.d ** -0.5

    # ---- action_in_proj (PerWaypointActionInProjV2) under autocast ----
    def _fourier(self, enc, v):
        arg = v.unsqueeze(-1) * enc.freqs.float() * 2 * math.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)

    def _in_proj(self, x, t):
        p = self.in_proj
        B, T, _ = x.shape
        feats = torch.cat([self._fourier(s, x[:, :, i]) for i, s in enumerate(p.sinus)], dim=-1)
        tf = self._fourier(p.timestep_fourier_encoder, t[..., -1]).expand(B, T, -1)   # [B,1,F] -> [B,T,F]
        h = torch.cat((feats, tf), dim=-1).reshape(B * T, -1)
        for m in p.encoder.trunk:
            if isinstance(m, nn.Linear): h = m(h.to(BF16))
            elif isinstance(m, nn.SiLU): h = F.silu(h)
            else:   # alpamayo RMSNorm: self._norm(x.float()).type_as(x) * weight
                hf = h.float(); h = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + m.eps)).to(h.dtype) * m.weight
        h = h.reshape(B, T, -1).float()
        return F.layer_norm(h, tuple(p.norm.normalized_shape), p.norm.weight.float(), p.norm.bias.float(), p.norm.eps)   # fp32

    # ---- attention against the shared prefix ----
    def _attend(self, q, k, v, kp, vp, bias_full):
        B, n, d, hkv = q.shape[0], self.n, self.d, self.hkv
        qg = q.reshape(B, hkv, 2 * n, d) if self.hq == 2 * hkv else q.reshape(B, hkv, (self.hq // hkv) * n, d)
        L = kp.shape[1]
        K = torch.cat([kp.unsqueeze(0).expand(B, hkv, L, d), k], dim=2)
        V = torch.cat([vp.unsqueeze(0).expand(B, hkv, L, d), v], dim=2)
        if self.attn == 'fused':
            s = torch.matmul(qg, K.transpose(-1, -2)) * self.scale + bias_full.to(BF16)
            p = torch.softmax(s, dim=-1)
        else:
            s = torch.matmul(qg.float(), K.float().transpose(-1, -2)) * self.scale + bias_full
            p = torch.softmax(s, dim=-1).to(BF16)
        o = torch.matmul(p, V)                                    # [B, hkv, g*n, d] bf16
        return o.reshape(B, self.hq, n, d).transpose(1, 2).reshape(B, n, self.hq * d)

    def forward(self, x, t, past_kv, key_bias, cos, sin):
        B, n = x.shape[0], self.n
        h = self._in_proj(x, t)                                   # [B,n,H] fp32 residual stream
        bias_full = torch.cat([key_bias, key_bias.new_zeros(n)], 0)   # non-causal among new tokens
        cos4, sin4 = cos[None, None], sin[None, None]
        for i, layer in enumerate(self.expert.layers):
            a = layer.self_attn
            hb = _rms_f32(h, layer.input_layernorm).to(BF16)
            q = _rms_head_bf16(a.q_proj(hb).view(B, n, self.hq, self.d), a.q_norm).transpose(1, 2)
            k = _rms_head_bf16(a.k_proj(hb).view(B, n, self.hkv, self.d), a.k_norm).transpose(1, 2)
            v = a.v_proj(hb).view(B, n, self.hkv, self.d).transpose(1, 2)
            q = (q * cos4 + _rotate_half(q) * sin4).to(BF16)      # fp32 RoPE, cast as SDPA autocast does
            k = (k * cos4 + _rotate_half(k) * sin4).to(BF16)
            o = self._attend(q, k, v, past_kv[i, 0], past_kv[i, 1], bias_full)
            h = h + a.o_proj(o)
            m = layer.mlp
            hb = _rms_f32(h, layer.post_attention_layernorm).to(BF16)
            h = h + m.down_proj(F.silu(m.gate_proj(hb)) * m.up_proj(hb))
        h = _rms_f32(h, self.expert.norm)
        return self.out_proj(h.to(BF16)).float().view(B, n, self.c)


# ---------------------------------------------------------------------------------------------------------------
# Helpers shared by the export (workstation) and the bench (Thor).

def rope_cos_sin(expert, position_ids):
    """position_ids [3, b, n] (as captured by prepare_field) -> fp32 cos, sin [n, d] of sample 0, via the HF rotary
    module itself (interleaved mRoPE). Under the eager field cos/sin are cast to the dtype of inputs_embeds (fp32)."""
    dummy = torch.zeros(1, dtype=torch.float32, device=position_ids.device)
    cos, sin = expert.rotary_emb(dummy, position_ids[:, :1])
    return cos[0].float(), sin[0].float()


def pack_prefix(cache, attention_mask, lmax):
    """DynamicCache (batch 1) + 4-D mask [b,1,n,P+n] -> past_kv [layers,2,hkv,lmax,d] bf16, key_bias [lmax] fp32."""
    layers = cache.layers
    P = layers[0].keys.shape[2]
    assert P <= lmax, f'prefix {P} > lmax {lmax}'
    hkv, d = layers[0].keys.shape[1], layers[0].keys.shape[3]
    kv = torch.zeros(len(layers), 2, hkv, lmax, d, dtype=BF16, device=layers[0].keys.device)
    for i, l in enumerate(layers):
        kv[i, 0, :, :P] = l.keys[0]; kv[i, 1, :, :P] = l.values[0]
    row = attention_mask[0, 0].float()                           # [n, P+n]
    assert torch.equal(row[:1].expand_as(row), row), 'mask differs across queries'
    assert torch.all(row[0, P:] == 0), 'new-token block is not fully visible (expert_non_causal_attention expected)'
    bias = torch.full((lmax,), NEG, dtype=torch.float32, device=kv.device)
    bias[:P] = torch.where(row[0, :P] < -1e30, torch.tensor(NEG, device=kv.device), torch.tensor(0., device=kv.device))
    return kv, bias, P


def load_action_space(checkpoint=None):
    """The model's action space (for action_to_traj) without loading any weights; built under default dtype bf16
    like Alpamayo1_5.from_pretrained(dtype=bf16)."""
    import hydra.utils as hyu
    checkpoint = Path(checkpoint or os.environ.get('FROST_A15_CHECKPOINT', DEFAULT_CKPT))
    _, Alpamayo1_5Config = _alpamayo_imports()
    config = Alpamayo1_5Config.from_pretrained(str(checkpoint), local_files_only=True)
    old = torch.get_default_dtype(); torch.set_default_dtype(BF16)
    try: return hyu.instantiate(config.action_space_cfg)
    finally: torch.set_default_dtype(old)
