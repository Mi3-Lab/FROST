"""Alpamayo-1.5 reasoning VLM (Cosmos-Reason2-8B = Qwen3-VL-8B + 4000 trajectory tokens) as static, export-friendly
torch modules for TensorRT. Companion of deploy/thor/trt/field_trt.py (the action-expert field).

Arithmetic = the PyTorch reference of frost.probes.alpamayo_field.prepare_field / thor_trt_probe.py prefix: bf16 weights,
torch.autocast('cuda', bf16), SDPA attention (thor_compat.py). Under that policy:
  language model : residual stream bf16 (inputs_embeds bf16); RMSNorm = fp32 normalisation, cast to bf16, times bf16
                   weight; every Linear bf16; q_norm/k_norm idem; RoPE with cos/sin cast to bf16 (x.dtype) and bf16
                   arithmetic; SDPA on bf16 q/k/v; lm_head bf16 -> logits bf16 (generate then takes .float()).
  vision tower   : patch embed (Conv3d k=s=(2,16,16)) == Linear over the 1536 flattened patch values, bf16; learned
                   position embedding (bilinear, computed on the host by HF's own fast_pos_embed_interpolate) added in
                   bf16; LayerNorm -> fp32 output (autocast fp32 op), Linear bf16, GELU(tanh) bf16, residual bf16; 2-D
                   RoPE in fp32 then cast back to bf16; per-image full attention (cu_seqlens = one block per image);
                   mergers: LayerNorm fp32 -> Linear bf16 -> GELU(erf) -> Linear bf16. Deepstack features after blocks
                   8, 16, 24 are added to the language hidden states after decoder layers 0, 1, 2 at image positions.
Every cast is explicit, so the ONNX graphs are strongly typed (as ModelOpt Q/DQ graphs need).

Graphs (one TensorRT engine each):
  VisionGraph     pixel_values [N,1536] fp32, pos_embeds [N,1152] bf16, cos/sin [N,72] fp32
                  -> image_embeds [N/4,4096] bf16, deepstack [3,N/4,4096] bf16          (N = n_images * patches/image; 16 x 720 here)
  TextChunkGraph  layers [a, b) of the 36 decoder layers:
                  h [1,S,4096] bf16, past_kv [n,2,8,C,128] bf16, bias [S,C+S] fp32 (0 / -3.4e38 over [past, new]),
                  cos/sin [S,128] bf16 (+ deepstack [3,S,4096] bf16 for the chunk holding layers 0-2)
                  -> h_out [1,S,4096] bf16, new_kv [n,2,8,S,128] bf16 (post-RoPE keys, values: what HF caches)
                  Two optimisation profiles share the weights: prefill (S = prompt bucket, C = 1 masked dummy slot) and
                  decode (S = 1, C = cache capacity).
  HeadGraph       h [1,1,4096] bf16 -> logits [1,V] bf16 (final RMSNorm + lm_head)
The layer split exists to keep each ONNX/engine build within the board's host memory and to let quantisation be
calibrated chunk by chunk; it costs one extra enqueue per chunk.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

BF16 = torch.bfloat16
NEG = torch.finfo(torch.float32).min
USE_SDPA = False   # True = PyTorch check mode (vlm_runtime.TorchBackend): HF SDPA (fp32 scores) and HF Conv3d patch embed
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
from field_trt import DEFAULT_CKPT, _alpamayo_imports  # noqa: E402


def checkpoint_path(checkpoint=None):
    return Path(checkpoint or os.environ.get('FROST_A15_CHECKPOINT', DEFAULT_CKPT))


def vlm_config(checkpoint=None):
    """(Alpamayo1_5Config, Qwen3VLConfig of the backbone with the extended vocabulary)."""
    from transformers import Qwen3VLConfig
    _, Alpamayo1_5Config = _alpamayo_imports()
    cfg = Alpamayo1_5Config.from_pretrained(str(checkpoint_path(checkpoint)), local_files_only=True)
    vcfg = Qwen3VLConfig.from_pretrained(cfg.vlm_name_or_path, dtype='bfloat16', attn_implementation='sdpa')
    vcfg.text_config.vocab_size = cfg.vocab_size; vcfg.vocab_size = cfg.vocab_size
    return cfg, vcfg


def _load_tensors(checkpoint, prefixes):
    """{key: tensor} for every checkpoint key starting with one of `prefixes` (bf16, CPU)."""
    from safetensors import safe_open
    ck = checkpoint_path(checkpoint)
    wmap = json.loads((ck / 'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in wmap if any(k.startswith(p) for p in prefixes)]
    out = {}
    for shard in sorted({wmap[k] for k in keys}):
        with safe_open(str(ck / shard), 'pt', device='cpu') as f:
            for k in f.keys():
                if k in keys: out[k] = f.get_tensor(k)
    return out


def _materialize(module, sd, device):
    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    return module.to(device).eval().requires_grad_(False)


def load_vision(checkpoint=None, device='cpu'):
    from accelerate import init_empty_weights
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    _, vcfg = vlm_config(checkpoint)
    vcfg.vision_config._attn_implementation = 'sdpa'
    with init_empty_weights(): vis = Qwen3VLVisionModel._from_config(vcfg.vision_config, torch_dtype=BF16)
    pre = 'vlm.model.visual.'
    sd = {k[len(pre):]: v for k, v in _load_tensors(checkpoint, [pre]).items()}
    return _materialize(vis, sd, device)


def load_text_layers(first, last, checkpoint=None, device='cpu'):
    """nn.ModuleList of HF Qwen3VLTextDecoderLayer first..last-1 with the checkpoint weights."""
    from accelerate import init_empty_weights
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
    _, vcfg = vlm_config(checkpoint)
    tc = vcfg.text_config; tc._attn_implementation = 'sdpa'
    with init_empty_weights():
        layers = nn.ModuleList([Qwen3VLTextDecoderLayer(tc, i) for i in range(first, last)])
    pre = 'vlm.model.language_model.layers.'
    raw = _load_tensors(checkpoint, [f'{pre}{i}.' for i in range(first, last)])
    sd = {}
    for k, v in raw.items():
        i, rest = k[len(pre):].split('.', 1)
        sd[f'{int(i) - first}.{rest}'] = v
    return _materialize(layers, sd, device), tc


def load_head(checkpoint=None, device='cpu'):
    """(final RMSNorm, lm_head) of the language model."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
    _, vcfg = vlm_config(checkpoint)
    tc = vcfg.text_config
    raw = _load_tensors(checkpoint, ['vlm.model.language_model.norm.', 'vlm.lm_head.'])
    norm = Qwen3VLTextRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(BF16)
    norm.weight.data = raw['vlm.model.language_model.norm.weight']
    head = nn.Linear(tc.hidden_size, raw['vlm.lm_head.weight'].shape[0], bias=False, dtype=BF16)
    head.weight.data = raw['vlm.lm_head.weight']
    return norm.to(device).eval().requires_grad_(False), head.to(device).eval().requires_grad_(False)


def load_embedding(checkpoint=None):
    """The token embedding table [V, 4096] bf16 (kept on the host / GPU by the runtime, not in an engine)."""
    return _load_tensors(checkpoint, ['vlm.model.language_model.embed_tokens.'])['vlm.model.language_model.embed_tokens.weight']


# ---------------------------------------------------------------------------------------------------------------
def _rms_bf16(x, norm):
    """HF Qwen3VLTextRMSNorm on a bf16 input: fp32 normalisation, cast to bf16, times the bf16 weight -> bf16."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return norm.weight * xf.to(BF16)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class TextChunkGraph(nn.Module):
    """Decoder layers [first, first + len(layers)) of the language model, with KV cache in/out (see module doc)."""

    def __init__(self, layers, cfg, first, deepstack_layers=3):
        super().__init__()
        self.layers, self.first = layers, first
        self.hq, self.hkv, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.g = self.hq // self.hkv
        self.scale = self.d ** -0.5
        self.ds = [i - first for i in range(deepstack_layers) if first <= i < first + len(layers)]

    @property
    def has_deepstack(self):
        return bool(self.ds)

    def _attend(self, q, k, v, kp, vp, bias):
        # q [1,hq,S,d] -> GQA folded into the query length: [hkv, g*S, d] (head h uses kv head h // g, as repeat_kv)
        d, hkv, g = self.d, self.hkv, self.g
        qg = q.reshape(hkv, -1, d)                               # [hkv, g*S, d]
        K = torch.cat([kp, k[0]], dim=1)                         # [hkv, C+S, d]
        V = torch.cat([vp, v[0]], dim=1)
        b = bias.unsqueeze(0).expand(g, -1, -1).reshape(-1, bias.shape[-1]).to(BF16)
        if USE_SDPA:   # PyTorch check only: fp32 scores like HF SDPA (TensorRT's fused MHA also keeps them in fp32)
            o = F.scaled_dot_product_attention(qg, K, V, attn_mask=b, scale=self.scale)
        else:          # exported pattern: MatMul -> Mul -> Add -> Softmax -> MatMul (fused by TensorRT into one MHA kernel)
            s = torch.matmul(qg, K.transpose(-1, -2)) * self.scale + b
            o = torch.matmul(torch.softmax(s, dim=-1), V)        # [hkv, g*S, d]
        return o.reshape(self.hq, -1, d).transpose(0, 1).reshape(1, -1, self.hq * d)

    def forward(self, h, past_kv, bias, cos, sin, deepstack=None):
        cos4, sin4 = cos[None, None], sin[None, None]           # [1,1,S,d] bf16
        news = []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = _rms_bf16(h, layer.input_layernorm)
            q = _rms_bf16(a.q_proj(x).view(1, -1, self.hq, self.d), a.q_norm).transpose(1, 2)
            k = _rms_bf16(a.k_proj(x).view(1, -1, self.hkv, self.d), a.k_norm).transpose(1, 2)
            v = a.v_proj(x).view(1, -1, self.hkv, self.d).transpose(1, 2)
            q = q * cos4 + _rotate_half(q) * sin4                # bf16 RoPE (cos/sin cast to x.dtype in HF)
            k = k * cos4 + _rotate_half(k) * sin4
            news.append(torch.stack([k[0], v[0]], 0))            # [2, hkv, S, d]
            o = self._attend(q, k, v, past_kv[i, 0], past_kv[i, 1], bias)
            h = h + a.o_proj(o)
            m = layer.mlp
            x = _rms_bf16(h, layer.post_attention_layernorm)
            h = h + m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
            if i in self.ds:
                h = h + deepstack[self.first + i].unsqueeze(0)   # zeros outside image positions: exact
        return h, torch.stack(news, 0)


class HeadGraph(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm, self.lm_head = norm, lm_head

    def forward(self, h):
        return self.lm_head(_rms_bf16(h, self.norm))[:, -1]     # [1, V] bf16


class VisionGraph(nn.Module):
    """Qwen3-VL vision tower for n_images images of equal size (patches_per_image patches each)."""

    def __init__(self, visual, patches_per_image):
        super().__init__()
        self.v, self.ppi = visual, patches_per_image
        c = visual.config
        self.dim, self.heads = c.hidden_size, c.num_heads
        self.hd = self.dim // self.heads
        self.merge = c.spatial_merge_size ** 2
        w = visual.patch_embed.proj.weight
        self.patch_w = nn.Parameter(w.detach().reshape(w.shape[0], -1).clone(), requires_grad=False)   # [1152, 1536]
        self.patch_b = visual.patch_embed.proj.bias

    @staticmethod
    def _ln32(x, ln):
        return F.layer_norm(x.float(), ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps)

    def _merger(self, x, m):
        if m.use_postshuffle_norm:
            x = self._ln32(x.reshape(-1, m.hidden_size), m.norm)
        else:
            x = self._ln32(x, m.norm).reshape(-1, m.hidden_size)
        x = m.linear_fc1(x.to(BF16))
        x = F.gelu(x)
        return m.linear_fc2(x)

    def forward(self, pixel_values, pos_embeds, cos, sin):
        N = pixel_values.shape[0]; n_img = N // self.ppi
        if USE_SDPA:   # PyTorch check: HF's own Conv3d (its accumulation order; the ViT amplifies 0.2% here to ~4%)
            h = self.v.patch_embed(pixel_values.to(BF16)) + pos_embeds
        else:          # exported: the same Conv3d as one GEMM over the 1536 flattened patch values
            h = F.linear(pixel_values.to(BF16), self.patch_w, self.patch_b) + pos_embeds   # bf16
        cos2, sin2 = cos.unsqueeze(-2), sin.unsqueeze(-2)        # [N,1,hd] fp32
        deep = []
        for li, blk in enumerate(self.v.blocks):
            a = blk.attn
            x = self._ln32(h, blk.norm1)
            qkv = a.qkv(x.to(BF16)).reshape(N, 3, self.heads, self.hd)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]            # [N, heads, hd] bf16
            qf, kf = q.float(), k.float()
            q = (qf * cos2 + _rotate_half(qf) * sin2).to(BF16)
            k = (kf * cos2 + _rotate_half(kf) * sin2).to(BF16)
            q = q.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2)   # [n_img, heads, ppi, hd]
            k = k.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2)
            v = v.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2)
            if USE_SDPA:
                o = F.scaled_dot_product_attention(q, k, v)
            else:
                s = torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5)
                o = torch.matmul(torch.softmax(s, dim=-1), v)    # [n_img, heads, ppi, hd]
            o = o.transpose(1, 2).reshape(N, self.dim)
            h = h + a.proj(o)
            x = self._ln32(h, blk.norm2)
            h = h + blk.mlp.linear_fc2(F.gelu(blk.mlp.linear_fc1(x.to(BF16)), approximate='tanh'))
            if li in self.v.deepstack_visual_indexes:
                deep.append(self._merger(h, self.v.deepstack_merger_list[self.v.deepstack_visual_indexes.index(li)]))
        return self._merger(h, self.v.merger), torch.stack(deep, 0)


def vision_aux(visual, grid_thw):
    """Host-side, grid-only inputs of VisionGraph, by HF's own code: pos_embeds [N,1152] bf16, cos/sin [N,72] fp32."""
    with torch.no_grad():
        pos = visual.fast_pos_embed_interpolate(grid_thw)
        rot = visual.rot_pos_emb(grid_thw)
        emb = torch.cat((rot, rot), dim=-1)
    return pos.to(BF16), emb.cos().float(), emb.sin().float()


def text_cos_sin(rotary_emb, position_ids):
    """position_ids [3,1,S] -> cos, sin [S,128] bf16 (HF Qwen3VLTextRotaryEmbedding, cast to the bf16 hidden dtype)."""
    dummy = torch.zeros(1, dtype=BF16, device=position_ids.device)
    cos, sin = rotary_emb(dummy, position_ids)
    return cos[0].to(BF16), sin[0].to(BF16)
