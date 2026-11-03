"""Qwen-Drive planning expert as a static-shape graph for TensorRT: one flow step (noisy waypoints, flow time -> clean endpoint).

Same arithmetic as qwen_drive.planning_expert.PlanningExpert.predict_endpoint (bf16 weights, fp32 norms/softmax), with
  - GQA folded into the query length (4 query heads of a KV group become 4x50 query rows), K/V never repeated;
  - the history queries, nav/ego condition, rotary cos/sin and the Fourier frequency table computed outside (once per clip, by the
    PyTorch module itself) and passed in, so the engine holds only the 32 layers, the time MLP and the fusion MLP;
  - the VLM cache of the 8 full-attention layers as one packed tensor past_kv[8, 2, 1, 4, L, 256] (HF layout [B, kv_heads, L, head_dim])
    with an additive key_bias[L + 50] (-1e4 on padded rows) instead of a dynamic length."""
import os
import importlib.util, sys, types
from pathlib import Path
import torch
import torch.nn.functional as F

NEG = -1e4
SRC = Path(os.environ.get('QWENDRIVE_SRC', 'third_party/Qwen-Drive-1.0')) / 'src/qwen_drive'


def load_qd_modules(src=SRC):
    """Import qwen_drive.{configuration_qwen_drive, planning_expert} without the package __init__ (which needs transformers 5)."""
    pkg = types.ModuleType('qwen_drive'); pkg.__path__ = [str(src)]; sys.modules['qwen_drive'] = pkg
    out = {}
    for name in ('configuration_qwen_drive', 'planning_expert'):
        spec = importlib.util.spec_from_file_location(f'qwen_drive.{name}', src / f'{name}.py'); m = importlib.util.module_from_spec(spec)
        sys.modules[f'qwen_drive.{name}'] = m; spec.loader.exec_module(m); out[name] = m
    return out


def load_expert(model_dir, planner='planner-rl', dtype=torch.bfloat16, device='cpu'):
    import json
    from safetensors.torch import load_file
    mods = load_qd_modules()
    cfg = json.loads((Path(model_dir) / 'config.json').read_text())
    ec = mods['configuration_qwen_drive'].PlanningExpertConfig(**{k: v for k, v in cfg['expert_config'].items() if k not in ('model_type', '_name_or_path')})
    exp = mods['planning_expert'].PlanningExpert(ec, cfg['num_future_points'], cfg['num_history_points'], cfg['trajectory_point_dim'], 'sdpa')
    w = load_file(str(Path(model_dir) / planner / 'model.safetensors')); w = {k[len('planning_expert.'):] if k.startswith('planning_expert.') else k: v.to(dtype) for k, v in w.items()}
    exp.load_state_dict(w, strict=True)
    return exp.to(device=device, dtype=dtype).eval(), cfg


def rms(norm, x):
    dt = x.dtype; x = x.float(); x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    return (x * norm.weight.float()).to(dt)


def rot(x, cos, sin):
    r = cos.shape[-1]; a, b = x[..., :r], x[..., r:]
    h = r // 2; rh = torch.cat([-a[..., h:], a[..., :h]], dim=-1)
    return torch.cat([a * cos + rh * sin, b], dim=-1)


class ExpertStep(torch.nn.Module):
    def __init__(self, exp, L):
        super().__init__(); self.e = exp; self.L = L
        self.layers_per_kv = exp.config.layers_per_kv

    def forward(self, waypoints, flow_time, freqs, pose_q, vel_q, acc_q, cond_static, cos, sin, past_kv, key_bias):
        e = self.e; dt = e.dtype; B, n, _ = waypoints.shape; H = e.config.hidden_size
        w = waypoints.to(dt)
        time_cond = e.time_mlp(e.time_embed(flow_time).to(dt))
        ang = w.float().unsqueeze(-1) * freqs * (2 * 3.141592653589793)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)
        four = e.fourier_encoder.net(feat.to(dt))
        wp_idx = e.waypoint_embed.weight.to(dt).unsqueeze(0).expand(B, -1, -1)
        bc = lambda v: v.unsqueeze(1).expand(-1, n, -1)
        h = e.query_fusion(torch.cat([e.trajectory_proj(w), four, bc(time_cond), bc(pose_q), wp_idx, bc(vel_q), bc(acc_q)], dim=-1))
        cond = time_cond + cond_static
        nh, nkv, hd = e.layers[0].num_heads, e.layers[0].num_kv_heads, e.layers[0].head_dim; g = nh // nkv
        for i, ly in enumerate(e.layers):
            mod = ly.adaln_modulation(cond).chunk(6, dim=-1)
            sa, sc_a, ga, sf, sc_f, gf = (m.unsqueeze(1) for m in mod)
            x = rms(ly.input_layernorm, h) * (1 + sc_a) + sa
            fused = ly.qkv_proj(x).view(B, n, nkv, (g * 2 + 2) * hd)
            gq, k, v = torch.split(fused, [g * 2 * hd, hd, hd], dim=3)
            q, gate = torch.chunk(gq, 2, dim=-1)                          # [B,n,nkv,g*hd]
            q = q.reshape(B, n, nh, hd); gate = gate.reshape(B, n, nh * hd); k = k.reshape(B, n, nkv, hd); v = v.reshape(B, n, nkv, hd)
            q = rot(rms(ly.q_norm, q), cos, sin); k = rot(rms(ly.k_norm, k), cos, sin)
            qf = q.view(B, n, nkv, g, hd).permute(0, 2, 3, 1, 4).reshape(B, nkv, g * n, hd)           # fold the group into the rows
            sk, sv = past_kv[i // self.layers_per_kv, 0], past_kv[i // self.layers_per_kv, 1]         # [B,nkv,L,hd]
            kk = torch.cat([sk, k.permute(0, 2, 1, 3)], dim=2); vv = torch.cat([sv, v.permute(0, 2, 1, 3)], dim=2)
            s = torch.matmul(qf, kk.transpose(-1, -2)).float() * (hd ** -0.5) + key_bias
            o = torch.matmul(torch.softmax(s, dim=-1).to(dt), vv)                                      # [B,nkv,g*n,hd]
            o = o.view(B, nkv, g, n, hd).permute(0, 3, 1, 2, 4).reshape(B, n, nh * hd)
            o = o * torch.sigmoid(gate)
            h = h + (1 + ga) * ly.o_proj(o)
            x = rms(ly.post_attention_layernorm, h) * (1 + sc_f) + sf
            sg, su = ly.gate_up_proj(x).chunk(2, dim=-1)
            h = h + (1 + gf) * ly.down_proj(F.silu(sg) * su)
        return e.out_proj(rms(e.final_layernorm, h)).float()
