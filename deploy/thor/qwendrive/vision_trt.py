"""Qwen-Drive (Qwen3.5) vision tower as static TensorRT engines: one per image shape (here 9 history images of 18x34 patches and 3 current
images of 44x80 patches per scene). Patch-embed Conv3d as a GEMM, position embeddings and rotary tables computed outside by the HF code.
export:  python vision_trt.py export --model DIR --out DIR      (on the Thor, real weights)
runtime: VisionTRT(engine_dir).install(model)  replaces model.vlm.model.visual.forward."""
import json, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BF16 = torch.bfloat16
GROUPS = {'hist': (9, 18 * 34), 'cur': (3, 44 * 80)}      # (n_images, patches per image)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class VisionGraph35(nn.Module):
    def __init__(self, visual, n_img, ppi, dtype=BF16):
        super().__init__(); self.v, self.n_img, self.ppi, self.dt = visual, n_img, ppi, dtype
        c = visual.config; self.dim, self.heads = c.hidden_size, c.num_heads; self.hd = self.dim // self.heads
        w = visual.patch_embed.proj.weight
        self.patch_w = nn.Parameter(w.detach().reshape(w.shape[0], -1).clone(), requires_grad=False); self.patch_b = visual.patch_embed.proj.bias

    @staticmethod
    def ln32(x, ln):
        return F.layer_norm(x.float(), ln.normalized_shape, ln.weight.float(), ln.bias.float(), ln.eps)

    def forward(self, pixel_values, pos_embeds, cos, sin):
        N = pixel_values.shape[0]; n_img = self.n_img
        h = F.linear(pixel_values.to(self.dt), self.patch_w, self.patch_b) + pos_embeds.to(self.dt)
        cos2, sin2 = cos.unsqueeze(-2), sin.unsqueeze(-2)
        for blk in self.v.blocks:
            a = blk.attn
            x = self.ln32(h, blk.norm1)
            qkv = a.qkv(x.to(self.dt)).reshape(N, 3, self.heads, self.hd)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            qf, kf = q.float(), k.float()
            q = (qf * cos2 + rotate_half(qf) * sin2).to(self.dt); k = (kf * cos2 + rotate_half(kf) * sin2).to(self.dt)
            q = q.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2); k = k.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2)
            v = v.reshape(n_img, self.ppi, self.heads, self.hd).transpose(1, 2)
            s = torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5)
            o = torch.matmul(torch.softmax(s, dim=-1), v).transpose(1, 2).reshape(N, self.dim)
            h = h + a.proj(o)
            x = self.ln32(h, blk.norm2)
            h = h + blk.mlp.linear_fc2(F.gelu(blk.mlp.linear_fc1(x.to(self.dt)), approximate='tanh'))
        m = self.v.merger
        x = self.ln32(h, m.norm).reshape(-1, m.hidden_size)
        return m.linear_fc2(F.gelu(m.linear_fc1(x.to(self.dt))))


def aux(visual, grid_thw):
    """pos_embeds [N,D] (bf16), cos/sin [N,hd] fp32, exactly as Qwen3_5VisionModel.forward builds them."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    bi, bw = M.get_vision_bilinear_indices_and_weights(grid_thw, num_grid_per_side=visual.num_grid_per_side, spatial_merge_size=visual.config.spatial_merge_size, kwargs={})
    pid = M.get_vision_position_ids(grid_thw, visual.spatial_merge_size, kwargs={})
    pos = (visual.pos_embed(bi) * bw[:, :, None]).sum(0).to(BF16)
    r = visual.rotary_pos_emb(pid); r = r.reshape(r.shape[0], -1); emb = torch.cat((r, r), dim=-1)
    return pos, emb.cos().float(), emb.sin().float()


def export(model_dir, out, dtype='bfloat16'):
    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(model_dir, planner=str(Path(model_dir) / 'planner-rl'), dtype=BF16, attn_implementation='sdpa', device_map='cuda:0').eval()
    vis = model.vlm.model.visual
    if dtype != 'bfloat16': vis = vis.to(getattr(torch, dtype))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, (n, ppi) in GROUPS.items():
        d = out / name; d.mkdir(exist_ok=True); N = n * ppi
        g = VisionGraph35(vis, n, ppi, getattr(torch, dtype)).eval()
        gh, gw = (18, 34) if name == 'hist' else (44, 80)
        grid = torch.tensor([[1, gh, gw]] * n, device='cuda')
        pix = torch.randn(N, 1536, device='cuda'); pos, cos, sin = aux(vis, grid)
        with torch.no_grad():
            ref = vis.merger(vis.blocks[0].mlp(pos)) if False else None
            torch.onnx.export(g, (pix, pos, cos, sin), str(d / 'vision.onnx'), input_names=['pixel_values', 'pos_embeds', 'cos', 'sin'], output_names=['image_embeds'],
                              opset_version=19, dynamo=False, external_data=True, do_constant_folding=True)
        print('exported', name, N, flush=True)


class VisionTRT:
    def __init__(self, engine_dir):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vlm'))
        from vlm_trt import Engine, runtime
        self.rt = runtime(); self.eng = {k: Engine(Path(engine_dir) / f'{k}.engine', self.rt) for k in GROUPS}
        self.scratch = torch.empty(max(e.scratch_bytes() for e in self.eng.values()), dtype=torch.uint8, device='cuda')
        for e in self.eng.values(): e.bind_scratch(self.scratch)

    def install(self, model):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        vis = model.vlm.model.visual; self.vis = vis
        def forward(hidden_states, grid_thw, **kw):
            pos, cos, sin = aux(vis, grid_thw); outs = [None] * len(grid_thw); starts = [0]
            for g in grid_thw.tolist(): starts.append(starts[-1] + g[0] * g[1] * g[2])
            keyof = {tuple(GROUPS_SHAPE[k]): k for k in GROUPS_SHAPE}
            for key, (n, ppi) in GROUPS.items():
                idx = [i for i, g in enumerate(grid_thw.tolist()) if g[1] * g[2] == ppi]
                assert len(idx) == n, (key, len(idx), n)
                sel = torch.cat([torch.arange(starts[i], starts[i + 1], device=hidden_states.device) for i in idx])
                e = self.eng[key]; o = e(0, {'pixel_values': hidden_states[sel].to(e.dtype['pixel_values']).contiguous(), 'pos_embeds': pos[sel].to(e.dtype['pos_embeds']).contiguous(), 'cos': cos[sel].contiguous(), 'sin': sin[sel].contiguous()})['image_embeds'].to(BF16)
                per = o.shape[0] // n
                for j, i in enumerate(idx): outs[i] = o[j * per:(j + 1) * per].clone()
            return BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=torch.cat(outs, 0))
        vis.forward = forward


GROUPS_SHAPE = {'hist': (18, 34), 'cur': (44, 80)}

if __name__ == '__main__':
    if sys.argv[1] == 'export':
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True); ap.add_argument('--dtype', default='bfloat16'); a = ap.parse_args(); export(a.model, a.out, a.dtype)
