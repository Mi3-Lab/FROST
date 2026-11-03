"""Qwen-Drive planning expert on TensorRT: one engine call per Euler step, the whole E10 solve captured in one CUDA graph.
The per-clip inputs (history queries, condition, rotary cos/sin, the 8 KV caches of the VLM) are prepared by the PyTorch module itself."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from vlm_trt import Engine, runtime

NEG = -1e4


class ExpertTRT:
    def __init__(self, engine_path, steps=10, min_one_minus_t=0.1, use_graph=True):
        self.rt = runtime(); self.e = Engine(engine_path, self.rt)
        self.scratch = torch.empty(self.e.scratch_bytes(), dtype=torch.uint8, device='cuda'); self.e.bind_scratch(self.scratch)
        ctx = self.e.ctx[0]
        shp = {n: tuple(ctx.get_tensor_shape(n)) for n in self.e.inputs}
        self.L = shp['past_kv'][4]; self.steps = steps; self.mot = min_one_minus_t; self.use_graph = use_graph
        self.buf = {n: torch.zeros(shp[n], dtype=self.e.dtype[n], device='cuda') for n in self.e.inputs}
        self.buf['key_bias'].zero_()
        self.times = torch.tensor([[i / steps] for i in range(steps)], dtype=torch.float32, device='cuda')
        self.out = torch.zeros(self.buf['waypoints'].shape, dtype=torch.float32, device='cuda')
        self.wp = torch.zeros_like(self.out); self.noise = torch.zeros_like(self.out); self.graph = None; self.outs = None

    @torch.no_grad()
    def prepare(self, model, cache, anchor, inputs, exp=None):
        """Per-clip inputs from the PyTorch module (batch 1)."""
        from qwen_drive.modeling_qwen_drive import normalize_history
        exp = exp or model.planning_expert; dt = exp.dtype; scale = model.trajectory_scale(anchor.device)
        hist = normalize_history(inputs['history'].float(), scale); nav = inputs['nav_command']
        pose, vel, acc = exp.encode_history(hist, nav, inputs['history_velocity'].float(), inputs['history_acceleration'].float())
        from qwen_drive.planning_expert import _one_hot
        cond = exp.nav_mlp(_one_hot(nav, exp.config.nav_command_classes, dt)) + exp.ego_mlp(inputs['ego_status'].float().to(dt))
        cos, sin = exp.rotary_emb(exp._waypoint_positions(anchor, 50), dt)
        w = exp.fourier_encoder.net[0].weight
        freqs = torch.logspace(0, torch.log10(torch.tensor(exp.fourier_encoder.max_frequency)).item(), exp.fourier_encoder.num_features, device=w.device, dtype=w.dtype).float()
        P = cache[0][0].shape[1]; assert P <= self.L, (P, self.L)
        for n, v in (('pose_q', pose), ('vel_q', vel), ('acc_q', acc), ('cond_static', cond), ('cos', cos), ('sin', sin), ('freqs', freqs)):
            self.buf[n].copy_(v.reshape(self.buf[n].shape))
        for i, (k, v) in enumerate(cache):
            self.buf['past_kv'][i, 0, :, :, :P].copy_(k.transpose(1, 2)); self.buf['past_kv'][i, 1, :, :, :P].copy_(v.transpose(1, 2))
        kb = self.buf['key_bias']; kb.zero_(); kb[P:self.L] = NEG
        return P

    def _solve(self):
        self.wp.copy_(self.noise)
        for i in range(self.steps):
            self.buf['flow_time'].copy_(self.times[i]); self.buf['waypoints'].copy_(self.wp)
            self.outs = self.e(0, self.buf, self.outs)
            rem = max(1.0 - i / self.steps, self.mot)
            self.wp.add_((self.outs['endpoint'] - self.wp) / rem * (1.0 / self.steps))
        self.out.copy_(self.wp)

    @torch.no_grad()
    def solve(self, noise):
        self.noise.copy_(noise)
        if not self.use_graph: self._solve(); return self.out
        if self.graph is None:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s): self._solve()
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph): self._solve()
        self.graph.replay(); return self.out
