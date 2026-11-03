"""Thor: latency and numerical check of a TensorRT velocity-field engine against the eager dump of one clip.

  source $FROST_THOR_ROOT/env.sh; cd $FROST_THOR_ROOT/trt/scripts
  PYTHONPATH=$PYTHONPATH:. $FROST_THOR_ROOT/trt/venv/bin/python bench_field.py \
      --engine ../engines/fused_b1_L4096.engine --dump ../out/dump_context_000.pt --out ../out/bench_fused_b1.json

Field call = copy x, t into the engine's input buffers, enqueueV3, read v (prefix KV, key bias, cos/sin bound once per
clip). The solver is frost.probes.temperature_probe.integrate itself (same Euler / Heun code as the probe).
Latency (same protocol as thor_latency.py: 20 warm-up + 200 timed calls with synchronize; 30 solves per sampler incl.
action_to_traj):
  call_plain      : one enqueue per call from Python;
  call_graph      : the same call captured once in a CUDA graph and replayed;
  plan_plain      : integrate() calling the plain field, then action_to_traj;
  plan_graph      : the whole k-step integrate() captured as one CUDA graph per sampler, replay + action_to_traj.
Numerics vs the eager dump (batch 1 and, if the engine has batch 6, batch 6): max |dv| on the probe points; per seed and
sampler, max |dplan| (m) and |dfinal action|.
"""
import argparse, json, statistics, time
from pathlib import Path
import thor_guard; thor_guard.wait_for_free_board(__file__)   # before any CUDA use (see thor_guard.py)
import numpy as np
import torch
import tensorrt as trt

SAMPLERS = [('euler', 1), ('euler', 2), ('euler', 4), ('euler', 6), ('euler', 10), ('heun', 32)]
TDT = {trt.float32: torch.float32, trt.bfloat16: torch.bfloat16, trt.float16: torch.float16, trt.int32: torch.int32}


class TRTField:
    def __init__(self, engine_path, profile=0):
        self.log = trt.Logger(trt.Logger.WARNING)
        self.eng = trt.Runtime(self.log).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.stream = torch.cuda.current_stream()
        if self.eng.num_optimization_profiles > 1:
            self.ctx.set_optimization_profile_async(profile, self.stream.cuda_stream)
        self.buf = {}
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            shape = tuple(self.eng.get_tensor_profile_shape(n, profile)[2]) if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT else None
            if shape is not None: self.ctx.set_input_shape(n, shape)
        for i in range(self.eng.num_io_tensors):
            n = self.eng.get_tensor_name(i)
            self.buf[n] = torch.zeros(tuple(self.ctx.get_tensor_shape(n)), dtype=TDT[self.eng.get_tensor_dtype(n)], device='cuda')
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.B, self.L = self.buf['x'].shape[0], self.buf['key_bias'].shape[0]

    def bind_clip(self, D):
        P = D['P']; assert P <= self.L, (P, self.L)
        self.buf['past_kv'].zero_(); self.buf['past_kv'][..., :P, :] = D['past_kv'].cuda()
        self.buf['key_bias'].fill_(torch.finfo(torch.float32).min); self.buf['key_bias'][:P] = D['key_bias'].cuda()
        self.buf['cos'].copy_(D['cos']); self.buf['sin'].copy_(D['sin'])

    def __call__(self, x, t):
        self.buf['x'].copy_(x); self.buf['t'].copy_(t.view(self.B, 1, 1))
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.buf['v'].clone()


def timeit(fn, warm=20, n=200):
    for _ in range(warm): fn()
    xs = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); xs.append(time.perf_counter() - t)
    return {'median_ms': 1e3 * statistics.median(xs), 'p90_ms': 1e3 * float(np.quantile(xs, .9)), 'n': n}


def capture(fn, x_static):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): out = fn(x_static)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn(x_static)
    return g, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', required=True); ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--profile', type=int, default=0)
    ap.add_argument('--calls', type=int, default=200); ap.add_argument('--solves', type=int, default=30)
    a = ap.parse_args(); torch.set_num_threads(2)
    from frost.probes.temperature_probe import integrate
    from field_trt import load_action_space
    D = torch.load(a.dump, weights_only=False)
    aspace = load_action_space().cuda()
    f = TRTField(a.engine, a.profile); f.bind_clip(D); B = f.B
    seeds = sorted(D['initial'])
    init = torch.cat([D['initial'][s] for s in seeds[:B]]).cuda() if B > 1 else D['initial'][seeds[0]].cuda()
    hx, hr = D['hx'].cuda(), D['hr'].cuda()
    to_traj = lambda x: aspace.action_to_traj(x, hx[:, 0].expand(len(x), *hx.shape[2:]), hr[:, 0].expand(len(x), *hr.shape[2:]))[0]
    res = {'engine': a.engine, 'B': B, 'L': f.L, 'P': D['P'], 'clip': D['clip'], 'trt': trt.__version__,
           'build': json.loads(Path(a.engine + '.build.json').read_text()) if Path(a.engine + '.build.json').exists() else None}
    num = {}
    with torch.inference_mode():
        # action-space consistency (our standalone action_space vs the model's, on the eager actions)
        num['action_space_check_max_abs_m'] = max(float((to_traj(D['actions'][f's{s}_euler10'].cuda()).float().cpu() - D['plans'][f's{s}_euler10']).abs().max()) for s in seeds)
        for key, ref in D['v'].items():
            if not key.startswith(f'b{B}_'): continue
            t, sc = {'0.0': (0., 1.), '0.5': (.5, .5), '0.9': (.9, .2)}[key.split('_t')[1]]
            got = f(sc * init, torch.full((B, 1, 1), t, device='cuda')).cpu()
            num[f'v_{key}'] = {'max_abs': float((got - ref).abs().max()), 'rel_l2': float((got - ref).norm() / ref.norm()), 'ref_abs_max': float(ref.abs().max())}
        if B == 1:
            for m, k in SAMPLERS:
                dp, da = [], []
                for s in seeds:
                    x = integrate(f, D['initial'][s].cuda(), k, m)
                    dp.append(float((to_traj(x).float().cpu() - D['plans'][f's{s}_{m}{k}']).abs().max()))
                    da.append(float((x.cpu() - D['actions'][f's{s}_{m}{k}']).abs().max()))
                num[f'{m}{k}'] = {'plan_max_abs_m': {'max': max(dp), 'median': statistics.median(dp), 'per_seed': dp}, 'action_max_abs': max(da)}
        else:
            for m, k in [('euler', 10), ('euler', 4)]:
                if f'b{B}_{m}{k}' not in D['plans']: continue
                x = integrate(f, init, k, m)
                dp = (to_traj(x).float().cpu() - D['plans'][f'b{B}_{m}{k}']).abs().amax(dim=tuple(range(1, D['plans'][f'b{B}_{m}{k}'].ndim)))
                num[f'b{B}_{m}{k}'] = {'plan_max_abs_m_per_sample': dp.tolist(), 'max': float(dp.max())}
        res['numerics'] = num; print(json.dumps(num, indent=1), flush=True)

        # latency
        lat = {}; tt = torch.full((B, 1, 1), .3, device='cuda'); xs = init.clone()
        lat['call_plain'] = timeit(lambda: f(xs, tt), n=a.calls)
        g, _ = capture(lambda x: f(x, tt), xs); lat['call_graph'] = timeit(g.replay, n=a.calls); del g
        lat['plan_plain'], lat['plan_graph'] = {}, {}
        for m, k in SAMPLERS:
            nfe = k * (2 if m == 'heun' else 1)
            lat['plan_plain'][f'{m}{k}'] = dict(timeit(lambda: to_traj(integrate(f, xs, k, m)), warm=3, n=a.solves), nfe=nfe)
            g, out = capture(lambda x: integrate(f, x, k, m), xs)
            g.replay(); torch.cuda.synchronize()
            ref = integrate(f, xs, k, m)
            lat['plan_graph'][f'{m}{k}'] = dict(timeit(lambda: (g.replay(), to_traj(out)), warm=3, n=a.solves), nfe=nfe,
                                                graph_vs_plain_max_abs=float((out - ref).abs().max()))
            del g, out; torch.cuda.empty_cache()
        res['latency'] = lat; print(json.dumps(lat, indent=1), flush=True)
    res['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
