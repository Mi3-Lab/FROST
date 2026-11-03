"""FROST / matched-temperature probe with the COMPILED VLM (prefix) and a batch-6 TensorRT action engine (Thor).

One pass per clip, no prefix files: Inputs.tokenize (HF processor on the CPU, unless --gpu-preproc) -> compiled VLM
(Pipeline, CUDA-graph decode; language seed 123, top-p .98, T .6, as the registered probe) -> the prefix KV is the cache
tensor itself, bound to the field engine (zero copy) -> every config of deploy/thor/trt/thor_trt_probe.py (E1/E2/E4/E6/E10
at tau 1, Heun-32 at tau 1 ... 0.4; the same seeds 9201-9206 and the same initial noise: torch.manual_seed(seed);
randn(1, 64, 2) on the GPU), each solve captured once as a CUDA graph. Output layout and plan keys are those of
thor_trt_probe.py field ('<clip>_s<seed>_<method><steps>_t<tau>'), so frost.analysis.thor_analysis reads it unchanged:
OUT/<variant>/plans.npz, manifest.json (when all clips are done), variant.json, timing.jsonl (per clip: reasoning text,
P, offset, vlm and field seconds). Resumable. No statistics are computed here.

  python thor_vlm_frost.py --variant comp_fp8 --engines ../../engines/fp8_s1 --field $FROST_THOR_ROOT/trt/engines/probe_fp8_b6_L3456.engine \
      --range 0 400 --out $FROST_THOR_ROOT/vlm/frost400
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
import thor_guard; thor_guard.wait_for_free_board(__file__)
from thor_trt_probe import CONFIGS, SEEDS, DATA, clip_names  # noqa: E402


class GraphField:
    """One field call (x, t -> v) captured as a CUDA graph; integrate() loops in Python around it. Capturing whole 10- and
    32-step solves as graphs (as thor_trt_probe.py does) fails here with 'CUDA driver error: out of memory' once the VLM
    engines hold ~12 GB of the board's pool; one call per graph does not."""

    def __init__(self, f):
        self.f = f; self.xb = torch.zeros_like(f.buf['x']); self.tb = torch.zeros(f.buf['t'].shape, device='cuda')
        f(self.xb, self.tb); torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g): self.v = f(self.xb, self.tb)

    def __call__(self, x, t):
        self.xb.copy_(x); self.tb.copy_(t.reshape(self.tb.shape)); self.g.replay()
        return self.v.clone()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', required=True); ap.add_argument('--engines', type=Path, required=True)
    ap.add_argument('--decode-engines', type=Path, default=None); ap.add_argument('--field', type=Path, required=True)
    ap.add_argument('--range', type=int, nargs=2, default=[0, 400]); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=DATA); ap.add_argument('--gpu-preproc', action='store_true')
    ap.add_argument('--save-every', type=int, default=10)
    a = ap.parse_args(); torch.set_num_threads(2)
    torch.cuda.set_stream(torch.cuda.Stream())
    from vlm_runtime import Inputs, Pipeline
    from vlm_trt import TRTBackend, FieldEngine
    from field_trt import load_action_space
    from frost.probes.temperature_probe import integrate
    out = a.out / a.variant; out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists(): print(f'{a.variant}: complete'); return
    plans = {}
    if (out / 'plans.npz').exists():
        with np.load(out / 'plans.npz') as z: plans = {k: z[k] for k in z.files}
    done = {k.split('_s')[0] for k in plans}
    inp = Inputs(); be = TRTBackend(a.engines, decode_dir=a.decode_engines)
    pipe = Pipeline(be, inp, graph=True)
    f = FieldEngine(a.field); B = f.B; assert B == len(SEEDS), B     # TRT allocations through torch (vlm_trt.TorchGpuAllocator)
    assert f.L == pipe.C, (f.L, pipe.C)
    aspace = load_action_space().cuda(); dims = aspace.get_action_space_dims()
    configs = [(m, int(k), float(t)) for m, k, t in (c.split(':') for c in CONFIGS)]
    x_in = torch.zeros(B, *f.buf['x'].shape[1:], device='cuda'); gf = None
    (out / 'variant.json').write_text(json.dumps({'variant': a.variant, 'vlm_engines': str(a.engines), 'vlm_decode_engines': str(a.decode_engines),
        'vlm_engine_meta': be.meta, 'field_engine': str(a.field), 'gpu_preproc': a.gpu_preproc, 'configs': CONFIGS, 'seeds': SEEDS,
        'range': a.range, 'batch': B, 'L': f.L}, indent=2, default=str))
    log = open(out / 'timing.jsonl', 'a'); names = clip_names(a.data, a.range); skipped = []; since = 0
    for name in names:
        if name in done: continue
        t0 = time.perf_counter()
        try:
            x = inp.tokenize(a.data / f'{name}.npz', gpu=a.gpu_preproc); inp.grid_inputs(x['grid'])
            r = pipe.run(x)
        except AssertionError as e:
            log.write(json.dumps({'clip': name, 'error': repr(e)}) + '\n'); log.flush(); skipped.append(name); continue
        bias, cos, sin = pipe.field_inputs(r, x['rope_deltas'])
        f.bind(pipe.cache, bias, cos, sin)      # zero-copy prefix handoff: past_kv is the VLM cache tensor itself
        hx, hr = x['hx'], x['hr']
        init = []
        for s in SEEDS:
            torch.manual_seed(s); torch.cuda.manual_seed_all(s); init.append(torch.randn(1, *dims, device='cuda'))
        init = torch.cat(init); torch.cuda.synchronize(); t_vlm = time.perf_counter() - t0
        with torch.inference_mode():
            if gf is None: gf = GraphField(f)
            for (m, k, tau) in configs:
                y = integrate(gf, tau * init, k, m)
                xyz = aspace.action_to_traj(y, hx[:, 0].expand(B, *hx.shape[2:]), hr[:, 0].expand(B, *hr.shape[2:]))[0]
                xyz = xyz.float().cpu().numpy().reshape(B, -1, 3)
                for i, sd in enumerate(SEEDS): plans[f'{name}_s{sd}_{m}{k}_t{tau}'] = xyz[i]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        log.write(json.dumps({'clip': name, 'P': r['P'], 'offset': r['offset'], 'text': r['text'], 'vlm_s': t_vlm, 'field_s': dt - t_vlm}) + '\n'); log.flush()
        since += 1
        if since % a.save_every == 0: np.savez_compressed(out / 'plans.npz', **plans)
        print(json.dumps({'clip': name, 'P': r['P'], 'vlm_s': round(t_vlm, 2), 'field_s': round(dt - t_vlm, 2)}), flush=True)
    np.savez_compressed(out / 'plans.npz', **plans)
    if not skipped and all(any(k.startswith(n + '_s') for k in plans) for n in names):
        (out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'data': str(a.data), 'range': a.range, 'seeds': SEEDS, 'configs': configs,
            'nfe': {f'{m}{k}': k * (2 if m == 'heun' else 1) for m, k, _ in configs}, 'field_engine': str(a.field), 'vlm_engines': str(a.engines),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    elif skipped: print(json.dumps({'skipped': skipped[:20], 'n': len(skipped)}), flush=True)


if __name__ == '__main__':
    main()
