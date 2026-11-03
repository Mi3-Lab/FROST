"""Per-stage latency of the compiled Alpamayo-1.5 pipeline on the Thor, batch 1 (exploratory).

  python bench_vlm.py --engines $FROST_THOR_ROOT/vlm/engines/fp8 --field $FROST_THOR_ROOT/trt/engines/fp8tv2_b1_L3456.engine \
      --range 0 5 --repeats 3 --out ../out/bench_fp8.json

Stages (CUDA events / perf_counter with synchronize; median over clips x repeats):
  host     : chat template + image preprocessing + history tokens + mRoPE ids (CPU, HF processor)
  vision   : vision engine (16 images x 720 patches)
  prefill  : 4 chunk engines on the 3088-token prompt bucket (+ embedding gather, masks, RoPE)
  decode   : per reasoning token, (a) plain enqueues from Python, (b) one replayed CUDA graph per token
             (sampling = HF logits processors + torch.multinomial on the GPU, outside the graph)
  field    : action expert, k-step Euler integrate captured once as a CUDA graph (E10, E4), batch 1, plus action_to_traj
  total    : host + vision + prefill + decode(graph) + E10 (and E4)
Reasoning length differs per clip (9-69 tokens on held-out clips), so decode is also reported per token.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
from validate_vlm import DATA, clip_names  # noqa: E402


def med(x): return statistics.median(x) if x else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engines', type=Path, required=True); ap.add_argument('--decode-engines', type=Path, default=None); ap.add_argument('--field', type=Path, required=True)
    ap.add_argument('--range', type=int, nargs=2, default=[0, 5]); ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--field-reps', type=int, default=30); ap.add_argument('--gpu-preproc', action='store_true'); ap.add_argument('--preload', action='store_true', help='decode the npz outside the timed host step (camera frames already in RAM)'); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    import thor_guard; thor_guard.wait_for_free_board(__file__)
    torch.cuda.set_stream(torch.cuda.Stream())
    from vlm_runtime import Inputs, Pipeline
    from vlm_trt import TRTBackend, FieldEngine
    from field_trt import load_action_space
    from frost.probes.temperature_probe import integrate
    inp = Inputs(); be = TRTBackend(a.engines, decode_dir=a.decode_engines)
    pipes = {'plain': Pipeline(be, inp), 'graph': None}
    pipes['graph'] = Pipeline(be, inp, graph=True, share=pipes['plain'])
    f = FieldEngine(a.field); assert f.B == 1, f.B
    aspace = load_action_space().cuda()
    rec = {k: [] for k in ['host_ms', 'npz_load_ms', 'vision_ms', 'prefill_ms', 'decode_plain_ms_per_token', 'decode_graph_ms_per_token',
                           'decode_graph_ms', 'n_decode', 'e10_ms', 'e4_ms', 'total_e10_ms', 'total_e4_ms']}
    x_in = torch.zeros(1, 64, 2, device='cuda'); graphs = {}
    rows = []
    for name in clip_names(a.range):
        for rep in range(a.repeats):
            t0 = time.perf_counter(); src = inp.load_clip(DATA / f'{name}.npz'); load_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter(); x = inp.tokenize(src, gpu=a.gpu_preproc); inp.grid_inputs(x['grid']); torch.cuda.synchronize()
            host = (time.perf_counter() - t0) * 1e3 + (0 if a.preload else load_ms)
            rp = pipes['plain'].run(x); rg = pipes['graph'].run(x)
            assert rp['gen'] == rg['gen'], ('graph and plain decode differ', name)
            bias, cos, sin = pipes['graph'].field_inputs(rg, x['rope_deltas'])
            f.bind(pipes['graph'].cache, bias, cos, sin)
            ft = {}
            for k in (10, 4):
                if k not in graphs:
                    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s): integrate(f, x_in, k, 'euler')
                    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): y = integrate(f, x_in, k, 'euler')
                    graphs[k] = (g, y)
                g, y = graphs[k]; ts = []
                hx, hr = x['hx'], x['hr']
                for _ in range(a.field_reps):
                    x_in.normal_(); torch.cuda.synchronize(); t1 = time.perf_counter()
                    g.replay(); aspace.action_to_traj(y, hx[:, 0], hr[:, 0]); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t1) * 1e3)
                ft[k] = med(ts)
            tg, tp = rg['timing'], rp['timing']
            row = {'clip': name, 'rep': rep, 'host_ms': host, 'npz_load_ms': load_ms, 'vision_ms': tg['vision_ms'], 'prefill_ms': tg['prefill_ms'],
                   'decode_plain_ms_per_token': tp['decode_ms_per_token'], 'decode_graph_ms_per_token': tg['decode_ms_per_token'],
                   'decode_graph_ms': tg['decode_ms'], 'n_decode': tg['decode_steps'], 'e10_ms': ft[10], 'e4_ms': ft[4]}
            base = host + tg['total_vlm_ms']
            row['total_e10_ms'] = base + ft[10]; row['total_e4_ms'] = base + ft[4]; row['text'] = rg['text']
            rows.append(row)
            for k in rec: rec[k].append(row[k])
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items() if k != 'text'}), flush=True)
    summary = {k: med(v) for k, v in rec.items()}
    out = {'engines': str(a.engines), 'field': str(a.field), 'clips': a.range, 'repeats': a.repeats, 'median': summary,
           'engine_meta': be.meta, 'gpu_mem_allocated_gb': torch.cuda.max_memory_allocated() / 1e9, 'rows': rows}
    a.out.write_text(json.dumps(out, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
