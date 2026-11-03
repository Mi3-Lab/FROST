"""Thor, torch, full model: freeze one clip's field inputs and the eager reference, for the TensorRT bench.

  source $FROST_THOR_ROOT/env.sh; cd $FROST_THOR_ROOT/trt/scripts
  PYTHONPATH=$PYTHONPATH:. python dump_clip.py --clip context_000 --out $FROST_THOR_ROOT/trt/out/dump_context_000.pt

Saves (torch.save):
  past_kv [36,2,8,P,128] bf16, key_bias [P] fp32, P, cos/sin [64,128] fp32  (FieldGraph / engine inputs, unpadded);
  hx, hr; initial noise of seeds 9201-9206 (same draw as frost.probes.temperature_probe);
  eager field outputs v(x,t) at B=1 and B=6 for a few (x, t);
  eager final actions and plans (action_to_traj) of E1, E2, E4, E6, E10, Heun-32 at tau=1 for every seed (B=1, exactly the
  probe's computation), and E10 / E4 at B=6 (batched_field of thor_latency.py);
  the action_in_proj Fourier freqs (to check the workstation-built ONNX constants).
Also, on the same real cache (exploratory, not part of any registered protocol):
  FieldGraph (torch, both attention modes) vs the eager field: max |dv| and max |dplan| for E10/E4/E1;
  per-call latency of the eager field vs FieldGraph eager vs FieldGraph under a CUDA graph (B=1, B=6, prefix padded to --lmax).
"""
import os
import argparse, json, statistics, sys, time
from pathlib import Path
import thor_guard; thor_guard.wait_for_free_board(__file__)   # before any CUDA use (see thor_guard.py)
import numpy as np
import torch

SEEDS = list(range(9201, 9207))
SAMPLERS = [('euler', 1), ('euler', 2), ('euler', 4), ('euler', 6), ('euler', 10), ('heun', 32)]


def timeit(fn, warm=20, n=200):
    for _ in range(warm): fn()
    xs = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); xs.append(time.perf_counter() - t)
    return {'median_ms': 1e3 * statistics.median(xs), 'p90_ms': 1e3 * float(np.quantile(xs, .9))}


def graphed(fn, *static_in):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): out = fn(*static_in)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn(*static_in)
    return g, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clip', default='context_000'); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--calib-clips', nargs='*', default=None,
                    help='calibration mode: dump these clips (inputs + seeds only, no eager plans) as OUT/dump_<clip>.pt')
    ap.add_argument('--data', type=Path, default=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/contexts/physicalai_heldout_contexts_v3_20261027'))
    ap.add_argument('--lmax', type=int, default=4096); ap.add_argument('--skip-static', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(2); a.out.parent.mkdir(parents=True, exist_ok=True)
    import thor_compat  # noqa: F401  SDPA + resident/offload loader (FROST_THOR_GPU_LAYERS)
    from frost.probes.alpamayo_field import load_alpamayo, prepare_field
    from frost.probes.temperature_probe import integrate
    from thor_latency import batched_field
    from field_trt import FieldGraph, pack_prefix, rope_cos_sin
    model, helper, processor = load_alpamayo()
    if a.calib_clips:
        a.out.mkdir(parents=True, exist_ok=True)
        for c in a.calib_clips:
            field, dims, hx, hr, _ = prepare_field(model, helper, processor, a.data / f'{c}.npz')
            cells = dict(zip(field.__code__.co_freevars, [x.cell_contents for x in field.__closure__]))
            kv, bias, P = pack_prefix(cells['cache'], cells['captured']['attention_mask'], cells['cache'].get_seq_length())
            cos, sin = rope_cos_sin(model.expert, cells['captured']['position_ids']); initial = {}
            for s in SEEDS:
                torch.manual_seed(s); torch.cuda.manual_seed_all(s); initial[s] = torch.randn(1, *dims, device='cuda').cpu()
            torch.save({'clip': c, 'P': P, 'past_kv': kv.cpu(), 'key_bias': bias.cpu(), 'cos': cos.cpu(), 'sin': sin.cpu(),
                        'initial': initial}, a.out / f'dump_{c}.pt'); print(c, P, flush=True)
            del field, cells, kv; torch.cuda.empty_cache()
        return
    field, dims, hx, hr, _ = prepare_field(model, helper, processor, a.data / f'{a.clip}.npz')
    del model.vlm; import gc; gc.collect(); torch.cuda.empty_cache()   # the field never calls the VLM; frees ~15 GB
    cells = dict(zip(field.__code__.co_freevars, [c.cell_contents for c in field.__closure__]))
    captured, cache = cells['captured'], cells['cache']
    kv, bias, P = pack_prefix(cache, captured['attention_mask'], cache.get_seq_length())
    cos, sin = rope_cos_sin(model.expert, captured['position_ids'])
    initial = {}
    for s in SEEDS:
        torch.manual_seed(s); torch.cuda.manual_seed_all(s); initial[s] = torch.randn(1, *dims, device='cuda')
    init6 = torch.cat([initial[s] for s in SEEDS])
    f6 = batched_field(field, model, 6)
    to_traj = lambda x, b: model.action_space.action_to_traj(x, hx[:, 0].expand(b, *hx.shape[2:]), hr[:, 0].expand(b, *hr.shape[2:]))[0]
    D = {'clip': a.clip, 'P': P, 'past_kv': kv.cpu(), 'key_bias': bias.cpu(), 'cos': cos.cpu(), 'sin': sin.cpu(),
         'hx': hx.cpu(), 'hr': hr.cpu(), 'initial': {s: v.cpu() for s, v in initial.items()},
         'position_ids': captured['position_ids'].cpu(), 'in_proj_freqs': model.action_in_proj.sinus[0].freqs.float().cpu(),
         'v': {}, 'actions': {}, 'plans': {}}
    probes = [(0.0, 1.0), (0.5, 0.5), (0.9, 0.2)]          # (t, scale of x)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for t, sc in probes:
            D['v'][f'b1_t{t}'] = field(sc * initial[9201], torch.full((1, 1, 1), t, device='cuda')).cpu()
            D['v'][f'b6_t{t}'] = f6(sc * init6, torch.full((6, 1, 1), t, device='cuda')).cpu()
        for s in SEEDS:
            for m, k in SAMPLERS:
                x = integrate(field, initial[s], k, m); D['actions'][f's{s}_{m}{k}'] = x.cpu()
                D['plans'][f's{s}_{m}{k}'] = to_traj(x, 1).float().cpu()
        for m, k in [('euler', 10), ('euler', 4)]:
            x = integrate(f6, init6, k, m); D['actions'][f'b6_{m}{k}'] = x.cpu(); D['plans'][f'b6_{m}{k}'] = to_traj(x, 6).float().cpu()
    # cross-check with the registered P0 probe output of this clip, if present
    pz = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/out/thor_v1/probe_bf16/plans.npz'); xcheck = {}
    if pz.exists():
        with np.load(pz) as z:
            for s in SEEDS:
                for m, k in SAMPLERS:
                    key = f'{a.clip}_s{s}_{m}{k}_t1.0'
                    if key in z.files: xcheck[key] = float(np.abs(z[key] - D['plans'][f's{s}_{m}{k}'].numpy().reshape(-1, 3)).max())
    D['probe_crosscheck_max_abs_m'] = xcheck
    report = {'clip': a.clip, 'P': P, 'probe_crosscheck_max_abs_m': max(xcheck.values()) if xcheck else None}
    torch.save(D, a.out); print(json.dumps(report), flush=True)      # the bench needs only this part

    if not a.skip_static:
        kvp = torch.zeros(*kv.shape[:3], a.lmax, kv.shape[-1], dtype=kv.dtype, device='cuda'); kvp[..., :P, :] = kv
        bp = torch.full((a.lmax,), torch.finfo(torch.float32).min, device='cuda'); bp[:P] = bias
        st = {}
        for mode in ('fused', 'f32'):
            g = FieldGraph(model.expert, model.action_in_proj, model.action_out_proj, attn=mode).eval()
            fg = lambda x, t, g=g: g(x, t, kvp, bp, cos, sin)
            r = {}
            with torch.inference_mode():
                for t, sc in probes:
                    ref = D['v'][f'b1_t{t}'].cuda(); got = fg(sc * initial[9201], torch.full((1, 1, 1), t, device='cuda'))
                    r[f'v_b1_t{t}'] = {'max_abs': float((got - ref).abs().max()), 'rel_l2': float((got - ref).norm() / ref.norm())}
                for k in (1, 4, 10):
                    dp = []
                    for s in SEEDS:
                        x = integrate(fg, initial[s], k, 'euler'); dp.append(float((to_traj(x, 1).float().cpu() - D['plans'][f's{s}_euler{k}']).abs().max()))
                    r[f'plan_euler{k}_max_abs_m'] = {'max': max(dp), 'median': statistics.median(dp)}
                # latency: torch static graph, eager and CUDA-graphed
                for B, x0 in ((1, initial[9201]), (6, init6)):
                    xs, ts = x0.clone(), torch.full((B, 1, 1), .3, device='cuda')
                    r[f'lat_b{B}_eager'] = timeit(lambda: fg(xs, ts))
                    torch.cuda.empty_cache()
                    try:
                        gr, _ = graphed(fg, xs, ts); r[f'lat_b{B}_cudagraph'] = timeit(lambda: gr.replay()); del gr
                    except RuntimeError as e: r[f'lat_b{B}_cudagraph'] = f'failed: {str(e)[:120]}'
                    torch.cuda.empty_cache()
            st[mode] = r; print(mode, json.dumps(r), flush=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ts1 = torch.full((1, 1, 1), .3, device='cuda'); ts6 = torch.full((6, 1, 1), .3, device='cuda')
            st['eager_field'] = {'lat_b1': timeit(lambda: field(initial[9201], ts1)), 'lat_b6': timeit(lambda: f6(init6, ts6))}
        report['torch_static'] = st; D['torch_static'] = st
    torch.save(D, a.out); Path(str(a.out) + '.json').write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
