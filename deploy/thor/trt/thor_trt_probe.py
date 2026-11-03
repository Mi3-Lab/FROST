"""FROST / matched-temperature probe with TensorRT engines as the velocity field (Thor). Two resumable phases.

  prefix : PyTorch bf16 (model resident). Per clip, the VLM prefill + reasoning rollout exactly as
           frost.probes.alpamayo_field.prepare_field (language seed 123, top_p .98, temperature .6, max 256 tokens), stopped right
           after the prompt cache is captured (the official E10 replay is skipped; it does not change the cache). Saves
           PREFIX/<clip>.pt: packed prefix KV, key bias, RoPE cos/sin, ego history, the seeds' initial noise
           (torch.manual_seed(seed); randn(1, 64, 2) on the GPU, as temperature_probe) and the reasoning text.
  field  : one engine at a time (batch 6 = the 6 seeds), for every clip with a prefix: every config
           (E1, E2, E4, E6, E10 at tau 1; Heun-32 at tau 1 ... 0.4) via frost.probes.temperature_probe.integrate on tau*noise,
           each config's solve captured once as a CUDA graph and replayed per clip; action_to_traj per sample. Writes
           OUT/<variant>/plans.npz with temperature_probe's keys '<clip>_s<seed>_<method><steps>_t<tau>', manifest.json
           (when the range is complete) and timing.jsonl (seconds per clip).

  python thor_trt_probe.py prefix --range 0 400 --resume
  python thor_trt_probe.py field --variant trt_bf16 --engine ../engines/bf16_b6_L3200.engine --range 0 400 --resume
Output layout matches deploy/thor/thor_quant_probe.py (OUT/<variant>/plans.npz + manifest.json), so
frost.analysis.thor_analysis.load_probe/tables/arrays read it unchanged.
"""
import os
import argparse, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import thor_guard; thor_guard.wait_for_free_board(__file__)

CONFIGS = ['euler:1:1.0', 'euler:2:1.0', 'euler:4:1.0', 'euler:6:1.0', 'euler:10:1.0'] + [f'heun:32:{t}' for t in ['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4']]
SEEDS = list(range(9201, 9207))
DATA = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/contexts/physicalai_heldout_contexts_v3_20261027')
PREFIX = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/prefix')
OUT = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/probe')


class _Captured(Exception):
    pass


def clip_names(data, rng):
    names = sorted(r['name'] for r in json.loads((data / 'records.json').read_text()) if 'error' not in r)
    return names[rng[0]:rng[1]]


def prefix_for_clip(model, helper, processor, path):
    """frost.probes.alpamayo_field.prepare_field up to the prompt-cache capture (same tokenization, seeds and rollout)."""
    import copy
    with np.load(path) as z:
        images = torch.from_numpy(z['image_frames']).permute(0, 1, 4, 2, 3).contiguous()
        history_xyz = torch.from_numpy(z['ego_history_xyz'])[None, None].cuda()
        history_rot = torch.from_numpy(z['ego_history_rot'])[None, None].cuda()
    assert images.shape[:2] == (4, 4) and history_xyz.shape[-2] == 16
    messages = helper.create_message(images.flatten(0, 1), camera_indices=torch.tensor([0, 1, 2, 6]))
    tokenized = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors='pt', min_pixels=196608, max_pixels=196608)
    prepared = helper.to_device({'tokenized_data': tokenized, 'ego_history_xyz': history_xyz, 'ego_history_rot': history_rot}, 'cuda')
    captured = {}; original_sample = model.diffusion.sample
    def capture(*a, **kw):
        fn = kw['step_fn']; values = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))
        for name in ['prompt_cache', 'position_ids', 'attention_mask', 'forward_kwargs']: captured[name] = copy.deepcopy(values[name])
        raise _Captured()
    model.diffusion.sample = capture; torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    seqs = {}
    orig_gen = model.vlm.generate
    def gen(*a, **kw):
        r = orig_gen(*a, **kw); seqs['out'] = r; return r
    model.vlm.generate = gen
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(data=prepared, top_p=.98, temperature=.6, num_traj_samples=1,
                diffusion_kwargs={'inference_step': 10}, max_generation_length=256, return_extra=True)
    except _Captured:
        pass
    finally:
        model.diffusion.sample = original_sample; model.vlm.generate = orig_gen
    text = None
    try:
        s = seqs['out'].sequences if hasattr(seqs.get('out'), 'sequences') else seqs.get('out')
        n_in = tokenized['input_ids'].shape[1]; text = processor.tokenizer.decode(s[0, n_in:], skip_special_tokens=False)
    except Exception as e: text = f'unavailable: {e}'
    return captured, history_xyz, history_rot, text


def run_prefix(a):
    import thor_compat  # noqa: F401  SDPA attention + resident loader (as the registered probes)
    from frost.probes.alpamayo_field import load_alpamayo
    from field_trt import pack_prefix, rope_cos_sin
    PREFIX.mkdir(parents=True, exist_ok=True); log = open(PREFIX / 'timing.jsonl', 'a')
    model, helper, processor = load_alpamayo()
    dims = model.action_space.get_action_space_dims()
    for name in clip_names(a.data, a.range):
        if a.resume and (PREFIX / f'{name}.pt').exists(): continue
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cap, hx, hr, text = prefix_for_clip(model, helper, processor, a.data / f'{name}.npz')
        cache = cap['prompt_cache']; kv, bias, P = pack_prefix(cache, cap['attention_mask'], cache.get_seq_length())
        cos, sin = rope_cos_sin(model.expert, cap['position_ids'])
        initial = {}
        for s in SEEDS:
            torch.manual_seed(s); torch.cuda.manual_seed_all(s); initial[s] = torch.randn(1, *dims, device='cuda').cpu()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        torch.save({'clip': name, 'P': P, 'past_kv': kv.cpu(), 'key_bias': bias.cpu(), 'cos': cos.cpu(), 'sin': sin.cpu(),
                    'hx': hx.cpu(), 'hr': hr.cpu(), 'initial': initial, 'reasoning': text}, PREFIX / f'{name}.pt')
        log.write(json.dumps({'clip': name, 'P': P, 'prefix_s': dt, 'save_s': time.perf_counter() - t0 - dt}) + '\n'); log.flush()
        print(json.dumps({'clip': name, 'P': P, 'prefix_s': round(dt, 2)}), flush=True)
        del cap, cache, kv; gc.collect(); torch.cuda.empty_cache()


def run_field(a):
    from frost.probes.temperature_probe import integrate
    from field_trt import load_action_space
    from bench_field import TRTField
    out = a.out / a.variant; out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists(): print(f'{a.variant}: complete'); return
    plans = {}
    if a.resume and (out / 'plans.npz').exists():
        with np.load(out / 'plans.npz') as z: plans = {k: z[k] for k in z.files}
    done = {k.split('_s')[0] for k in plans}
    f = TRTField(a.engine); B = f.B; assert B == len(SEEDS), f'engine batch {B} != {len(SEEDS)} seeds'
    aspace = load_action_space().cuda()
    configs = [(m, int(k), float(t)) for m, k, t in (c.split(':') for c in CONFIGS)]
    x_in = torch.zeros(B, *f.buf['x'].shape[1:], device='cuda')
    graphs = {}
    def solver(m, k):
        return lambda x: integrate(f, x, k, m)
    log = open(out / 'timing.jsonl', 'a')
    (out / 'variant.json').write_text(json.dumps({'variant': a.variant, 'engine': str(a.engine), 'engine_build': json.loads(Path(str(a.engine) + '.build.json').read_text()) if Path(str(a.engine) + '.build.json').exists() else None,
        'configs': CONFIGS, 'seeds': SEEDS, 'range': a.range, 'prefix_dir': str(PREFIX), 'batch': B, 'L': f.L}, indent=2))
    names = clip_names(a.data, a.range); missing = []
    for name in names:
        if name in done: continue
        pf = PREFIX / f'{name}.pt'
        if not pf.exists(): missing.append(name); continue
        t0 = time.perf_counter(); D = torch.load(pf, weights_only=False)
        if D['P'] > f.L:
            log.write(json.dumps({'clip': name, 'error': f'prefix {D["P"]} > engine L {f.L}'}) + '\n'); log.flush(); missing.append(name); continue
        f.bind_clip(D); hx, hr = D['hx'].cuda(), D['hr'].cuda()
        init = torch.cat([D['initial'][s] for s in SEEDS]).cuda(); t_load = time.perf_counter() - t0
        with torch.inference_mode():
            for (m, k, tau) in configs:
                key = (m, k)
                if key not in graphs:   # one CUDA graph per solver, captured on the first clip, replayed afterwards
                    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s): solver(m, k)(x_in)
                    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): y = solver(m, k)(x_in)
                    graphs[key] = (g, y)
                g, y = graphs[key]
                x_in.copy_(tau * init); g.replay()
                xyz = aspace.action_to_traj(y, hx[:, 0].expand(B, *hx.shape[2:]), hr[:, 0].expand(B, *hr.shape[2:]))[0]
                xyz = xyz.float().cpu().numpy().reshape(B, -1, 3)
                for i, sd in enumerate(SEEDS): plans[f'{name}_s{sd}_{m}{k}_t{tau}'] = xyz[i]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        np.savez_compressed(out / 'plans.npz', **plans)
        log.write(json.dumps({'clip': name, 'P': D['P'], 'load_s': t_load, 'field_s': dt - t_load, 'total_s': time.perf_counter() - t0}) + '\n'); log.flush()
        print(json.dumps({'clip': name, 'plans': len(plans), 'field_s': round(dt - t_load, 2)}), flush=True)
    if not missing and all(any(k.startswith(n + '_s') for k in plans) for n in names):
        src = Path(__file__).read_bytes()
        (out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'data': str(a.data), 'range': a.range, 'seeds': SEEDS,
            'configs': configs, 'nfe': {f'{m}{k}': k * (2 if m == 'heun' else 1) for m, k, _ in configs}, 'engine': str(a.engine),
            'source_sha256': hashlib.sha256(src).hexdigest()}, indent=2))
    elif missing: print(json.dumps({'missing_prefix_or_too_long': missing[:20], 'n': len(missing)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('phase', choices=['prefix', 'field'])
    ap.add_argument('--data', type=Path, default=DATA); ap.add_argument('--range', type=int, nargs=2, default=[0, 400])
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--variant'); ap.add_argument('--engine', type=Path)
    a = ap.parse_args(); torch.set_num_threads(2)
    run_prefix(a) if a.phase == 'prefix' else run_field(a)


if __name__ == '__main__':
    main()
