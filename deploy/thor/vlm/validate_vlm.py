"""Validate the compiled VLM against the PyTorch bf16 reference on held-out clips (Thor). Exploratory, not registered.

  source $FROST_THOR_ROOT/env.sh; cd $FROST_THOR_ROOT/vlm/scripts
  python validate_vlm.py --backend trt --engines ../engines/bf16 --range 0 20 --field ../../trt/engines/probe_bf16_b6_L3456.engine \
      --field-ref $FROST_THOR_ROOT/trt/frost400/trt_bf16/plans.npz --out ../out/val_bf16.json

Reference per clip = $FROST_THOR_ROOT/trt/prefix/<clip>.pt, written by deploy/thor/trt/thor_trt_probe.py prefix (PyTorch
bf16 eager, SDPA, seed 123, top_p .98, temperature .6): packed prefix KV, key bias, action-token RoPE, reasoning text,
and the 6 seeds' initial noise. Reference plans = the TRT FROST probe output on that prefix with the same field engine.

Per clip, two runs of the compiled VLM:
  forced : the reference reasoning tokens are fed (teacher forcing). Isolates numerics: prefix-KV agreement (relative
           L2 over the P cached rows, worst layer), per-step agreement of the token the compiled model samples with the
           same RNG stream, and the plans of the field engine on the compiled prefix vs on the reference prefix.
  free   : normal rollout (seed 123). Reasoning text exact match, P, and plan deviation including any change of reasoning.
Plans: E10 and E4 (Euler, tau 1) for the 6 seeds through the same action-expert engine; max |dplan| over the 64
waypoints, in metres (temperature_probe.integrate, action_to_traj), compared to the probe's plans.npz keys.
"""
import os
import argparse, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trt'))
DATA = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/contexts/physicalai_heldout_contexts_v3_20261027')
PREFIX = Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/prefix')
SEEDS = list(range(9201, 9207))
SAMPLERS = [('euler', 10), ('euler', 4)]


def clip_names(rng):
    names = sorted(r['name'] for r in json.loads((DATA / 'records.json').read_text()) if 'error' not in r)
    return names[rng[0]:rng[1]]


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


class Field:
    """B-seed plans from one prefix through an action-expert engine (vlm_trt.FieldEngine, shared allocator)."""

    def __init__(self, engine, cache):
        from vlm_trt import FieldEngine
        from field_trt import load_action_space
        from frost.probes.temperature_probe import integrate
        self.f = FieldEngine(engine); self.integrate = integrate
        assert self.f.L == cache.shape[3], (self.f.L, cache.shape)
        self.aspace = load_action_space().cuda()

    def plans(self, kv, bias, cos, sin, init, hx, hr):
        self.f.bind(kv, bias, cos, sin)
        out = {}
        with torch.inference_mode():
            for m, k in SAMPLERS:
                y = self.integrate(self.f, init.clone(), k, m)
                B = y.shape[0]
                xyz = self.aspace.action_to_traj(y, hx[:, 0].expand(B, *hx.shape[2:]), hr[:, 0].expand(B, *hr.shape[2:]))[0]
                out[f'{m}{k}'] = xyz.float().cpu().numpy().reshape(B, -1, 3)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', choices=['torch', 'trt'], required=True); ap.add_argument('--engines', type=Path); ap.add_argument('--decode-engines', type=Path, default=None, help='chunk engines used for the decode profile (mixed precision: prefill from --engines)')
    ap.add_argument('--range', type=int, nargs=2, default=[0, 10]); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--field', type=Path, default=None, help='action-expert engine (batch 6, L = cache capacity)')
    ap.add_argument('--field-ref', type=Path, default=None, help="probe plans.npz of the same engine on the reference prefixes")
    ap.add_argument('--capacity', type=int, default=3456); ap.add_argument('--chunks', type=int, default=4)
    ap.add_argument('--modes', nargs='*', default=['forced', 'free'])
    ap.add_argument('--gpu-preproc', action='store_true', help='image resize/normalise on the GPU (Inputs.tokenize gpu=True)')
    ap.add_argument('--graph', action='store_true', help='decode steps as one replayed CUDA graph')
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    import thor_guard; thor_guard.wait_for_free_board(__file__)
    from vlm_runtime import Inputs, Pipeline, TorchBackend
    torch.cuda.set_stream(torch.cuda.Stream())    # TensorRT enqueues on a non-default stream (no implicit syncs)
    inp = Inputs()
    if a.backend == 'torch': be = TorchBackend(a.chunks)
    else:
        from vlm_trt import TRTBackend
        be = TRTBackend(a.engines, decode_dir=a.decode_engines)
    pipe = Pipeline(be, inp, capacity=a.capacity, graph=a.graph)
    field = Field(a.field, pipe.cache) if a.field else None
    ref_plans = np.load(a.field_ref) if a.field_ref else None
    res = {'args': {k: str(v) for k, v in vars(a).items()}, 'clips': []}
    all_plans = {}   # compiled-prefix plans [6 seeds, 64, 3] per clip, mode and sampler (for spread / tau-style checks)
    for name in clip_names(a.range):
        D = torch.load(PREFIX / f'{name}.pt', weights_only=False)
        x = inp.tokenize(DATA / f'{name}.npz', gpu=a.gpu_preproc)
        n = x['input_ids'].shape[1]
        ref_ids = inp.tok(D['reasoning'], add_special_tokens=False)['input_ids']
        row = {'clip': name, 'P_ref': D['P'], 'n_prompt': n, 'n_gen_ref': len(ref_ids), 'ref_text': D['reasoning'],
               'retok_ok': n + len(ref_ids) - 1 == D['P']}
        init = torch.cat([D['initial'][s] for s in SEEDS]).cuda(); hx, hr = D['hx'].cuda(), D['hr'].cuda()
        ref_bias, ref_cos, ref_sin = D['key_bias'].cuda(), D['cos'].cuda(), D['sin'].cuda()
        for mode in a.modes:
            if mode == 'forced' and not row['retok_ok']: row['forced'] = {'skipped': 'reference text does not re-tokenize to P'}; continue
            r = pipe.run(x, forced=ref_ids if mode == 'forced' else None)
            out = {k: r[k] for k in ['P', 'offset', 'text', 'timing']}
            if mode == 'forced':
                P = D['P']; kv = pipe.cache[:, :, :, :P]; ref = D['past_kv'].cuda()
                per_layer = [rel(kv[l], ref[l]) for l in range(kv.shape[0])]
                out.update({'kv_rel_l2': rel(kv, ref), 'kv_rel_l2_worst_layer': max(per_layer), 'kv_rel_l2_layers': per_layer,
                            'kv_rel_l2_prompt': rel(kv[..., :n, :], ref[..., :n, :]),
                            # the token after <|traj_future_start|> is never cached (HF pads it): excluded
                            'token_agree': float(np.mean([s == f for s, f in zip(r['sampled'][:-1], r['gen'][:-1])])),
                            'first_disagree': next((i for i, (s, f) in enumerate(zip(r['sampled'][:-1], r['gen'][:-1])) if s != f), None)})
                del ref
            else:
                cut = lambda t: t.split('<|traj_future_start|>')[0]
                out.update({'text_match': cut(r['text']) == cut(D['reasoning']), 'P_match': r['P'] == D['P'],
                            'gen_agree_prefix': next((i for i, (s, f) in enumerate(zip(r['gen'][:-1], ref_ids[:-1])) if s != f), min(len(r['gen']), len(ref_ids)) - 1)})
            bias, cos, sin = pipe.field_inputs(r, x['rope_deltas'])
            if mode == 'forced':
                out['field_inputs_match'] = {'key_bias': bool(torch.equal(bias[:D['P']], ref_bias[:D['P']])),
                                             'cos_max_abs': float((cos - ref_cos).abs().max()), 'sin_max_abs': float((sin - ref_sin).abs().max())}
            if field is not None:
                pl = field.plans(pipe.cache, bias, cos, sin, init, hx, hr)
                for key, arr in pl.items(): all_plans[f'{name}_{mode}_{key}'] = arr
                d = {}
                for key, arr in pl.items():
                    errs = []
                    for i, sd in enumerate(SEEDS):
                        rk = f'{name}_s{sd}_{key}_t1.0'
                        if ref_plans is not None and rk in ref_plans.files:
                            errs.append(float(np.abs(arr[i] - ref_plans[rk]).max()))
                    if errs: d[key] = {'median': statistics.median(errs), 'max': max(errs), 'per_seed': errs}
                out['dplan_m'] = d
            row[mode] = out
        res['clips'].append(row)
        print(json.dumps({'clip': name, **{m: {k: v for k, v in row.get(m, {}).items() if k not in ('kv_rel_l2_layers', 'text')} for m in a.modes}}, default=str), flush=True)
        a.out.write_text(json.dumps(res, indent=1, default=str))
        np.savez_compressed(a.out.with_suffix('.plans.npz'), **all_plans)
    a.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
