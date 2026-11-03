"""Replicate the numerical-temperature (FROST) probe on SimWAM-PAI-AV, a non-NVIDIA flow-matching driving policy.

SimWAM (HUST; Wan2.2-5B video expert + action DiT, rectified flow, Euler sampler WanContinuousFlowMatchScheduler,
shift 5, 10 eval steps) prefills the current front frame once into a KV cache and only the action expert iterates.
We integrate the same velocity field with our own integrators on SimWAM's own shifted-sigma grid
(sigma: 1 = noise -> 0 = data, x <- x + delta * v, v = noise - sample):
  euler{k} tau=1, heun{k} tau=1, and a Heun-32 reference started from tau*z, tau in TAUS.
Plan keys and config names match temperature_probe ('{clip}_s{seed}_{method}{steps}_t{tau}'); plans are (40,3) arrays
[x, y, 0] in metres, ego frame at t0 (heading goes to heading.npz). Noise z comes from SimWAM's own generator
(CPU randn, seed) and is shared across configs.

Numerics: --numerics fp32 (default) keeps the ODE state and step sizes in float32 so that sampler differences are not
confounded by bf16 state rounding; weights/activations stay bf16. --numerics native reproduces SimWAM's infer_action
loop bit for bit (bf16 state, bf16 timesteps and deltas) and is used by --verify.

Adapter (our 400 held-out PhysicalAI-AV contexts, 5.1 s into each clip), matching experiments/physicalai/eval.py and
configs/sim_physicalai_eval.yaml + configs/task/physicalai_uncond_front_384x672_1e-4.yaml (video_size [384,672]):
front-wide camera (eval.py CAM), the frame at the anchor (last of our 4 cached frames = t0), tensor bilinear antialias resize
(_preprocess_image), fixed T5 prompt (use_dynamic_prompt=false, context_len 256), 40 pts at 10 Hz. Ego state: velocity and
acceleration at t0 from the official PhysicalAI-AV egomotion interpolator, world frame rotated into the anchor frame
(frost.data.simwam_ego_inputs; reproduces the released physicalai_train.jsonl ego_velocity/acceleration to 4 decimals and
future_traj_10hz x, y [left = +y], heading). Command: --command official = SimWAM's eval rule (_command_from_gt_traj,
0.31 rad on the logged future); --command straight = (0,1,0,0) for every clip (sensitivity, no future information).
The command each clip received is saved in inputs.npz ('{clip}_command').
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

TAUS = [1., .9, .8, .7, .6, .5, .4]
CONFIGS = [('euler', k, 1.) for k in [1, 2, 4, 6, 10]] + [('heun', 5, 1.)] + [('heun', 32, tau) for tau in TAUS]
ARTIFACTS = Path(os.environ.get('FROST_ARTIFACTS', 'artifacts'))
SIMWAM = Path(os.environ.get('SIMWAM_SRC', 'third_party/simwam'))
ASSETS = Path(os.environ.get('SIMWAM_ASSETS', 'third_party/simwam_assets'))
HUB = os.environ.get('FROST_SIMWAM_HUB', os.path.join(os.environ.get('HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')), 'hub'))
CKPT_GLOB = os.path.join(HUB, 'models--H-EmbodVis--SimWAM', 'snapshots', '*', 'weights', 'SimWAM-PAI-AV.pt')
STATS_GLOB = os.path.join(HUB, 'models--H-EmbodVis--SimWAM', 'snapshots', '*', 'data', 'physicalai_dataset_stats.json')
HORIZON, VIDEO_HW, FRONT_CAMERA, CMD_TH = 40, (384, 672), 1, 0.31


def setup_paths():
    os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(ASSETS / 'checkpoints'))
    os.environ.setdefault('DIFFSYNTH_DOWNLOAD_SOURCE', 'huggingface')
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'true')
    sys.path.insert(0, str(SIMWAM / 'src'))


def fixed_prompt():
    return ('A high-quality, photorealistic dashboard camera view of autonomous driving. Based on the past 2 seconds videos, '
            'predict and generate the next 4 seconds of realistic driving continuation, Maintain temporal consistency, '
            'stable camera perspective, natural motion flow without jitter or artifacts, clear details, and realistic physics. ')


def load_model(device, ckpt):
    """Instantiate SimWAM from its own Hydra config (random init, no Wan DiT download) and load the PAI-AV checkpoint."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from simwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    with initialize_config_dir(str(SIMWAM / 'configs'), version_base='1.3'):
        cfg = compose('sim_physicalai_eval', overrides=['task=physicalai_uncond_front_384x672_1e-4',
            'model.skip_dit_load_from_pretrain=true', 'model.redirect_common_files=false', 'model.load_text_encoder=false'])
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    payload = torch.load(ckpt, map_location='cpu', mmap=True)
    inc = model.mot.load_state_dict(payload['mot'], strict=False)
    inc2 = model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    print(json.dumps({'ckpt_missing': len(inc.missing_keys), 'ckpt_unexpected': len(inc.unexpected_keys), 'missing_head': inc.missing_keys[:3]}), flush=True)
    assert len(inc.missing_keys) == 0 and len(inc.unexpected_keys) == 0, (inc.missing_keys[:5], inc.unexpected_keys[:5])
    return model.to(device).eval(), cfg


def fixed_context(cache):
    """T5 context of the fixed prompt (length 256), computed once on CPU with Wan2.2's umt5-xxl encoder and cached."""
    if Path(cache).exists():
        p = torch.load(cache, map_location='cpu')
    else:
        from simwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from simwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
        _, text_cfg, _, tok_cfg = _resolve_configs('Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B', redirect_common_files=False)
        text_cfg.download_if_necessary(); tok_cfg.download_if_necessary()
        enc = _load_registered_model(text_cfg.path, 'wan_video_text_encoder', torch_dtype=torch.bfloat16, device='cpu')
        tok = HuggingfaceTokenizer(name=tok_cfg.path, seq_len=256, clean='whitespace')
        with torch.no_grad():
            ids, mask = tok([fixed_prompt()], return_mask=True, add_special_tokens=True)
            ctx = enc(ids, mask.bool())
        p = {'context': ctx[0].cpu(), 'mask': mask[0].bool().cpu(), 'prompt_sha256': hashlib.sha256(fixed_prompt().encode()).hexdigest()}
        Path(cache).parent.mkdir(parents=True, exist_ok=True); torch.save(p, cache)
        del enc
    context = p['context'].to(torch.bfloat16); mask = p['mask'].bool(); context[~mask] = 0
    return context.contiguous(), torch.ones_like(mask)


def denorm_stats():
    import glob
    stats = json.load(open(sorted(glob.glob(STATS_GLOB))[0])); t = stats.get('future_traj_10hz', stats)
    q = {c: (float(t[c]['q1']), float(t[c]['q99'])) for c in ('x', 'y', 'heading')}
    bias = torch.tensor([-q[c][0] for c in q]); rng = torch.tensor([q[c][1] - q[c][0] for c in q])
    return bias, rng


def command_from_gt(gt_xy):
    """SimWAM's own rule (experiments/physicalai/eval.py::_command_from_gt_traj): [left, straight, right, unknown]."""
    heading = float(np.arctan2(gt_xy[-1, 1] - gt_xy[-2, 1], gt_xy[-1, 0] - gt_xy[-2, 0]))
    return np.array([1., 0, 0, 0] if heading >= CMD_TH else [0, 0, 1., 0] if heading <= -CMD_TH else [0, 1., 0, 0], np.float32)


def load_inputs(path, ego, command_mode):
    """Adapter: one cached context + official ego inputs -> (image [1,3,384,672], proprio [8], gt_xy [40,2], command)."""
    import torchvision.transforms.functional as F
    name = Path(path).stem
    with np.load(path) as z:
        frame = z['image_frames'][FRONT_CAMERA, -1]
    t = F.to_tensor(frame)  # same preprocessing as experiments/physicalai/eval.py::_preprocess_image
    t = F.normalize(F.resize(t, size=list(VIDEO_HW), interpolation=F.InterpolationMode.BILINEAR, antialias=True), [.5] * 3, [.5] * 3)
    gt = ego[f'{name}_gt_xyh'][:, :2].astype(np.float32)
    cmd = command_from_gt(gt) if command_mode == 'official' else np.array([0, 1., 0, 0], np.float32)
    return t.unsqueeze(0).contiguous(), torch.from_numpy(np.concatenate([ego[f'{name}_vel'], ego[f'{name}_acc'], cmd]).astype(np.float32)), gt, cmd


@torch.no_grad()
def prepare_field(model, image, proprio, context, context_mask, device):
    """Condition prefill, copied from SimWAM.infer_action (action-only path): returns velocity(x, timestep) with the KV cache fixed."""
    dtype = model.torch_dtype
    first = model._encode_input_image_latents_tensor(input_image=image.to(device=device, dtype=dtype), tiled=False)
    ctx = context.unsqueeze(0).to(device=device, dtype=dtype); msk = context_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
    ctx, msk = model._append_proprio_to_context(context=ctx, context_mask=msk, proprio=proprio.unsqueeze(0).to(device=device, dtype=dtype))
    t0 = torch.zeros((first.shape[0],), dtype=first.dtype, device=device)
    pre = model.video_expert.pre_dit(x=first, timestep=t0, context=ctx, context_mask=msk, action=None,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, 'fuse_vae_embedding_in_latents', False)))
    seq = int(pre['tokens'].shape[1])
    mask = model._build_mot_attention_mask(video_seq_len=seq, action_seq_len=HORIZON, video_tokens_per_frame=int(pre['meta']['tokens_per_frame']), device=pre['tokens'].device)
    cache = model.mot.prefill_video_cache(video_tokens=pre['tokens'], video_freqs=pre['freqs'], video_t_mod=pre['t_mod'],
        video_context_payload={'context': pre['context'], 'mask': pre['context_mask']}, video_attention_mask=mask[:seq, :seq])

    def velocity(x, timestep):
        with torch.autocast(device.split(':')[0], dtype=torch.bfloat16, enabled=timestep.dtype == torch.float32):  # fp32 timestep embedding -> bf16 Linear
            return velocity_(x, timestep)

    def velocity_(x, timestep):
        return model._predict_action_noise_with_cache(latents_action=x, timestep_action=timestep.reshape(1).expand(x.shape[0]), context=ctx,
            context_mask=msk, video_kv_cache=cache, attention_mask=mask, video_seq_len=seq)
    return velocity


@torch.no_grad()
def integrate(model, velocity, z, method, steps, tau, numerics, device):
    """Euler/Heun on SimWAM's shifted-sigma grid, started from tau*z at sigma=1."""
    native = numerics == 'native'; dt = torch.bfloat16 if native else torch.float32
    ts, ds = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=steps, device=device, dtype=dt, shift_override=None)
    x = (tau * z).to(device=device, dtype=dt)
    sigma = torch.cat([ts / 1000., torch.zeros(1, device=device, dtype=dt)]) if not native else None
    for i in range(steps):
        v = velocity(x.to(torch.bfloat16), ts[i]).to(dt)
        if method == 'euler':
            x = x + v * ds[i]
        else:  # Heun: predictor to sigma_{i+1}, corrector with the trapezoid rule
            xp = x + v * ds[i]
            v2 = velocity(xp.to(torch.bfloat16), (sigma[i + 1] * 1000.).to(dt)).to(dt)
            x = x + .5 * ds[i] * (v + v2)
    return x.float()


def to_plan(action, bias, rng):
    xyh = ((action.cpu().float() + 1) / 2 * rng - bias).numpy().reshape(HORIZON, 3)
    return np.stack([xyh[:, 0], xyh[:, 1], np.zeros(HORIZON, np.float32)], 1), xyh[:, 2]


def noise(seed):
    return torch.randn((HORIZON, 3), generator=torch.Generator(device='cpu').manual_seed(int(seed)), device='cpu', dtype=torch.float32).unsqueeze(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=ARTIFACTS / 'physicalai_heldout_contexts_v3_20261027'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--range', type=int, nargs=2, default=[0, 20]); p.add_argument('--clips', nargs='+', default=None, help='context names (e.g. context_007) instead of --range')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(9201, 9207))); p.add_argument('--device', default='cuda')
    p.add_argument('--numerics', choices=['fp32', 'native'], default='fp32'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--configs', nargs='+', default=None, help='method:steps:tau triples replacing the default grid')
    p.add_argument('--verify', action='store_true', help='check our native Euler-10 against SimWAM.infer_action and exit')
    p.add_argument('--ckpt', default=None); p.add_argument('--command', choices=['official', 'straight'], default='official')
    p.add_argument('--ego', type=Path, default=ARTIFACTS / 'simwam_ego_inputs_20261031/ego.npz'); p.add_argument('--converge', action='store_true', help='Heun-32/64/128 convergence at tau 1 and 0.7, then exit')
    p.add_argument('--extend', action='store_true', help='Heun-128/256/512 and adaptive RK45 (rtol=atol 1e-5, 1e-7) on the same ODE, tau 1 and 0.7, then exit'); p.add_argument('--max-nfe', type=int, default=6000)
    p.add_argument('--turns', action='store_true', help='with --verify/--converge: also report x,y sign vs logged future')
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    import glob
    setup_paths(); torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=args.resume or args.verify or args.extend)
    source = Path(__file__).read_bytes(); (args.out / 'source_snapshot.py').write_bytes(source)
    records = sorted((r for r in json.loads((args.data / 'records.json').read_text()) if 'error' not in r), key=lambda r: r['name'])
    records = [r for r in records if r['name'] in set(args.clips)] if args.clips else records[args.range[0]:args.range[1]]
    ego = dict(np.load(args.ego))
    ckpt = args.ckpt or sorted(glob.glob(CKPT_GLOB))[0]
    model, cfg = load_model(args.device, ckpt); bias, rng = denorm_stats()
    context, cmask = fixed_context(ASSETS / 'text_embeds_cache/fixed_prompt.t5_len256.wan22ti2v5b.pt')
    configs = [(m, int(k), float(t)) for m, k, t in (c.split(':') for c in args.configs)] if args.configs else CONFIGS
    if args.verify:
        report = []
        for r in records:
            image, proprio, gt, cmd = load_inputs(args.data / f"{r['name']}.npz", ego, args.command); vel = prepare_field(model, image, proprio, context, cmask, args.device)
            for seed in args.seeds[:1]:
                theirs = model.infer_action(prompt=None, input_image=image, proprio=proprio, context=context, context_mask=cmask, action_horizon=HORIZON,
                    num_inference_steps=10, sigma_shift=None, rand_device='cpu', tiled=False, seed=seed, num_samples=1)['action']
                ours = {n: integrate(model, vel, noise(seed), 'euler', 10, 1., n, args.device)[0] for n in ['native', 'fp32']}
                pt, _ = to_plan(theirs, bias, rng)
                row = {'clip': r['name'], 'seed': seed, 'max_abs_diff_native_vs_infer_action_normalized': float((ours['native'].cpu() - theirs).abs().max()),
                       'bit_identical_native': bool(torch.equal(ours['native'].cpu(), theirs)),
                       'fp32_vs_infer_action_max_m': float(np.abs(to_plan(ours['fp32'], bias, rng)[0] - pt).max()),
                       'fp32_vs_infer_action_mean_m': float(np.linalg.norm(to_plan(ours['fp32'], bias, rng)[0][:, :2] - pt[:, :2], axis=1).mean()),
                       'ade_vs_logged_future_m': float(np.linalg.norm(pt[:, :2] - gt, axis=1).mean()), 'plan_end_xy': pt[-1, :2].tolist(), 'gt_end_xy': gt[-1].tolist(),
                       'proprio': proprio.tolist()}
                report.append(row); print(json.dumps(row), flush=True)
        (args.out / 'verify.json').write_text(json.dumps(report, indent=2)); return
    if args.extend:
        from frost.core.decisions import decisions
        from frost.probes.adaptive_ref import rk45
        rep = []
        for r in records:
            image, proprio, gt, cmd = load_inputs(args.data / f"{r['name']}.npz", ego, args.command); vel = prepare_field(model, image, proprio, context, cmask, args.device)
            speed = float(np.linalg.norm(proprio[:2].numpy()))
            ts0, _ = model.infer_action_scheduler.build_inference_schedule(num_inference_steps=128, device=args.device, dtype=torch.float32, shift_override=None)
            sigma0 = float(ts0[0]) / 1000.
            for tau in (1., .7):
                for seed in args.seeds:
                    z = noise(seed); pl = {}; info = {}
                    for k in (128, 256, 512):
                        pl[f'heun{k}'] = to_plan(integrate(model, vel, z, 'heun', k, tau, args.numerics, args.device)[0], bias, rng)[0]
                    for tol in (1e-5, 1e-7):
                        def f(s, y):
                            x = torch.from_numpy(np.asarray(y, np.float32)).reshape(1, HORIZON, 3).to(args.device)
                            return vel(x.to(torch.bfloat16), torch.tensor(s * 1000., dtype=torch.float32, device=args.device)).float().cpu().numpy().astype(np.float64).ravel()
                        res = rk45(f, (tau * z).numpy(), sigma0, tol, tol, args.max_nfe); info[f'rk45_{tol:g}'] = {k: res[k] for k in ('nfe', 'ok', 'steps', 'reason')}
                        if res['y'] is not None:
                            pl[f'rk45_{tol:g}'] = to_plan(torch.from_numpy(res['y'].astype(np.float32)).reshape(1, HORIZON, 3), bias, rng)[0]
                    ref = pl['heun512']; cls = {k: decisions(v, speed) for k, v in pl.items()}
                    row = {'clip': r['name'], 'tau': tau, 'seed': seed, 'speed': speed, 'sigma0': sigma0, 'adaptive': info, 'end_xy_heun512': ref[-1, :2].tolist(), 'gt_end_xy': gt[-1].tolist(),
                           'dist': {k: float(np.abs(v - ref).max()) for k, v in pl.items()},
                           'dist_128_256': float(np.abs(pl['heun128'] - pl['heun256']).max()), 'dist_256_512': float(np.abs(pl['heun256'] - pl['heun512']).max()),
                           'class': {k: [c['longitudinal'], c['lateral']] for k, c in cls.items()}}
                    rep.append(row); print(json.dumps(row), flush=True)
        (args.out / 'extend.json').write_text(json.dumps(rep, indent=2)); return
    if args.converge:
        from frost.core.decisions import decisions
        rep = []
        for r in records:
            image, proprio, gt, cmd = load_inputs(args.data / f"{r['name']}.npz", ego, args.command); vel = prepare_field(model, image, proprio, context, cmask, args.device)
            speed = float(np.linalg.norm(proprio[:2].numpy())); row = {'clip': r['name'], 'speed': speed, 'command': cmd.tolist(), 'by_tau': {}}
            for tau in (1., .7):
                for seed in args.seeds:
                    pl = {k: to_plan(integrate(model, vel, noise(seed), 'heun', k, tau, args.numerics, args.device)[0], bias, rng)[0] for k in (32, 64, 128)}
                    d = {k: decisions(v, speed) for k, v in pl.items()}
                    row['by_tau'].setdefault(str(tau), []).append({'seed': seed, 'max_abs_32_64_m': float(np.abs(pl[32] - pl[64]).max()), 'max_abs_64_128_m': float(np.abs(pl[64] - pl[128]).max()),
                        'long_agree_32_128': d[32]['longitudinal'] == d[128]['longitudinal'], 'lat_agree_32_128': d[32]['lateral'] == d[128]['lateral'],
                        'long_agree_64_128': d[64]['longitudinal'] == d[128]['longitudinal'], 'lat_agree_64_128': d[64]['lateral'] == d[128]['lateral'],
                        'class_128': [d[128]['longitudinal'], d[128]['lateral']], 'end_xy_128': pl[128][-1, :2].tolist(), 'gt_end_xy': gt[-1].tolist()})
            rep.append(row); print(json.dumps(row), flush=True)
        (args.out / 'converge.json').write_text(json.dumps(rep, indent=2)); return
    plans, headings, inputs = {}, {}, {}
    if args.resume and (args.out / 'plans.npz').exists():
        with np.load(args.out / 'plans.npz') as z: plans = {k: z[k] for k in z.files}
        with np.load(args.out / 'heading.npz') as z: headings = {k: z[k] for k in z.files}
        with np.load(args.out / 'inputs.npz') as z: inputs = {k: z[k] for k in z.files}
    done = {k.split('_s')[0] for k in plans}; times = {}
    for r in records:
        name = r['name']
        if name in done: continue
        tic = time.time()
        image, proprio, gt, cmd = load_inputs(args.data / f'{name}.npz', ego, args.command); inputs[f'{name}_gt'] = gt; inputs[f'{name}_proprio'] = proprio.numpy(); inputs[f'{name}_command'] = cmd
        velocity = prepare_field(model, image, proprio, context, cmask, args.device)
        for seed in args.seeds:
            z = noise(seed)
            for method, steps, tau in configs:
                plan, hdg = to_plan(integrate(model, velocity, z, method, steps, tau, args.numerics, args.device)[0], bias, rng)
                plans[f'{name}_s{seed}_{method}{steps}_t{tau}'] = plan; headings[f'{name}_s{seed}_{method}{steps}_t{tau}'] = hdg
        times[name] = time.time() - tic
        np.savez_compressed(args.out / 'plans.npz', **plans); np.savez_compressed(args.out / 'heading.npz', **headings); np.savez_compressed(args.out / 'inputs.npz', **inputs)
        print(json.dumps({'context': name, 'plans': len(plans), 'seconds': round(times[name], 1)}), flush=True)
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    (args.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'model': 'SimWAM-PAI-AV', 'data': str(args.data), 'range': args.range, 'seeds': args.seeds,
        'command': args.command, 'ego_inputs': str(args.ego), 'configs': configs, 'nfe': {f'{m}{k}': k * (2 if m == 'heun' else 1) for m, k, _ in configs}, 'numerics': args.numerics, 'ckpt': ckpt,
        'simwam_commit': os.popen(f'git -C {SIMWAM} rev-parse HEAD').read().strip(), 'seconds_per_clip': times, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'limitations': ['Logged-future-derived command in the proprio vector, as in SimWAM own PhysicalAI eval', 
                        'frames decoded from our cached 1080p contexts, not SimWAM extracted jpgs']}, indent=2))


if __name__ == '__main__':
    main()
