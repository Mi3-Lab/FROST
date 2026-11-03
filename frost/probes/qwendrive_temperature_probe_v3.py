"""Qwen-Drive-1.0 temperature probe v3 for the cross-model comparison (protocols/fair_comparison.md).

Same samplers/configs and reference as v2 (release Euler 1/2/4/10 ending on the endpoint at t = 0.9; Heun-32 reference curve on
[0, 0.9] at tau 1..0.3; Heun-64 at tau 1), reasoning_planning with greedy reasoning shared per clip, fp32 expert. Differences from v2:
 - clips: the 400 held-out PhysicalAI-AV v3 contexts (records.json), t0 = 5.1 s, one window per clip;
 - seeds 9201..9206 (the release's per-sample seed is noise_seed + k; noise_seed is set to 9201 here, so sample k uses 9201 + k);
 - navigation command by SimWAM's rule (0.31 rad on the last segment of the 4 s future) by default, so every model that receives a
   command gets it from the same rule (--command legacy gives v2's 0.35 rad at 5 s);
 - the logged future is saved at the full 5 s horizon (50 points); the 4 s common-horizon sensitivity uses its first 40 points.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import frost.probes.qwendrive_temperature_probe as v1
import frost.probes.qwendrive_temperature_probe_v2 as v2

SIMWAM_TH = .31


def command(future, rule):
    if rule == 'legacy':
        return v1.nav_command(future)
    xy = future[:40, :2]; h = float(np.arctan2(xy[-1, 1] - xy[-2, 1], xy[-1, 0] - xy[-2, 0]))
    return (1, [1, 0, 0, 0]) if h >= SIMWAM_TH else (2, [0, 0, 1, 0]) if h <= -SIMWAM_TH else (0, [0, 1, 0, 0])


def build_scene(avdi, clip, t0, rule):
    from qwen_drive import CameraFrame, DrivingScene
    ego = avdi.get_clip_feature(clip, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    history, hist_vel, hist_acc = v1.ego_series(ego, t0, np.arange(-15, 1) / 10)
    future, _, _ = v1.ego_series(ego, t0, np.arange(1, 51) / 10); history[-1] = 0.
    cmd, driving = command(future, rule)
    stamps = (t0 + np.array([-1.5, -1., -.5, 0.]) * 1e6).astype(np.int64); views = {}
    for tag, feature in v1.CAMERAS.items():
        camera = avdi.get_clip_feature(clip, getattr(avdi.features.CAMERA, feature), maybe_stream=True)
        frames, _ = camera.decode_images_from_timestamps(stamps)
        views[tag] = [CameraFrame(Image.fromarray(np.asarray(f))) for f in frames]
    return DrivingScene(views=views, history=history, history_velocity=hist_vel, history_acceleration=hist_acc, ego_velocity=tuple(hist_vel[-1]),
                        ego_acceleration=tuple(hist_acc[-1]), driving_command=driving, nav_command=cmd, token=clip), future


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--records', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--planner', default='planner-rl'); p.add_argument('--range', type=int, nargs=2, default=[0, 400])
    p.add_argument('--samples', type=int, default=6); p.add_argument('--seed0', type=int, default=9201)
    p.add_argument('--command', choices=['simwam', 'legacy'], default='simwam'); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=args.resume)
    source = Path(__file__).read_bytes(); (args.out / 'source_snapshot.py').write_bytes(source)
    import physical_ai_av
    from qwen_drive import QwenDriveForPlanning
    records = sorted((r for r in json.loads(args.records.read_text()) if 'error' not in r), key=lambda r: r['name'])[args.range[0]:args.range[1]]
    model = QwenDriveForPlanning.from_pretrained(args.model, planner=str(Path(args.model) / args.planner), dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2').cuda().eval()
    model.config.noise_seed = args.seed0
    model.planning_expert.float()
    for module in model.planning_expert.modules():
        if getattr(module, 'attn_implementation', None) == 'flash_attention_2': module.attn_implementation = 'sdpa'
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(); seeds = [args.seed0 + k for k in range(args.samples)]
    plans, traces = {}, {}
    if args.resume and (args.out / 'plans.npz').exists():
        with np.load(args.out / 'plans.npz') as z: plans = {k: z[k] for k in z.files}
        if (args.out / 'reasoning.json').exists(): traces = json.loads((args.out / 'reasoning.json').read_text())
    done = {k.split('_s')[0] for k in plans}
    for record in records:
        name = record['name']
        if name in done: continue
        scene, future = build_scene(avdi, record['clip_id'], record['t0_us'], args.command)
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in model.processor(scene, with_reasoning=True, device='cpu').items()}
        with torch.no_grad():
            cache, anchor, traces[name] = model._prefill_with_reasoning(inputs, model.config.max_reasoning_tokens)
            for method, steps, tau in v2.CONFIGS:
                for i, trajectory in enumerate(v2.plan(model, cache, anchor, inputs, args.samples, method, steps, tau, True)):
                    plans[f'{name}_s{seeds[i]}_{method}{steps}_t{tau}'] = trajectory
        plans[f'{name}_human'] = future
        np.savez_compressed(args.out / 'plans.npz', **plans); (args.out / 'reasoning.json').write_text(json.dumps(traces, indent=1))
        print(json.dumps({'context': name, 'nav_command': scene.nav_command, 'plans': len(plans)}), flush=True); torch.cuda.empty_cache()
    (args.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'model': args.model, 'planner': args.planner, 'mode': 'reasoning_planning', 'expert_fp32': True,
        'noise_init_std': model.config.noise_init_std, 'min_one_minus_t': model.config.min_one_minus_t, 't_end': v2.T_END, 'range': args.range, 'seeds': seeds,
        'configs': v2.CONFIGS, 'reference': 'heun32_t1.0', 'command_rule': args.command, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'limitations': ['Trajectories are (x, y, heading) at 10 Hz over 5 s; first 40 points = 4 s common horizon', 'command derived from the logged future by the SimWAM rule']}, indent=2))


if __name__ == '__main__':
    main()
