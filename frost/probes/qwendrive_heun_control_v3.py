"""Heun-k control for Qwen-Drive-1.0 on the v3 held-out probe (protocols/qwen_drive.md, addendum 1).

Same scene, command rule, greedy reasoning, fp32 expert, seeds (9201..9206) and initial noise as
frost.probes.qwendrive_temperature_probe_v3; only the sampler list changes. Default configs: Heun-3 and Heun-5 at tau = 1 (the
controls), plus Euler-10 and Heun-32 at tau = 1, regenerated so the run can be checked against the stored v3 plans
(reproduction check R2) and so a same-run reference exists. Heun-k integrates [0, 0.9] with k steps (2k evaluations)
and lands on the endpoint predicted at t = 0.9 (one more evaluation), as in frost.probes.qwendrive_temperature_probe_v2.plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

import frost.probes.qwendrive_temperature_probe_v2 as v2
from frost.probes.qwendrive_temperature_probe_v3 import build_scene

CONFIGS = [('euler', 10, 1.), ('heun', 3, 1.), ('heun', 5, 1.), ('heun', 32, 1.)]


def parse_configs(text):
    out = []
    for item in text.split(','):
        method, steps, tau = item.split(':'); out.append((method, int(steps), float(tau)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--records', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--planner', default='planner-rl'); p.add_argument('--range', type=int, nargs=2, default=[0, 400])
    p.add_argument('--samples', type=int, default=6); p.add_argument('--seed0', type=int, default=9201)
    p.add_argument('--command', choices=['simwam', 'legacy'], default='simwam'); p.add_argument('--resume', action='store_true')
    p.add_argument('--configs', type=parse_configs, default=CONFIGS, help='method:steps:tau,... (default: euler:10:1,heun:3:1,heun:5:1,heun:32:1)')
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
            for method, steps, tau in args.configs:
                for i, trajectory in enumerate(v2.plan(model, cache, anchor, inputs, args.samples, method, steps, tau, True)):
                    plans[f'{name}_s{seeds[i]}_{method}{steps}_t{tau}'] = trajectory
        plans[f'{name}_human'] = future
        np.savez_compressed(args.out / 'plans.npz', **plans); (args.out / 'reasoning.json').write_text(json.dumps(traces, indent=1))
        print(json.dumps({'context': name, 'nav_command': scene.nav_command, 'plans': len(plans)}), flush=True); torch.cuda.empty_cache()
    (args.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'model': args.model, 'planner': args.planner, 'mode': 'reasoning_planning', 'expert_fp32': True,
        'noise_init_std': model.config.noise_init_std, 'min_one_minus_t': model.config.min_one_minus_t, 't_end': v2.T_END, 'range': args.range, 'seeds': seeds,
        'configs': args.configs, 'reference': 'heun32_t1.0', 'command_rule': args.command, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'gpu': torch.cuda.get_device_name(0), 'limitations': ['Trajectories are (x, y, heading) at 10 Hz over 5 s; first 40 points = 4 s common horizon']}, indent=2))


if __name__ == '__main__':
    main()
