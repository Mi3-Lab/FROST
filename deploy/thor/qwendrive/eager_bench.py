"""Qwen-Drive-1.0-4B release path (PyTorch eager) on the Thor from dumped scenes: per-stage latency (batch 1, frames in RAM) and the
plans/reasoning for the fidelity comparison with the cluster reference (qwendrive_v3 probe).
Usage: python eager_bench.py --model DIR --scenes DIR --range 0 40 --out DIR [--samples 6] [--expert-fp32]"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
try:
    import thor_patches  # noqa
except ImportError:
    pass
from PIL import Image


def load_scene(path):
    from qwen_drive import CameraFrame, DrivingScene
    z = np.load(path)
    views = {str(t): [CameraFrame(Image.fromarray(f)) for f in z['frames'][i]] for i, t in enumerate(z['tags'])}
    return DrivingScene(views=views, history=z['history'], history_velocity=z['history_velocity'], history_acceleration=z['history_acceleration'],
                        ego_velocity=tuple(z['ego_velocity']), ego_acceleration=tuple(z['ego_acceleration']), driving_command=list(z['driving_command']),
                        nav_command=int(z['nav_command']), token=str(z['token']))


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--range', type=int, nargs=2, default=[0, 40]); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=6); p.add_argument('--seed0', type=int, default=9201); p.add_argument('--attn', default='sdpa')
    p.add_argument('--expert-fp32', action='store_true'); p.add_argument('--warmup', type=int, default=1)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from qwen_drive import QwenDriveForPlanning
    t = time.time()
    model = QwenDriveForPlanning.from_pretrained(a.model, planner=str(Path(a.model) / 'planner-rl'), dtype=torch.bfloat16,
                                                 attn_implementation=a.attn, device_map='cuda:0').eval()
    model.config.noise_seed = a.seed0
    print('loaded', round(time.time() - t, 1), 's; gpu mem GB', torch.cuda.memory_allocated() / 1e9, flush=True)
    if a.expert_fp32:
        model.planning_expert.float()
        for m in model.planning_expert.modules():
            if getattr(m, 'attn_implementation', None) == 'flash_attention_2': m.attn_implementation = 'sdpa'
    rows, plans, texts = [], {}, {}
    names = [f'context_{i:03d}' for i in range(*a.range)]
    for k, name in enumerate(names):
        scene = load_scene(a.scenes / f'{name}.npz')
        t0 = sync()
        inputs = model.processor(scene, with_reasoning=True, device='cpu'); t1 = sync()
        inputs = {k_: v.to(model.device) if torch.is_tensor(v) else v for k_, v in inputs.items()}; t2 = sync()
        with torch.no_grad():
            cache, anchor, text = model._prefill_with_reasoning(inputs, model.config.max_reasoning_tokens); t3 = sync()
            traj = model._plan_from_cache(cache, anchor, inputs, num_samples=1, num_steps=10, seed=a.seed0); t4 = sync()
            # six samples with the reference seeds, for fidelity (not part of the one-plan latency)
            traj6 = model._plan_from_cache(cache, anchor, inputs, num_samples=a.samples, num_steps=10, seed=a.seed0)
            # prefill alone (extra pass, for the stage split)
            ta = sync(); model._prefill(inputs); tb = sync()
        ntok = int(len(model.processor.tokenizer(text)['input_ids']))
        rows.append(dict(name=name, host_s=t1 - t0, h2d_s=t2 - t1, vlm_prefill_plus_decode_s=t3 - t2, prefill_only_s=tb - ta, expert_e10_b1_s=t4 - t3,
                         plan_total_s=t4 - t0, prompt_tokens=int(inputs['input_ids'].shape[1]), reasoning_tokens=ntok, warm=k >= a.warmup))
        texts[name] = text
        for i in range(a.samples): plans[f'{name}_s{a.seed0 + i}'] = np.asarray(traj6[i])
        print(json.dumps(rows[-1]), flush=True)
        (a.out / 'rows.json').write_text(json.dumps(rows)); (a.out / 'reasoning.json').write_text(json.dumps(texts, indent=1))
        np.savez_compressed(a.out / 'plans.npz', **plans)
    print('peak gpu GB', torch.cuda.max_memory_allocated() / 1e9)


if __name__ == '__main__':
    main()
