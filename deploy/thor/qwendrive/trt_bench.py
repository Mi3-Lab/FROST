"""Qwen-Drive on the Thor: PyTorch (eager) VLM + TensorRT/CUDA-graph planning expert. Same outputs/timings schema as eager_bench.py.
python trt_bench.py --model DIR --scenes DIR --engine step.engine --range 0 40 --out DIR"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import thor_patches  # noqa
from eager_bench import load_scene, sync
from expert_trt import ExpertTRT


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--scenes', type=Path, required=True); p.add_argument('--engine', required=True)
    p.add_argument('--range', type=int, nargs=2, default=[0, 40]); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=6); p.add_argument('--seed0', type=int, default=9201); p.add_argument('--no-graph', action='store_true'); p.add_argument('--vision-engines', default=None); p.add_argument('--graph-decode', action='store_true'); p.add_argument('--fp32-prep', action='store_true', help='per-clip queries/condition from an fp32 copy of the expert (for an fp32 engine)')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive.modeling_qwen_drive import denormalize_trajectory
    model = QwenDriveForPlanning.from_pretrained(a.model, planner=str(Path(a.model) / 'planner-rl'), dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval()
    model.config.noise_seed = a.seed0
    if a.vision_engines:
        from vision_trt import VisionTRT
        VisionTRT(a.vision_engines).install(model)
    ex = ExpertTRT(a.engine, use_graph=not a.no_graph); print('gpu mem GB after engine', torch.cuda.memory_allocated() / 1e9, flush=True)
    scale = model.trajectory_scale(torch.device('cuda'))
    gd = None
    if a.graph_decode:
        from graph_decode import GraphDecoder
        gd = GraphDecoder(model); first = {k_: v.to(model.device) if torch.is_tensor(v) else v for k_, v in model.processor(load_scene(a.scenes / 'context_000.npz'), with_reasoning=True, device='cpu').items()}; gd.capture(first); print('decode graph captured', flush=True)
    import copy
    exp32 = copy.deepcopy(model.planning_expert).float() if a.fp32_prep else None
    rows, plans, texts, errs = [], {}, {}, []
    for k, name in enumerate(f'context_{i:03d}' for i in range(*a.range)):
        scene = load_scene(a.scenes / f'{name}.npz')
        t0 = sync(); inputs = model.processor(scene, with_reasoning=True, device='cpu'); t1 = sync()
        inputs = {k_: v.to(model.device) if torch.is_tensor(v) else v for k_, v in inputs.items()}; t2 = sync()
        with torch.no_grad():
            if gd is None: cache, anchor, text = model._prefill_with_reasoning(inputs, model.config.max_reasoning_tokens)
            else: cache, anchor, text, ntok = gd.run(inputs, model.config.max_reasoning_tokens)
            t3 = sync()
            ex.prepare(model, cache, anchor, inputs, exp=exp32); t3b = sync()
            noise = model._initial_noise(1, 50, a.seed0, anchor.device); out = ex.solve(noise); traj = denormalize_trajectory(out.clone(), scale).cpu().numpy(); t4 = sync()
            trajs = []
            for i in range(a.samples):
                nz = model._initial_noise(1, 50, a.seed0 + i, anchor.device); trajs.append(denormalize_trajectory(ex.solve(nz).clone(), scale).cpu().numpy()[0])
            if exp32 is None: ref = model._plan_from_cache(cache, anchor, inputs, num_samples=a.samples, num_steps=10, seed=a.seed0)   # eager bf16 expert on the same cache
        e = [float(np.linalg.norm(trajs[i][:, :2] - ref[i][:, :2], axis=-1).max()) for i in range(a.samples)] if exp32 is None else [0.0]; errs.append(float(np.median(e)))
        rows.append(dict(name=name, host_s=t1 - t0, h2d_s=t2 - t1, vlm_prefill_plus_decode_s=t3 - t2, prepare_s=t3b - t3, expert_e10_s=t4 - t3b, plan_total_s=t4 - t0,
                         expert_vs_eager_cm=errs[-1] * 100, gd=(gd.t if gd else None), warm=k >= 1))
        texts[name] = text
        for i in range(a.samples): plans[f'{name}_s{a.seed0 + i}'] = trajs[i]
        print(json.dumps(rows[-1]), flush=True)
        (a.out / 'rows.json').write_text(json.dumps(rows)); (a.out / 'reasoning.json').write_text(json.dumps(texts, indent=1)); np.savez_compressed(a.out / 'plans.npz', **plans)
    print('peak gpu GB', torch.cuda.max_memory_allocated() / 1e9)


if __name__ == '__main__':
    main()
