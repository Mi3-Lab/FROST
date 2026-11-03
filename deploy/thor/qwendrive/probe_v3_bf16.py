"""Qwen-Drive-1.0-4B temperature probe v3 (same design as frost/probes/qwendrive_temperature_probe_v3.py) on the Thor with the release bf16 expert.

Pipeline = the best-fidelity Thor recipe ("trt40_fp32_gd" family): HF eager vision tower, Triton/FLA Gated-DeltaNet prefill, SDPA full attention
with the enable_gqa patch, CUDA-graph greedy reasoning decode on a StaticCache. Differences from trt40_fp32_gd: the planning expert is NOT a TensorRT
fp32 engine; it is the PyTorch module in bf16 (release precision), run through the probe's own `plan` (copied verbatim from
frost/probes/qwendrive_temperature_probe_v2.py) for every config: release Euler 1/2/4/10 (ending at t = 0.9), Heun-32 curve at tau 1..0.3, Heun-64 at tau 1.
The Heun reference is therefore bf16 as well. --expert-fp32 reproduces the cluster's fp32 setting (used only for the 2-clip parity check).
Scenes: dumped by make_scenes.py (same builder as the v3 probe), 6 seeds 9201..9206, greedy reasoning shared per clip.
Per-clip checkpoints in OUT/parts/<name>.npz (+ .json); `--merge` / end of run writes OUT/plans.npz, reasoning.json, manifest.json in the v3 key format.
python probe_v3_bf16.py --model DIR --scenes DIR --out DIR [--range 0 400] [--expert-fp32] [--wait-scenes]"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import thor_patches  # noqa
from eager_bench import load_scene, sync

T_END = .9
TAUS = [1., .9, .8, .7, .6, .5, .4, .3]
CONFIGS = [('euler', k, 1.) for k in [1, 2, 4, 10]] + [('heun', 32, t) for t in TAUS] + [('heun', 64, 1.)]


def plan(model, cache, anchor, inputs, num_samples, method, steps, tau, fp32):
    """Verbatim from frost/probes/qwendrive_temperature_probe_v2.py (normalized trajectories from one sampler)."""
    from qwen_drive.modeling_qwen_drive import normalize_history, denormalize_trajectory
    expert = model.planning_expert; device = anchor.device; scale = model.trajectory_scale(device)
    tile = lambda x: x.repeat_interleave(num_samples, dim=0)
    history = tile(normalize_history(inputs['history'].float(), scale)); nav = tile(inputs['nav_command'])
    ego = tile(inputs['ego_status'].float()); position = anchor.expand(-1, num_samples)
    if fp32: cache = [(k.float(), v.float()) for k, v in cache]
    noise = tau * model.config.noise_init_std * model._initial_noise(num_samples, model.config.num_future_points, model.config.noise_seed, device)
    if method == 'euler':
        out = expert.sample(scene_cache=cache, position_anchor=position, history=history, history_velocity=tile(inputs['history_velocity'].float()),
            history_acceleration=tile(inputs['history_acceleration'].float()), nav_command=nav, ego_status=ego, noise=noise, num_steps=steps,
            min_one_minus_t=model.config.min_one_minus_t)
        return denormalize_trajectory(out, scale).cpu().numpy()
    queries = expert.encode_history(history, nav, tile(inputs['history_velocity'].float()), tile(inputs['history_acceleration'].float()))
    endpoint = lambda x, t: expert.predict_endpoint(x, torch.full((x.shape[0],), t, device=device, dtype=torch.float32), queries, cache, position, nav, ego)
    velocity = lambda x, t: (endpoint(x, t) - x) / (1. - t)
    x = noise.float(); h = T_END / steps
    for i in range(steps):
        t = i * h; v = velocity(x, t); x = x + .5 * h * (v + velocity(x + h * v, t + h))
    return denormalize_trajectory(endpoint(x, T_END), scale).cpu().numpy()


def patch_expert_attention():
    """The expert's SDPA branch passes enable_gqa=True, which on this torch build takes the same ~15x slower kernel as in thor_patches.py.
    Equivalent attention with the 4 KV heads repeated (repeat_interleave = enable_gqa's grouping) so the efficient kernel is used.
    Returns the original function so the check can compare the two."""
    import torch.nn.functional as F
    from qwen_drive.planning_expert import PlanningExpertLayer
    orig = PlanningExpertLayer._attend

    def _attend(self, query, key, value):
        if self.attn_implementation == 'flash_attention_2': return orig(self, query, key, value)
        g = self.num_heads // self.num_kv_heads
        if g > 1: key, value = key.repeat_interleave(g, dim=2), value.repeat_interleave(g, dim=2)
        return F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
    PlanningExpertLayer._attend = _attend
    return orig


def merge(out, names, manifest=None):
    plans, traces = {}, {}
    for n in names:
        f = out / 'parts' / f'{n}.npz'
        if not f.exists(): continue
        with np.load(f) as z: plans.update({k: z[k] for k in z.files})
        traces[n] = json.loads((out / 'parts' / f'{n}.json').read_text())['reasoning']
    np.savez_compressed(out / 'plans.npz', **plans); (out / 'reasoning.json').write_text(json.dumps(traces, indent=1))
    return len(traces)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--model', required=True); p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--range', type=int, nargs=2, default=[0, 400]); p.add_argument('--samples', type=int, default=6)
    p.add_argument('--seed0', type=int, default=9201); p.add_argument('--expert-fp32', action='store_true'); p.add_argument('--wait-scenes', action='store_true')
    p.add_argument('--check-release', action='store_true', help='also compare plan(euler10) with model._plan_from_cache on each clip (structural parity)')
    p.add_argument('--merge-only', action='store_true'); p.add_argument('--no-gqa-patch', action='store_true')
    a = p.parse_args(); (a.out / 'parts').mkdir(parents=True, exist_ok=True)
    names = [f'context_{i:03d}' for i in range(*a.range)]
    if a.merge_only: print('merged', merge(a.out, names)); return
    source = Path(__file__).read_bytes(); (a.out / 'source_snapshot.py').write_bytes(source)
    from qwen_drive import QwenDriveForPlanning
    model = QwenDriveForPlanning.from_pretrained(a.model, planner=str(Path(a.model) / 'planner-rl'), dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda:0').eval()
    model.config.noise_seed = a.seed0
    if a.expert_fp32:
        model.planning_expert.float()
    for m in model.planning_expert.modules():
        if getattr(m, 'attn_implementation', None) == 'flash_attention_2': m.attn_implementation = 'sdpa'
    orig_attend = None if a.no_gqa_patch else patch_expert_attention()
    expert_dtype = str(next(model.planning_expert.parameters()).dtype)
    from graph_decode import GraphDecoder
    scene_path = lambda n: a.scenes / f'{n}.npz'
    while not scene_path(names[0]).exists(): time.sleep(5)
    gd = GraphDecoder(model); first = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in model.processor(load_scene(scene_path(names[0])), with_reasoning=True, device='cpu').items()}
    gd.capture(first); print('decode graph captured; gpu GB', torch.cuda.memory_allocated() / 1e9, 'expert dtype', expert_dtype, flush=True)
    seeds = [a.seed0 + k for k in range(a.samples)]; rows = []
    for k, name in enumerate(names):
        if (a.out / 'parts' / f'{name}.npz').exists(): continue
        while not scene_path(name).exists():
            if not a.wait_scenes: raise SystemExit(f'missing {scene_path(name)}')
            time.sleep(10)
        time.sleep(1 if a.wait_scenes else 0)
        z = np.load(scene_path(name)); future = z['future']
        t0 = sync(); inputs = model.processor(load_scene(scene_path(name)), with_reasoning=True, device='cpu')
        inputs = {k_: v.to(model.device) if torch.is_tensor(v) else v for k_, v in inputs.items()}; t1 = sync()
        out, rel = {}, None
        with torch.no_grad():
            cache, anchor, text, ntok = gd.run(inputs, model.config.max_reasoning_tokens); t2 = sync()
            for method, steps, tau in CONFIGS:
                for i, traj in enumerate(plan(model, cache, anchor, inputs, a.samples, method, steps, tau, a.expert_fp32)):
                    out[f'{name}_s{seeds[i]}_{method}{steps}_t{tau}'] = traj
            t3 = sync()
            if a.check_release:
                from qwen_drive.planning_expert import PlanningExpertLayer as _L
                patched = _L._attend
                if orig_attend is not None: _L._attend = orig_attend          # release attention (enable_gqa=True) for the reference
                ref = model._plan_from_cache(cache, anchor, inputs, num_samples=a.samples, num_steps=10, seed=a.seed0)
                _L._attend = patched
                rel = float(max(np.abs(np.asarray(ref[i]) - out[f'{name}_s{seeds[i]}_euler10_t1.0']).max() for i in range(a.samples)))
        out[f'{name}_human'] = future
        bad = [kk for kk, v in out.items() if not np.isfinite(v).all()]
        row = dict(name=name, host_h2d_s=t1 - t0, vlm_prefill_decode_s=t2 - t1, probe_expert_s=t3 - t2, clip_s=t3 - t0, reasoning_tokens=ntok, nonfinite=len(bad),
                   n_plans=len(out), release_vs_probe_euler10_maxabs=rel, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        np.savez_compressed(a.out / 'parts' / f'{name}.tmp.npz', **out); os.replace(a.out / 'parts' / f'{name}.tmp.npz', a.out / 'parts' / f'{name}.npz.new')
        (a.out / 'parts' / f'{name}.json').write_text(json.dumps(dict(reasoning=text, row=row)))
        os.replace(a.out / 'parts' / f'{name}.npz.new', a.out / 'parts' / f'{name}.npz')   # the .npz appears last: its existence marks a finished clip
        print(json.dumps(row), flush=True); rows.append(row)
        if (k + 1) % 20 == 0: merge(a.out, names)
    n = merge(a.out, names)
    manifest = dict(scope=__doc__, model=a.model, planner='planner-rl', mode='reasoning_planning', expert_fp32=a.expert_fp32, noise_init_std=model.config.noise_init_std,
        min_one_minus_t=model.config.min_one_minus_t, t_end=T_END, range=a.range, seeds=seeds, configs=CONFIGS, reference='heun32_t1.0', command_rule='simwam',
        source_sha256=hashlib.sha256(source).hexdigest(), clips_done=n,
        precision=dict(platform='NVIDIA DRIVE AGX Thor (Blackwell, unified memory), torch 2.8.0+thor101, transformers 5.14.1', vision_tower='HF eager, bf16 weights/activations (no TensorRT)',
            vlm_lm='bf16, SDPA full attention (enable_gqa patch), Gated-DeltaNet via Triton/FLA kernels, prefill + CUDA-graph greedy decode on a bf16 StaticCache',
            scene_kv_cache='bf16 (float32 only when --expert-fp32)', expert_weights_dtype=expert_dtype, expert_attention='SDPA with KV heads repeated (equivalent to enable_gqa=True, efficient kernel) unless --no-gqa-patch; the release uses flash_attention_2, not available on Thor',
            sampler='probe_v2.plan verbatim: noise float32; Euler via expert.sample (release), Heun via expert.predict_endpoint with x in float32',
            reference_curve='Heun-32/64 on the same bf16 expert (release precision end to end)'),
        limitations=['Trajectories are (x, y, heading) at 10 Hz over 5 s; first 40 points = 4 s common horizon', 'command derived from the logged future by the SimWAM rule',
                     'VLM numerics differ slightly from the cluster run (different GPU/kernels): reasoning text can differ on a few clips'])
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)); print('done', n)


if __name__ == '__main__':
    main()
