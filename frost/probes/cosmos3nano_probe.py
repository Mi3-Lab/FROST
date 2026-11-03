"""Cosmos 3 Nano official AV-policy path (diffusers Cosmos3OmniPipeline, mode='policy', domain 'av', 60 actions at 10 Hz) on
the cached PhysicalAI-AV held-out contexts: smoke test, reference-convergence study and sampler pilot.

Official inputs (Cosmos 3 Nano model card, AV policy mode): one conditioning frame (front wide camera at t0, full
1080x1920 given to the pipeline, which fits it into the tier-480 canvas), a task prompt, view_point 'ego_view'. The diffusers
CosmosActionCondition has no ego-state field (the NIM 'use_state' option is not in this pipeline), so no state is passed.
Prompt modes (protocols/fair_comparison.md rule 2): 'neutral' = the card's "You are an autonomous vehicle planning system."
(no maneuver information, matching Alpamayo without navigation); 'maneuver' = that sentence plus an instruction derived from the
logged future by SimWAM's command rule (0.31 rad on the last segment of the 4 s future).
Sampler: the official UniPC (flow_shift 10, guidance 1.0). Reference/other solvers integrate the same joint (video, action) flow
ODE x_sigma = sigma z + (1 - sigma) x0 on the official shifted sigma grid sigma = s u / (1 + (s - 1) u), u uniform, starting at
tau * z (tau scales all noise, video and action) at the top of the grid; the model timestep is sigma * 1000.
Output plan: [60, 2] (x forward, y left) in the ego frame at t0 from the integrated pose deltas; the camera-frame axes
(z forward, x right) are mapped x = z_model, y = -x_model and the sign is checked against ego_future_xyz in the analysis.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

SHIFT = 10.0
FRONT_WIDE, T_NOW = 1, -1
SIMWAM_TH = 0.31
BASE_PROMPT = 'You are an autonomous vehicle planning system.'
MANEUVER = {'left': 'Turn left onto the road and continue driving in the leftmost lane.',
            'right': 'Turn right onto the road and continue driving in the rightmost lane.',
            'straight': 'Continue driving straight in the current lane.'}


def rot6d_to_so3(rot6d):
    v = np.asarray(rot6d, dtype=np.float32); c0, c1 = v[:3], v[3:]
    u, _, vt = np.linalg.svd(np.stack((c0, c1, np.cross(c0, c1)), axis=-1)); r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1; r = u @ vt
    return r.astype(np.float32)


def plan_xy(action):
    """[60, 9] relative pose deltas (3 translation + 6D rotation) -> [60, 2] waypoints (x forward, y left)."""
    cur = np.eye(4, dtype=np.float32); pts = []
    for row in np.asarray(action, np.float32):
        d = np.eye(4, dtype=np.float32); d[:3, :3] = rot6d_to_so3(row[3:9]); d[:3, 3] = row[:3]
        cur = cur @ d; pts.append(cur[:3, 3].copy())
    P = np.stack(pts)
    return np.stack([P[:, 2], -P[:, 0]], 1).astype(np.float32)


def simwam_command(future_xy):
    xy = future_xy[:40]
    h = float(np.arctan2(xy[-1, 1] - xy[-2, 1], xy[-1, 0] - xy[-2, 0]))
    return 'left' if h >= SIMWAM_TH else 'right' if h <= -SIMWAM_TH else 'straight'


class SolverSched:
    """Euler / Heun on dx/dsigma = v, one pipeline step() call per model evaluation (the pipeline deep-copies the scheduler for
    the action stream, so state is per instance)."""
    order = 1
    class config: num_train_timesteps = 1000
    def __init__(self, kind, grid):
        self.kind, self.grid = kind, np.asarray(grid, np.float64)
        self._eval = self.grid[:-1] if kind == 'euler' else np.stack([self.grid[:-1], self.grid[1:]], 1).reshape(-1)
        self.n = 0; self.timesteps = None; self.sigmas = None
    def set_timesteps(self, num_inference_steps, device=None, sigmas=None):
        assert num_inference_steps == len(self._eval), (num_inference_steps, len(self._eval))
        self.timesteps = torch.tensor(self._eval * 1000.0, dtype=torch.float32, device=device)
        self.sigmas = np.concatenate([self._eval, [0.0]]).astype(np.float32); self.n = 0
    def step(self, v, t, x, return_dict=False):
        k = self.n; self.n += 1
        if self.kind == 'euler':
            return (x + (self.grid[k + 1] - self.grid[k]) * v,)
        s, j = divmod(k, 2); d = self.grid[s + 1] - self.grid[s]
        if j == 0:
            self._x0, self._v1 = x, v
            return (x + d * v,)
        return (self._x0 + d * 0.5 * (self._v1 + v),)
    def nfe(self): return len(self._eval)


def shifted_grid(N, top=1 - 1e-3):
    u = np.linspace(1.0, 0.0, N + 1)
    return top * SHIFT * u / (1 + (SHIFT - 1) * u)


class Runner:
    def __init__(self, model_dir, tier=480):
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        t = time.time()
        self.pipe = Cosmos3OmniPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, enable_safety_checker=False)
        self.pipe.to('cuda'); self.tier = tier
        self.unipc = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=SHIFT, use_karras_sigmas=False)
        self.load_s = time.time() - t; self.tau = 1.0
        orig = self.pipe.prepare_latents
        def wrapped(*a, **k):
            out = list(orig(*a, **k))
            if self.tau != 1.0:
                lat, act, mv, ma = out[0], out[2], out[5], out[7]
                out[0] = mv * lat + (1 - mv) * self.tau * lat
                out[2] = ma * act + (1 - ma) * self.tau * act
            return tuple(out)
        self.pipe.prepare_latents = wrapped

    def solve(self, image, prompt, seed, kind, N, tau=1.0):
        """kind unipc|euler|heun. Returns (action [60, 9], seconds, nfe)."""
        from diffusers import CosmosActionCondition
        if kind == 'unipc':
            sched, nfe, n_arg = self.unipc, N, N
        else:
            sched = SolverSched(kind, shifted_grid(N)); nfe = n_arg = sched.nfe()
        object.__setattr__(self.pipe, 'scheduler', sched); self.tau = tau
        cond = CosmosActionCondition(mode='policy', chunk_size=60, domain_name='av', resolution_tier=self.tier, image=image, view_point='ego_view')
        gen = torch.Generator(device='cuda').manual_seed(seed)
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            r = self.pipe(prompt=prompt, action=cond, fps=10, num_inference_steps=n_arg, guidance_scale=1.0, generator=gen,
                          use_system_prompt=False, output_type='latent', enable_safety_check=False)
        torch.cuda.synchronize(); dt = time.time() - t; self.tau = 1.0
        a = r.action[0].float().numpy()
        assert a.shape == (60, 9) and np.isfinite(a).all(), a.shape
        return a, dt, nfe


def load_context(data, name):
    with np.load(data / f'{name}.npz') as z:
        image = Image.fromarray(z['image_frames'][FRONT_WIDE, T_NOW]); fut = z['ego_future_xyz'][:, :2].astype(np.float32)
    return image, fut


def prompt_for(mode, fut):
    return BASE_PROMPT if mode == 'neutral' else f'{BASE_PROMPT} {MANEUVER[simwam_command(fut)]}'


def pilot_configs(ref_n):
    c = [('unipc30_t1.0', 'unipc', 30, 1.0), ('unipc8_t1.0', 'unipc', 8, 1.0), ('unipc4_t1.0', 'unipc', 4, 1.0),
         ('euler4_t1.0', 'euler', 4, 1.0), ('euler10_t1.0', 'euler', 10, 1.0)]
    return c + [(f'heun{ref_n}_t{t}', 'heun', ref_n, t) for t in [1.0, .9, .8, .7, .6, .5, .4]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['smoke', 'conv', 'pilot', 'diag', 'ext'])
    ap.add_argument('--model', required=True); ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--clips', type=int, nargs='*', default=[0, 1, 2, 3, 4])
    ap.add_argument('--seeds', type=int, nargs='*', default=[9201])
    ap.add_argument('--prompt-mode', choices=['neutral', 'maneuver'], default='neutral')
    ap.add_argument('--taus', type=float, nargs='*', default=[1.0, 0.8, 0.6])
    ap.add_argument('--heun-n', type=int, nargs='*', default=[32, 64, 128, 256])
    ap.add_argument('--unipc-steps', type=int, nargs='*', default=[30, 8, 4])
    ap.add_argument('--max-nfe', type=int, default=1500); ap.add_argument('--ref-n', type=int, default=64); ap.add_argument('--tier', type=int, default=480)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes(); (a.out / 'source_snapshot.py').write_bytes(source)
    rn = Runner(a.model, a.tier); print('loaded', rn.load_s, flush=True)
    (a.out / 'parts').mkdir(exist_ok=True); tl = (a.out / 'timing.jsonl').open('a')
    names = [f'context_{c:03d}' for c in a.clips]

    def run(name, image, prompt, seed, kind, N, tau):
        act, dt, nfe = rn.solve(image, prompt, seed, kind, N, tau)
        tl.write(json.dumps(dict(name=name, seed=seed, kind=kind, N=N, tau=tau, seconds=dt, nfe=nfe, peak_gb=torch.cuda.max_memory_allocated() / 1e9)) + '\n'); tl.flush()
        return act, nfe

    if a.mode == 'smoke':
        res = {}
        for name in names[:2]:
            image, fut = load_context(a.data, name); prompt = prompt_for(a.prompt_mode, fut)
            for steps in a.unipc_steps:
                act, nfe = run(name, image, prompt, a.seeds[0], 'unipc', steps, 1.0); p = plan_xy(act)
                print(name, 'unipc', steps, 'plan end', p[-1], 'gt end', fut[59], flush=True)
                res[f'{name}_unipc{steps}'] = p; res[f'{name}_gt'] = fut
        np.savez(a.out / 'smoke_plans.npz', **res)
    elif a.mode == 'diag':
        # numerical diagnostics at tau = 1: run-to-run repeatability of identical solves and sensitivity to a 0.1% change of the noise scale
        for name in names:
            image, fut = load_context(a.data, name); prompt = prompt_for(a.prompt_mode, fut); res = {}
            for kind, N in [('unipc', 30), ('heun', 32), ('heun', 64)]:
                for rep in range(2):
                    act, _ = run(name, image, prompt, a.seeds[0], kind, N, 1.0); res[f'{kind}{N}_rep{rep}'] = plan_xy(act)
                act, _ = run(name, image, prompt, a.seeds[0], kind, N, 1.001); res[f'{kind}{N}_tau1.001'] = plan_xy(act)
            for k, v in res.items():
                base = res[k.split('_')[0] + '_rep0']
                print(name, k, 'max|d| vs rep0 %.4f' % np.abs(v - base).max(), flush=True)
            np.savez(a.out / 'parts' / f'{name}_diag.npz', **res, gt=fut)
    elif a.mode == 'ext':
        # convergence extension: Heun-512 and adaptive Dormand-Prince 5(4) (rtol = atol 1e-5, 1e-7, step control on the action channels), tau scaling as in conv
        from frost.probes.adaptive_ref import AdaptiveController
        from diffusers import CosmosActionCondition
        for name in names:
            image, fut = load_context(a.data, name); prompt = prompt_for(a.prompt_mode, fut)
            for tau in a.taus:
                f = a.out / 'parts' / f'{name}_t{tau}_ext.npz'
                if f.exists(): continue
                plans, info = {}, {}
                for tol in (1e-5, 1e-7):
                    ctl = AdaptiveController(tol, tol, a.max_nfe)
                    object.__setattr__(rn.pipe, 'scheduler', ctl); rn.tau = tau
                    cond = CosmosActionCondition(mode='policy', chunk_size=60, domain_name='av', resolution_tier=rn.tier, image=image, view_point='ego_view')
                    gen = torch.Generator(device='cuda').manual_seed(a.seeds[0]); t0 = time.time()
                    with torch.inference_mode():
                        r = rn.pipe(prompt=prompt, action=cond, fps=10, num_inference_steps=a.max_nfe, guidance_scale=1.0, generator=gen,
                                    use_system_prompt=False, output_type='latent', enable_safety_check=False)
                    rn.tau = 1.0; torch.cuda.synchronize()
                    plans[f'rk45_{tol:g}'] = plan_xy(r.action[0].float().numpy()); info[f'rk45_{tol:g}'] = dict(ctl.info, seconds=time.time() - t0)
                    print(name, 'tau', tau, f'rk45 {tol:g}', ctl.info, flush=True)
                act, _ = run(name, image, prompt, a.seeds[0], 'heun', 512, tau); plans['heun512'] = plan_xy(act)
                for k in [k for k in plans if k.startswith('rk45')]: print(name, 'tau', tau, k, 'max|d| vs heun512 %.3f' % np.abs(plans[k] - plans['heun512']).max(), flush=True)
                np.savez(f.with_suffix('.tmp.npz'), **plans, gt=fut, info=json.dumps(info)); os.replace(f.with_suffix('.tmp.npz'), f); print('done', name, tau, time.strftime('%H:%M:%S'), flush=True)
    elif a.mode == 'conv':
        for name in names:
            image, fut = load_context(a.data, name); prompt = prompt_for(a.prompt_mode, fut)
            for tau in a.taus:
                f = a.out / 'parts' / f'{name}_t{tau}.npz'
                if f.exists(): continue
                plans, acts = {}, {}
                for N in a.heun_n:
                    act, _ = run(name, image, prompt, a.seeds[0], 'heun', N, tau); plans[f'heun{N}'] = plan_xy(act); acts[f'heun{N}'] = act
                    print(name, 'tau', tau, f'heun{N}', 'max|d| vs prev', flush=True)
                    np.savez(f.with_suffix('.partial.npz'), **plans, **{'act_' + k: v for k, v in acts.items()}, gt=fut)
                if tau == 1.0:
                    for s in a.unipc_steps:
                        act, _ = run(name, image, prompt, a.seeds[0], 'unipc', s, 1.0); plans[f'unipc{s}'] = plan_xy(act); acts[f'unipc{s}'] = act
                np.savez(f.with_suffix('.tmp.npz'), **plans, **{'act_' + k: v for k, v in acts.items()}, gt=fut); os.replace(f.with_suffix('.tmp.npz'), f)
                print('done', name, tau, time.strftime('%H:%M:%S'), flush=True)
    else:
        cf = pilot_configs(a.ref_n)
        for name in names:
            f = a.out / 'parts' / f'{name}.npz'
            if f.exists(): continue
            image, fut = load_context(a.data, name); prompt = prompt_for(a.prompt_mode, fut); plans = {}
            for seed in a.seeds:
                for cname, kind, N, tau in cf:
                    act, _ = run(name, image, prompt, seed, kind, N, tau); plans[f'{name}_s{seed}_{cname}'] = plan_xy(act)
            plans[f'{name}_human'] = fut
            np.savez(f.with_suffix('.tmp.npz'), **plans); os.replace(f.with_suffix('.tmp.npz'), f); print('done', name, time.strftime('%H:%M:%S'), flush=True)
    (a.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'mode': a.mode, 'model': a.model, 'tier': a.tier, 'prompt_mode': a.prompt_mode,
        'prompt': BASE_PROMPT, 'maneuver_sentences': MANEUVER, 'maneuver_rule': f'simwam, {SIMWAM_TH} rad, last segment of 4 s future',
        'seeds': a.seeds, 'taus': a.taus, 'heun_n': a.heun_n, 'flow_shift': SHIFT, 'guidance_scale': 1.0, 'clips': a.clips,
        'source_sha256': hashlib.sha256(source).hexdigest(),
        'limitations': ['no ego-state input (not in the diffusers CosmosActionCondition)', 'maneuver sentences other than "right" are mirrored/own wording',
                        'tau scales the initial (video and action) noise at the top of the official grid', 'one front frame, no history']}, indent=2))


if __name__ == '__main__':
    main()
