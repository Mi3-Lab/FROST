"""Qwen-Drive reference-convergence diagnostic (no tau_eff, residual or flip statistics): Heun-32/64/128/256/512 on [0, 0.9] (landing on the
endpoint predicted at t = 0.9, as the v2/v3 reference) and adaptive RK45 on the same ODE at rtol = atol 1e-5 and 1e-7 (NFE budget), tau 1 and 0.6,
seeds 9201..9206 (noise_seed = 9201), first clips of the held-out v3 list, reasoning_planning with fp32 expert, SimWAM command rule."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import frost.probes.qwendrive_temperature_probe_v2 as v2
from frost.probes.qwendrive_temperature_probe_v3 import build_scene
from frost.probes.adaptive_ref import rk45
from frost.core.decisions import decisions


def setup(model, cache, anchor, inputs, S, tau):
    from qwen_drive.modeling_qwen_drive import normalize_history, denormalize_trajectory
    expert = model.planning_expert; device = anchor.device; scale = model.trajectory_scale(device)
    tile = lambda x: x.repeat_interleave(S, dim=0)
    history = tile(normalize_history(inputs['history'].float(), scale)); nav = tile(inputs['nav_command']); ego = tile(inputs['ego_status'].float())
    position = anchor.expand(-1, S); cache = [(k.float(), v.float()) for k, v in cache]
    noise = tau * model.config.noise_init_std * model._initial_noise(S, model.config.num_future_points, model.config.noise_seed, device)
    queries = expert.encode_history(history, nav, tile(inputs['history_velocity'].float()), tile(inputs['history_acceleration'].float()))
    endpoint = lambda x, t: expert.predict_endpoint(x, torch.full((x.shape[0],), t, device=device, dtype=torch.float32), queries, cache, position, nav, ego)
    vel = lambda x, t: (endpoint(x, t) - x) / (1. - t)
    return noise.float(), endpoint, vel, lambda y: denormalize_trajectory(y, scale).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--records', type=Path, required=True); p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--range', type=int, nargs=2, default=[0, 20]); p.add_argument('--max-nfe', type=int, default=3000)
    p.add_argument('--taus', type=float, nargs='+', default=[1., .6]); p.add_argument('--heun', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import physical_ai_av
    from qwen_drive import QwenDriveForPlanning
    records = sorted((r for r in json.loads(a.records.read_text()) if 'error' not in r), key=lambda r: r['name'])[a.range[0]:a.range[1]]
    model = QwenDriveForPlanning.from_pretrained(a.model, planner=str(Path(a.model) / 'planner-rl'), dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda().eval()
    model.config.noise_seed = 9201; model.planning_expert.float()
    for m in model.planning_expert.modules():
        if getattr(m, 'attn_implementation', None) == 'flash_attention_2': m.attn_implementation = 'sdpa'
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(); S = 6; rows = []; T = v2.T_END
    for rec in records:
        scene, future = build_scene(avdi, rec['clip_id'], rec['t0_us'], 'simwam')
        inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in model.processor(scene, with_reasoning=True, device='cpu').items()}
        with torch.no_grad():
            cache, anchor, _ = model._prefill_with_reasoning(inputs, model.config.max_reasoning_tokens)
            for tau in a.taus:
                noise, endpoint, vel, den = setup(model, cache, anchor, inputs, S, tau); pl = {}; info = {}
                for N in a.heun:
                    x = noise.clone(); h = T / N
                    for i in range(N):
                        t = i * h; v = vel(x, t); x = x + .5 * h * (v + vel(x + h * v, t + h))
                    pl[f'heun{N}'] = den(endpoint(x, T))
                for tol in (1e-5, 1e-7):
                    shape = noise.shape
                    def f(t, y):
                        x = torch.from_numpy(np.asarray(y, np.float32)).reshape(shape).to(noise.device)
                        return vel(x, float(t)).float().cpu().numpy().astype(np.float64).ravel()
                    # rk45 integrates from sigma0 down to 0; here time runs 0 -> T, so integrate in s = T - t with dy/ds = -v
                    res = rk45(lambda s, y: -f(T - s, y), noise.cpu().numpy(), T, tol, tol, a.max_nfe); info[f'rk45_{tol:g}'] = {k: res[k] for k in ('nfe', 'ok', 'steps', 'reason')}
                    if res['y'] is not None:
                        pl[f'rk45_{tol:g}'] = den(endpoint(torch.from_numpy(res['y'].astype(np.float32)).reshape(shape).to(noise.device), T))
                speed = rec['speed_mps']; ref = pl[f'heun{a.heun[-1]}']
                for s in range(S):
                    cls = {k: (lambda q: (q['longitudinal'], q['lateral']))(decisions(v[s], speed)) for k, v in pl.items()}
                    d = lambda x, y: float(np.abs(x[s][:, :2] - y[s][:, :2]).max())
                    row = dict(clip=rec['name'], tau=tau, seed=9201 + s, speed=speed,
                               **{f'd{a.heun[i]}_{a.heun[i+1]}': d(pl[f'heun{a.heun[i]}'], pl[f'heun{a.heun[i+1]}']) for i in range(len(a.heun) - 1)},
                               dist_to_ref={k: d(v, ref) for k, v in pl.items()}, agree_ref={k: c == cls[f'heun{a.heun[-1]}'] for k, c in cls.items()},
                               adaptive=info, end_xy_ref=ref[s][-1, :2].tolist(), gt_end_xy=future[-1, :2].tolist())
                    rows.append(row)
                print(json.dumps({'clip': rec['name'], 'tau': tau, 'adaptive': info, 'd256_512_max': max(r['d256_512'] for r in rows if r['clip'] == rec['name'] and r['tau'] == tau)}), flush=True)
                (a.out / 'rows.json').write_text(json.dumps(rows))
        torch.cuda.empty_cache()
    (a.out / 'manifest.json').write_text(json.dumps({'scope': __doc__, 'heun': a.heun, 'range': a.range, 'taus': a.taus, 'max_nfe': a.max_nfe}, indent=1))


if __name__ == '__main__':
    main()
