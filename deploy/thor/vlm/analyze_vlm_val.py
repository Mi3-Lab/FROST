"""Summarise validate_vlm.py outputs per precision (workstation or board; numpy only). Exploratory.

  python analyze_vlm_val.py --val DIR/val40_bf16.json DIR/val40_fp8.json DIR/val40_nvfp4.json \
      --ref DIR/ref_plans_bf16.npz DIR/ref_plans_fp8.npz DIR/ref_plans_nvfp4.npz --out summary.json

Per precision, over clips (each clip = one unit; CIs are bootstraps over clips, 2000 resamples):
  reasoning   : free-rollout text identical up to <|traj_future_start|>; forced per-step sampled-token agreement
  prefix KV   : relative L2 over the P cached rows (median, max over clips)
  dplan       : per clip the median over the 6 seeds of max_t |plan_compiled - plan_reference| (m), forced and free;
                reported as median / p90 / max over clips
  spread ratio: the tau-relevant check. Seed spread of a clip's 6 plans = mean over seeds of max_t distance to the
                seed-mean plan. Ratio = spread(compiled prefix) / spread(reference prefix), same field engine and noise.
                A VLM that shifted the effective temperature would move this ratio away from 1 systematically
                (geometric mean over clips, 95% bootstrap CI).
"""
import argparse, json
from pathlib import Path
import numpy as np

SEEDS = list(range(9201, 9207))


def spread(p):                       # p [6, 64, 3]
    m = p.mean(0, keepdims=True)
    return float(np.linalg.norm(p - m, axis=-1).max(-1).mean())


def boot_ci(x, f, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    v = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarise(val, ref, plans):
    d = json.loads(Path(val).read_text()); R = np.load(ref); C = np.load(plans)
    rows = d['clips']; out = {'n_clips': len(rows), 'clips': [r['clip'] for r in rows]}
    free = [r['free'] for r in rows]; forced = [r['forced'] for r in rows if 'kv_rel_l2' in r.get('forced', {})]
    out['text_identical'] = f"{sum(f['text_match'] for f in free)}/{len(free)}"
    out['P_match'] = f"{sum(f['P_match'] for f in free)}/{len(free)}"
    out['token_agree_forced_mean'] = float(np.mean([f['token_agree'] for f in forced]))
    out['clips_all_tokens_agree_forced'] = f"{sum(f['token_agree'] == 1.0 for f in forced)}/{len(forced)}"
    kv = [f['kv_rel_l2'] for f in forced]
    out['kv_rel_l2'] = {'median': float(np.median(kv)), 'max': float(np.max(kv))}
    for mode in ('forced', 'free'):
        for key in ('euler10', 'euler4'):
            v = [r[mode]['dplan_m'][key]['median'] for r in rows if key in r.get(mode, {}).get('dplan_m', {})]
            if v: out[f'dplan_{mode}_{key}_m'] = {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)), 'max': float(np.max(v)),
                                                   'median_ci95': boot_ci(v, np.median)}
            ratios = []
            for r in rows:
                k = f"{r['clip']}_{mode}_{key}"
                if k not in C.files: continue
                rp = np.stack([R[f"{r['clip']}_s{s}_{key}_t1.0"] for s in SEEDS])
                ratios.append(spread(C[k]) / spread(rp))
            if ratios:
                lr = np.log(ratios)
                out[f'spread_ratio_{mode}_{key}'] = {'geomean': float(np.exp(lr.mean())), 'ci95': [float(np.exp(b)) for b in boot_ci(lr, np.mean)],
                                                     'min': float(min(ratios)), 'max': float(max(ratios))}
    t = [f['timing'] for f in free]
    out['timing_ms_median'] = {k: float(np.median([x[k] for x in t])) for k in ('vision_ms', 'prefill_ms', 'decode_ms_per_token', 'decode_ms', 'total_vlm_ms')}
    out['reasoning_tokens_median'] = float(np.median([x['decode_steps'] + 1 for x in t]))
    out['text_mismatch_examples'] = [{'clip': r['clip'], 'ref': r['ref_text'].split('<|traj_future_start|>')[0][:160],
                                      'compiled': r['free']['text'].split('<|traj_future_start|>')[0][:160]} for r in rows if not r['free']['text_match']][:6]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val', nargs='+', required=True); ap.add_argument('--ref', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for v, r in zip(a.val, a.ref):
        name = Path(v).stem
        res[name] = summarise(v, r, str(Path(v).with_suffix('.plans.npz')))
        s = res[name]
        print(name, s['text_identical'], 'kv', s['kv_rel_l2'], 'E10 forced', s.get('dplan_forced_euler10_m'), 'E10 free', s.get('dplan_free_euler10_m'),
              'spread E10 free', s.get('spread_ratio_free_euler10'), s['timing_ms_median'])
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
