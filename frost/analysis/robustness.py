"""EXPLORATORY / POST HOC robustness analyses of FROST (2026-11-02). NOT pre-registered.

Uses only existing plans (no new inference):
  Alpamayo-1.5/-R1/-2: the pooled 400 held-out probes (temperature_probe_{a15,r1,a2}_heldout_pooled400_v3_20261028),
    the Heun-3/5 controls of R1/A2 (heun_controls_{r1,a2}_heldout_v3_20261028; flips against that run's own Heun-32 tau=1,
    tau_eff on the main curve, as frost.analysis.heun_controls_analysis), the scaled-noise Euler-10 runs (A1.5 pooled probe;
    scaled_noise_{r1,a2}_combined_v1_20261031) and Heun-64/128 at tau=1 (convergence A18/A19/A20).
  Qwen-Drive-1.0 RL: the v3 held-out probe (qwendrive_v3_rl_heldout_20261101_c0..c3), native 5 s horizon.

Estimator and conventions are the paper's: per-clip statistics averaged over clips; tau_eff by linear interpolation of the
matching statistic on the reference curve (Heun-32 from tau*z); residual = sampler minus the curve at tau_eff; 2000 clip
resamples re-solving tau_eff, drawn exactly as frost.analysis.paper_numbers (seed 20261026, the same draws for the three Alpamayo
models, i.e. joint) and frost.analysis.qwendrive_v3_analysis (seed 20261102); TOST at 90% with margins 0.05 m (ADE1) and
2 points (flips). The baseline reproduces paper_numbers / heun_controls_analysis / qwendrive_v3_analysis (asserted).
Decision classes: frost.core.decisions.decisions, evaluated here in vectorized form so the thresholds can be
perturbed; the vectorized classifier is asserted equal to decisions() on every plan at the original thresholds.

Questions (keys of summary.json):
  q1_matching   tau_eff and verdicts when matching by longitudinal-only, lateral-only, median pairwise spread, or the
                tau minimizing the energy distance between the sampler's plan set and the reference's (parabolic vertex).
  q2_fraction   (effect - residual) / effect with bootstrap intervals; margins as a fraction of the sampler's effect.
  q3_noises     every subset of k in {3,4,5} of the 6 noises: tau_eff spread and verdict stability.
  q4_thresholds class thresholds scaled jointly by 0.8/0.9/1.1/1.2 and one at a time by 0.8/1.2.
  q5_heun       per-noise plan distance to the reference (Heun-3/5 vs E10 vs the cold reference), boundary proximity of
                flipped decisions, distance of Heun-5 from Heun-32/64 under the same noise.
  q6_planted    (a) reference at known tau with disjoint noises as the 'sampler'; (b) Euler-10 from tau'*z;
                (c) inventory of tau > 1 data.
  q7_e1         one-step Euler spread decomposed (identity: the exact one-step map has no spread).
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from frost.analysis import paper_numbers as pn
from frost.analysis import qwendrive_v3_analysis as qv3
from frost.analysis.qwendrive_analysis import TAUS as QTAUS, CURVE as QCURVE
from frost.core.decisions import decisions, DT, STOP_EXTENT_M, HEADING_DEG, OFFSET_M

ART = Path('artifacts')
DATA = ART / 'physicalai_heldout_contexts_v3_20261027'
PROBES = {'alpamayo15': 'temperature_probe_a15_heldout_pooled400_v3_20261028', 'alpamayo_r1': 'temperature_probe_r1_heldout_pooled400_v3_20261028',
          'alpamayo2': 'temperature_probe_a2_heldout_pooled400_v3_20261028'}
CONTROLS = {'alpamayo_r1': 'heun_controls_r1_heldout_v3_20261028', 'alpamayo2': 'heun_controls_a2_heldout_v3_20261028'}
SCALED = {'alpamayo15': PROBES['alpamayo15'], 'alpamayo_r1': 'scaled_noise_r1_combined_v1_20261031', 'alpamayo2': 'scaled_noise_a2_combined_v1_20261031'}
HEUN64 = {'alpamayo15': ['convergence_a18_a15_heldout_v3_20261029', 'convergence_a19_a15_heun64_heldout_v3_20261030'],
          'alpamayo_r1': ['convergence_a20_r1_heun64_heldout_v3_20261030'], 'alpamayo2': ['convergence_a20_a2_heun64_heldout_v3_20261030']}
QPROBES = [f'qwendrive_v3_rl_heldout_20261101_c{i}' for i in range(4)]
POOLED = ART / 'paper_numbers_heldout_pooled400_v2_20261028/summary.json'
HCTRL = {'alpamayo_r1': ART / 'heun_controls_analysis_r1_v1_20261028/summary.json', 'alpamayo2': ART / 'heun_controls_analysis_a2_v1_20261028/summary.json'}
QSUM = ART / 'qwendrive_v3_analysis_20261102/summary.json'
MARGINS = pn.MARGINS
THR0 = {'stop': STOP_EXTENT_M, 'band_frac': .15, 'band_floor': 1., 'heading': HEADING_DEG, 'offset': OFFSET_M}
STATS = ['spread', 'spread_long', 'spread_lat', 'spread_median', 'energy']


# ---------------------------------------------------------------- decisions (vectorized; asserted equal to decisions())
def features(P):
    """P (..., T, 2) -> extent, plan speed, tail heading (deg), final lateral offset, as in decisions()."""
    extent = np.linalg.norm(P, axis=-1).max(-1)
    speed = np.linalg.norm(np.diff(P[..., 19:30, :], axis=-2), axis=-1).mean(-1) / DT
    tail = P[..., -1, :] - P[..., -6, :]
    heading = np.where(np.linalg.norm(tail, axis=-1) > .2, np.degrees(np.arctan2(tail[..., 1], tail[..., 0])), 0.)
    return {'extent': extent, 'speed': speed, 'heading': heading, 'offset': P[..., -1, 1]}


def classify(F, v, thr=THR0):
    """Joint class code long*5+lat with LONG=[stop,decelerate,keep,accelerate], LAT=[left,shift_left,straight,shift_right,right]."""
    v = v.reshape((-1,) + (1,) * (F['extent'].ndim - 1)); band = np.maximum(thr['band_floor'], thr['band_frac'] * v)
    lon = np.where(F['extent'] <= thr['stop'], 0, np.where(F['speed'] < v - band, 1, np.where(F['speed'] > v + band, 3, 2)))
    lat = np.where(F['heading'] > thr['heading'], 0, np.where(F['heading'] < -thr['heading'], 4,
                   np.where(F['offset'] > thr['offset'], 1, np.where(F['offset'] < -thr['offset'], 3, 2))))
    return lon * 5 + lat


def margins(F, v):
    """Normalized distance to the nearest class boundary (decisions()' long_margin, lat_margin)."""
    v = v.reshape((-1,) + (1,) * (F['extent'].ndim - 1)); band = np.maximum(1., .15 * v)
    lm = np.minimum(np.abs(F['extent'] - STOP_EXTENT_M) / STOP_EXTENT_M, np.abs(np.abs(F['speed'] - v) - band) / band)
    am = np.minimum(np.abs(np.abs(F['heading']) - HEADING_DEG) / HEADING_DEG, np.abs(np.abs(F['offset']) - OFFSET_M) / OFFSET_M)
    return lm, am


LONG = ['stop', 'decelerate', 'keep', 'accelerate']; LAT = ['left', 'shift_left', 'straight', 'shift_right', 'right']


def check_classifier(P, v, full):
    """Assert the vectorized classifier and margins equal decisions() on every plan of P (N,S,T,3) at THR0."""
    F = features(P[..., :2]); c = classify(F, v); lm, am = margins(F, v)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            d = decisions(full[i, j], v[i])
            assert c[i, j] == LONG.index(d['longitudinal']) * 5 + LAT.index(d['lateral'])
            assert abs(lm[i, j] - d['long_margin']) < 1e-9 and abs(am[i, j] - d['lat_margin']) < 1e-9


# ---------------------------------------------------------------- loading
class Model:
    """Plans as arrays (N clips, S seeds, T, 2); `full` keeps (x,y,z/heading) for the decisions() check."""

    def __init__(self, key, names, speed, human, P, full, taus, curve, ref, seeds):
        self.key, self.names, self.speed, self.human, self.P, self.full = key, names, speed, human, P, full
        self.taus, self.curve, self.ref, self.seeds = np.array(taus), curve, ref, seeds
        self.samplers = {}  # name -> (X, R): sampler plans and the reference plans its flips/effect are measured against
        self.F = {}

    def feat(self, X):
        k = id(X)
        if k not in self.F: self.F[k] = features(X)
        return self.F[k]


def npz(path, names, seeds, cfgs, cols=2):
    with np.load(path / 'plans.npz') as z:
        return {c: np.stack([np.stack([z[f'{n}_s{s}_{c}'][:, :cols] for s in seeds]) for n in names]).astype(float) for c in cfgs}


def load_alpamayo(key, records):
    names = [f'context_{i:03d}' for i in range(400)]; path = ART / PROBES[key]
    seeds = json.loads((path / 'manifest.json').read_text())['seeds']
    with np.load(path / 'plans.npz') as z: cfgs = sorted({k.split('_', 3)[-1] for k in z.files if k.startswith('context_000_s9201_')})
    full = npz(path, names, seeds, cfgs, cols=3); P = {c: a[..., :2] for c, a in full.items()}
    human = np.stack([np.load(DATA / f'{n}.npz')['ego_future_xyz'][:, :2] for n in names]).astype(float)
    speed = np.array([records[n]['speed_mps'] for n in names], float)
    m = Model(key, names, speed, human, P, full, pn.TAUS, pn.CURVE, pn.REFERENCE, seeds)
    for c in ['euler1_t1.0', 'euler2_t1.0', 'euler4_t1.0', 'euler6_t1.0', 'euler10_t1.0', 'heun3_t1.0', 'heun5_t1.0']:
        if c in P: m.samplers[c] = (P[c], P[pn.REFERENCE])
    if key in CONTROLS:
        ctrl = npz(ART / CONTROLS[key], names, seeds, ['heun3_t1.0', 'heun5_t1.0', pn.REFERENCE], cols=3)
        m.ctrl_full = ctrl
        for c in ['heun3_t1.0', 'heun5_t1.0']: m.samplers[c] = (ctrl[c][..., :2], ctrl[pn.REFERENCE][..., :2])
    return m


def load_qwen(records):
    plans, _ = qv3.load([ART / p for p in QPROBES], records)
    names = sorted(k.rsplit('_human', 1)[0] for k in plans if k.endswith('_human'))
    seeds = sorted({int(k.split('_s')[1].split('_')[0]) for k in plans if not k.endswith('_human')})
    cfgs = sorted({k.split('_', 3)[3] for k in plans if not k.endswith('_human')})
    full = {c: np.stack([np.stack([plans[f'{n}_s{s}_{c}'] for s in seeds]) for n in names]).astype(float) for c in cfgs}
    P = {c: a[..., :2] for c, a in full.items()}
    human = np.stack([plans[f'{n}_human'][:, :2] for n in names]).astype(float)
    m = Model('qwendrive', names, np.array([records[n]['speed_mps'] for n in names], float), human, P, full, QTAUS, QCURVE, qv3.REFERENCE, seeds)
    for c in ['euler4_t1.0', 'euler10_t1.0']: m.samplers[c] = (P[c], P[qv3.REFERENCE])
    return m


# ---------------------------------------------------------------- per-clip statistics
def pair_d(X, idx):
    """Per-clip pairwise mean-over-time distances among seeds idx: full, |dx|, |dy| -> three (N, k, k) arrays."""
    X = X[:, idx]; d = X[:, :, None] - X[:, None]
    return np.linalg.norm(d, axis=-1).mean(-1), np.abs(d[..., 0]).mean(-1), np.abs(d[..., 1]).mean(-1)


def spreads(X, idx):
    D, Dx, Dy = pair_d(X, idx); iu = np.triu_indices(len(idx), 1)
    return {'spread': D[:, iu[0], iu[1]].mean(1), 'spread_long': Dx[:, iu[0], iu[1]].mean(1), 'spread_lat': Dy[:, iu[0], iu[1]].mean(1),
            'spread_median': np.median(D[:, iu[0], iu[1]], axis=1)}


def energy(X, Y, idx):
    """Per-clip energy distance between plan sets {X_i} and {Y_j}, seeds idx, distinct-noise pairs only (i != j) so that
    shared noise does not enter: 2 E|X-Y'| - E|X-X'| - E|Y-Y'|."""
    X, Y = X[:, idx], Y[:, idx]; off = ~np.eye(len(idx), dtype=bool)
    xy = np.linalg.norm(X[:, :, None] - Y[:, None], axis=-1).mean(-1)[:, off].mean(1)
    xx = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1).mean(-1)[:, off].mean(1)
    yy = np.linalg.norm(Y[:, :, None] - Y[:, None], axis=-1).mean(-1)[:, off].mean(1)
    return 2 * xy - xx - yy


def clip_stats(m, X, R, idx, thr=THR0, with_energy_to=None):
    s = spreads(X, idx)
    s['ade1'] = np.linalg.norm(X[:, idx] - m.human[:, None], axis=-1).mean(-1).mean(1)
    cx = classify(m.feat(X), m.speed, thr)[:, idx]; cr = classify(m.feat(R), m.speed, thr)[:, idx]
    s['flip'] = (cx != cr).mean(1)
    if with_energy_to is not None: s['energy'] = np.stack([energy(X, Y, idx) for Y in with_energy_to])
    return s


# ---------------------------------------------------------------- matching + bootstrap
def draws(m, boot):
    N = len(m.names)
    if m.key == 'qwendrive':
        rng = np.random.default_rng(qv3.SEED); D = [rng.integers(0, N, N) for _ in range(boot)]
    else:
        rng = np.random.default_rng(20261026); D = [rng.choice(N, N) for _ in range(boot)]  # == rng.choice(names, N), asserted in main
    W = np.ones((boot + 1, N))
    for b, d in enumerate(D): W[b + 1] = np.bincount(d, minlength=N)
    return W / N


def match(W, taus, curve, samp, stat, ref):
    """curve: per-config clip stats (list over the tau grid); samp, ref: clip stats of sampler and of the tau=1 reference.
    Returns per-draw tau_eff, in_range, matched curve values, sampler values and reference values (row 0 = point)."""
    met = ['ade1', 'flip']
    CM = {k: np.stack([W @ c[k] for c in curve], 1) for k in met}; SM = {k: W @ samp[k] for k in met}; RM = {k: W @ ref[k] for k in met}
    B = W.shape[0]; tau = np.empty(B); inr = np.empty(B, bool); at = {k: np.empty(B) for k in met}
    if stat == 'energy':
        E = W @ samp['energy'].T  # (B, K): mean energy distance to each grid config
        o = np.argsort(taus); t = taus[o]; E = E[:, o]; h = t[1] - t[0]
        for b in range(B):
            j = int(np.argmin(E[b]))
            if 0 < j < len(t) - 1:
                y0, y1, y2 = E[b, j - 1:j + 2]; den = y0 - 2 * y1 + y2
                tau[b] = t[j] + (.5 * h * (y0 - y2) / den if den > 0 else 0.); inr[b] = True
            else: tau[b] = t[j]; inr[b] = False
            for k in met: at[k][b] = np.interp(tau[b], t, CM[k][b, o])
    else:
        CS = np.stack([W @ c[stat] for c in curve], 1); S = W @ samp[stat]
        for b in range(B):
            o = np.argsort(CS[b]); tau[b] = np.interp(S[b], CS[b, o], taus[o]); inr[b] = CS[b].min() <= S[b] <= CS[b].max()
            for k in met: at[k][b] = np.interp(S[b], CS[b, o], CM[k][b, o])
    return {'tau': tau, 'in_range': inr, 'at': at, 'samp': SM, 'ref': RM}


def summarize(r, require='none'):
    q = lambda x, p: np.quantile(x[1:], p).tolist()
    e = {'tau_eff': float(r['tau'][0]), 'tau_eff_ci95': q(r['tau'], [.025, .975]), 'in_range': bool(r['in_range'][0]),
         'boot_in_range': float(r['in_range'][1:].mean())}
    ok = {'none': True, 'in_range': e['in_range'], 'interpretable': e['in_range'] and e['boot_in_range'] >= .95}[require]
    for k in ['ade1', 'flip']:
        res = r['samp'][k] - r['at'][k]; eff = r['samp'][k] - r['ref'][k]
        with np.errstate(divide='ignore', invalid='ignore'): frac = 1 - res / eff
        ci90 = q(res, [.05, .95])
        e[k] = {'sampler': float(r['samp'][k][0]), 'matched': float(r['at'][k][0]), 'residual': float(res[0]), 'residual_ci95': q(res, [.025, .975]),
                'residual_ci90': ci90, 'effect': float(eff[0]), 'effect_ci95': q(eff, [.025, .975]),
                'equivalent': bool(ok and -MARGINS[k] < ci90[0] and ci90[1] < MARGINS[k]),
                'fraction_reproduced': float(frac[0]) if abs(eff[0]) > 1e-9 else None,
                'fraction_reproduced_ci95': q(frac, [.025, .975]) if np.all(np.abs(eff) > 1e-9) else None,
                'margin_over_effect': float(MARGINS[k] / abs(eff[0])) if abs(eff[0]) > 1e-9 else None}
    return e


def require_rule(m, s):
    if m.key == 'qwendrive': return 'interpretable'
    return 'in_range' if s.startswith('heun') and m.key in CONTROLS else 'none'


def run(m, W, s, stat='spread', idx=None, thr=THR0):
    idx = list(range(len(m.seeds))) if idx is None else list(idx)
    X, R = m.samplers[s]; ref = m.P[m.ref]
    curve = [clip_stats(m, m.P[c], ref, idx, thr) for c in m.curve]
    samp = clip_stats(m, X, R, idx, thr, with_energy_to=[m.P[c] for c in m.curve] if stat == 'energy' else None)
    refs = clip_stats(m, R, R, idx, thr)
    return summarize(match(W, m.taus, curve, samp, stat, refs), require_rule(m, s))


def slim(e):
    return {'tau_eff': round(e['tau_eff'], 4), 'tau_eff_ci95': [round(x, 4) for x in e['tau_eff_ci95']], 'in_range': e['in_range'],
            **{k: {'residual': e[k]['residual'], 'ci90': e[k]['residual_ci90'], 'ci95': e[k]['residual_ci95'], 'equivalent': e[k]['equivalent']}
               for k in ['ade1', 'flip']}}


# ---------------------------------------------------------------- checks against the paper's artifacts
def check_baseline(models, W):
    pooled = json.loads(POOLED.read_text())['models']; q = json.loads(QSUM.read_text())['horizons']['native_5s']['samplers']; out = {}
    for m in models:
        for s in m.samplers:
            e = run(m, W[m.key], s)
            if m.key == 'qwendrive': ref = q[s]; tci = ref['tau_eff_ci95']; rr = {k: [ref[k]['residual'], *ref[k]['residual_ci95']] for k in ['ade1', 'flip']}; eq = {k: ref[k]['equivalent'] for k in ['ade1', 'flip']}
            elif s.startswith('heun') and m.key in HCTRL:
                ref = json.loads(HCTRL[m.key].read_text())['samplers'][s]; tci = ref['tau_eff_ci95']
                rr = {k: [ref[k]['residual'], *ref[k]['residual_ci95']] for k in ['ade1', 'flip']}; eq = {k: ref[k]['equivalent'] for k in ['ade1', 'flip']}
            else:
                ref = pooled[m.key][s]; tci = ref['tau_eff'][1:]; rr = {k: ref[k]['residual'] for k in ['ade1', 'flip']}
                eq = {k: ref[k].get('equivalent') for k in ['ade1', 'flip']}
            t0 = ref['tau_eff'][0] if isinstance(ref['tau_eff'], list) else ref['tau_eff']
            assert abs(e['tau_eff'] - t0) < 1e-6, (m.key, s)  # float64 here vs float32 sums in the stored analyses
            assert np.allclose(e['tau_eff_ci95'], tci, atol=1e-6), (m.key, s, e['tau_eff_ci95'], tci)
            for k in ['ade1', 'flip']:
                assert np.allclose([e[k]['residual'], *e[k]['residual_ci95']], rr[k], atol=1e-6), (m.key, s, k)
                assert eq[k] is None or eq[k] == e[k]['equivalent'], (m.key, s, k)
            out.setdefault(m.key, {})[s] = 'reproduced'
    return out


# ---------------------------------------------------------------- Q5: Heun
def per_noise_dist(A, B): return np.linalg.norm(A - B, axis=-1).mean(-1)  # (N, S)


def q5(m, extra):
    """Per-noise distances to the tau=1 reference of the same run, boundary proximity of flips, convergence of Heun-5."""
    out = {}; v = m.speed
    if m.key in CONTROLS:
        cref = m.ctrl_full[pn.REFERENCE][..., :2]
        same = np.abs(cref - m.P[pn.REFERENCE]).max((1, 2, 3)) < 1e-6  # clips whose control-run reference equals the main one
        out['clips_identical_reference'] = int(same.sum())
    else: cref = m.P[m.ref]; same = np.ones(len(m.names), bool)
    lm, am = margins(m.feat(cref), v); marg = np.minimum(lm, am); cls_ref = classify(m.feat(cref), v)
    rows = {}
    cand = {'euler10_t1.0': m.P['euler10_t1.0'], 'euler4_t1.0': m.P['euler4_t1.0'], 'heun32_t0.9': m.P['heun32_t0.9'], 'heun32_t0.8': m.P['heun32_t0.8'],
            'heun32_t0.6': m.P['heun32_t0.6'], 'heun3_t1.0': m.samplers['heun3_t1.0'][0], 'heun5_t1.0': m.samplers['heun5_t1.0'][0]}
    for name, X in cand.items():
        mask = same if not name.startswith('heun') or name.startswith('heun32') else np.ones(len(m.names), bool)
        if m.key in CONTROLS and name.startswith('heun') and not name.startswith('heun32'): R = cref
        else: R = m.P[m.ref]
        d = per_noise_dist(X, R)[mask]; c = classify(m.feat(X), v)[mask]; f = c != classify(m.feat(R), v)[mask]
        mg = marg[mask] if R is cref else np.minimum(*margins(m.feat(R), v))[mask]
        # mean shift (bias) vs spread change: distance between the 6-sample means and the spread ratio
        mean_shift = np.linalg.norm(X[mask].mean(1) - R[mask].mean(1), axis=-1).mean(-1)
        sp = spreads(X[mask], range(len(m.seeds)))['spread']; spr = spreads(R[mask], range(len(m.seeds)))['spread']
        F = m.feat(X); FR = m.feat(R)
        rows[name] = {'clips': int(mask.sum()), 'per_noise_dist_median_m': float(np.median(d)), 'per_noise_dist_mean_m': float(d.mean()),
                      'per_noise_dist_p90_m': float(np.quantile(d, .9)), 'per_noise_frac_above_1m': float(np.mean(d > 1.)),
                      'flips_with_dist_above_1m': float(np.mean(d[f] > 1.)) if f.any() else None,
                      'flip_rate': float(f.mean()), 'flips': int(f.sum()), 'flip_clips': int(f.any(1).sum()),
                      'ref_margin_median_flipped': float(np.median(mg[f])) if f.any() else None, 'ref_margin_median_all': float(np.median(mg)),
                      'flips_ref_margin_below_0.1': float(np.mean(mg[f] < .1)) if f.any() else None,
                      'flips_ref_margin_below_0.25': float(np.mean(mg[f] < .25)) if f.any() else None,
                      'all_ref_margin_below_0.25': float(np.mean(mg < .25)),
                      'flip_rate_given_ref_margin_above_0.25': float(f[mg >= .25].mean()),
                      'flip_rate_given_ref_margin_below_0.25': float(f[mg < .25].mean()),
                      'long_flip_rate': float((c // 5 != classify(FR, v)[mask] // 5).mean()), 'lat_flip_rate': float((c % 5 != classify(FR, v)[mask] % 5).mean()),
                      'mean_shift_m': float(mean_shift.mean()), 'spread_ratio': float(sp.mean() / spr.mean()),
                      'plan_speed_shift_mps': float((F['speed'] - FR['speed'])[mask].mean()), 'abs_heading_shift_deg': float(np.abs(F['heading'] - FR['heading'])[mask].mean()),
                      'abs_offset_shift_m': float(np.abs(F['offset'] - FR['offset'])[mask].mean())}
    out['vs_reference'] = rows
    # convergence: same noise, tau=1. Heun-5 and Heun-32 from the same run; Heun-64/128 from the convergence runs.
    h5, h32 = (m.samplers['heun5_t1.0'][0], cref)
    conv = {'heun5_vs_heun32_median_m': float(np.median(per_noise_dist(h5, h32))), 'heun3_vs_heun32_median_m': float(np.median(per_noise_dist(m.samplers['heun3_t1.0'][0], h32))),
            'euler10_vs_heun32_median_m': float(np.median(per_noise_dist(m.P['euler10_t1.0'][same], m.P[m.ref][same])))}
    if extra.get('heun64') is not None:
        H64, ok = extra['heun64']  # ok: clips where the Heun-64 run's numerics match the main run (see main)
        conv['heun64_clips'] = int(ok.sum())
        conv['heun32_vs_heun64_median_m'] = float(np.median(per_noise_dist(m.P[m.ref][ok], H64[ok])))
        ok5 = ok & same  # control-run Heun-5 only where the control run's reference equals the main one
        conv['heun5_heun64_clips'] = int(ok5.sum())
        conv['heun5_vs_heun64_median_m'] = float(np.median(per_noise_dist(h5[ok5], H64[ok5])))
        conv['heun32_vs_heun64_class_agreement'] = float((classify(m.feat(m.P[m.ref]), v)[ok] == classify(features(H64), v)[ok]).mean())
    if extra.get('heun128') is not None:
        H128, ok = extra['heun128']; conv['heun128_clips'] = int(ok.sum())
        conv['heun32_vs_heun128_median_m'] = float(np.median(per_noise_dist(m.P[m.ref][ok], H128[ok])))
        conv['heun5_vs_heun128_median_m'] = float(np.median(per_noise_dist(h5[ok], H128[ok])))
    out['convergence'] = conv
    return out


# ---------------------------------------------------------------- Q6
def energy_sets(X, a, Y, b):
    """Energy distance between plan sets X[:, a] and Y[:, b] on disjoint noises (all cross pairs; within-set i != j)."""
    X, Y = X[:, list(a)], Y[:, list(b)]; ox = ~np.eye(X.shape[1], dtype=bool); oy = ~np.eye(Y.shape[1], dtype=bool)
    xy = np.linalg.norm(X[:, :, None] - Y[:, None], axis=-1).mean(-1).mean((1, 2))
    xx = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1).mean(-1)[:, ox].mean(1); yy = np.linalg.norm(Y[:, :, None] - Y[:, None], axis=-1).mean(-1)[:, oy].mean(1)
    return 2 * xy - xx - yy


def q6a(m, W, rng):
    """Planted recovery: the reference at a known tau0 used as the 'sampler'.
    (i) clip split (sanity): the curve is built on a random half of the clips, the planted sampler is the other half (same six
        noises); 200 random splits, point estimates; tau_eff should return tau0 up to clip heterogeneity.
    (ii) noise split (diagnostic): curve on noises A, planted sampler on the disjoint noises B (3+3, all 10 splits), all clips;
        spread matching and energy distance. The six noises are the SAME vectors for every clip (torch.manual_seed(seed) in the
        probes), so differences between noise sets do not average out over clips."""
    S = len(m.seeds); N = len(m.names); ref = m.P[m.ref]; full = list(range(S)); out = {'clip_split': {}, 'noise_split': {}}
    base = {c: clip_stats(m, m.P[c], ref, full) for c in m.curve}
    splits = [rng.permutation(N) for _ in range(200)]
    for t0 in [.9, .8, .7, .6]:
        c0 = f'heun32_t{t0}'; taus, ra, rf = [], [], []
        for perm in splits:
            A, B = perm[:N // 2], perm[N // 2:]
            WA = np.zeros((1, N)); WA[0, A] = 1 / len(A); WB = np.zeros((1, N)); WB[0, B] = 1 / len(B)
            cs = np.array([WA @ base[c]['spread'] for c in m.curve])[:, 0]; o = np.argsort(cs); x = float(WB @ base[c0]['spread'])
            t = float(np.interp(x, cs[o], m.taus[o])); taus.append(t)
            for k, lst in (('ade1', ra), ('flip', rf)):
                cm = np.array([WA @ base[c][k] for c in m.curve])[:, 0]; lst.append(float(WB @ base[c0][k]) - float(np.interp(x, cs[o], cm[o])))
        out['clip_split'][str(t0)] = {'splits': len(splits), 'tau_eff_mean': float(np.mean(taus)), 'tau_eff_q025_q975': np.quantile(taus, [.025, .975]).tolist(),
                                      'ade1_residual_q025_q975': np.quantile(ra, [.025, .975]).tolist(), 'flip_residual_q025_q975': np.quantile(rf, [.025, .975]).tolist()}
    nsplits = [a for a in itertools.combinations(range(S), S // 2) if 0 in a]
    sr = []
    for a in nsplits:
        b = [i for i in full if i not in a]; sr.append(float(spreads(ref, b)['spread'].mean() / spreads(ref, a)['spread'].mean()))
    out['noise_split']['ref_spread_ratio_B_over_A'] = {'min': min(sr), 'max': max(sr)}
    for t0 in [.9, .8, .7, .6]:
        c0 = f'heun32_t{t0}'; per = []
        for a in nsplits:
            b = [i for i in full if i not in a]
            curve = [clip_stats(m, m.P[c], ref, a) for c in m.curve]; samp = clip_stats(m, m.P[c0], ref, b); refs = clip_stats(m, ref, ref, b)
            samp['energy'] = np.stack([energy_sets(m.P[c0], b, m.P[c], a) for c in m.curve])
            r = match(W[:1], m.taus, curve, samp, 'spread', refs); re_ = match(W[:1], m.taus, curve, samp, 'energy', refs)
            per.append({'tau_spread': float(r['tau'][0]), 'tau_energy': float(re_['tau'][0]),
                        'ade1_residual': float(r['samp']['ade1'][0] - r['at']['ade1'][0]), 'flip_residual': float(r['samp']['flip'][0] - r['at']['flip'][0])})
        out['noise_split'][str(t0)] = {'splits': len(per), **{f'{k}_{f}': g([p[k] for p in per]) for k in ['tau_spread', 'tau_energy', 'ade1_residual', 'flip_residual']
                                                               for f, g in (('min', min), ('max', max), ('mean', lambda x: float(np.mean(x))))}}
    return out


def two_way(m, W, s, seed=20261103):
    """Clip x noise bootstrap (exploratory): each draw resamples clips (the paper's draws) AND the six noises with replacement;
    spreads use pairs of distinct noise indices only (duplicated noises would add zero distances), weighted by multiplicity."""
    S = len(m.seeds); iu = np.triu_indices(S, 1); X, R = m.samplers[s]; ref = m.P[m.ref]; rng = np.random.default_rng(seed)  # same noise draws for every model (joint)
    def per(Xc, Rc):
        D = np.linalg.norm(Xc[:, :, None] - Xc[:, None], axis=-1).mean(-1)[:, iu[0], iu[1]]  # (N, 15)
        cx = classify(m.feat(Xc), m.speed); cr = classify(m.feat(Rc), m.speed)
        return D, np.linalg.norm(Xc - m.human[:, None], axis=-1).mean(-1), (cx != cr).astype(float)
    curve = [per(m.P[c], ref) for c in m.curve]; samp = per(X, R); rf = per(R, R)
    B = W.shape[0]; cnt = np.ones((B, S)); cnt[1:] = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(B - 1)])
    pw = cnt[:, iu[0]] * cnt[:, iu[1]]; pw = pw / pw.sum(1, keepdims=True); sw = cnt / S
    def agg(t):
        D, A, F = t; Wc = W  # (B, N)
        return {'spread': np.einsum('bn,np,bp->b', Wc, D, pw), 'ade1': np.einsum('bn,ns,bs->b', Wc, A, sw), 'flip': np.einsum('bn,ns,bs->b', Wc, F, sw)}
    C = [agg(t) for t in curve]; Sm = agg(samp); Rm = agg(rf)
    CS = np.stack([c['spread'] for c in C], 1); tau = np.empty(B); at = {k: np.empty(B) for k in ['ade1', 'flip']}; inr = np.empty(B, bool)
    for b in range(B):
        o = np.argsort(CS[b]); tau[b] = np.interp(Sm['spread'][b], CS[b, o], m.taus[o]); inr[b] = CS[b].min() <= Sm['spread'][b] <= CS[b].max()
        for k in at: at[k][b] = np.interp(Sm['spread'][b], CS[b, o], np.array([c[k][b] for c in C])[o])
    return slim(summarize({'tau': tau, 'in_range': inr, 'at': at, 'samp': {k: Sm[k] for k in at}, 'ref': {k: Rm[k] for k in at}}, require_rule(m, s)))


def q6b(key, records, W, e10_tau):
    """Euler-10 started from tau'*z, matched on the Heun-32 curve of the same run: tau_eff vs tau' * tau_eff(E10 @ tau'=1)."""
    names = [f'context_{i:03d}' for i in range(400)]; path = ART / SCALED[key]; seeds = json.loads((path / 'manifest.json').read_text())['seeds']
    cfgs = pn.CURVE + [f'euler10_t{t}' for t in pn.TAUS]
    P = npz(path, names, seeds, cfgs)
    human = np.stack([np.load(DATA / f'{n}.npz')['ego_future_xyz'][:, :2] for n in names]).astype(float)
    m = Model(key, names, np.array([records[n]['speed_mps'] for n in names], float), human, P, None, pn.TAUS, pn.CURVE, pn.REFERENCE, seeds)
    out = {}; base = None
    for t in pn.TAUS:
        m.samplers[f'euler10_t{t}'] = (P[f'euler10_t{t}'], P[pn.REFERENCE]); e = run(m, W, f'euler10_t{t}')
        if t == 1.: base = e['tau_eff']
        pred = t * base; ok = pred >= min(pn.TAUS)
        out[str(t)] = {**slim(e), 'predicted_tau_eff': pred, 'prediction_in_grid': bool(ok), 'tau_eff_minus_predicted': e['tau_eff'] - pred,
                       'ratio_tau_eff_over_tau_prime': e['tau_eff'] / t}
    out['euler10_tau1_tau_eff_this_run'] = base; out['euler10_tau1_tau_eff_pooled'] = e10_tau
    return out


def q6c():
    """Inventory of tau > 1 plans in the probes used here."""
    found = []
    for p in list(PROBES.values()) + list(CONTROLS.values()) + [SCALED['alpamayo_r1'], SCALED['alpamayo2']] + QPROBES:
        with np.load(ART / p / 'plans.npz') as z:
            found += [k for k in z.files if '_t' in k and not k.endswith('_human') and float(k.rsplit('_t', 1)[1]) > 1]
    return {'plans_with_tau_above_1': len(found)}


# ---------------------------------------------------------------- Q7
def q7(m):
    """E1 spread relative to the reference, by axis. With x_t=(1-t)z+t x1 the exact field at t=0 is E[x1|c]-z, so the exact
    one-step map z+v*(z,0)=E[x1|c] is constant in z: any E1 spread equals the spread of the network's t=0 field error."""
    idx = range(len(m.seeds)); e1 = spreads(m.P['euler1_t1.0'], idx); r = spreads(m.P[m.ref], idx)
    out = {k: float(e1[k].mean() / r[k].mean()) for k in ['spread', 'spread_long', 'spread_lat']}
    out['e1_spread_m'] = float(e1['spread'].mean()); out['ref_spread_m'] = float(r['spread'].mean())
    # where does it sit on the reference curve? fraction of the reference spread at the curve's lowest tau
    low = spreads(m.P[m.curve[int(np.argmin(m.taus))]], idx)['spread'].mean(); out['lowest_tau_curve_spread_ratio'] = float(low / r['spread'].mean())
    out['e1_mean_to_ref_mean_m'] = float(np.linalg.norm(m.P['euler1_t1.0'].mean(1) - m.P[m.ref].mean(1), axis=-1).mean())
    return out


# ---------------------------------------------------------------- main
THR_VARIANTS = {**{f'all_x{f}': {k: v * f for k, v in THR0.items()} for f in [.8, .9, 1.1, 1.2]},
                **{f'{g}_x{f}': {**THR0, **{k: THR0[k] * f for k in ks}} for g, ks in [('stop', ['stop']), ('band', ['band_frac', 'band_floor']),
                                                                                   ('heading', ['heading']), ('offset', ['offset'])] for f in [.8, 1.2]}}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True); p.add_argument('--boot', type=int, default=2000)
    p.add_argument('--skip-check', action='store_true'); a = p.parse_args()
    records = {r['name']: r for r in json.loads((DATA / 'records.json').read_text()) if 'error' not in r}
    names = [f'context_{i:03d}' for i in range(400)]
    r1 = np.random.default_rng(20261026); r2 = np.random.default_rng(20261026)
    assert all((np.array([names.index(x) for x in r1.choice(names, 400)]) == r2.choice(400, 400)).all() for _ in range(3))
    models = [load_alpamayo(k, records) for k in PROBES] + [load_qwen(records)]
    W = {m.key: draws(m, a.boot) for m in models}
    if not a.skip_check:
        for m in models:
            check_classifier(m.full['euler10_t1.0'], m.speed, m.full['euler10_t1.0']); check_classifier(m.full[m.ref], m.speed, m.full[m.ref])
            check_classifier(m.full[m.curve[3]], m.speed, m.full[m.curve[3]])
    res = {'status': 'EXPLORATORY / POST HOC (not pre-registered); robustness questions, 2026-11-02', 'boot': a.boot, 'margins': MARGINS,
           'draws': {'alpamayo': 'seed 20261026, rng.choice(names) as frost.analysis.paper_numbers; same draws for the three models (joint)',
                     'qwendrive': 'seed 20261102, rng.integers as frost.analysis.qwendrive_v3_analysis'},
           'thresholds_original': THR0, 'baseline_check': check_baseline(models, W)}
    print('baseline reproduced', flush=True)
    res['q1_matching'] = {m.key: {s: {st: slim(run(m, W[m.key], s, st)) for st in STATS} for s in m.samplers if not s.startswith(('euler1_', 'euler2_'))} for m in models}
    print('q1 done', flush=True)
    q2 = {}
    for m in models:
        for s in m.samplers:
            if s.startswith(('euler1_', 'euler2_')): continue
            e = run(m, W[m.key], s); q2.setdefault(m.key, {})[s] = {k: {x: e[k][x] for x in ['effect', 'effect_ci95', 'residual', 'residual_ci95', 'fraction_reproduced',
                                                                                            'fraction_reproduced_ci95', 'margin_over_effect', 'equivalent']} for k in ['ade1', 'flip']}
            q2[m.key][s]['tau_eff'] = e['tau_eff']
    res['q2_fraction'] = q2; print('q2 done', flush=True)
    q3 = {}
    for m in models:
        S = len(m.seeds)
        for s in m.samplers:
            if s.startswith(('euler1_', 'euler2_')): continue
            full = run(m, W[m.key], s); ent = {'full': slim(full)}
            for k in [3, 4, 5]:
                runs = [run(m, W[m.key], s, idx=c) for c in itertools.combinations(range(S), k)]
                taus = [r['tau_eff'] for r in runs]
                ent[f'k{k}'] = {'subsets': len(runs), 'tau_eff_min': min(taus), 'tau_eff_max': max(taus), 'tau_eff_sd': float(np.std(taus)),
                                **{f'{q}_equivalent_frac': float(np.mean([r[q]['equivalent'] for r in runs])) for q in ['ade1', 'flip']},
                                **{f'{q}_same_verdict_frac': float(np.mean([r[q]['equivalent'] == full[q]['equivalent'] for r in runs])) for q in ['ade1', 'flip']},
                                **{f'{q}_residual_range': [min(r[q]['residual'] for r in runs), max(r[q]['residual'] for r in runs)] for q in ['ade1', 'flip']},
                                **{f'{q}_ci90_hi_max': max(max(abs(x) for x in r[q]['residual_ci90']) for r in runs) for q in ['ade1', 'flip']}}
            q3.setdefault(m.key, {})[s] = ent
        print('q3', m.key, flush=True)
    res['q3_noises'] = q3
    res['q3_two_way_bootstrap'] = {m.key: {s: two_way(m, W[m.key], s) for s in m.samplers if s in ('euler4_t1.0', 'euler10_t1.0', 'heun5_t1.0')} for m in models}
    print('two-way done', flush=True)
    q4 = {}
    for m in models:
        for s in m.samplers:
            if s.startswith(('euler1_', 'euler2_', 'euler6_')): continue
            q4.setdefault(m.key, {})[s] = {name: {**{k: v for k, v in slim(run(m, W[m.key], s, thr=thr)).items() if k in ('tau_eff', 'flip')},
                                                  'flip_sampler': None} for name, thr in {'original': THR0, **THR_VARIANTS}.items()}
            for name, thr in {'original': THR0, **THR_VARIANTS}.items():
                X, R = m.samplers[s]; q4[m.key][s][name]['flip_sampler'] = float((classify(m.feat(X), m.speed, thr) != classify(m.feat(R), m.speed, thr)).mean())
        print('q4', m.key, flush=True)
    res['q4_thresholds'] = q4
    # Q5 (Alpamayo only: Qwen-Drive Heun-3/5 not available yet)
    q5r = {}
    for m in models[:3]:
        extra = {}
        if m.key == 'alpamayo15':
            h64 = np.concatenate([npz(ART / HEUN64['alpamayo15'][0], names[:100], m.seeds, ['heun64_t1.0'])['heun64_t1.0'],
                                  npz(ART / HEUN64['alpamayo15'][1], names[100:], m.seeds, ['heun64_t1.0'])['heun64_t1.0']])
            h128 = npz(ART / HEUN64['alpamayo15'][0], names[:100], m.seeds, ['heun128_t1.0'])['heun128_t1.0']
            extra['heun64'] = (h64, np.ones(400, bool))
            extra['heun128'] = (np.concatenate([h128, np.full((300,) + h128.shape[1:], np.nan)]), np.arange(400) < 100)
        else:
            z = npz(ART / HEUN64[m.key][0], names, m.seeds, ['heun64_t1.0', 'euler10_t1.0'])
            ok = np.abs(z['euler10_t1.0'] - m.P['euler10_t1.0']).max((1, 2, 3)) < 1e-6  # same numerics as the main run on this clip
            extra['heun64'] = (z['heun64_t1.0'], ok)
        q5r[m.key] = q5(m, extra)
    res['q5_heun'] = q5r; print('q5 done', flush=True)
    rng3 = np.random.default_rng(20261104)
    res['q6_planted'] = {'a_reference_as_sampler': {m.key: q6a(m, W[m.key], rng3) for m in models},
                         'b_scaled_noise_euler10': {k: q6b(k, records, W[k], json.loads(POOLED.read_text())['models'][k]['euler10_t1.0']['tau_eff'][0]) for k in PROBES},
                         'c_heated': q6c()}
    print('q6 done', flush=True)
    res['q7_e1'] = {m.key: q7(m) for m in models}
    a.out.mkdir(parents=True, exist_ok=True); (a.out / 'summary.json').write_text(json.dumps(res, indent=1))
    (a.out / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes()); print('wrote', a.out)


if __name__ == '__main__': main()
