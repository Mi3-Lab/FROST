"""Summaries of the convergence extensions (diagnostics; no paper statistics).
  cosmos: parts of cosmos3nano_probe.py ext (heun512 + adaptive) with heun128/256 from the conv parts of the same clip and tau.
  simwam: extend.json rows from simwam_temperature_probe.py --extend (or rows parsed from slurm logs).
Reports the distances 128->256->512, decision-class agreement with Heun-512, and the adaptive solver NFE/ok/distance to Heun-512."""
import argparse, json
from pathlib import Path
import numpy as np
from frost.core.decisions import decisions


def cosmos(ext_dirs, conv_dir, records):
    rec = {r['name']: r for r in json.loads(Path(records).read_text()) if 'error' not in r}; rows = []
    for d in ext_dirs:
        for f in sorted((Path(d) / 'parts').glob('context_*_t*_ext.npz')):
            name, tau = f.stem[:-4].rsplit('_t', 1); z = np.load(f); c = np.load(Path(conv_dir) / 'parts' / f'{name}_t{tau}.npz'); sp = rec[name]['speed_mps']
            info = json.loads(str(z['info'])); pl = {'heun128': c['heun128'], 'heun256': c['heun256'], 'heun512': z['heun512']}
            for k in z.files:
                if k.startswith('rk45'): pl[k] = z[k]
            cls = {k: (lambda q: (q['longitudinal'], q['lateral']))(decisions(v, sp)) for k, v in pl.items()}; ref = pl['heun512']
            rows.append(dict(clip=name, tau=float(tau), d128_256=float(np.abs(pl['heun128'] - pl['heun256']).max()), d256_512=float(np.abs(pl['heun256'] - ref).max()),
                             d128_512=float(np.abs(pl['heun128'] - ref).max()), agree128=cls['heun128'] == cls['heun512'], agree256=cls['heun256'] == cls['heun512'],
                             adaptive={k: dict(nfe=v['nfe'], ok=v['ok'], dist_to_heun512=float(np.abs(pl[k] - ref).max()), agree=cls[k] == cls['heun512']) for k, v in info.items()}))
    return rows


def simwam(paths):
    rows = []
    for p in paths:
        txt = Path(p).read_text(errors='ignore')
        if p.endswith('.json'): rows += json.loads(txt)
        else: rows += [json.loads(l) for l in txt.splitlines() if l.startswith('{"clip"') and '"adaptive"' in l]
    seen = {}; [seen.setdefault((r['clip'], r['tau'], r['seed']), r) for r in rows]
    out = []
    for r in seen.values():
        c = r['class']
        out.append(dict(clip=r['clip'], tau=r['tau'], seed=r['seed'], d128_256=r['dist_128_256'], d256_512=r['dist_256_512'], d128_512=r['dist']['heun128'],
                        agree128=c['heun128'] == c['heun512'], agree256=c['heun256'] == c['heun512'],
                        adaptive={k: dict(nfe=v['nfe'], ok=v['ok'], dist_to_heun512=r['dist'].get(k), agree=(c.get(k) == c['heun512']) if k in c else None) for k, v in r['adaptive'].items()}))
    return out


def qwen(dirs):
    rows = []
    for d in dirs:
        for r in json.loads((Path(d) / 'rows.json').read_text()):
            dr, ag = r['dist_to_ref'], r['agree_ref']; ad = r['adaptive']
            rows.append(dict(clip=r['clip'], tau=r['tau'], seed=r['seed'], d128_256=r['d128_256'], d256_512=r['d256_512'], d128_512=dr['heun128'],
                             d32_64=r['d32_64'], agree128=ag['heun128'], agree256=ag['heun256'], agree32=ag['heun32'], agree64=ag['heun64'],
                             adaptive={k: dict(nfe=v['nfe'], ok=v['ok'], dist_to_heun512=dr.get(k), agree=ag.get(k)) for k, v in ad.items()}))
    seen = {}; [seen.setdefault((r['clip'], r['tau'], r['seed']), r) for r in rows]
    return list(seen.values())


def table(rows):
    for tau in sorted({r['tau'] for r in rows}):
        rs = [r for r in rows if r['tau'] == tau]; n = len(rs)
        q = lambda k: f"median {np.median([r[k] for r in rs]):.2f} max {max(r[k] for r in rs):.2f}"
        print(f"tau {tau} n={n}: 128-256 {q('d128_256')} | 256-512 {q('d256_512')} | 128-512 {q('d128_512')} | class agree vs H512: H128 {sum(r['agree128'] for r in rs)}/{n}, H256 {sum(r['agree256'] for r in rs)}/{n}")
        for tol in sorted({k for r in rs for k in r['adaptive']}):
            a = [r['adaptive'][tol] for r in rs if tol in r['adaptive']]; done = [x for x in a if x['ok']]
            print(f"   {tol}: completed {len(done)}/{len(a)}, NFE median {np.median([x['nfe'] for x in a]):.0f} max {max(x['nfe'] for x in a)}"
                  + (f", dist to H512 (completed) median {np.median([x['dist_to_heun512'] for x in done]):.2f} max {max(x['dist_to_heun512'] for x in done):.2f}, class agree {sum(bool(x['agree']) for x in done)}/{len(done)}" if done else ''))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['cosmos', 'simwam', 'qwen']); ap.add_argument('--ext', nargs='+', required=True)
    ap.add_argument('--conv'); ap.add_argument('--records'); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    rows = cosmos(a.ext, a.conv, a.records) if a.kind == 'cosmos' else simwam(a.ext) if a.kind == 'simwam' else qwen(a.ext)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / 'summary.json').write_text(json.dumps(rows, indent=1)); table(rows)
