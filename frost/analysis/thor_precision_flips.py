"""Exploratory (not registered): do lower-precision TensorRT engines on DRIVE AGX Thor change
driving decisions at the same noise? For each clip, seed and sampler, compare the decision class (longitudinal, lateral;
frost.core.decisions.decisions, as in the paper's flip metric) of a plan with the plan of another run under the
same noise. Pairs: FP8 vs bf16 and NVFP4 vs bf16 on the board (TensorRT), and, as a floor, board bf16 vs the desktop GPU.
Per-clip flip shares are averaged over clips; 95% intervals from 2000 clip resamples (seed 20261031).

Usage: python -m frost.analysis.thor_precision_flips --out artifacts/thor_trt_precision_flips_v1_<date>
"""
import argparse
import json
from pathlib import Path
import numpy as np
from frost.analysis.thor_analysis import load_probe
from frost.core.decisions import decisions

CONFIGS={'E10':'euler10_t1.0','E4':'euler4_t1.0','Heun32':'heun32_t1.0'}
SEED=20261031


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--thor',type=Path,default=Path('artifacts/thor_trt_frost400_20261031'))
    p.add_argument('--workstation',type=Path,default=Path('artifacts/temperature_probe_a15_heldout_pooled400_v3_20261028'))
    p.add_argument('--data',type=Path,default=Path('artifacts/physicalai_heldout_contexts_v3_20261027'))
    p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--boot',type=int,default=2000)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    speed={r['name']:r['speed_mps'] for r in json.loads((a.data/'records.json').read_text()) if 'error' not in r}
    runs={'bf16':load_probe(a.thor/'trt_bf16'),'fp8':load_probe(a.thor/'trt_fp8'),'nvfp4':load_probe(a.thor/'trt_nvfp4'),'desktop':load_probe(a.workstation)}
    seeds=runs['bf16'][1];assert all(r[1]==seeds for r in runs.values())
    names=[n for i in range(*a.range) for n in [f'context_{i:03d}'] if n in speed
           and all(f'{n}_s{s}_{c}' in r[0] for r in runs.values() for s in seeds for c in CONFIGS.values())]
    key=lambda t,n:(lambda d:(d['longitudinal'],d['lateral']))(decisions(t,speed[n]))
    cls={v:{c:np.array([[key(r[0][f'{n}_s{s}_{c}'],n) for s in seeds] for n in names],dtype=object) for c in CONFIGS.values()} for v,r in runs.items()}
    rng=np.random.default_rng(SEED);idx=[rng.integers(0,len(names),len(names)) for _ in range(a.boot)]
    res={'note':'exploratory, not registered','clips':len(names),'seeds':seeds,'boot':a.boot,'boot_seed':SEED,'pairs':{}}
    for x,y in [('fp8','bf16'),('nvfp4','bf16'),('bf16','desktop')]:
        res['pairs'][f'{x}_vs_{y}']={}
        for lab,c in CONFIGS.items():
            per=np.array([[tuple(u)!=tuple(w) for u,w in zip(cls[x][c][i],cls[y][c][i])] for i in range(len(names))]).mean(axis=1)*100
            bt=[per[j].mean() for j in idx]
            res['pairs'][f'{x}_vs_{y}'][lab]={'flip_pct':float(per.mean()),'ci95':np.quantile(bt,[.025,.975]).tolist(),'clips_with_any_flip':int((per>0).sum())}
    # context: the samplers' own flips against the reference A_1 under the same noise, on board bf16
    res['context_bf16_flips_vs_A1']={}
    for lab in ['E10','E4']:
        per=np.array([[tuple(u)!=tuple(w) for u,w in zip(cls['bf16'][CONFIGS[lab]][i],cls['bf16'][CONFIGS['Heun32']][i])] for i in range(len(names))]).mean(axis=1)*100
        res['context_bf16_flips_vs_A1'][lab]=float(per.mean())
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(res,indent=2))
    (a.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in res.items() if k!='seeds'},indent=1))


if __name__=='__main__':
    main()
