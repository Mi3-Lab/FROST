"""SmolVLA/LIBERO appendix numbers, computed from the grid's per-episode eval_info.json files and the smoke probe."""
import re
import json,random,re,collections
from pathlib import Path

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro_paths import RESULTS, TABLES as OUT  # noqa: E402
from repro_paths import ARTIFACTS  # noqa: E402
GRID=RESULTS/'robot_grid_smolvla_v1_20261022';PROBE=ARTIFACTS/'robot_smoke_20261022/probe.npz'
CONTRACTION=RESULTS/'robot_smoke_20261022/contraction.json'  # contraction() on PROBE, cached (the probe is 16 MB)
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
B=10000


WORDS={'0':'Zero','1':'One','2':'Two','3':'Three','4':'Four','5':'Five','6':'Six','7':'Seven','8':'Eight','9':'Nine'}
def safe(name):
    return re.sub(r'\d+',lambda m:'Ten' if m.group()=='10' else 'Sixteen' if m.group()=='16' else ''.join(WORDS[c] for c in m.group()),name)


def load(cfg):
    """Success indicator per (suite, task, episode) for one configuration."""
    out={}
    for s in SUITES:
        for t in json.loads((GRID/cfg/s/'eval_info.json').read_text())['per_task']:
            for e,ok in enumerate(t['metrics']['successes']):out[(s,t['task_id'],e)]=float(bool(ok))
    return out


def paired(a,b,suites):
    """Mean difference (points) a-b on shared episodes; 95% bootstrap interval resampling tasks."""
    keys=[k for k in a if k[0] in suites];tasks=collections.defaultdict(list)
    for k in keys:tasks[(k[0],k[1])].append(a[k]-b[k])
    groups=list(tasks.values());mean=lambda gs:100*sum(map(sum,gs))/sum(map(len,gs))
    rng=random.Random(0);bs=sorted(mean([rng.choice(groups) for _ in groups]) for _ in range(B))
    return mean(groups),bs[int(.025*B)],bs[int(.975*B)]


def sig(x,d=1):
    t=f'{x:+.{d}f}'
    return t.replace('-','$-$') if float(t)!=0 else f'{0:.{d}f}'


def contraction():
    """Spread ratio and contracted share of E_k against Heun-16 on the smoke probe; read from the cache unless the probe exists."""
    if not PROBE.exists():
        c=json.loads(CONTRACTION.read_text());return {int(k):tuple(v) for k,v in c['res'].items()},c['calls']
    import numpy as np
    d=np.load(PROBE);calls=collections.defaultdict(lambda:collections.defaultdict(dict))
    for k in d.files:
        m=re.match(r'c(\d+)_s(\d+)_(.*)',k);calls[int(m[1])][m[3]][int(m[2])]=d[k].reshape(-1)
    def spread(v,cfg):
        return np.sqrt(np.mean(np.var([v[cfg][s] for s in sorted(v[cfg])],0,ddof=1)))
    res={}
    for k in [10,4,2,1]:
        cfg=f'euler{k}_t1.0';tau=[];shrunk=[]
        for v in calls.values():
            tau.append(spread(v,cfg)/spread(v,'heun16_t1.0'))
            mean=np.mean([v['heun16_t0.2'][s] for s in sorted(v[cfg])],0)
            shrunk+=[np.linalg.norm(v[cfg][s]-mean)<np.linalg.norm(v['heun16_t1.0'][s]-mean) for s in sorted(v[cfg])]
        res[k]=(float(np.mean(tau)),100*float(np.mean(shrunk)),len(shrunk))
    return res,len(calls)


def main():
    cfgs={c.name:load(c.name) for c in GRID.iterdir() if c.is_dir()}
    names={'10':'Ten','4':'Four','2':'Two','1':'One'};tn={'1.0':'One','0.9':'Nine','0.75':'SevenFive','0.5':'Five','0.25':'TwoFive'}
    L=[];ref=cfgs['heun16_t1.0'];allS=set(SUITES)
    L.append(f"\\newcommand{{\\RobotEpisodes}}{{{len(ref)}}}")
    L.append(f"\\newcommand{{\\RobotEpisodesPerSuite}}{{{len(ref)//len(SUITES)}}}")
    L.append(f"\\newcommand{{\\RobotTasks}}{{{len({k[:2] for k in ref})}}}")
    res,ncalls=contraction()
    L.append(f"\\newcommand{{\\RobotProbeCalls}}{{{ncalls}}}")
    for k,(t,sh,n) in res.items():
        L.append(f"\\newcommand{{\\RobotTauEff{names[str(k)]}}}{{{t:.2f}}}")
        L.append(f"\\newcommand{{\\RobotShrunk{names[str(k)]}}}{{{sh:.0f}}}")
    L.append(f"\\newcommand{{\\RobotProbeSamples}}{{{res[10][2]}}}")
    def rate(c,s=allS):
        ks=[k for k in cfgs[c] if k[0] in s];return 100*sum(cfgs[c][k] for k in ks)/len(ks)
    def put(name,r):
        L.append(f"\\newcommand{{\\Robot{name}}}{{{sig(r[0])}\\,[{sig(r[1])},\\,{sig(r[2])}]}}")
    rows=[]
    order=[('Euler10','euler10_t1.0','$E_{10}$'),('Euler4','euler4_t1.0','$E_4$'),('Euler2','euler2_t1.0','$E_2$'),('Euler1','euler1_t1.0','$E_1$')]+[(f'HeunTau{tn[t]}',f'heun16_t{t}',f'Heun-16, $\\tau{{=}}{t}$') for t in ['1.0','0.9','0.75','0.5','0.25']]
    for n,c,lab in order:
        k=c.split('_')[0][5:] if c.startswith('euler') else None
        cells=[lab,f'{res[int(k)][0]:.2f}' if k else '--',f'{rate(c):.1f}']
        for s in SUITES:cells.append(f'{rate(c,{s}):.1f}')
        d=paired(cfgs[c],ref,allS);cells.append(sig(d[0])+f'\\,[{sig(d[1])},\\,{sig(d[2])}]' if c!='heun16_t1.0' else '--')
        d10=paired(cfgs[c],ref,{'libero_10'});cells.append(sig(d10[0])+f'\\,[{sig(d10[1])},\\,{sig(d10[2])}]' if c!='heun16_t1.0' else '--')
        rows.append(' & '.join(cells)+r' \\')
        if c!='heun16_t1.0':
            put(f'VsRef{n}',d);put(f'VsRefTen{n}',d10)
        L.append(f"\\newcommand{{\\RobotRate{n}}}{{{rate(c):.1f}}}")
        L.append(f"\\newcommand{{\\RobotRateTen{n}}}{{{rate(c,{'libero_10'}):.1f}}}")
    # Euler-k against the explicit temperature of matching tau_eff (nearest grid point)
    for n,a,b in [('EulerTenVsTauNine','euler10_t1.0','heun16_t0.9'),('EulerFourVsTauSevenFive','euler4_t1.0','heun16_t0.75'),('EulerTwoVsTauFive','euler2_t1.0','heun16_t0.5')]:
        put(n,paired(cfgs[a],cfgs[b],allS))
    # largest absolute paired difference among the central configurations (tau in [0.5,1] and Euler 10/4/2)
    central=['euler10_t1.0','euler4_t1.0','euler2_t1.0','heun16_t0.9','heun16_t0.75','heun16_t0.5']
    pairs=[(a,b) for i,a in enumerate(central+['heun16_t1.0']) for b in (central+['heun16_t1.0'])[i+1:]]
    ds=[(paired(cfgs[a],cfgs[b],allS),a,b) for a,b in pairs]
    L.append(f"\\newcommand{{\\RobotCentralMaxAbs}}{{{max(abs(d[0][0]) for d in ds):.1f}}}")
    L.append(f"\\newcommand{{\\RobotCentralPairs}}{{{len(ds)}}}")
    L.append(f"\\newcommand{{\\RobotCentralExclZero}}{{{sum(d[0][1]>0 or d[0][2]<0 for d in ds)}}}")
    # TeX control words cannot contain digits: spell them out (Euler10 -> EulerTen)
    L=[re.sub(r'\\newcommand\{(\\Robot[A-Za-z0-9]+)\}',lambda m:'\\newcommand{'+safe(m.group(1))+'}',x) for x in L]
    (OUT/'smolvla_numbers.tex').write_text('% generated by make_smolvla_numbers.py; do not edit\n'+'\n'.join(L)+'\n')
    tab=r"""\begin{table}[!htbp]
  \caption{\textbf{SmolVLA on LIBERO: success is nearly flat except at the coldest settings on long tasks.} Success rate (\%) per suite and over all suites; $\Delta$: difference from Heun-16 at $\tau{=}1$ in points, paired by episode, with 95\% intervals from a bootstrap over the 40 tasks. $\taueff$: effective temperature of $E_k$ in the offline probe. 100 episodes per suite and 400 per configuration, with the same environment seeds in every configuration.}
  \label{tab:smolvla}
  \centering\small\setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lcccccccc@{}}
    \toprule
    & $\taueff$ & All & Spat. & Obj. & Goal & LIBERO-10 & $\Delta$ all & $\Delta$ LIBERO-10 \\
    \midrule
    """+'\n    '.join(rows)+r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    (OUT/'tab_smolvla.tex').write_text(tab)


if __name__=='__main__':
    main()
