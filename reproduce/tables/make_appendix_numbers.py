"""Macros and tables for the toy-model and schedule appendices, computed from artifact JSON (no hand-typed numbers)."""
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro_paths import RESULTS, TABLES as OUT  # noqa: E402
ART=RESULTS


def num(x,digits=2):
    """Typeset a number with a true minus sign and no signed zero."""
    text=f'{x:.{digits}f}'
    if float(text)==0:text=f'{0:.{digits}f}'
    return text.replace('-','$-$')


def interval(x,digits):return f'[{num(x[0],digits)},\\,{num(x[1],digits)}]'


def load(path):return json.loads((ART/path).read_text())


def macro(name,value):return f'\\newcommand{{\\{name}}}{{{value}}}'


# ---- Solvable mixtures -------------------------------------------------------------------
SOLVERS=[('euler2','$E_2$'),('euler4','$E_4$'),('euler8','$E_8$'),('heun2','Heun-2'),('heun4','Heun-4'),('rk41','RK4 (1 step)'),('rk42','RK4 (2 steps)'),('cached8','Cached-8')]


def toy():
    data=load('mode_collapse_theory_v1_20261022/summary.json');res=data['results'];lines=[]
    lines+=[macro('ToyConfigs',len(res)),macro('ToyKs',', '.join(str(k) for k in sorted({r['k'] for r in res}))),
        macro('ToyDims',', '.join(str(k) for k in sorted({r['dim'] for r in res}))),
        macro('ToyEntropyFrac',num(100*data['summary']['entropy_decreased_fraction'],0))]
    rows=[];names={'euler2':'ETwo','euler4':'EFour','euler8':'EEight','heun2':'HTwo','heun4':'HFour','rk41':'RKOne','rk42':'RKTwo','cached8':'Cached'}
    for key,label in SOLVERS:
        flips=sum(r['solvers'][key]['flips'] for r in res);more=sum(r['solvers'][key]['to_more_probable'] for r in res)
        decreased=sum(r['solvers'][key]['entropy_change_bits']<0 for r in res)
        majority=sum(r['solvers'][key]['to_more_probable']>r['solvers'][key]['to_less_probable'] for r in res)
        share=100*more/flips
        lines+=[macro(f'Toy{names[key]}Share',num(share,0)),macro(f'Toy{names[key]}Entropy',decreased)]
        rows.append(f'{label} & {flips} & {num(share,0)} & {majority} & {decreased} \\\\')
    eu=[k for k,_ in SOLVERS if k.startswith('euler')]   # range over every Euler row, so the text cannot pick endpoints
    ent=[sum(r['solvers'][k]['entropy_change_bits']<0 for r in res) for k in eu]
    shr=[round(100*sum(r['solvers'][k]['to_more_probable'] for r in res)/sum(r['solvers'][k]['flips'] for r in res)) for k in eu]
    lines+=[macro('ToyEulerEntropyMin',min(ent)),macro('ToyEulerEntropyMax',max(ent)),macro('ToyEulerShareMin',min(shr)),macro('ToyEulerShareMax',max(shr))]
    flips_all=sum(r['solvers'][s]['flips'] for r in res for s in r['solvers'])
    lines.append(macro('ToyFlipsAll',flips_all))
    table=r"""\begin{table}[!htbp]
  \caption{\textbf{In solvable mixtures, every solver's flips lean toward likelier modes.} Flips in the """+str(len(res))+r""" configurations ($K\in\{"""+', '.join(map(str,sorted({r['k'] for r in res})))+r"""\}$ modes, dimension $\in\{"""+', '.join(map(str,sorted({r['dim'] for r in res})))+r"""\}$, two mode scales, three random draws each): decisions that differ from a 400-step RK4 solution under the same noise, """+str(flips_all)+r""" in total. Euler and Heun with $k$ steps, RK4 with one and two steps; Cached-8: eight Euler steps with the velocity of step 2 reused on steps 3 to 6. ``To likelier'' is the share of flips that end in a mode with higher weight than the exact one. Configurations are counted over the """+str(len(res))+r""" runs.}
  \label{tab:toy}
  \centering\small\setlength{\tabcolsep}{8pt}
  \begin{tabular}{@{}lrrrr@{}}
    \toprule
    Solver & Flips & To likelier (\%) & Configs, net toward likelier & Configs, entropy down \\
    \midrule
    """+'\n    '.join(rows)+r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    (OUT/'tab_toy.tex').write_text(table)
    return lines


# ---- Schedules ----------------------------------------------------------------------------
def confirmation(lines):
    path=ART/'schedule_eval_physicalai_v2_20261022'
    manifest=json.loads((path/'manifest.json').read_text());rows=json.loads((path/'rows.json').read_text())
    clips=sorted({r['context'] for r in rows});nfes=(4,6,8)

    def per_clip(method,key):
        acc=defaultdict(list)
        for r in rows:
            if r['method']==method:acc[(r['context'],r['nfe'])].append(float(r[key]))
        mean={k:sum(v)/len(v) for k,v in acc.items()}
        return {c:sum(mean[(c,n)] for n in nfes)/len(nfes) for c in clips}

    def paired(other,key):
        a=per_clip('decision_info_plus_uniform',key);b=per_clip(other,key);v=[a[c]-b[c] for c in clips]
        rng=random.Random(0);boots=sorted(sum(rng.choice(v) for _ in v)/len(v) for _ in range(2000))
        return sum(v)/len(v),(boots[49],boots[1949])  # 2.5% and 97.5% of 2000 resamples; seed 0

    lines+=[macro('SchedConfClips',len(clips)),macro('SchedConfSeeds',len(manifest['seeds']))]
    for other,tag in [('uniform','Unif'),('rescaled_entropic','Entr')]:
        for key,short,scale,digits in [('flip_vs_reference','Flip',100,2),('ade_human','AdeHuman',1,3)]:
            m,c=paired(other,key)
            lines+=[macro(f'SchedConf{short}{tag}',num(scale*m,digits)),macro(f'SchedConf{short}{tag}CI',interval([scale*c[0],scale*c[1]],digits))]
    return lines


def matched(lines):
    data=load('schedule_matched_report_v1_20261022/summary.json');res=data['results']
    lines+=[macro('SchedMatchClips',data['clips']),macro('SchedMatchSeeds',len(data['seeds']))]
    words={4:'Four',6:'Six',8:'Eight'}
    for n in (4,6,8):
        u=res[f'uniform_{n}'];d=res[f'decision_info_plus_uniform_{n}'];r=d['flip_residual_minus_uniform']
        lines+=[macro(f'SchedTauUnif{words[n]}',num(u['tau_eff'][0])),macro(f'SchedTauInfo{words[n]}',num(d['tau_eff'][0])),
            macro(f'SchedResid{words[n]}',num(100*r[0],1)),macro(f'SchedResid{words[n]}CI',interval([100*r[1],100*r[2]],1))]
    # all schedule-by-NFE comparisons against uniform, one row per grid family
    tested=[(k,v['flip_residual_minus_uniform']) for k,v in res.items() if 'flip_residual_minus_uniform' in v]
    excl=[(k,v) for k,v in tested if v[1]>0 or v[2]<0]
    lines+=[macro('SchedNTested',len(tested)),macro('SchedNExcl',len(excl))]
    names={'decision_info':'Decision information','decision_info_plus_uniform':'Information + uniform','field_turn':'Field turning','rescaled_entropic':'Rescaled entropic time','flashdrive_cached8':'Velocity cache (8 steps)'}
    def split(k):
        fam,n=k.rsplit('_',1);return fam,int(n)
    for k,v in excl:
        fam,n=split(k)
        lines+=[macro('SchedExclFlip',num(100*v[0],1)),macro('SchedExclCI',interval([100*v[1],100*v[2]],1)),macro('SchedExclName',f'{names[fam].lower()} at NFE {n}')]
    cell=lambda k:(lambda v:f'{num(100*v[0],1)}\\,{interval([100*v[1],100*v[2]],1)}')(res[k]['flip_residual_minus_uniform'])
    tcell=lambda k:num(res[k]['tau_eff'][0])
    rows=['Uniform & '+' & '.join(tcell(f'uniform_{n}') for n in (4,6,8))+' & \\multicolumn{3}{c}{reference} \\\\','\\midrule']
    for fam in ('decision_info','decision_info_plus_uniform','field_turn','rescaled_entropic','flashdrive_cached8'):
        ns=[n for n in (4,6,8) if f'{fam}_{n}' in res]
        t=[tcell(f'{fam}_{n}') if n in ns else '--' for n in (4,6,8)];c=[cell(f'{fam}_{n}') if n in ns else '--' for n in (4,6,8)]
        rows.append(f'{names[fam]} & '+' & '.join(t+c)+' \\\\')
    table=r"""\begin{table}[!htbp]
  \caption{\textbf{At matched temperature the schedule gain is not distinguishable from zero.} Effective temperature $\taueff$ of each grid and the flip residual of the grid against the uniform grid after matching temperature (points, 95\% interval), for all """+str(len(tested))+r""" schedule-by-NFE comparisons. The velocity cache has one budget (4 evaluations). """+f"{data['clips']} clips, {len(data['seeds'])} seeds"+r""".}
  \label{tab:sched}
  \centering\small\setlength{\tabcolsep}{5pt}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    & \multicolumn{3}{c}{$\taueff$ at NFE} & \multicolumn{3}{c}{Flip residual vs.\ uniform at NFE} \\
    \cmidrule(lr){2-4}\cmidrule(l){5-7}
    Grid & 4 & 6 & 8 & 4 & 6 & 8 \\
    \midrule
    """+'\n    '.join(rows)+r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    (OUT/'tab_sched.tex').write_text(table)
    return lines


def overlap(lines):
    d=load('split_overlap_v1_20261027/summary.json')['samples']['main']
    return lines+[macro('SchedMainTrain',d['partition']['train']),macro('SchedMainClips',d['clips'])]


if __name__=='__main__':
    lines=toy();lines=confirmation(lines);lines=matched(lines);lines=overlap(lines)
    (OUT/'appendix_numbers.tex').write_text('\n'.join(lines)+'\n');print('ok')
