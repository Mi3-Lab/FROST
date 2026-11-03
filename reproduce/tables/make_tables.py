"""Generate the paper's LaTeX tables directly from artifact JSON, so no number is typed by hand."""
import json
import re
from pathlib import Path

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repro_paths import RESULTS, TABLES as OUT  # noqa: E402
POOLED=RESULTS/'paper_numbers_heldout_pooled400_v2_20261028'  # primary evidence since v5.32: 400 held-out clips
ORIG=RESULTS/'paper_numbers_v4_20261026'  # original 60 test-split clips (appendix)
MODELS=[('alpamayo15','Alpamayo-1.5'),('alpamayo_r1','Alpamayo-R1'),('alpamayo2','Alpamayo-2')]
SAMPLERS=[('euler4_t1.0','$E_4$'),('euler6_t1.0','$E_6$'),('shrink4_t1.0','Shrink$_4$'),('euler10_t1.0','$E_{10}$'),('shrink10_t1.0','Shrink$_{10}$'),('heun3_t1.0','Heun-3'),('heun5_t1.0','Heun-5')]


def num(x,digits=2):
    """Typeset a number with a true minus sign and no signed zero."""
    text=f'{x:.{digits}f}'
    if float(text)==0:text=f'{0:.{digits}f}'
    return text.replace('-','$-$')


def ci(x):
    """Interval bound in points: two decimals near zero so that the sign stays visible."""
    return num(x,2) if abs(x)<0.1 else num(x,1)


HEUN_CTRL={'alpamayo_r1':'heun_controls_analysis_r1_v1_20261028','alpamayo2':'heun_controls_analysis_a2_v1_20261028'}
TAG={'alpamayo15':'AOneFive','alpamayo_r1':'ROne','alpamayo2':'ATwo'}


def dd_tau_text(r):
    """DiffusionDrive effective-temperature interval with out-of-range resamples kept as tau_eff > 1 (audit A3, 29/10)."""
    c=r['tau_eff_ci95_censored'];assert c[1]!=c[1]  # upper quantile censored above the grid
    return f"95\\% interval {c[0]:.2f} to above 1; {100*r['boot_in_range']:.0f}\\% of resamples in range"


def load_pooled():
    """Pooled-400 summary with Heun-3/5 of Alpamayo-R1 and -2 (heun_controls_analysis) added in the same layout.
    Residual triples are (point, lo95, hi95) as in paper_numbers; the controls store their own 95% intervals."""
    stats=json.loads((POOLED/'summary.json').read_text())
    for key,art in HEUN_CTRL.items():
        h=json.loads((RESULTS/art/'summary.json').read_text())['samplers'];ref=stats['models'][key]['euler10_t1.0']
        for k in ('heun3_t1.0','heun5_t1.0'):
            e=h[k];stats['models'][key][k]={'tau_eff':[e['tau_eff'],*e['tau_eff_ci95']],'in_range':e['in_range'],
                'ade1':{'reference':ref['ade1']['reference'],'sampler':e['ade1']['sampler'],'matched':e['ade1']['matched'],'residual':[e['ade1']['residual'],*e['ade1']['residual_ci95']],'equivalent':e['ade1']['equivalent']},
                'flip':{'reference':0.,'sampler':e['flip']['sampler'],'matched':e['flip']['matched'],'residual':[e['flip']['residual'],*e['flip']['residual_ci95']],'equivalent':e['flip']['equivalent']}}
    return stats


def matched_table(src=None,label='tab:matched',orig=False):
    stats=json.loads((src/'summary.json').read_text()) if src else load_pooled()
    rows=[]
    for key,mlabel in MODELS:
        model=stats['models'][key];first=True
        for sampler,name in SAMPLERS:
            if sampler not in model:continue
            e=model[sampler];ade=e['ade1'];flip=e['flip']
            mark='' if e['in_range'] else '$^\\dagger$'
            cells=[mlabel if first else '',name,num(e['tau_eff'][0])+mark,
                num(ade['sampler']),num(ade['matched']),
                f"{sgn(ade['residual'][0],3)}\\,[{num(ade['residual'][1],3)},\\,{num(ade['residual'][2],3)}]",
                num(100*flip['sampler'],1),num(100*flip['matched'],1),
                f"{num(100*flip['residual'][0],1)}\\,[{ci(100*flip['residual'][1])},\\,{ci(100*flip['residual'][2])}]"]
            rows.append(' & '.join(cells)+r' \\');first=False
        rows.append(r'\midrule')
    body='\n    '.join(rows[:-1])
    if orig:
        cap=rf"""\textbf{{The original sample gives a similar picture with wider intervals.}} Columns as in \cref{{tab:matched}}. {stats['clips']} clips of the PhysicalAI-AV test split, most of them in the Alpamayo training partition, $\times$ 6 noises; Heun was run for Alpamayo-1.5 only."""
    else:
        cap=rf"""\textbf{{Every sampler's residual on the {stats['clips']} held-out clips (the numbers behind main-text \cref{{M-tab:headline}}).}} $S$: the sampler; $A_{{\taueff}}$: the reference at the sampler's effective temperature; $S-A_{{\taueff}}$: the residual, with its 95\% clip-bootstrap interval. Flips: share of samples whose decision differs from $\Aref$ under the same noise. $\Aref$ has ADE$_1$ {', '.join(num(stats['models'][k]['euler10_t1.0']['ade1']['reference']) for k,_ in MODELS)}\,m in Alpamayo-1.5, -R1 and -2. Shrink$_k$: each $\Aref$ sample moved linearly toward $E_1$ until the spread equals $E_k$'s (\cref{{app:shrink}}). Heun for Alpamayo-R1 and -2 comes from the registered Heun controls, which have their own $\Aref$ (\cref{{app:hardware}}). $^\dagger$: $\taueff$ at the upper edge of the measured range."""
    return rf"""\begin{{table}}[!htbp]
  \caption{{{cap}}}
  \label{{{label}}}
  \centering\small\setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{@{{}}llcccc@{{\hspace{{14pt}}}}ccc@{{}}}}
    \toprule
    & & & \multicolumn{{3}}{{c}}{{ADE$_1$ (m)}} & \multicolumn{{3}}{{c}}{{Decision flips (\%)}} \\
    \cmidrule(lr){{4-6}}\cmidrule(l){{7-9}}
    Model & Sampler $S$ & $\taueff$ & $S$ & $A_{{\taueff}}$ & $S-A_{{\taueff}}$ & $S$ & $A_{{\taueff}}$ & $S-A_{{\taueff}}$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def contraction_table():
    stats=json.loads((POOLED/'summary.json').read_text())
    g=json.loads((RESULTS/'gaussian_contraction_v1_20261026/summary.json').read_text())
    rows=[]
    for sampler,name in [('euler4_t1.0','$E_4$'),('euler10_t1.0','$E_{10}$'),('heun5_t1.0','Heun-5')]:
        cells=[name]
        for key,_ in MODELS:
            e=stats['models'][key].get(sampler)
            cells.append(f"{num(e['contraction']['median'])} ({100*e['contraction']['fraction_below_one']:.0f}\\%)" if e else '--')
        k={'euler4_t1.0':'4','euler10_t1.0':'10'}.get(sampler)
        cells.append(num(g['steps'][k]['predicted_median']) if k else '--')
        rows.append(' & '.join(cells)+r' \\')
    body='\n    '.join(rows)
    return rf"""\begin{{table}}[!htbp]
  \caption{{\textbf{{Euler samplers contract toward the one-step mean as the Gaussian factor predicts; Heun barely contracts.}} Median ratio of a sample's distance to $E_1$ over the distance of the reference sample from the same noise; in parentheses, the share of samples with ratio below 1. {stats['clips']} held-out clips $\times$ 6 noises. Gaussian: the factor of main-text \cref{{M-eq:contraction}} computed from Alpamayo-1.5's conditional action variance (median over a separate sample of 200 clips; it does not predict the per-clip ranking).}}
  \label{{tab:contraction}}
  \centering\small\setlength{{\tabcolsep}}{{8pt}}
  \begin{{tabular}}{{@{{}}lcccc@{{}}}}
    \toprule
    Sampler & Alpamayo-1.5 & Alpamayo-R1 & Alpamayo-2 & Gaussian \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def table_macros():
    """Macros for numbers quoted in the text that come from the same artifacts as the tables."""
    stats=json.loads((POOLED/'summary.json').read_text())
    g=json.loads((RESULTS/'gaussian_contraction_v1_20261026/summary.json').read_text())
    a=stats['models']['alpamayo15'];lines=['% Generated by paper/tables/make_tables.py; do not edit by hand.']
    for sampler,tag in [('heun3_t1.0','HeunThree'),('heun5_t1.0','HeunFive')]:
        e=a[sampler]
        lines+=[f'\\newcommand{{\\AOneFive{tag}Tau}}{{{e["tau_eff"][0]:.2f}}}',f'\\newcommand{{\\AOneFive{tag}Flip}}{{{100*e["flip"]["sampler"]:.1f}}}',
            f'\\newcommand{{\\AOneFive{tag}FlipMatched}}{{{100*e["flip"]["matched"]:.1f}}}']
    lines+=[f'\\newcommand{{\\GaussEFour}}{{{g["steps"]["4"]["predicted_median"]:.2f}}}',f'\\newcommand{{\\GaussETen}}{{{g["steps"]["10"]["predicted_median"]:.2f}}}']
    # Ranges over the three models and the in-range released-style samplers (E4, E10), so the text never quotes a range by hand.
    cells=[stats['models'][key][s] for key,_ in MODELS for s in ('euler4_t1.0','euler10_t1.0')]
    expl=[100*e[m]['fraction_explained'] for e in cells for m in ('ade1','flip')]
    flip=[100*e['flip']['residual'][0] for e in cells];minade=[e['minade6']['residual'][0] for e in cells]
    ent=[e['entropy']['residual'][0] for e in cells];one=[100*stats['models'][key]['euler1_t1.0']['flip']['residual'][0] for key,_ in MODELS]
    a2=stats['models']['alpamayo2']['euler10_t1.0']['flip']['residual']
    lines+=[f'\\newcommand{{\\ExplMin}}{{{min(expl):.0f}}}',f'\\newcommand{{\\ExplMax}}{{{max(expl):.0f}}}',f'\\newcommand{{\\ExplRatioMin}}{{{min(expl)/100:.2f}}}',f'\\newcommand{{\\ExplRatioMax}}{{{max(expl)/100:.2f}}}',
        f'\\newcommand{{\\FlipResidMaxAbs}}{{{max(map(abs,flip)):.2f}}}',
        f'\\newcommand{{\\MinadeResidMin}}{{{num(min(minade)).replace('$-$','-')}}}',f'\\newcommand{{\\MinadeResidMax}}{{{max(minade):.2f}}}',
        f'\\newcommand{{\\MinadeResidPos}}{{{sum(r>0.005 for r in minade)}}}',f'\\newcommand{{\\NCells}}{{{len(cells)}}}',
        f'\\newcommand{{\\EntropyResidMin}}{{{min(ent):.2f}}}',f'\\newcommand{{\\EntropyResidMax}}{{{max(ent):.2f}}}',
        f'\\newcommand{{\\EOneFlipResidMin}}{{{min(one):.0f}}}',f'\\newcommand{{\\EOneFlipResidMax}}{{{max(one):.0f}}}',
        f'\\newcommand{{\\ATwoETenFlipResid}}{{{100*a2[0]:.1f}}}',f'\\newcommand{{\\ATwoETenFlipResidLo}}{{{100*a2[1]:.2f}}}',
        f'\\newcommand{{\\ATwoETenFlipResidHi}}{{{100*a2[2]:.1f}}}']
    rng_=lambda v,d=2,f=1:f'{f*min(v):.{d}f}' if f'{f*min(v):.{d}f}'==f'{f*max(v):.{d}f}' else f'{f*min(v):.{d}f}--{f*max(v):.{d}f}'
    for k,tag in ((4,'Four'),(10,'Ten')):
        eu=[stats['models'][key][f'euler{k}_t1.0']['contraction'] for key,_ in MODELS];sh=[stats['models'][key][f'shrink{k}_t1.0']['contraction'] for key,_ in MODELS]
        lines+=[f'\\newcommand{{\\SigEuler{tag}}}{{{rng_([c["median"] for c in eu])}}}',f'\\newcommand{{\\SigTemp{tag}}}{{{rng_([c["matched_median"] for c in eu])}}}',
            f'\\newcommand{{\\SigShrink{tag}}}{{{rng_([c["median"] for c in sh])}}}']
    eu=[stats['models'][key][f'euler{k}_t1.0']['contraction'] for key,_ in MODELS for k in (4,10)]
    lines+=[f'\\newcommand{{\\SigEulerFrac}}{{{rng_([c["fraction_below_one"] for c in eu],0,100)}}}',f'\\newcommand{{\\SigTempFrac}}{{{rng_([c["matched_fraction_below_one"] for c in eu],0,100)}}}']
    recs={r['name']:r for r in json.loads((RESULTS/'physicalai_heldout_contexts_v3_20261027/records.json').read_text()) if 'error' not in r}
    lines.append(f'\\newcommand{{\\StopClipMaxSpeed}}{{{max(recs[c]["speed_mps"] for c in stats["stop_clips"]):.1f}}}')
    for k in ('4','10'):
        g=stats['signature'][f'E{k}'];tag='Four' if k=='4' else 'Ten'
        for name,mac,scale,d in (('euler_minus_temp_median','SigDiffTemp',1,3),('euler_minus_shrink_median','SigDiffShrink',1,3),('euler_minus_temp_frac','SigDiffFrac',100,1)):
            v=[scale*x for x in g[name]]
            lines.append(f'\\newcommand{{\\{mac}{tag}}}{{{num(v[0],d)}\\,[{num(v[1],d)},\\,{num(v[2],d)}]}}')
    shr=[stats['models'][key][f'shrink{k}_t1.0'] for key,_ in MODELS for k in (4,10)]
    lines+=[f'\\newcommand{{\\ShrinkFlipResidMin}}{{{min(100*e["flip"]["residual"][0] for e in shr):.1f}}}',
        f'\\newcommand{{\\ShrinkFlipResidMax}}{{{max(100*e["flip"]["residual"][0] for e in shr):.1f}}}',
        f'\\newcommand{{\\ShrinkContrFour}}{{{rng_([e["contraction"]["median"] for e in shr[0::2]])}}}',
        f'\\newcommand{{\\ShrinkContrTen}}{{{rng_([e["contraction"]["median"] for e in shr[1::2]])}}}']
    fd=json.loads((RESULTS/'flashdrive_analysis_nvidia_v2_20261026/summary.json').read_text());c,e10=fd['samplers']['flashcache8_t1.0'],fd['samplers']['euler10_t1.0']
    lines+=[f'\\newcommand{{\\FDClips}}{{{fd["clips"]}}}',f'\\newcommand{{\\FDTau}}{{{c["tau_eff"]:.2f}}}',f'\\newcommand{{\\FDGain}}{{{fd["curve"]["1.0"][0]-c["ade1"]:.2f}}}',
        f'\\newcommand{{\\FDTauCI}}{{[{c["tau_ci90"][0]:.2f},\\,{c["tau_ci90"][1]:.2f}]}}',f'\\newcommand{{\\FDResidHi}}{{{c["residual_ci90"][1]:.3f}}}',
        f'\\newcommand{{\\FDGainEEight}}{{{-fd["paired_ade1"]["flashcache8_t1.0-euler8_t1.0"]["mean"]:.2f}}}',
        f'\\newcommand{{\\FDVsEFour}}{{{num(fd["paired_ade1"]["flashcache8_t1.0-euler4_t1.0"]["mean"],3)}\\,[{num(fd["paired_ade1"]["flashcache8_t1.0-euler4_t1.0"]["ci90"][0],3)},\\,{num(fd["paired_ade1"]["flashcache8_t1.0-euler4_t1.0"]["ci90"][1],3)}]}}',
        f'\\newcommand{{\\FDGainEEightCI}}{{[{-fd["paired_ade1"]["flashcache8_t1.0-euler8_t1.0"]["ci90"][1]:.2f},\\,{-fd["paired_ade1"]["flashcache8_t1.0-euler8_t1.0"]["ci90"][0]:.2f}]}}',
        f'\\newcommand{{\\FDEEightResid}}{{{num(fd["samplers"]["euler8_t1.0"]["residual"],3)}\\,[{num(fd["samplers"]["euler8_t1.0"]["residual_ci90"][0],3)},\\,{num(fd["samplers"]["euler8_t1.0"]["residual_ci90"][1],3)}]}}',
        f'\\newcommand{{\\FDResidPt}}{{{c["residual"]:.3f}}}',
        f'\\newcommand{{\\FDEFourTau}}{{{fd["samplers"]["euler4_t1.0"]["tau_eff"]:.2f}}}',f'\\newcommand{{\\FDEFourResidHi}}{{{fd["samplers"]["euler4_t1.0"]["residual_ci90"][1]:.3f}}}',f'\\newcommand{{\\FDEFourResid}}{{{num(fd["samplers"]["euler4_t1.0"]["residual"],3)}\\,[{num(fd["samplers"]["euler4_t1.0"]["residual_ci90"][0],3)},\\,{num(fd["samplers"]["euler4_t1.0"]["residual_ci90"][1],3)}]}}',
        f'\\newcommand{{\\FDResid}}{{{num(c["residual"],3)}\\,[{num(c["residual_ci90"][0],3)},\\,{num(c["residual_ci90"][1],3)}]}}',
        f'\\newcommand{{\\FDEEightTau}}{{{fd["samplers"]["euler8_t1.0"]["tau_eff"]:.2f}}}',f'\\newcommand{{\\FDETenTau}}{{{e10["tau_eff"]:.2f}}}',f'\\newcommand{{\\FDETenResid}}{{{num(e10["residual"],3)}\\,[{num(e10["residual_ci90"][0],3)},\\,{num(e10["residual_ci90"][1],3)}]}}']
    fz=json.loads((RESULTS/'flashdrive_analysis_zlab_v1_20261027/summary.json').read_text());cz=fz['samplers']['flashcache8_t1.0'];gz=fz['paired_ade1']['flashcache8_t1.0-euler8_t1.0']
    lines+=[f'\\newcommand{{\\FDZClips}}{{{fz["clips"]}}}',f'\\newcommand{{\\FDZTau}}{{{cz["tau_eff"]:.2f}\\,[{cz["tau_ci90"][0]:.2f},\\,{cz["tau_ci90"][1]:.2f}]}}',
        f'\\newcommand{{\\FDZGainEEight}}{{{-gz["mean"]:.2f}\\,[{-gz["ci90"][1]:.2f},\\,{-gz["ci90"][0]:.2f}]}}',
        f'\\newcommand{{\\FDZResid}}{{{num(cz["residual"],3)}\\,[{num(cz["residual_ci90"][0],3)},\\,{num(cz["residual_ci90"][1],3)}]}}',
        f'\\newcommand{{\\FDZEFourTau}}{{{fz["samplers"]["euler4_t1.0"]["tau_eff"]:.2f}}}']
    ov=json.loads((RESULTS/'split_overlap_v1_20261027/summary.json').read_text())['samples']['main']['partition']
    lines+=[f'\\newcommand{{\\NClipsInTrain}}{{{ov.get("train",0)}}}']
    qdall=json.loads((RESULTS/'qwendrive_analysis_v2_20261027/summary.json').read_text())['planners'];qd=qdall['qwendrive_probe_rl_v2_20261027']
    qs=qdall['qwendrive_probe_sft_v2_20261027']['samplers']['euler4_t1.0']
    q4,q10=qd['samplers']['euler4_t1.0'],qd['samplers']['euler10_t1.0'];band=lambda x,d=3:f'{num(x[0],d)},\\,{num(x[1],d)}'
    lines+=[f'\\newcommand{{\\QDClips}}{{{qd["clips"]}}}',f'\\newcommand{{\\QDConvMM}}{{{1000*qd["heun64_vs_heun32_max_m"]:.1f}}}',f'\\newcommand{{\\QDRefSpread}}{{{qd["curve_spread"]["heun32_t1.0"]:.2f}}}',
        f'\\newcommand{{\\QDRefADE}}{{{q4["ade1"]["reference"]:.2f}}}',f'\\newcommand{{\\QDETenTau}}{{{q10["tau_eff"]:.2f}}}',
        f'\\newcommand{{\\QDETenResid}}{{{num(q10["ade1"]["residual"],3)}\\,[{band(q10["ade1"]["residual_ci90"])}]}}',
        f'\\newcommand{{\\QDEFourTau}}{{{q4["tau_eff"]:.2f}\\,[{band(q4["tau_eff_ci90"],2)}]}}',
        f'\\newcommand{{\\QDEFourResid}}{{{num(q4["ade1"]["residual"],3)}\\,[{band(q4["ade1"]["residual_ci90"])}]}}',
        f'\\newcommand{{\\QDEFourMatchedGain}}{{{q4["ade1"]["reference"]-q4["ade1"]["matched"]:.3f}}}',
        f'\\newcommand{{\\QDEFourFlipResid}}{{{num(100*q4["flip"]["residual"],1)}\\,[{band([100*x for x in q4["flip"]["residual_ci90"]],1)}]}}',f'\\newcommand{{\\QDSftEFourTau}}{{{qs["tau_eff"]:.2f}\\,[{band(qs["tau_eff_ci90"],2)}]}}',
        f'\\newcommand{{\\QDSftEFourResid}}{{{num(qs["ade1"]["residual"],3)}\\,[{band(qs["ade1"]["residual_ci90"])}]}}',
        f'\\newcommand{{\\QDSftEFourFlipResid}}{{{num(100*qs["flip"]["residual"],1)}\\,[{band([100*x for x in qs["flip"]["residual_ci90"]],1)}]}}',
        f'\\newcommand{{\\QDSftEFourMatchedGain}}{{{qs["ade1"]["reference"]-qs["ade1"]["matched"]:.3f}}}',f'\\newcommand{{\\QDEFourFlip}}{{{100*q4["flip"]["sampler"]:.1f}}}',f'\\newcommand{{\\QDEFourFlipMatched}}{{{100*q4["flip"]["matched"]:.1f}}}']
    # held-out replication (held-out replication): 95% intervals, TOST verdicts at 90% as stored by paper_numbers
    ho={k:json.loads((RESULTS/f'paper_numbers_heldout_{t}_v1_20261027/summary.json').read_text()) for k,t in [('alpamayo15','a15'),('alpamayo_r1','r1'),('alpamayo2','a2')]}
    tags={'alpamayo15':'AOneFive','alpamayo_r1':'ROne','alpamayo2':'ATwo'};e10={k:ho[k]['models'][k]['euler10_t1.0'] for k in ho}
    lines+=[f'\\newcommand{{\\HOClips}}{{{ho["alpamayo2"]["clips"]}}}',
        f'\\newcommand{{\\HOAdePass}}{{{sum(e["ade1"]["equivalent"] for e in e10.values())}}}',
        f'\\newcommand{{\\HOBothPass}}{{{sum(e["ade1"]["equivalent"] and e["flip"]["equivalent"] for e in e10.values())}}}']
    for k,e in e10.items():
        a,f=e['ade1']['residual'],[100*x for x in e['flip']['residual']]
        lines+=[f'\\newcommand{{\\HO{tags[k]}ETenTau}}{{{e["tau_eff"][0]:.2f}}}',
            f'\\newcommand{{\\HO{tags[k]}ETenAdeResid}}{{{num(a[0],3)}\\,[{num(a[1],3)},\\,{num(a[2],3)}]}}',
            f'\\newcommand{{\\HO{tags[k]}ETenFlipResid}}{{{num(f[0],1)}\\,[{num(f[1],1)},\\,{num(f[2],1)}]}}']
    # closed loop (AlpaSim, protocols/closed_loop_alpasim.md): scene-paired McNemar
    cl=json.loads((RESULTS/'closedloop_analysis_v1_20261028/summary.json').read_text());r=cl['rates']
    inc={'Ref':'heun32_t1.0','TauEightFour':'heun32_t0.84','TauSixFive':'heun32_t0.65','ETen':'euler10_t1.0','EFour':'euler4_t1.0'}
    lines+=[f'\\newcommand{{\\CLScenes}}{{{cl["common_scenes"]}}}',f'\\newcommand{{\\CLMinP}}{{{min(t["p_exact_mcnemar"] for t in cl["tests"]):.2f}}}',
        f'\\newcommand{{\\CLMaxDiscordant}}{{{max(t["only_a"]+t["only_b"] for t in cl["tests"])}}}']
    lines+=[f'\\newcommand{{\\CLInc{k}}}{{{100*r[c]["incident"]:.0f}}}' for k,c in inc.items()]
    # closed-loop comfort (comfort): scene-bootstrap 95% intervals of paired differences
    cf=json.loads((RESULTS/'closedloop_comfort_v1_20261028/summary.json').read_text())
    test=lambda lab,m:next(t for t in cf['tests'] if t['label']==lab and t['metric']==m)
    for key,lab,m in [('CLJerkTau','C1','jerk_rms'),('CLDecelTau','C1','max_decel'),('CLJerkETen','C3 unmatched','jerk_rms'),('CLJerkETenMatched','C3 matched','jerk_rms')]:
        # three decimals: two would print the bounds +0.005 and -0.002 as 0.00
        t=test(lab,m);lines.append(f'\\newcommand{{\\{key}}}{{{num(t["diff"],3)}\\,[{num(t["diff_ci95"][0],3)},\\,{num(t["diff_ci95"][1],3)}]}}')
    hb=[cf['means'][c]['hard_brake'] for c in cf['means']]
    lines+=[f'\\newcommand{{\\CLHardBrakeMin}}{{{100*min(hb):.0f}}}',f'\\newcommand{{\\CLHardBrakeMax}}{{{100*max(hb):.0f}}}']
    # held-out extension context_120-399 (extension): R1 and A2
    ex=json.loads((RESULTS/'paper_numbers_heldout_ext_v1_20261028/summary.json').read_text())
    lines.append(f'\\newcommand{{\\EXClips}}{{{ex["clips"]}}}')
    lines.append(f'\\newcommand{{\\EXFlipPass}}{{{sum(ex["models"][k]["euler10_t1.0"]["flip"]["equivalent"] and ex["models"][k]["euler10_t1.0"]["ade1"]["equivalent"] for k in ex["models"])}}}')
    for k in ex['models']:
        for s,st in [('euler10_t1.0','ETen'),('euler4_t1.0','EFour')]:
            e=ex['models'][k][s];a,f=e['ade1']['residual'],[100*x for x in e['flip']['residual']]
            lines+=[f'\\newcommand{{\\EX{tags[k]}{st}AdeResid}}{{{num(a[0],3)}\\,[{num(a[1],3)},\\,{num(a[2],3)}]}}',
                f'\\newcommand{{\\EX{tags[k]}{st}FlipResid}}{{{num(f[0],1)}\\,[{num(f[1],1)},\\,{num(f[2],1)}]}}']
    # margins as a share of E10's raw held-out effects; E4's held-out ADE1 verdicts
    pct=[100*.05/abs(e['ade1']['effect'][0]) for e in e10.values()]+[100*.02/abs(e['flip']['effect'][0]) for e in e10.values()]
    e4={k:ho[k]['models'][k]['euler4_t1.0'] for k in ho}
    lines+=[f'\\newcommand{{\\HOMarginPctMin}}{{{min(pct):.0f}}}',f'\\newcommand{{\\HOMarginPctMax}}{{{max(pct):.0f}}}',
        f'\\newcommand{{\\HOEFourAdePass}}{{{sum(e["ade1"]["equivalent"] for e in e4.values())}}}']
    for key,lab,m in [('CLJerkEFourMatched','C2 matched','jerk_rms'),('CLJerkEFour','C2 unmatched','jerk_rms'),('CLDecelETenMatched','C3 matched','max_decel'),('CLDecelETen','C3 unmatched','max_decel'),('CLDecelEFourMatched','C2 matched','max_decel'),('CLDecelEFour','C2 unmatched','max_decel')]:
        t=test(lab,m);lines.append(f'\\newcommand{{\\{key}}}{{{num(t["diff"],3)}\\,[{num(t["diff_ci95"][0],3)},\\,{num(t["diff_ci95"][1],3)}]}}')
    hbs={c:round(cf['means'][c]['hard_brake']*cf['scenes']) for c in cf['means']}
    lines+=[f'\\newcommand{{\\CLHardBrakeETen}}{{{hbs["euler10_t1.0"]}}}',f'\\newcommand{{\\CLHardBrakeRef}}{{{hbs["heun32_t1.0"]}}}',
        f'\\newcommand{{\\CLHardBrakeETenP}}{{{test("C3 unmatched","hard_brake")["p_exact_mcnemar"]:.2f}}}']
    # FlashDrive flips and minADE6 (exploratory, FlashDrive): 90% clip-bootstrap intervals
    ff=json.loads((RESULTS/'flashdrive_flips_v1_20261028/summary.json').read_text())
    for tag,pr in [('FDF','flashdrive_probe_nvidia_v1'),('FDZF','flashdrive_probe_zlab_v1')]:
        c=ff[pr]['samplers']['flashcache8_t1.0'];fr=[100*x for x in c['flip_resid_ci90']]
        lines+=[f'\\newcommand{{\\{tag}FlipResid}}{{{num(100*c["flip_resid"],1)}\\,[{num(fr[0],1)},\\,{num(fr[1],1)}]}}',
            f'\\newcommand{{\\{tag}Flip}}{{{100*c["flip"]:.1f}}}',
            f'\\newcommand{{\\{tag}MinadeRef}}{{{c["minade6_ref"]:.2f}}}',f'\\newcommand{{\\{tag}Minade}}{{{c["minade6"]:.2f}}}',
            f'\\newcommand{{\\{tag}MinadeResid}}{{{num(c["minade6_resid"],3)}\\,[{num(c["minade6_resid_ci90"][0],3)},\\,{num(c["minade6_resid_ci90"][1],3)}]}}']
    return '\n'.join(lines)+'\n'


# ---------------------------------------------------------------------------------------------------------------------
# v5.32: pooled 400 held-out clips as primary evidence, Heun controls, NAVSIM second family.
def sgn(x,d=1,scale=1):
    """Signed number with a true minus sign."""
    t=num(scale*x,d);return t if t.startswith('$-$') or float(t)==0 else '$+$'+t


def interval(r,d=1,scale=1):
    return f'{sgn(r[0],d,scale)}\\,[{num(scale*r[1],d)},\\,{num(scale*r[2],d)}]'


def pooled_extra():
    st=load_pooled();L=['% Generated by paper/tables/make_tables.py (pooled 400 held-out clips, Heun controls, NAVSIM); do not edit by hand.']
    add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    # numbers.tex: the pooled-400 macro file of paper_numbers, copied verbatim
    (OUT/'numbers.tex').write_text('% Copied by make_tables.py from artifacts/paper_numbers_heldout_pooled400_v2_20261028.macros.tex; do not edit by hand.\n'
        +''.join(l for l in (RESULTS/'paper_numbers_heldout_pooled400_v2_20261028.macros.txt').read_text().splitlines(True) if not l.startswith('%')))
    o=json.loads((ORIG/'summary.json').read_text());add('OrigClips',o['clips'])
    tv=[st['models'][k]['euler10_t1.0']['tau_eff'][0] for k in TAG];add('PETenTauMean',num(sum(tv)/3))
    names={'euler10':'ETen','euler4':'EFour','shrink10':'ShrTen','shrink4':'ShrFour'};cnt={}
    for key,t in TAG.items():
        for s,n in names.items():
            e=st['models'][key][f'{s}_t1.0']
            add(f'P{t}{n}Flip',interval(e['flip']['residual'],1,100));add(f'P{t}{n}Ade',interval(e['ade1']['residual'],3))
            add(f'P{t}{n}FlipPt',sgn(e['flip']['residual'][0],1,100))
            for m in ('flip','ade1'):
                c=cnt.setdefault((n,m),{'eq':0,'excl':0})
                c['eq']+=bool(e[m]['equivalent']);c['excl']+=e[m]['residual'][1]>0 or e[m]['residual'][2]<0
        for k,h in ((3,'Three'),(5,'Five')):
            e=st['models'][key][f'heun{k}_t1.0'];add(f'P{t}Heun{h}Flip',interval(e['flip']['residual'],1,100));add(f'P{t}Heun{h}FlipPt',sgn(e['flip']['residual'][0],1,100))
            add(f'P{t}Heun{h}Ade',interval(e['ade1']['residual'],3));add(f'P{t}Heun{h}Tau',num(e['tau_eff'][0]))
    for (n,m),c in cnt.items():
        mm='Flip' if m=='flip' else 'Ade';add(f'P{n}{mm}Equiv',c['eq']);add(f'P{n}{mm}Excl',c['excl'])
    for k,h in ((3,'Three'),(5,'Five')):
        v=[100*st['models'][key][f'heun{k}_t1.0']['flip']['residual'][0] for key in TAG]
        add(f'PHeun{h}FlipMin',f'{min(v):.1f}');add(f'PHeun{h}FlipMax',f'{max(v):.1f}')
        v=[100*st['models'][key][f'heun{k}_t1.0']['flip']['residual'][1] for key in TAG];add(f'PHeun{h}FlipLoMin',f'{min(v):.1f}')
        v=[st['models'][key][f'heun{k}_t1.0']['tau_eff'][0] for key in TAG];add(f'PHeun{h}TauMin',num(min(v)));add(f'PHeun{h}TauMax',num(max(v)))
    # the pre-registered held-out history (primary 120 clips): E10 flip equivalence in how many models
    ho={k:json.loads((RESULTS/f'paper_numbers_heldout_{t}_v1_20261027/summary.json').read_text()) for k,t in [('alpamayo15','a15'),('alpamayo_r1','r1'),('alpamayo2','a2')]}
    add('HOFlipPass',sum(ho[k]['models'][k]['euler10_t1.0']['flip']['equivalent'] for k in ho))
    # Heun A2 reference sensitivity (identical reference on clips 200-399)
    a2=json.loads((RESULTS/'heun_controls_analysis_a2_v1_20261028/summary.json').read_text())
    idr=json.loads((RESULTS/'heun_controls_analysis_a2_identicalref_200_400_v1_20261028/summary.json').read_text())
    add('IDClips',idr['clips']);add('IDRefDiffMax',f'{a2["reference_max_abs_diff_m"]:.1f}')
    for k,h in ((3,'Three'),(5,'Five')):
        e=idr['samplers'][f'heun{k}_t1.0'];add(f'ID{h}Flip',interval([e['flip']['residual'],*e['flip']['residual_ci95']],1,100))
        add(f'ID{h}Ade',interval([e['ade1']['residual'],*e['ade1']['residual_ci95']],3))
        add(f'ID{h}Tau',num(e['tau_eff']))
    add('IDMinFlip',f'{min(100*idr["samplers"][f"heun{k}_t1.0"]["flip"]["residual_ci95"][0] for k in (3,5)):.1f}')
    # NAVSIM
    cfg={'ddim2_lead_eta0_t1.0':'ETwo','ddim4_lead_eta0_t1.0':'EFour','ddim5_lead_eta0_t1.0':'EFive','ddim10_lead_eta0_t1.0':'ETen',
         'ddim20_lead_eta0_t1.0':'ETwenty','ddim50_lead_eta0_t1.0':'EFifty','release5_lead_eta1_t1.0':'Rel'}
    nv={}
    for fam in ('il','rl'):
        d=json.loads((RESULTS/f'navsim_analysis_recogdrive_{fam}_addendum2_v1_20261028/summary.json').read_text());nv[fam]=d;F=fam.upper()
        add(f'NV{F}Scenes',d['scenes']);add(f'NV{F}NOneP','passes' if d['N1_prime'] else 'fails')
        for c,n in cfg.items():
            e=d['samplers'][c];ok=e['in_range']
            add(f'NV{F}{n}Tau',num(e['tau_eff']) if ok else 'out of range')
            if ok:
                add(f'NV{F}{n}Flip',interval([e['flip']['residual'],*[x for x in e['flip']['residual_ci90']]],1,100));add(f'NV{F}{n}FlipPt',sgn(e['flip']['residual'],1,100))
                add(f'NV{F}{n}Pdms',sgn(e['pdms']['residual'],2,100));add(f'NV{F}{n}Ade',sgn(e['ade1']['residual'],3))
        grid=[d['samplers'][f'ddim{k}_lead_eta0_t1.0'] for k in (4,5,10,20,50)]
        add(f'NV{F}GridEquiv',sum(all(e[m].get('equivalent') for m in ('pdms','ade1','flip')) for e in grid))
        add(f'NV{F}RelEquivPdmsAde','yes' if all(d['samplers']['release5_lead_eta1_t1.0'][m]['equivalent'] for m in ('pdms','ade1')) else 'no')
    for fam in ('il','rl'):
        F=fam.upper();n2=nv[fam]['N2_prime'];lead=[nv[fam]['samplers'][f'ddim{k}_lead_eta0_t1.0'] for k in (2,4,5,10,20,50)]
        add(f'NV{F}LeadEq',sum(bool(e['in_range']) and all(e[m]['equivalent'] for m in ('pdms','ade1','flip')) for e in lead))
        add(f'NV{F}NTwoFiveRates','yes' if n2['ddim5_lead_eta0_t1.0']['pdms'] and n2['ddim5_lead_eta0_t1.0']['ade1'] else 'no')
        add(f'NV{F}NTwoTenRates','yes' if n2['ddim10_lead_eta0_t1.0']['pdms'] and n2['ddim10_lead_eta0_t1.0']['ade1'] else 'no')
    add('NVLeadN',6)
    n2i,n2r=nv['il']['N2_prime'],nv['rl']['N2_prime']
    assert all(n2i[k]['pdms'] and n2i[k]['ade1'] and n2i[k]['flip'] for k in n2i)   # IL: E5, E10 equivalent in all three; flips against our prediction
    assert n2r['ddim5_lead_eta0_t1.0']['pdms'] and n2r['ddim5_lead_eta0_t1.0']['ade1'] and not n2r['ddim5_lead_eta0_t1.0']['flip']
    assert not any(n2r['ddim10_lead_eta0_t1.0'].values()) and not nv['rl']['samplers']['ddim10_lead_eta0_t1.0']['in_range']
    assert nv['il']['samplers']['ddim2_lead_eta0_t1.0']['in_range']
    rl=nv['rl']['samplers'];assert all(not rl[f'ddim{k}_lead_eta0_t1.0']['in_range'] and rl[f'ddim{k}_lead_eta0_t1.0']['spread']>rl['ddim50_lead_eta0_t1.0']['spread'] for k in (10,20))
    # trailing grid (registered v1 analysis, decision bug for 8-point plans) vs corrected flips (v2)
    v1=json.loads((RESULTS/'navsim_analysis_recogdrive_rl_v1_20261028/summary.json').read_text())['samplers']
    v2=json.loads((RESULTS/'navsim_analysis_recogdrive_rl_v2_20261028/summary.json').read_text())['samplers']
    for c,n in (('ddim5_trail_eta0_t1.0','TrailFive'),('release5_lead_eta1_t1.0','RelOld')):
        add(f'NVRL{n}FlipRegistered',sgn(v1[c]['flip']['residual'],1,100));add(f'NVRL{n}FlipCorrected',sgn(v2[c]['flip']['residual'],1,100))
    add('NVRLTrailFivePdms',sgn(v1['ddim5_trail_eta0_t1.0']['pdms']['residual'],1,100));add('NVRLTrailFiveAde',sgn(v1['ddim5_trail_eta0_t1.0']['ade1']['residual'],2))
    # DiffusionDrive
    dd=json.loads((RESULTS/'navsim_analysis_diffusiondrive_v3_20261029/summary.json').read_text())['samplers']
    r,c5=dd['release2_lead_eta0_t1.0'],dd['ddim5_int_eta0_t1.0'];assert not r['in_range']
    add('DDAdeEffect',sgn(r['ade1']['effect'],3));add('DDPdmsEffect',sgn(r['pdms']['effect'],2,100))
    add('DDFlip',f'{100*r["flip"]["sampler"]:.1f}');add('DDFlipFloor',f'{100*c5["flip"]["sampler"]:.1f}')
    c=r['tau_eff_ci95_censored'];add('DDTauCI',f'{c[0]:.2f} to above 1')  # out-of-range resamples kept (audit 29/10)
    il=nv['il']['samplers'];allil=list(il.values())
    add('NVILRawPdmsMax',f"{100*max(abs(e['pdms']['effect']) for e in allil):.2f}");add('NVILRawAdeMax',f"{max(abs(e['ade1']['effect']) for e in allil):.3f}")
    assert all(abs(e['pdms']['effect'])<=nv['il']['margins']['pdms'] and abs(e['ade1']['effect'])<=nv['il']['margins']['ade1'] for e in allil)
    add('NVILFlipFailN',sum(not e['flip']['equivalent'] for e in allil if e['in_range']))
    rls=[e for e in nv['rl']['samplers'].values() if e['in_range']]
    add('NVRLRawOutN',sum(abs(e['pdms']['effect'])>nv['rl']['margins']['pdms'] or abs(e['ade1']['effect'])>nv['rl']['margins']['ade1'] for e in rls))
    # v5.36: raw flips inside the margin on IL; RL ADE1 residuals of E4/E5 (better than the matched temperature)
    for n,c in (('ETwenty','ddim20_lead_eta0_t1.0'),('EFifty','ddim50_lead_eta0_t1.0')):
        e=il[c];assert abs(e['flip']['effect'])<=nv['il']['margins']['flip']
        add(f'NVILRawFlip{n}',f"{100*e['flip']['effect']:.1f}")
    for n,c in (('EFour','ddim4_lead_eta0_t1.0'),('EFive','ddim5_lead_eta0_t1.0')):
        e=nv['rl']['samplers'][c];assert e['ade1']['residual_ci90'][1]<0
        add(f'NVRL{n}AdeResid',interval([e['ade1']['residual'],*e['ade1']['residual_ci90']],3))
    add('NVILRawFlipInsideN',sum(abs(e['flip']['effect'])<=nv['il']['margins']['flip'] for e in il.values()))
    add('NVRawFlipMargin',f"{100*nv['il']['margins']['flip']:.0f}")
    # spreads behind the IL E2 effective temperature; reference sizes of the original and corrected analyses
    for n,c in (('ETwo','ddim2_lead_eta0_t1.0'),('EFour','ddim4_lead_eta0_t1.0'),('EFifty','ddim50_lead_eta0_t1.0')):add(f'NVILSpread{n}',num(il[c]['spread'],3))
    steps=lambda d:re.match(r'ddim(\d+)',d['reference']).group(1)
    add('NVRefStepsCorrected',steps(nv['rl']));v2d=json.loads((RESULTS/'navsim_analysis_recogdrive_rl_v2_20261028/summary.json').read_text());add('NVRefStepsOriginal',steps(v2d))
    (OUT/'numbers_extra.tex').write_text('\n'.join(L)+'\n')
    (OUT/'tab_navsim.tex').write_text(navsim_table(nv,dd))


def navsim_table(nv,dd):
    rows=[];fmt=lambda e,m,d,sc:f"{sgn(e[m]['residual'],d,sc)}"
    def cell(e,m,d,sc,ci=False):
        t=fmt(e,m,d,sc)
        if ci:
            bd=lambda x:num(x,2) if abs(abs(x)-2)<0.05 else num(x,1)
            t+=f"\\,{{\\scriptsize[{bd(100*e[m]['residual_ci90'][0])},\\,{bd(100*e[m]['residual_ci90'][1])}]}}"
        return t if e[m]['equivalent'] else f'\\textbf{{{t}}}'
    spec=[('IL','il',[('ddim2_lead_eta0_t1.0','$E_2$'),('ddim4_lead_eta0_t1.0','$E_4$'),('ddim5_lead_eta0_t1.0','$E_5$'),('ddim10_lead_eta0_t1.0','$E_{10}$'),('ddim20_lead_eta0_t1.0','$E_{20}$'),('ddim50_lead_eta0_t1.0','$E_{50}$'),('release5_lead_eta1_t1.0','release')]),
          ('RL','rl',[('ddim2_lead_eta0_t1.0','$E_2$'),('ddim4_lead_eta0_t1.0','$E_4$'),('ddim5_lead_eta0_t1.0','$E_5$'),('ddim10_lead_eta0_t1.0','$E_{10}$'),('ddim20_lead_eta0_t1.0','$E_{20}$'),('ddim50_lead_eta0_t1.0','$E_{50}$'),('release5_lead_eta1_t1.0','release')])]
    blocks={}
    for mlabel,fam,cs in spec:
        first=True;rows=[]
        for c,name in cs:
            e=nv[fam]['samplers'][c];lab=f'ReCog.-{mlabel}' if first else '';first=False
            raw=lambda m,d,sc,mg:(lambda t:t if abs(e[m]['effect'])<=mg else f'\\textbf{{{t}}}')(sgn(e[m]['effect'],d,sc))
            if e['in_range']:rows.append(f"{lab} & {name} & {num(e['tau_eff'])} & {raw('pdms',2,100,nv[fam]['margins']['pdms'])} & {cell(e,'pdms',2,100)} & {raw('ade1',3,1,nv[fam]['margins']['ade1'])} & {cell(e,'ade1',3,1)} & {raw('flip',1,100,nv[fam]['margins']['flip'])} & {cell(e,'flip',1,100,True)} \\\\")
            else:
                why='spread below the measured range' if e['spread']<nv[fam]['samplers']['ddim50_lead_eta0_t1.0']['spread'] else 'spread above the reference'
                rows.append(f"{lab} & {name} & \\multicolumn{{7}}{{c}}{{no $\\taueff$ ({why})}} \\\\")
        blocks[fam]=rows
    r=dd['release2_lead_eta0_t1.0']
    blocks['rl']=blocks['rl']+[r'\midrule',f"DiffDrive & release & \\multicolumn{{7}}{{c}}{{no $\\taueff$ (bootstrap {dd_tau_text(r)})}} \\\\"]
    head=r"""  \centering\footnotesize\setlength{\tabcolsep}{3.2pt}
  \begin{tabular}{@{}llccccccc@{}}
    \toprule
    Model & Sampler & $\taueff$ & PDMS raw & PDMS resid. & ADE$_1$ raw & ADE$_1$ resid. & Flips raw & Flips resid. \\
    \midrule
    """
    cap1=rf"""\textbf{{ReCogDrive before RL: leading-grid DDIM with 4 to 50 steps is temperature-like (the numbers behind \cref{{fig:navsim}}).}} Raw: the sampler's own change against the reference at $\tau{{=}}1$ (PDMS, flips in points, ADE$_1$ in m). Resid.: what remains against the 100-step reference at its effective temperature (flips in points, 90\% interval). Every raw PDMS and ADE$_1$ effect is already inside the margin, and so are the raw flips of $E_{{20}}$ and $E_{{50}}$, so equivalence there is uninformative; the flips of the other samplers carry the claim. Bold: outside the pre-registered margin (1 point PDMS, 0.05\,m, 2 points flips). Release: 5 steps, $\eta{{=}}1$. {nv['il']['scenes']} NAVSIM navtest scenes $\times$ 6 noises."""
    cap2=r"""\textbf{ReCogDrive after RL fine-tuning, and DiffusionDrive: temperature no longer accounts for the sampler.} Columns and bold as in \cref{tab:navsim}. The negative ADE$_1$ residuals of $E_4$ and $E_5$ mean better than the matched temperature."""
    b_il='\n    '.join(blocks['il']);b_rl='\n    '.join(blocks['rl'])
    t1=rf"""\begin{{table}}[!htb]
  \caption{{{cap1}}}
  \label{{tab:navsim}}
{head}{b_il}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    t2=rf"""\begin{{table}}[!htb]
  \caption{{{cap2}}}
  \label{{tab:navsim-rl}}
{head}{b_rl}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    return t1+t2


def v533_extra():
    """v5.33: A2 Blackwell sensitivity, extension with all three models, scaled-noise control (scaled-noise check), minADE6 support."""
    L=['% Generated by paper/tables/make_tables.py (v5.33 additions); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    st=load_pooled()
    # minADE6 on the pooled 400 clips: how many of the six E4/E10 residual intervals exclude zero (claim dropped if none)
    cells=[st['models'][k][s]['minade6']['residual'] for k in TAG for s in ('euler4_t1.0','euler10_t1.0')]
    add('MinadeExclCount',sum(c[1]>0 or c[2]<0 for c in cells))
    # extension (280 clips): registered models R1/A2 and Alpamayo-1.5 run afterwards
    ex=json.loads((RESULTS/'paper_numbers_heldout_ext_v2_20261028/summary.json').read_text())
    ok=lambda k:ex['models'][k]['euler10_t1.0']['flip']['equivalent'] and ex['models'][k]['euler10_t1.0']['ade1']['equivalent']
    add('EXAllPass',sum(ok(k) for k in ex['models']));add('EXRegPass',sum(ok(k) for k in ('alpamayo_r1','alpamayo2')))
    add('HOPrimaryClips',json.loads((RESULTS/'paper_numbers_heldout_a2_v1_20261027/summary.json').read_text())['clips'])
    # A2 extension on Blackwell only (Blackwell sensitivity)
    bw=json.loads((RESULTS/'paper_numbers_heldout_ext_a2blackwell_v1_20261028/summary.json').read_text());e=bw['models']['alpamayo2']['euler10_t1.0']
    add('BWClips',bw['clips']);add('HTwoHundredClips',400-bw['clips']);add('BWTau',num(e['tau_eff'][0]));add('BWAdeResid',interval(e['ade1']['residual'],3));add('BWFlipResid',interval(e['flip']['residual'],1,100))
    add('BWEquiv','equivalent' if e['ade1']['equivalent'] and e['flip']['equivalent'] else 'not equivalent')
    # scaled-noise control
    sn=json.loads((RESULTS/'scaled_noise_control_a15_v1_20261029/summary.json').read_text());A,B=sn['curves']['A_heun32'],sn['curves']['B_euler10']
    add('SNClips',sn['clips']);add('SNSpreadRef',num(A['heun32_t1.0']['spread']));add('SNSpreadEuler',num(B['euler10_t1.0']['spread']))
    add('SNSpreadDeficit',f"{100*(1-B['euler10_t1.0']['spread']/A['heun32_t1.0']['spread']):.0f}")
    names={'euler2_t1.0':'Two','euler4_t1.0':'Four','euler6_t1.0':'Six'};rows=[]
    for sk,n in names.items():
        for c,cl in (('A_heun32','A'),('B_euler10','B')):
            e=sn['samplers'][sk][c];add(f'SN{n}{cl}Tau',num(e['tau_eff']))
            add(f'SN{n}{cl}Ade',interval([e['ade1']['residual'],*e['ade1']['residual_ci95']],3));add(f'SN{n}{cl}Flip',interval([e['flip']['residual'],*e['flip']['residual_ci95']],1,100))
    add('SNMatchedEquiv',sum(sn['samplers'][s][c][m]['equivalent'] for s in ('euler4_t1.0','euler6_t1.0') for c in ('A_heun32','B_euler10') for m in ('ade1','flip')))
    hdr=[]
    for sk,n in names.items():
        k={'Two':2,'Four':4,'Six':6}[n]
        for c,cl,lab in (('A_heun32','A','Heun-32'),('B_euler10','B','Euler-10')):
            e=sn['samplers'][sk][c];f=lambda m,d,sc:(lambda t:t if e[m]['equivalent'] else f'\\textbf{{{t}}}')(interval([e[m]['residual'],*e[m]['residual_ci95']],d,sc))
            rows.append(f"$E_{k}$ & {lab} & {num(e['tau_eff'])} & {f('ade1',3,1)} & {f('flip',1,100)} \\\\")
        rows.append(r'\midrule')
    body='\n    '.join(rows[:-1])
    (OUT/'tab_scaled.tex').write_text(rf"""\begin{{table}}[!htbp]
  \caption{{\textbf{{Matching the spread, on either temperature curve, reproduces $E_4$ and $E_6$.}} Each $E_k$ is placed, at the $\tau$ that matches its spread, on the temperature curve of our reference (Heun-32) and on that of Euler-10 started from scaled noise (analogous to the scaled-noise baseline of \citet{{yang2026coupling}}, who match the endpoint radius rather than the spread). The effective temperature depends on the curve. Residuals with 95\% clip-bootstrap intervals; bold: not inside the margin. A post hoc check on plans already collected; Alpamayo-1.5 only, {sn['clips']} held-out clips $\times$ 6 noises; the lowest temperature measured is $\tau{{=}}0.4$. This single-model bootstrap can differ in the last digit from the joint intervals of \cref{{tab:matched}}.}}
  \label{{tab:scaled}}
  \centering\small\setlength{{\tabcolsep}}{{8pt}}
  \begin{{tabular}}{{@{{}}llccc@{{}}}}
    \toprule
    Sampler & Temperature curve & $\taueff$ & ADE$_1$ residual (m) & Flip residual (points) \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    (OUT/'numbers_controls.tex').write_text('\n'.join(L)+'\n')


def lowtau_extra():
    """Addendum 16: reference at tau 0.3/0.2/0.1, per model as results arrive; LTModelsDone counts them."""
    L=['% Generated by paper/tables/make_tables.py (low-temperature extension); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    done=0;counts={};trows=[]
    for key,tag,mname in (('a15','AOneFive','Alpamayo-1.5'),('r1','ROne','Alpamayo-R1'),('a2','ATwo','Alpamayo-2')):
        path=RESULTS/f'lowtau_analysis_{key}_v1_20261029/summary.json'
        if not path.exists():continue
        lt=json.loads(path.read_text());done+=1
        for s,st in (('euler1_t1.0','EOne'),('euler2_t1.0','ETwo')):
            m=lt['samplers'][s];add(f'LT{tag}{st}Tau',num(m['tau_eff']));add(f'LT{tag}{st}TauCI',f"[{num(m['tau_eff_ci95'][0],2)},\\,{num(m['tau_eff_ci95'][1],2)}]")
            add(f'LT{tag}{st}AdeResid',interval([m['ade1']['residual']]+m['ade1']['residual_ci95'],3))
            add(f'LT{tag}{st}FlipResid',interval([m['flip']['residual']]+m['flip']['residual_ci95'],1,100))
            add(f'LT{tag}{st}AdePt',sgn(m['ade1']['residual'],3));add(f'LT{tag}{st}FlipPt',sgn(m['flip']['residual'],1,100))
            b=lambda t,ok:t if ok else f'\\textbf{{{t}}}'
            trows.append(f"{mname if st=='EOne' else ''} & {'$E_1$' if st=='EOne' else '$E_2$'} & {num(m['tau_eff'])} & {b(interval([m['ade1']['residual']]+m['ade1']['residual_ci95'],3),m['ade1']['equivalent'])} & {b(interval([m['flip']['residual']]+m['flip']['residual_ci95'],1,100),m['flip']['equivalent'])} \\\\")
        trows.append(r'\midrule')
        for k,nm in (('L1','LOne'),('L2','LTwo'),('L3','LThree')):
            if k in lt: add(f'LT{tag}{nm}',('holds' if lt[k] else 'fails'));counts.setdefault(nm,[]).append(bool(lt[k]))
        # shape of the reference ADE1 curve in tau: where it is lowest, and whether it is monotone
        cv={float(k.split('_t')[1]):v['ade1'] for k,v in lt['curve'].items()};ts=sorted(cv);lo=min(cv.values());near=[t for t in ts if cv[t]-lo<=0.01]
        add(f'LT{tag}RefAdeOne',num(cv[1.0]));add(f'LT{tag}RefAdeMin',num(lo));add(f'LT{tag}RefAdeLow',num(cv[ts[0]]))
        add(f'LT{tag}RefAdeMinTau',f'{min(cv,key=cv.get):.1f}')
        add(f'LT{tag}RefAdeNearMinTau',f'{min(near):.1f}' if len(near)==1 else f'{min(near):.1f}--{max(near):.1f}')
        add(f'LT{tag}RefAdeMonotone','yes' if all(cv[a]>=cv[b] for a,b in zip(ts[1:],ts[:-1])) else 'no')
    for nm,v in counts.items():add(f'LT{nm}Holds',sum(v))
    add('LTModelsDone',done)
    body='\n    '.join(trows[:-1])
    (OUT/'tab_lowtau.tex').write_text(rf"""\begin{{table}}[H]
  \caption{{\textbf{{One and two Euler steps are not temperature in either model; the sign of the error residual differs.}} Residual of $E_1$ and $E_2$ against the reference at their effective temperature, on the 400 held-out clips, with the reference extended down to $\tau{{=}}0.1$: ADE$_1$ in m, flips in points, 95\% clip-bootstrap intervals. Bold: the 90\% interval is not inside the margin ($\pm0.05$\,m, $\pm2$ points).}}
  \label{{tab:lowtau}}
  \centering\small\setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{@{{}}llccc@{{}}}}
    \toprule
    Model & Sampler & $\taueff$ & ADE$_1$ residual (m) & Flip residual (points) \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    (OUT/'numbers_lowtau.tex').write_text('\n'.join(L)+'\n')

def bestref_extra():
    """Exploratory: E_k ADE_1 against the best reference ADE_1 over tau in 1..0.1, both for single reference samples and for the
    per-clip mean of the six reference samples (mean-of-reference control). The clip bootstrap re-takes the minimum over tau."""
    L=['% Generated by paper/tables/make_tables.py (best reference over temperature, mean-of-reference control); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    rows=[];fl=lambda t:'\\textbf{%s}'%t
    for key,tag,mname in (('a15','AOneFive','Alpamayo-1.5'),('r1','ROne','Alpamayo-R1')):
        d=json.loads((RESULTS/f'lowtau_best_reference_{key}_v1_20261029/summary.json').read_text())
        mb=json.loads((RESULTS/f'lowtau_mean_baseline_{key}_v1_20261029/summary.json').read_text());assert mb['clips']==400
        first=True
        for s_,st,lab in (('euler1_t1.0','EOne','$E_1$'),('euler2_t1.0','ETwo','$E_2$'),('euler4_t1.0','EFour','$E_4$')):
            m=d['samplers'][s_];lo,hi=m['diff_ci95'];below=hi<0
            b=mb['samplers'][s_]['single_minus_best_ref_mean6'];blo,bhi=b['ci95'];bbelow=bhi<0
            assert abs(mb['samplers'][s_]['single_ade1']-m['ade1'])<1e-6
            add(f'BR{tag}{st}Ade',num(m['ade1'],3));add(f'BR{tag}{st}Diff',sgn(m['diff'],3))
            add(f'BR{tag}{st}DiffCI',interval([m['diff'],lo,hi],3));add(f'BR{tag}{st}Below','yes' if below else 'no')
            add(f'BM{tag}{st}Diff',sgn(b['diff'],3));add(f'BM{tag}{st}DiffCI',interval([b['diff'],blo,bhi],3));add(f'BM{tag}{st}Below','yes' if bbelow else 'no')
            if first:
                add(f'BR{tag}BestRef',num(m['best_reference_ade1'],3));add(f'BR{tag}BestTau',f"{m['best_reference_tau']:.1f}")
                add(f'BM{tag}BestRef',num(b['best_ref_mean6_ade1'],3));add(f'BM{tag}BestTau',f"{b['best_tau']:.1f}")
            c1=interval([m['diff'],lo,hi],3);c1=fl(c1) if below else c1
            c2=interval([b['diff'],blo,bhi],3);c2=fl(c2) if bbelow else c2
            rows.append(f"{mname if first else ''} & {lab} & {num(m['ade1'],3)} & {num(m['best_reference_ade1'],3)} ($\\tau{{=}}{m['best_reference_tau']:.1f}$) & {c1} & {num(b['best_ref_mean6_ade1'],3)} ($\\tau{{=}}{b['best_tau']:.1f}$) & {c2} \\\\");first=False
        rows.append(r'\midrule')
    # point predictions of the held-out replication protocol, evaluated on the registered primary clips
    ho={k:json.loads((RESULTS/f'paper_numbers_heldout_{t}_v1_20261027/summary.json').read_text())['models'][k] for k,t in (('alpamayo15','a15'),('alpamayo_r1','r1'),('alpamayo2','a2'))}
    word=lambda ok:'held' if ok else 'failed'
    p1=all(.78<=m['euler10_t1.0']['tau_eff'][0]<=.90 and .55<=m['euler4_t1.0']['tau_eff'][0]<=.72 and m['euler1_t1.0']['tau_eff'][0]<=.45 and m['euler2_t1.0']['tau_eff'][0]<=.45 for m in ho.values())
    p2=all(abs(m['euler10_t1.0']['ade1']['residual'][0])<.05 and m['euler10_t1.0']['ade1']['effect'][0]<0 for m in ho.values())
    p3=[ho[k]['euler10_t1.0']['flip']['equivalent'] for k in ('alpamayo15','alpamayo_r1')]
    p4=ho['alpamayo15']['heun3_t1.0']['tau_eff'][0]>=.85 and ho['alpamayo15']['heun5_t1.0']['tau_eff'][0]>=.85
    for k,v in (('One',p1),('Two',p2),('Three',all(p3)),('Four',p4)):add(f'HOPred{k}',word(v))
    add('HOPredThreeModelsHeld',sum(p3))
    (OUT/'numbers_bestref.tex').write_text('\n'.join(L)+'\n')
    body='\n    '.join(rows[:-1])
    (OUT/'tab_bestref.tex').write_text(rf"""\begin{{table}}[H]
  \caption{{\textbf{{On Alpamayo-R1, one and two Euler steps beat the best single reference sample but not the mean of six reference samples.}} Single-sample error ADE$_1$ (m) of $E_k$ against the lowest ADE$_1$ the reference reaches over the grid $\tau\in\{{1,0.9,\dots,0.1\}}$, for a single reference sample and for the per-clip mean of the six reference samples (a point estimate of the conditional mean), with 95\% clip-bootstrap intervals (2000 resamples, the minimum over $\tau$ re-taken in each; a minimum over a noisy grid is biased downward, so the differences lean toward $E_k$ being worse). Bold: the interval lies below zero. Exploratory; not registered. 400 held-out clips.}}
  \label{{tab:bestref}}
  \centering\small\setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{@{{}}llccccc@{{}}}}
    \toprule
    & & & \multicolumn{{2}}{{c}}{{Best single reference sample}} & \multicolumn{{2}}{{c}}{{Best six-sample mean}} \\
    \cmidrule(lr){{4-5}}\cmidrule(l){{6-7}}
    Model & Sampler & $E_k$ & Reference & Difference & Reference & Difference \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def wod_extra():
    """v5.43: WOD-E2E long-tail study (registered primary sample) -> macros and the appendix table."""
    d=wod_official('280')  # v6.3a: the official Waymo RFS is the reported score (same plans, same registered pipeline)
    assert d['n_segments']==280==d['n_distinct_segments'] and not d['metric_uninformative']
    L=['% Generated by paper/tables/make_tables.py (v5.43 WOD-E2E study); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    S=d['samplers'];C=d['configs'];W=d['W'];R=d['reference_levels']
    iv=lambda lo,hi,dg:f'[{num(lo,dg)},\\,{num(hi,dg)}]'
    add('WODN',d['n_segments']);add('WODSeeds',len(d['seeds']));add('WODRfsRange',num(d['rfs_range_over_configs'],2))
    add('WODLogRfs',num(R['log_future'],2));add('WODBestRater',num(R['best_rater'],2))
    for k,t in (('euler10_t1.0','ETen'),('euler4_t1.0','EFour'),('euler1_t1.0','EOne')):
        s=S[k];r=s['rfs'];a=s['ade1'];f=s['flip']
        add(f'WOD{t}Tau',num(s['tau_eff'],2));add(f'WOD{t}TauCI',iv(*s['tau_eff_ci95'],2))
        add(f'WOD{t}RfsEffect',f"{sgn(r['raw_effect'],2)}\\,{iv(*r['raw_effect_ci95'],2)}")
        add(f'WOD{t}RfsResid',f"{sgn(r['residual'],3)}\\,{iv(*r['residual_ci90'],3)}");add(f'WOD{t}RfsResidHi',num(r['residual_ci90'][1],4))
        add(f'WOD{t}AdeResid',f"{sgn(a['residual'],3)}\\,{iv(*a['residual_ci90'],3)}");add(f'WOD{t}AdeResidPlain',num(a['residual'],2))
        add(f'WOD{t}AdeEffect',f"{sgn(a['raw_effect'],2)}\\,{iv(*a['raw_effect_ci95'],2)}")
        add(f'WOD{t}FlipResid',f"{sgn(100*f['residual'],1)}\\,{iv(100*f['residual_ci90'][0],100*f['residual_ci90'][1],1)}")
        add(f'WOD{t}FlipPct',num(100*f['sampler'],0))
    assert S['euler10_t1.0']['rfs']['equivalent'] and not S['euler4_t1.0']['rfs']['equivalent'] and S['euler10_t1.0']['flip']['equivalent'] and S['euler4_t1.0']['flip']['equivalent']
    assert S['euler10_t1.0']['ade1']['equivalent'] and not S['euler4_t1.0']['ade1']['equivalent'] and S['euler4_t1.0']['ade1']['residual']<0
    assert not S['euler1_t1.0']['in_range'] and not S['euler1_t1.0']['rfs']['equivalent']
    assert not W['W1'] and W['W2'] and not W['W3'] and W['W4_E10'] and W['W5']  # converged 10k-resample verdicts (audit 29/10): W1 fails on E4
    w5=d['W5_detail'];add('WODWFiveEffect',f"{sgn(w5['rfs_t0.4_minus_t1'],2)}\\,{iv(*w5['ci95'],2)}")
    add('WODETenRfs',num(C['euler10_t1.0']['rfs'],2))
    (OUT/'numbers_wod.tex').write_text('\n'.join(L)+'\n')
    # appendix table: v6.2 shows the registered primary sample and the registered secondary analysis of all rated frames
    (OUT/'tab_wod.tex').write_text(wod_table())


def v542_extra():
    """v5.42: reference floor (two independent tau=1 references), original-sample A2 flip residual, schedule-confirmation partition."""
    L=['% Generated by paper/tables/make_tables.py (v5.42 additions); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    r1=json.loads((RESULTS/'heun_controls_analysis_r1_v1_20261028/summary.json').read_text())
    a2=json.loads((RESULTS/'heun_controls_analysis_a2_identicalref_200_400_v1_20261028/summary.json').read_text())
    add('RFClipsROne',r1['clips']);add('RFMaxDiffROne',f"{r1['reference_max_abs_diff_m']:.1f}")
    add('RFClipsATwo',a2['clips']);add('RFMaxDiffATwo',f"{a2['reference_max_abs_diff_m']:.1f}")
    o=json.loads((ORIG/'summary.json').read_text())['models']['alpamayo2']['euler10_t1.0']['flip']
    assert o['residual'][1]>0;add('OrigATwoETenFlip',f"{sgn(o['residual'][0],1,100)}\\,[{ci(100*o['residual'][1])},\\,{ci(100*o['residual'][2])}]")
    sp=json.loads((RESULTS/'split_overlap_confirmation_v1_20261029/summary.json').read_text())['samples']['confirmation']
    add('SchedConfTrain',sp['partition']['train']);add('SchedConfTest',sp['partition']['test']);add('SchedConfAlpasim',sp['partition']['alpasim'])
    h5=load_pooled()['models']['alpamayo15']['heun5_t1.0']['contraction'];add('AOneFiveHeunFiveContraction',num(h5['median']));add('AOneFiveHeunFiveFracContracted',f"{100*h5['fraction_below_one']:.0f}")
    (OUT/'numbers_reference_floor.tex').write_text('\n'.join(L)+'\n')


QV3=RESULTS/'qwendrive_v3_analysis_20261102/summary.json'  # Qwen-Drive, registered v3 (protocols/qwen_drive.md)
QHEUN=RESULTS/'qwendrive_v3_heun_analysis_20261102/summary.json'  # Qwen-Drive Heun-3/5 control (same protocol, addendum 1)
QE6=RESULTS/'qwendrive_v3_e6_analysis_20261102/summary.json'  # Qwen-Drive Euler-6 (same protocol, addendum 2)


def headline_table():
    """Main-body table: the matched-temperature residual of E10, E4 and Heun-5 in each model (pooled held-out clips).
    v7.1: Qwen-Drive columns from the registered v3 analysis (native 5 s horizon). Verdict rule is the same in every
    column: bold when the 90% clip-bootstrap interval of the residual is not inside the margin (pooled Alpamayo summary
    stores 95% intervals and the 90% TOST flag; the Qwen-Drive summary stores both). v7.2: Qwen-Drive Heun-5 from the
    registered control (protocols/qwen_drive.md, addendum 1; artifacts/qwendrive_v3_heun_analysis_20261102). Its spread
    lies above the curve at tau=1, so, as for Alpamayo-1.5's Heun-5, tau_eff is truncated at 1.00 and the residual is taken
    against tau=1; the bold rule is the same 90% interval ('equivalent_table'). v7.4: E6 row for all four models (Alpamayo
    from the pooled summary; Qwen-Drive from the registered E6 run, addendum 2, artifacts/qwendrive_v3_e6_analysis_20261102)."""
    st=load_pooled();rows=[];b=lambda t,ok:t if ok else f'\\textbf{{{t}}}'
    q=json.loads(QV3.read_text());qs=q['horizons']['native_5s']['samplers']
    qh=json.loads(QHEUN.read_text());assert qh['R2']['pass'] and qh['clips']==q['clips'] and qh['margins']==q['margins']
    qs={**qs,'heun5_t1.0':qh['horizons']['native_5s']['samplers']['heun5_t1.0']}
    q6=json.loads(QE6.read_text());assert q6['R2']['pass'] and q6['clips']==q['clips'] and q6['margins']==q['margins'] and not q6['R2']['missing_in_e6_run']
    qs['euler6_t1.0']=q6['horizons']['native_5s']['samplers']['euler6_t1.0']
    assert q['distinct_clip_ids']==q['clips']==st['clips'] and q['margins']['ade1']==st['margins']['ade1'] and q['margins']['flip']==st['margins']['flip']
    for sampler,name in (('euler10_t1.0','$E_{10}$'),('euler6_t1.0','$E_6$'),('euler4_t1.0','$E_4$'),('heun5_t1.0','Heun-5')):
        cells=[name]
        for key,_ in MODELS:
            e=st['models'][key][sampler]
            cells+=[num(e['tau_eff'][0]),b(sgn(e['ade1']['residual'][0],3),e['ade1']['equivalent']),b(sgn(e['flip']['residual'][0],1,100),e['flip']['equivalent'])]
        e=qs.get(sampler)
        if e is None:cells+=['--']*3
        else:
            key='equivalent' if sampler!='heun5_t1.0' else 'equivalent_table'
            if sampler!='heun5_t1.0':assert e['in_range'] and e['interpretable']
            else:assert e['above_grid'] and e['tau_eff']==1.  # above the grid: truncated at 1, residual = effect
            for m in ('ade1','flip'):  # recheck the stored flag against the 90% interval
                lo,hi=e[m]['residual_ci90'];assert e[m][key]==(-q['margins'][m]<lo and hi<q['margins'][m])
                e[m]['equivalent']=e[m][key]
            cells+=[num(e['tau_eff']),b(sgn(e['ade1']['residual'],3),e['ade1']['equivalent']),b(sgn(e['flip']['residual'],1,100),e['flip']['equivalent'])]
        rows.append(' & '.join(cells)+r' \\')
    body='\n    '.join(rows)
    (OUT/'tab_headline.tex').write_text(rf"""\begin{{table*}}[t]
  \caption{{\textbf{{The released ten-step sampler leaves residuals within the margins; in the Alpamayo models Heun-5 leaves a large one.}} Effective temperature $\taueff$ and residual against the reference at $\taueff$ in single-sample error (ADE$_1$, m) and decision flips (points), the same {st['clips']} held-out clips for every model. Bold: the 90\% clip-bootstrap interval of the residual is not inside the margin ($\pm0.05$\,m, $\pm2$ points); the rule is the same for all four models. Alpamayo 95\% intervals in \cref{{S-tab:matched}} (\cref{{S-app:details}}); Qwen-Drive 90\% intervals in \cref{{S-app:family}}. Heun-5 spreads more than the reference at $\tau=1$ in Alpamayo-1.5 and Qwen-Drive; there $\taueff$ is shown as 1.00 and the residual is taken at $\tau=1$. Qwen-Drive's $E_{{10}}$ barely departs from its reference ($\taueff=\QVETenTau$), so its residuals are zero to the reported precision.}}
  \label{{tab:headline}}
  \centering\small\setlength{{\tabcolsep}}{{3.6pt}}
  \begin{{tabular}}{{@{{}}lcccccccccccc@{{}}}}
    \toprule
    & \multicolumn{{3}}{{c}}{{Alpamayo-1.5}} & \multicolumn{{3}}{{c}}{{Alpamayo-R1}} & \multicolumn{{3}}{{c}}{{Alpamayo-2}} & \multicolumn{{3}}{{c}}{{Qwen-Drive}} \\
    \cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-10}}\cmidrule(l){{11-13}}
    Sampler & $\taueff$ & ADE$_1$ & Flips & $\taueff$ & ADE$_1$ & Flips & $\taueff$ & ADE$_1$ & Flips & $\taueff$ & ADE$_1$ & Flips \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}
""")


def margin_sensitivity():
    """Equivalence of E10 and E4 at 0.5x, 1x and 2x the registered margins, from the stored 95% residual intervals
    (conservative relative to the 90% intervals of the registered tests; identical verdicts at 1x are asserted)."""
    st=load_pooled();base={'ade1':0.05,'flip':0.02};rows=[];tot={f:0 for f in (0.5,1,2)};n=0;lines=['% Generated by paper/tables/make_tables.py (margin sensitivity); do not edit by hand.']
    for key,mlabel in MODELS:
        first=True
        for sampler,name in (('euler10_t1.0','$E_{10}$'),('euler4_t1.0','$E_4$')):
            cells=[mlabel if first else '',name];first=False
            for m in ('ade1','flip'):
                r=st['models'][key][sampler][m]['residual']
                for f in (0.5,1,2):
                    ok=-f*base[m]<r[1] and r[2]<f*base[m];tot[f]+=ok
                    if f==1:assert ok==st['models'][key][sampler][m]['equivalent']
                    cells.append('yes' if ok else '\\textbf{no}')
            n+=2;rows.append(' & '.join(cells)+r' \\')
        rows.append(r'\midrule')
    body='\n    '.join(rows[:-1])
    for f,t in (('Half',0.5),('One',1),('Two',2)):lines.append(f'\\newcommand{{\\Margin{f}Equiv}}{{{tot[t]}}}')
    lines.append(f'\\newcommand{{\\MarginCells}}{{{n}}}')
    (OUT/'numbers_margin.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'tab_margin.tex').write_text(rf"""\begin{{table}}[!htbp]
  \caption{{\textbf{{Equivalence of $E_{{10}}$ and $E_4$ to their matched temperature at half, once and twice the registered margins.}} Yes: the residual interval lies inside the margin scaled by the column factor (registered: 0.05\,m for ADE$_1$, 2 points for flips). The intervals are the 95\% intervals of \cref{{tab:matched}}, wider than the registered 90\% ones, so the half-margin column is conservative; at $\times1$ the verdicts equal the registered ones. {st['clips']} held-out clips.}}
  \label{{tab:margin}}
  \centering\small\setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{@{{}}llcccccc@{{}}}}
    \toprule
    & & \multicolumn{{3}}{{c}}{{ADE$_1$ margin}} & \multicolumn{{3}}{{c}}{{Flip margin}} \\
    \cmidrule(lr){{3-5}}\cmidrule(l){{6-8}}
    Model & Sampler & $\times0.5$ & $\times1$ & $\times2$ & $\times0.5$ & $\times1$ & $\times2$ \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def v60_tables():
    """v6.0: merged appendix tables (NAVSIM IL+RL+DiffusionDrive in one table; one/two-step residuals with the best-reference baselines)."""
    nv={fam:json.loads((RESULTS/f'navsim_analysis_recogdrive_{fam}_addendum2_v1_20261028/summary.json').read_text()) for fam in ('il','rl')}
    dd=json.loads((RESULTS/'navsim_analysis_diffusiondrive_v3_20261029/summary.json').read_text())['samplers']
    def cell(e,m,d,sc,ci=False):
        t=sgn(e[m]['residual'],d,sc)
        if ci:
            bd=lambda x:num(x,2) if abs(abs(x)-2)<0.05 else num(x,1)
            t+=f"\\,[{bd(100*e[m]['residual_ci90'][0])},\\,{bd(100*e[m]['residual_ci90'][1])}]"
        return t if e[m]['equivalent'] else f'\\textbf{{{t}}}'
    cs=[('ddim2_lead_eta0_t1.0','$E_2$'),('ddim4_lead_eta0_t1.0','$E_4$'),('ddim5_lead_eta0_t1.0','$E_5$'),('ddim10_lead_eta0_t1.0','$E_{10}$'),('ddim20_lead_eta0_t1.0','$E_{20}$'),('ddim50_lead_eta0_t1.0','$E_{50}$'),('release5_lead_eta1_t1.0','release')]
    rows=[]
    for mlabel,fam in (('IL','il'),('RL','rl')):
        first=True
        for c,name in cs:
            e=nv[fam]['samplers'][c];lab=f'ReCog.-{mlabel}' if first else '';first=False;mg=nv[fam]['margins']
            raw=lambda m,d,sc:(lambda t:t if abs(e[m]['effect'])<=mg[m] else f'\\textbf{{{t}}}')(sgn(e[m]['effect'],d,sc))
            if e['in_range']:rows.append(f"{lab} & {name} & {num(e['tau_eff'])} & {raw('pdms',2,100)} & {cell(e,'pdms',2,100)} & {raw('ade1',3,1)} & {cell(e,'ade1',3,1)} & {raw('flip',1,100)} & {cell(e,'flip',1,100,True)} \\\\")
            else:
                why='spread below the measured range' if e['spread']<nv[fam]['samplers']['ddim50_lead_eta0_t1.0']['spread'] else 'spread above the reference'
                rows.append(f"{lab} & {name} & \\multicolumn{{7}}{{c}}{{no $\\taueff$ ({why})}} \\\\")
        rows.append(r'\midrule')
    r=dd['release2_lead_eta0_t1.0'];assert not r['in_range']
    rows.append(f"DiffDrive & release & \\multicolumn{{7}}{{c}}{{no $\\taueff$ (bootstrap {dd_tau_text(r)})}} \\\\")
    body='\n    '.join(rows)
    (OUT/'tab_navsim_merged.tex').write_text(rf"""\begin{{table}}[!htbp]
  \caption{{\textbf{{ReCogDrive: leading-grid DDIM with 4 to 50 steps is temperature-like before RL fine-tuning, not after.}} Raw: the sampler's change against the reference at $\tau{{=}}1$. Resid.: against the {nv['rl']['reference'].split('_')[0].replace('ddim','')}-step reference at the sampler's effective temperature, with the 90\% scene-bootstrap interval for flips. PDMS and flips in points, ADE$_1$ in m. Bold: not inside the margin (1 point of PDMS, 0.05\,m, 2 points of flips). Before RL every raw PDMS and ADE$_1$ effect, and the raw flips of $E_{{20}}$ and $E_{{50}}$, are already inside the margin, so the flips of $E_4$ to $E_{{10}}$ carry the claim. Release: 5 steps, $\eta{{=}}1$. {nv['il']['scenes']} navtest scenes $\times$ 6 noises.}}
  \label{{tab:navsim}}
  \centering\small\setlength{{\tabcolsep}}{{4.5pt}}
  \begin{{tabular}}{{@{{}}llccccccc@{{}}}}
    \toprule
    Model & Sampler & $\taueff$ & PDMS raw & PDMS resid. & ADE$_1$ raw & ADE$_1$ resid. & Flips raw & Flips resid. \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")
    # one and two steps: residual against the extended reference, and ADE_1 against the best single-sample and best mean-of-six reference
    rows=[];rows2=[]
    for key,mname in (('a15','Alpamayo-1.5'),('r1','Alpamayo-R1')):
        lt=json.loads((RESULTS/f'lowtau_analysis_{key}_v1_20261029/summary.json').read_text())
        d=json.loads((RESULTS/f'lowtau_best_reference_{key}_v1_20261029/summary.json').read_text())
        mb=json.loads((RESULTS/f'lowtau_mean_baseline_{key}_v1_20261029/summary.json').read_text());assert mb['clips']==400
        first=True
        for s_,lab in (('euler1_t1.0','$E_1$'),('euler2_t1.0','$E_2$'),('euler4_t1.0','$E_4$')):
            b=lambda t,ok:t if ok else f'\\textbf{{{t}}}'
            if s_ in lt['samplers'] and s_!='euler4_t1.0':
                m=lt['samplers'][s_]
                left=f"{num(m['tau_eff'])} & {b(interval([m['ade1']['residual']]+m['ade1']['residual_ci95'],3),m['ade1']['equivalent'])} & {b(interval([m['flip']['residual']]+m['flip']['residual_ci95'],1,100),m['flip']['equivalent'])}"
            else:left='\\multicolumn{3}{c}{see \\cref{tab:matched}}'
            m=d['samplers'][s_];lo,hi=m['diff_ci95'];bm=mb['samplers'][s_]['single_minus_best_ref_mean6'];blo,bhi=bm['ci95']
            mm=mb['samplers'][s_]['mean6_minus_best_ref_mean6'];mlo,mhi=mm['ci95']
            assert abs(mb['samplers'][s_]['single_ade1']-m['ade1'])<1e-6
            star=lambda t,below:t+('$^\\ast$' if below else '')
            if s_!='euler4_t1.0':rows.append(f"{mname if first else ''} & {lab} & {left} \\\\")
            rows2.append(f"{mname if first else ''} & {lab} & {num(m['ade1'],3)} & {star(interval([m['diff'],lo,hi],3),hi<0)} & {star(interval([bm['diff'],blo,bhi],3),bhi<0)} & {num(mb['samplers'][s_]['mean6_ade1'],3)} & {star(interval([mm['diff'],mlo,mhi],3),mhi<0)} \\\\");first=False
        rows.append(r'\midrule');rows2.append(r'\midrule')
    body='\n    '.join(rows[:-1]);body2='\n    '.join(rows2[:-1])
    (OUT/'tab_onetwo.tex').write_text(rf"""\begin{{table}}[!htbp]
  \caption{{\textbf{{One and two Euler steps are not a temperature.}} Residual against the reference at $\taueff$, with the temperature grid extended to $\tau{{=}}0.1$; 95\% clip-bootstrap intervals; bold: the 90\% interval is not inside the margin. $E_4$: \cref{{tab:matched}}. 400 held-out clips.}}
  \label{{tab:onetwo}}
  \centering\small\setlength{{\tabcolsep}}{{6pt}}
  \begin{{tabular}}{{@{{}}llccc@{{}}}}
    \toprule
    Model & Sampler & $\taueff$ & ADE$_1$ residual (m) & Flip residual (points) \\
    \midrule
    {body}
    \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[!htbp]
  \caption{{\textbf{{Against the mean of six reference samples, the single-sample error gain of one and two steps on Alpamayo-R1 disappears; mean for mean, two steps remain better.}} ADE$_1$ (m) of $E_k$ minus the lowest ADE$_1$ the reference reaches on $\tau\in\{{1,0.9,\dots,0.1\}}$: left, one $E_k$ sample against the best single reference sample and against the best per-clip mean of six reference samples; right, the per-clip mean of six $E_k$ samples against the same six-sample reference mean. 95\% intervals with the minimum re-taken in each resample, which biases the differences against $E_k$; $^\ast$: interval below zero. Exploratory; 400 held-out clips.}}
  \label{{tab:onetwo-best}}
  \centering\small\setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{@{{}}llccc@{{\hspace{{14pt}}}}cc@{{}}}}
    \toprule
    & & \multicolumn{{3}}{{c}}{{One $E_k$ sample}} & \multicolumn{{2}}{{c}}{{Mean of six $E_k$ samples}} \\
    \cmidrule(lr){{3-5}}\cmidrule(l){{6-7}}
    Model & Sampler & ADE$_1$ & $-$ best sample & $-$ best six-sample mean & ADE$_1$ & $-$ best six-sample mean \\
    \midrule
    {body2}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""")


def v61_extra():
    """v6.1: macros for the mean-vs-mean baseline, DiffusionDrive decisions and per-model shrinkage separation."""
    L=['% Generated by paper/tables/make_tables.py (v6.1 additions); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    for key,tag in (('r1','ROne'),('a15','AOneFive')):
        mb=json.loads((RESULTS/f'lowtau_mean_baseline_{key}_v1_20261029/summary.json').read_text())['samplers']
        for s_,st in (('euler1_t1.0','EOne'),('euler2_t1.0','ETwo'),('euler4_t1.0','EFour')):
            m=mb[s_]['mean6_minus_best_ref_mean6'];add(f'MM{tag}{st}DiffCI',interval([m['diff'],*m['ci95']],3))
    m=json.loads((RESULTS/'lowtau_mean_baseline_r1_v1_20261029/summary.json').read_text())['samplers']['euler2_t1.0']['mean6_minus_best_ref_mean6']
    assert m['ci95'][1]<0  # two steps on R1 stay better mean for mean
    dd=json.loads((RESULTS/'navsim_analysis_diffusiondrive_v3_20261029/summary.json').read_text())['samplers']['release2_lead_eta0_t1.0']
    f=dd['flip'];g=dd['flip_registered']
    add('DDFlipEffectCI',interval([f['effect'],*f['effect_ci95']],1,100));add('DDFlipEffectHiNinety',num(100*f['effect_ci90'][1],1))
    add('DDFlipRegEffectCI',interval([g['effect'],*g['effect_ci95']],1,100))
    add('DDBootInRange',f"{100*dd['boot_in_range']:.0f}");add('DDAdeEffectCI',interval([dd['ade1']['effect'],*dd['ade1']['effect_ci90']],3))
    add('DDPdmsEffectCI',interval([dd['pdms']['effect'],*dd['pdms']['effect_ci90']],2,100))
    sig=json.loads((POOLED/'summary.json').read_text())['signature']['E4']['per_model']
    for k,tag in (('alpamayo15','AOneFive'),('alpamayo_r1','ROne'),('alpamayo2','ATwo')):
        v=sig[k]['euler_minus_shrink_median'];assert v[1]>0;add(f'SigDiffShrinkFour{tag}',f'{num(v[0],3)}\\,[{num(v[1],3)},\\,{num(v[2],3)}]')
    (OUT/'numbers_baselines.tex').write_text('\n'.join(L)+'\n')


WOD_PRIMARY=RESULTS/'wod_e2e_analysis_primary_converged10k_v2_20261029'
WOD_ALL=RESULTS/'wod_e2e_analysis_all479_converged10k_v2_20261029'
WOD_OFF=RESULTS/'wod_e2e_analysis_official_rfs_v1_20261030'  # v6.3a: official Waymo rater_feedback_utils, unmodified
FLIPAG=RESULTS/'flip_agreement_v1_20261030'  # v6.3a: exploratory flip agreement (not registered)


def wod_official(n):
    """One sample ('280' primary, '479' secondary) of the official-RFS re-analysis, with the shared run settings attached.
    Error and flip residuals do not depend on the score, so they must equal the registered proxy analysis exactly."""
    o=json.loads((WOD_OFF/'summary.json').read_text());d=dict(o['samples'][n]);d.update(seeds=o['seeds'],boot=o['boot'],margins=o['margins'])
    assert not d['verdict_changes'] and d['W']==d['ours_recomputed']['W']
    prox=json.loads(((WOD_PRIMARY if n=='280' else WOD_ALL)/'summary.json').read_text())
    for k in ('euler10_t1.0','euler4_t1.0','euler1_t1.0'):
        for m in ('ade1','flip'):
            a,b=d['samplers'][k][m],prox['samplers'][k][m]
            assert abs(a['residual']-b['residual'])<1e-9 and all(abs(x-y)<1e-9 for x,y in zip(a['residual_ci90'],b['residual_ci90'])),(n,k,m)
    return d


def wod_table():
    """v6.2: long-tail appendix table, registered primary (first 280 segments) and secondary (all 479 rated frames)."""
    iv=lambda lo,hi,dg:f'[{num(lo,dg)},\\,{num(hi,dg)}]'
    bold=lambda ok,t:t if ok else '\\textbf{%s}'%t
    T=['\\begin{tabular}{@{}lccccccc@{}}','\\toprule','Config & Rater score & ADE$_1$ (m) & Flips (\\%) & $\\taueff$ & Score residual & ADE$_1$ residual (m) & Flip residual (pt) \\\\']
    for n,head in (('280','Registered primary: first %d segments'),('479','Registered secondary: all %d rated frames')):
        d=wod_official(n);S=d['samplers'];C=d['configs'];assert d['boot']==10000
        T+=['\\midrule',f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{head%d["n_segments"]}}}}} \\\\','\\addlinespace[1pt]']
        for lab,k in (('Reference, $\\tau{=}1$','heun32_t1.0'),('Reference, $\\tau{=}0.4$','heun32_t0.4')):
            T.append(f"{lab} & {num(C[k]['rfs'],2)} & {num(C[k]['ade1'],2)} & {num(100*C[k]['flip'],0)} & & & & \\\\")
        for lab,k in (('$E_{10}$','euler10_t1.0'),('$E_4$','euler4_t1.0'),('$E_1$','euler1_t1.0')):
            s=S[k];r=s['rfs'];a=s['ade1'];f=s['flip'];tau=('$\\le$0.1' if not s['in_range'] else num(s['tau_eff'],2))
            T.append(f"{lab} & {num(C[k]['rfs'],2)} & {num(C[k]['ade1'],2)} & {num(100*C[k]['flip'],0)} & {tau} & "
                     +bold(r['equivalent'],f"{sgn(r['residual'],3)}\\,{iv(*r['residual_ci90'],3)}")+' & '
                     +bold(a['equivalent'],f"{sgn(a['residual'],3)}\\,{iv(*a['residual_ci90'],3)}")+' & '
                     +bold(f['equivalent'],f"{sgn(100*f['residual'],1)}\\,{iv(100*f['residual_ci90'][0],100*f['residual_ci90'][1],1)}")+' \\\\')
    T+=['\\bottomrule','\\end{tabular}']
    return '\n'.join(T)+'\n'


def v62_extra():
    """v6.2: registered secondary long-tail analysis (all rated frames) and FlashDrive released-stack Euler rows."""
    L=['% Generated by paper/tables/make_tables.py (v6.2 additions); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    iv=lambda lo,hi,dg:f'[{num(lo,dg)},\\,{num(hi,dg)}]'
    d=wod_official('479');S=d['samplers'];W=d['W'];R=d['reference_levels']
    assert d['n_segments']==479 and d['boot']==10000 and not json.loads((WOD_ALL/'summary.json').read_text())['dropped_nan'] and not d['metric_uninformative']
    assert W['W1'] and W['W2'] and W['W3'] and W['W4_E10'] and W['W5']  # all registered predictions hold on all rated frames
    for k in ('euler10_t1.0','euler4_t1.0'):
        assert all(S[k][m]['equivalent'] for m in ('rfs','ade1','flip'))
    assert not S['euler1_t1.0']['flip']['equivalent'] and not S['euler1_t1.0']['in_range']
    add('WODAllN',d['n_segments']);add('WODAllSegs',d['n_distinct_segments'])
    add('WODAllLogRfs',num(R['log_future'],2))
    for k,t in (('euler10_t1.0','ETen'),('euler4_t1.0','EFour'),('euler1_t1.0','EOne')):
        s=S[k];r=s['rfs'];a=s['ade1'];f=s['flip']
        add(f'WODAll{t}Tau',num(s['tau_eff'],2))
        add(f'WODAll{t}RfsEffect',f"{sgn(r['raw_effect'],2)}\\,{iv(*r['raw_effect_ci95'],2)}")
        add(f'WODAll{t}RfsResid',f"{sgn(r['residual'],3)}\\,{iv(*r['residual_ci90'],3)}")
        add(f'WODAll{t}AdeResid',f"{sgn(a['residual'],3)}\\,{iv(*a['residual_ci90'],3)}")
        add(f'WODAll{t}FlipResid',f"{sgn(100*f['residual'],1)}\\,{iv(100*f['residual_ci90'][0],100*f['residual_ci90'][1],1)}")
    w5=d['W5_detail'];add('WODAllWFiveEffect',f"{sgn(w5['rfs_t0.4_minus_t1'],2)}\\,{iv(*w5['ci95'],2)}")
    p=wod_official('280')
    add('WODETenRfsPlain',num(p['configs']['euler10_t1.0']['rfs'],2));add('WODETenRfsEffectPlain',num(p['samplers']['euler10_t1.0']['rfs']['raw_effect'],2))
    # FlashDrive released optimized stack: plain Euler rows (error residual only), 90% intervals
    fz=json.loads((RESULTS/'flashdrive_analysis_zlab_v1_20261027/summary.json').read_text());M=0.05
    for k,t in (('euler4_t1.0','EFour'),('euler8_t1.0','EEight'),('euler10_t1.0','ETen')):
        s=fz['samplers'][k];lo,hi=s['residual_ci90'];assert hi>M  # every Euler error interval on this stack reaches past the margin
        add(f'FDZ{t}TauPlain',num(s['tau_eff'],2));add(f'FDZ{t}Resid',f"{num(s['residual'],3)}\\,[{num(lo,3)},\\,{num(hi,3)}]");add(f'FDZ{t}ResidHi',num(hi,3))
    c=fz['samplers']['flashcache8_t1.0'];assert c['residual_ci90'][1]>M
    v=fz['paired_ade1']['flashcache8_t1.0-euler4_t1.0'];add('FDZVsEFour',f"{sgn(v['mean'],3)}\\,{iv(*v['ci90'],3)}")
    (OUT/'numbers_wod_secondary.tex').write_text('\n'.join(L)+'\n')


def v63_extra():
    """v6.3a: official-RFS agreement with the registered re-implementation, and the exploratory flip-agreement table."""
    L=['% Generated by paper/tables/make_tables.py (v6.3a additions); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    iv=lambda lo,hi,dg:f'[{num(lo,dg)},\\,{num(hi,dg)}]'
    o=json.loads((WOD_OFF/'summary.json').read_text());shifts=[];diffs=[]
    for n,t in (('280',''),('479','All')):
        d=o['samples'][n];ag=d['agreement_official_vs_ours']['pooled_all_configs'];assert not d['verdict_changes']
        add(f'WODOff{t}R',num(ag['pearson'],2));shifts.append(ag['mean_shift_official_minus_ours'])
        prox=json.loads(((WOD_PRIMARY if n=='280' else WOD_ALL)/'summary.json').read_text())['samplers']
        for k,tk in (('euler10_t1.0','ETen'),('euler4_t1.0','EFour')):
            c=d['comparison_rfs'][k];lo,hi=c['residual_diff_ci95'];assert lo<0<hi  # official minus proxy residual: CI contains 0
            assert c['equivalent_official']==c['equivalent_ours'];diffs.append(abs(c['residual_diff_official_minus_ours']))
            r=prox[k]['rfs'];assert abs(r['residual']-c['rfs_residual_ours'])<1e-9
            add(f'WODProx{t}{tk}RfsResid',f"{sgn(r['residual'],3)}\\,{iv(*r['residual_ci90'],3)}")
            add(f'WODProx{t}{tk}RfsEffect',num(r['raw_effect'],2))
    add('WODOffShiftLo',num(min(shifts),2));add('WODOffShiftHi',num(max(shifts),2));add('WODOffDiffMax',num(max(diffs),3))
    add('WODOffBoth',f"{o['samples']['280']['n_segments']} and {o['samples']['479']['n_segments']}")
    # exploratory flip agreement (E10 vs the two reference temperatures bracketing its tau_eff)
    f=json.loads((FLIPAG/'summary.json').read_text());assert f['status'].startswith('EXPLORATORY') and f['clips']==400
    T=['\\begin{tabular}{@{}lcccccccc@{}}','\\toprule',
       '& \\multicolumn{3}{c}{Flips (of %d pairs)} & & \\multicolumn{2}{c}{Agreement with $E_{10}$} & & \\\\' % 2400,
       '\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}',
       'Model ($\\taueff$) & $E_{10}$, ref.\\ 0.8 & shared & chance & Share (0.8 / 0.9) & ref.\\ 0.8 & ref.\\ 0.9 & Ref.\\ pair & Keep$\\to$stop \\\\','\\midrule']
    share=[];chance=[];both=[];clips=[];tvs=[];tvref=[]
    for key,name in MODELS:
        m=f['models'][key];P=m['point'];c=m['ci95'];lo,hi=P['lo'],P['hi'];n=sum(lo['table'].values());assert n==2400 and m['neighbours']=={'lo':0.8,'hi':0.9}
        M=P['transition_long']['matrices'];ks=lambda nm:M[nm][2][0]*n  # row: class at tau=1 (keep); column: class under the sampler (stop)
        dl=lambda q,nm:f"{num(q['agree_joint'],3)}"
        T.append(f"{name} ({num(m['tau_eff'],2)}) & {lo['e10_flip']*n:.0f}, {lo['cold_flip']*n:.0f} & {lo['table']['both']:.0f} & {lo['both_expected_indep']:.0f} & "
                 f"{100*lo['p_cold_given_e10']:.0f}\\% / {100*hi['p_cold_given_e10']:.0f}\\% & {dl(lo,'lo')} & {dl(hi,'hi')} & {num(P['agree_ref_0.9_0.8_joint'],3)} & {ks('e10'):.0f} / {ks('mix'):.1f} \\\\")
        share.append(lo['p_cold_given_e10']);chance.append(lo['table']['both']/lo['both_expected_indep']);both.append(lo['table']['both']);clips.append(m['distinct_clips']['both_lo_same_seed'])
        for ax in ('long','lat'):tvs.append(P['transition_'+ax]['tv_e10_mix']);tvref.append(P['transition_'+ax]['tv_lo_hi'])
        tag=TAG[key];dlo=P['lo']['agree_minus_bracket_joint'];cl=c['lo.agree_minus_bracket_joint']
        add(f'FA{tag}DiffLo',f"{sgn(dlo,3)}\\,{iv(*cl,3)}");add(f'FA{tag}DiffHi',f"{sgn(P['hi']['agree_minus_bracket_joint'],3)}\\,{iv(*c['hi.agree_minus_bracket_joint'],3)}")
        to=P['transition_long'];add(f'FA{tag}OffLong',num(to['tv_offdiag_e10_mix'],2));add(f'FA{tag}OffLongRef',num(to['tv_offdiag_lo_hi'],2))
        add(f'FA{tag}KSEten',f"{ks('e10'):.0f}");add(f'FA{tag}KSMix',f"{ks('mix'):.1f}")
    T+=['\\bottomrule','\\end{tabular}']
    assert all(a<b for a,b in zip(tvs,tvref))  # per model and axis, E10's full transition matrix is closer to the interpolated reference than the two grid references are to each other
    add('FAShareMin',f"{100*min(share):.0f}");add('FAShareMax',f"{100*max(share):.0f}");add('FAChanceMin',f"{min(chance):.0f}");add('FAChanceMax',f"{max(chance):.0f}")
    add('FABothMin',f"{min(both):.0f}");add('FABothMax',f"{max(both):.0f}");add('FAClipsMin',min(clips));add('FAClipsMax',max(clips))
    add('FATVMax',num(max(tvs),3));add('FATVRefMin',num(min(tvref),3));add('FATVRefMax',num(max(tvref),3))
    (OUT/'tab_flipagree.tex').write_text('\n'.join(T)+'\n')
    # Addendum 19: E10 against the converged reference (Heun-64), Alpamayo-1.5, pooled 400 clips
    a=json.loads((RESULTS/'convergence_a19_analysis_v1_20261030/results.json').read_text());assert a['clips']==400 and a['D1']['holds'] and a['D2']['holds'] and a['D3']['holds']
    add('CVClips',a['clips']);add('CVFlipResid',f"{sgn(a['D1']['flip_residual_pt'],2)}\\,{iv(*a['D1']['tost90_pt'],2)}")
    add('CVAdeResid',f"{sgn(a['D2']['ade1_residual_m'],3)}\\,{iv(*a['D2']['tost90_m'],3)}")
    add('CVTauSixtyFour',num(a['D3']['tau_eff']['heun64'][0],3));add('CVTauThirtyTwo',num(a['D3']['tau_eff']['heun32'][0],3))
    # Addendum 20: E10 against Heun-64, Alpamayo-R1 (A2 pending)
    r=json.loads((RESULTS/'convergence_a20_analysis_r1_v1_20261030/results.json').read_text())['primary'];assert r['F1']['holds'] and r['F2']['holds'] and r['F3']['holds']
    add('CVRFlipResid',f"{sgn(r['F1']['flip_residual_pt'],2)}\\,{iv(*r['F1']['tost90_pt'],2)}");add('CVRAdeResid',f"{sgn(r['F2']['ade1_residual_m'],3)}\\,{iv(*r['F2']['tost90_m'],3)}")
    add('CVRTauSixtyFour',num(r['F3']['tau_eff']['heun64'][0],3));add('CVRTauThirtyTwo',num(r['F3']['tau_eff']['heun32'][0],3))
    (OUT/'numbers_wod_official.tex').write_text('\n'.join(L)+'\n')


def thor_extra():
    """Thor addendum 2 (protocols/thor.md): FROST with TensorRT engines on DRIVE AGX Thor, pooled 400 clips;
    per-call and per-plan latency from the batch-1 CUDA-graph benches on context_000 (exploratory)."""
    L=['% Generated by paper/tables/make_tables.py (Thor addendum 2); do not edit by hand.'];add=lambda k,v:L.append(f'\\newcommand{{\\{k}}}{{{v}}}')
    t=json.loads((RESULTS/'thor_trt_frost400_analysis_v0v1v2_v1_20261031/summary.json').read_text())
    assert t['boot_seed']==20261031 and all(t['probes'][v]=={'complete':True,'clips':400} for v in ('P0','P1','P3'))
    V=t['verdicts'];assert V['T2']['verdict']=='pass' and V['T3']['verdict']=='pass' and V['T4']['verdict']=='pass'  # U2, U3 (ADE), U1
    P=t['precisions'];add('THClips',400)
    for v,tag in (('P0','BF'),('P1','FP'),('P3','NV')):
        for s,st in (('euler10_t1.0','ETen'),('euler4_t1.0','EFour')):
            e=P[v][s];add(f'TH{tag}{st}Tau',num(e['tau_eff'][0],2))
            if s=='euler10_t1.0':assert e['ade1']['equivalent'] and e['flip']['equivalent']  # U3 (V0, V1), U4 (V2)
    pc=list(t['hardware']['max_abs_dplan_m']['per_clip_median'].values());add('THTailClips',sum(v>1 for v in pc));add('THDplanMedCm',f"{100*t['hardware']['max_abs_dplan_m']['median_of_clip_medians']:.0f}")
    dt=[abs(t['shift_vs_P0'][v][s]['dtau_eff']) for v in ('P1','P3') for s in ('euler10_t1.0','euler4_t1.0')]
    assert all(t['shift_vs_P0'][v][s]['equivalent'] for v in ('P1','P3') for s in ('euler10_t1.0','euler4_t1.0'))
    add('THDtauMax',num(max(dt),3) if max(dt)>=5e-4 else '0.001')
    add('THWsTau',num(t['hardware']['tau_eff']['euler10_t1.0']['workstation'],2))
    B=RESULTS/'thor_trt_bench_20261031';eager=json.loads((RESULTS/'thor_v1_20261030/latency/P0/summary.json').read_text())['batch1']
    add('THEagerCall',f"{eager['call_ms']['median']:.0f}");add('THEagerEFour',f"{1e3*eager['plan']['euler4']['plan_s']['median']:.0f}")
    for n,tag in (('bf16_b1_L3200','BF'),('fp8tv2_b1_L3200','FP'),('nvfp4ne_b1_L3200','NV')):
        b=json.loads((B/f'bench_{n}.json').read_text())['latency'];add(f'TH{tag}Call',f"{b['call_graph']['median_ms']:.0f}")
        add(f'TH{tag}EFour',f"{b['plan_plain']['euler4']['median_ms']:.0f}");add(f'TH{tag}ETenMs',f"{b['plan_plain']['euler10']['median_ms']:.0f}")
    # same-noise decision flips between precisions (exploratory; frost.analysis.thor_precision_flips)
    F=json.loads((RESULTS/'thor_trt_precision_flips_v1_20261031/summary.json').read_text());assert F['clips']==400
    fp=F['pairs']['fp8_vs_bf16'];nv=F['pairs']['nvfp4_vs_bf16'];ws=F['pairs']['bf16_vs_desktop']
    add('THFlipFP',num(max(fp[c]['flip_pct'] for c in ('E10','E4')),1));add('THFlipNV',num(max(nv[c]['flip_pct'] for c in ('E10','E4')),1))
    add('THFlipNVHi',num(max(nv[c]['ci95'][1] for c in ('E10','E4')),1));add('THFlipDesk',num(ws['E10']['flip_pct'],0))
    add('THFlipEFourRef',num(F['context_bf16_flips_vs_A1']['E4'],0))
    # Thor v1 T5 (P2: language model and field in FP8, PyTorch, 60 clips), registered analysis, seed 20261029
    T=json.loads((RESULTS/'thor_v1_analysis_p0p1p2_v1_20261031/summary.json').read_text())
    assert T['probes']['P2']=={'complete':True,'clips':60} and T['verdicts']['T5']['verdict']=='pass'
    add('THFiveDtau',num(T['shift_vs_P0']['P2']['euler10_t1.0']['dtau_eff'],3));add('THFiveReason',f"{100*T['reasoning_P2_vs_bf16']['fraction_different']:.0f}")
    add('THVLMs',f"{T['latency']['P0']['prepare_field_s']['median']:.1f}");add('THVLMsFP',f"{T['latency']['P2']['prepare_field_s']['median']:.1f}")
    (OUT/'numbers_thor.tex').write_text('\n'.join(L)+'\n')


if __name__=="__main__":
    (OUT/'numbers_tables.tex').write_text(table_macros())
    (OUT/'tab_matched.tex').write_text(matched_table());(OUT/'tab_matched_orig.tex').write_text(matched_table(ORIG,'tab:matched-orig',True));(OUT/'tab_contraction.tex').write_text(contraction_table());pooled_extra();v533_extra();lowtau_extra();bestref_extra();v542_extra();wod_extra();headline_table();margin_sensitivity();v60_tables();v61_extra();v62_extra();v63_extra();thor_extra();print('ok')
