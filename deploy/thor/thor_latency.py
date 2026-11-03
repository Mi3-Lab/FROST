"""Thor v1 latency per sampler (protocols/thor.md, section 4; registered 29/10/2026 14:59 PDT).

On each of context_000-009 (prefix cache fixed by prepare_field), for batch 1 (seed 9201) and batch 6 (seeds 9201-9206
stacked, the prefix cache shared by the 6 samples):
  - 20 warm-up calls of the velocity field, then 200 timed calls with torch.cuda.synchronize before and after each;
  - 30 complete solves per sampler (E1, E2, E4, E6, E10, Heun-32 at tau=1), each including action_to_traj.
Reported: median and p90 per call and per plan, per variant (P0, P1, P2, P3 if validated). Also the GPU devfreq
governor/clock and one tegrastats line (temperatures) at start and end. The VLM prefill time (prepare_field) is recorded
but is a latency only in P2, where the model is resident (P0/P1/P3 stream 34 language layers from host memory).

Usage: python thor_latency.py --variant {P0,P1,P2,P3} --out $FROST_THOR_ROOT/out/thor_v1   (resumable per clip)
"""
import argparse, copy, gc, glob, json, os, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

# Legacy hook (documented in run_thor_v1.sh): the run_thor_v1.sh started 29/10 22:00 UTC ends with
# `python thor_latency.py --out OUT` (no --variant). That call hands the board over to run_thor_v1_next.sh, which runs
# latency, P1, P2 and P3 with the board-free gate. The runner is replaced by exec, so the old script waits on it.
if __name__=='__main__' and '--variant' not in sys.argv and os.environ.get('FROST_THOR_RUNNER')!='next':
    nxt=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/run_thor_v1_next.sh')
    if nxt.exists():print(f'thor_latency: legacy call, exec {nxt}',flush=True);os.execv('/bin/bash',['bash',str(nxt)])
    print('thor_latency: legacy call and no run_thor_v1_next.sh: nothing to do',flush=True);sys.exit(2)

import numpy as np
import torch

SAMPLERS=[('euler',1),('euler',2),('euler',4),('euler',6),('euler',10),('heun',32)]
SEEDS=list(range(9201,9207))
DATA=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/contexts/physicalai_heldout_contexts_v3_20261027')


def board_state():
    s={'utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'devfreq':{}}
    for d in sorted(glob.glob('/sys/class/devfreq/*')):
        try:s['devfreq'][os.path.basename(d)]={k:open(f'{d}/{k}').read().strip() for k in ['governor','cur_freq']}
        except OSError:pass
    try:s['tegrastats']=subprocess.run(['timeout','3','tegrastats','--interval','1000'],capture_output=True,text=True).stdout.splitlines()[0]
    except Exception as e:s['tegrastats']=f'unavailable: {e}'
    return s


def batched_field(field,model,batch):
    """Same computation as prepare_field's field, with the prefix cache, RoPE ids and mask shared by `batch` samples.
    The cached keys/values are broadcast views (no copy); after each call they are restored instead of cropped, so the
    per-call work (concatenating the new tokens to the prefix in every layer) is the same as with a materialized cache."""
    cells=dict(zip(field.__code__.co_freevars,[c.cell_contents for c in field.__closure__]))
    captured,cache,n,dims=cells['captured'],cells['cache'],cells['n'],cells['dims']
    pos=captured['position_ids'].expand(-1,batch,-1);mask=captured['attention_mask'].expand(batch,-1,-1,-1)
    base=[(l.keys,l.values) for l in cache.layers]
    views=[(k.expand(batch,*k.shape[1:]),v.expand(batch,*v.shape[1:])) for k,v in base]
    def restore(kv):
        for l,(k,v) in zip(cache.layers,kv):l.keys,l.values=k,v
    def f(x,t):
        restore(views)
        try:
            embeds=model.action_in_proj(x,t)
            if embeds.ndim==2:embeds=embeds.view(len(x),n,-1)
            r=model.expert(inputs_embeds=embeds,position_ids=pos,past_key_values=cache,attention_mask=mask,use_cache=True,**captured['forward_kwargs'])
            return model.action_out_proj(r.last_hidden_state[:,-n:]).view(-1,*dims).float()
        finally:restore(base)
    return f


def stats(xs):
    xs=np.asarray(xs);return {'median':float(np.median(xs)),'p90':float(np.quantile(xs,.9)),'mean':float(xs.mean()),'n':int(len(xs))}


def run_clip(model,helper,processor,path,a,variant):
    from frost.probes.alpamayo_field import prepare_field
    from frost.probes.temperature_probe import integrate
    sync=torch.cuda.synchronize;res={'clip':path.stem,'variant':variant,'start':board_state()}
    sync();t=time.perf_counter();field,dims,hx,hr,_=prepare_field(model,helper,processor,path);sync()
    res['prepare_field_s']=time.perf_counter()-t;res['prepare_field_is_latency']=variant=='P2'
    for batch in a.batches:
        torch.cuda.reset_peak_memory_stats();entry={}
        try:
            f=field if batch==1 else batched_field(field,model,batch);noise=[]
            for s in SEEDS[:batch]:
                torch.manual_seed(s);torch.cuda.manual_seed_all(s);noise.append(torch.randn(1,*dims,device='cuda'))
            initial=torch.cat(noise);hxb=hx[:,0].expand(batch,*hx.shape[2:]);hrb=hr[:,0].expand(batch,*hr.shape[2:])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(a.warmup):f(initial,torch.full((batch,1,1),(i%10)/10,device='cuda'))
                calls=[]
                for i in range(a.calls):
                    tt=torch.full((batch,1,1),(i%10)/10,device='cuda');sync();t0=time.perf_counter();f(initial,tt);sync();calls.append(time.perf_counter()-t0)
                entry['call_s']=calls;entry['solves']={}
                for method,steps in SAMPLERS:
                    xs=[]
                    for _ in range(a.solves):
                        sync();t0=time.perf_counter()
                        model.action_space.action_to_traj(integrate(f,initial,steps,method),hxb,hrb);sync();xs.append(time.perf_counter()-t0)
                    entry['solves'][f'{method}{steps}']={'nfe':steps*(2 if method=='heun' else 1),'solve_s':xs}
            entry['peak_mem_gb']=torch.cuda.max_memory_allocated()/1e9
        except torch.OutOfMemoryError as e:entry['error']=f'OOM: {str(e)[:300]}'
        res[f'batch{batch}']=entry;gc.collect();torch.cuda.empty_cache()
    del field;gc.collect();torch.cuda.empty_cache();res['end']=board_state()
    return res


def summarize(out,variant,clips,batches):
    rows=[json.loads((out/f'{c}.json').read_text()) for c in clips if (out/f'{c}.json').exists()]
    summ={'variant':variant,'clips':[r['clip'] for r in rows],'prepare_field_s':stats([r['prepare_field_s'] for r in rows]) if rows else None,
        'prepare_field_is_latency':variant=='P2','board_first':rows[0]['start'] if rows else None,'board_last':rows[-1]['end'] if rows else None}
    for b in batches:
        ok=[r[f'batch{b}'] for r in rows if 'call_s' in r.get(f'batch{b}',{})]
        d={'clips_ok':len(ok),'errors':[r[f'batch{b}'].get('error') for r in rows if 'error' in r.get(f'batch{b}',{})]}
        if ok:
            d['call_ms']={k:v*1e3 for k,v in stats([x for e in ok for x in e['call_s']]).items() if k!='n'}
            d['call_ms_per_clip_median']=[1e3*statistics.median(e['call_s']) for e in ok];d['plan']={}
            for s in ok[0]['solves']:
                xs=[x for e in ok for x in e['solves'][s]['solve_s']];nfe=ok[0]['solves'][s]['nfe']
                d['plan'][s]={'nfe':nfe,'plan_s':stats(xs),'per_nfe_ms_median':1e3*float(np.median(xs))/nfe,
                    'nfe_times_call_s':nfe*d['call_ms']['median']/1e3}
            d['peak_mem_gb']=max(e['peak_mem_gb'] for e in ok)
        summ[f'batch{b}']=d
    (out/'summary.json').write_text(json.dumps(summ,indent=2));return summ


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=['P0','P1','P2','P3'],required=True)
    p.add_argument('--out',type=Path,required=True,help='base output dir (out/thor_v1); writes OUT/latency/<variant>/')
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--clips',type=int,default=10)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--calls',type=int,default=200);p.add_argument('--solves',type=int,default=30)
    p.add_argument('--batches',type=int,nargs='+',default=[1,6]);a=p.parse_args();torch.set_num_threads(2)
    out=a.out/'latency'/a.variant;out.mkdir(parents=True,exist_ok=True);clips=[f'context_{i:03d}' for i in range(a.clips)]
    if (out/'summary.json').exists() and all((out/f'{c}.json').exists() for c in clips):print(f'latency {a.variant}: already complete');return
    layout=None
    if a.variant=='P3':
        val=json.loads((a.out/'P3'/'layout_validation.json').read_text())
        if not val.get('passed'):print('latency P3: NVFP4 layout validation did not pass; skipped');sys.exit(3)
        layout=tuple(val['layout'].split('/'))
    from thor_quant_probe import build_model
    t=time.time();model,helper,processor,report=build_model(a.variant,layout);load_s=time.time()-t
    (out/'variant.json').write_text(json.dumps({'variant':a.variant,'layout':layout,'load_s':load_s,'torch':torch.__version__,
        'device':torch.cuda.get_device_name(0),'device_map':getattr(model,'_thor_device_map',None),
        'quant_summary':report['summary'] if report else None,'args':{k:str(v) for k,v in vars(a).items()}},indent=2))
    for c in clips:
        if (out/f'{c}.json').exists():continue
        r=run_clip(model,helper,processor,a.data/f'{c}.npz',a,a.variant);(out/f'{c}.json').write_text(json.dumps(r))
        print(json.dumps({'clip':c,**{f'b{b}_call_ms':1e3*statistics.median(r[f'batch{b}']['call_s']) for b in a.batches if 'call_s' in r[f'batch{b}']}}),flush=True)
    print(json.dumps(summarize(out,a.variant,clips,a.batches),indent=1))


if __name__=='__main__':main()
