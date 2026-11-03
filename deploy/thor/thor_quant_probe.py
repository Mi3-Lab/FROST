"""Thor v1 precision variants P1/P2/P3 (protocols/thor.md, sections 2, 3 and 7; registered 29/10/2026 14:59 PDT).

  P1  FP8-expert : every nn.Linear of the action expert -> FP8 e4m3, weights per-tensor scale, activations dynamic
                   per-tensor scale, torch._scaled_mm with bf16 output. VLM bf16 with the same partial offload as P0.
  P2  FP8-total  : P1 plus every nn.Linear of the VLM (vision, language model, lm_head); whole model resident on the GPU
                   (no offload). Loaded on the host, quantized layer by layer onto the GPU.
  P3  NVFP4-expert: expert linears in NVFP4 (e2m1, block-16 e4m3 scales, per-tensor fp32 global scale, W4A4). Runs only
                   after the layout validation test (nvfp4_layout_validation) passes; otherwise exits with code 3.
Unchanged in every variant: embeddings, norms, action_in_proj, action_out_proj (not nn.Linear of vlm/expert, or excluded).

Before the probe each variant records the relative error ||y_q - y_bf16|| / ||y_bf16|| of every quantized linear on real
bf16-model activations of context_000 (the prefill/reasoning rollout and the official E10 replay inside prepare_field);
linears above 5% are flagged and so is the variant. Then the unchanged frost.probes.temperature_probe runs with the protocol
configs and seeds on clips [0,60) with --resume, writing to OUT/<variant>/. The VLM reasoning text of every clip is also
logged (reasoning.jsonl) so that P2 can be compared with the bf16 VLM of P1 (prediction T5).

Usage: python thor_quant_probe.py --variant {P1,P2,P3} --out $FROST_THOR_ROOT/out/thor_v1 [--validate-only]
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

# Legacy hook: the first run_thor_v1.sh (started 29/10 22:00 UTC) ends by calling `python thor_quant_probe.py --out OUT`
# right after `python thor_latency.py --out OUT`; the latter hands over to run_thor_v1_next.sh, which runs this step.
if __name__=='__main__' and '--variant' not in sys.argv and os.environ.get('FROST_THOR_RUNNER')!='next':
    print('thor_quant_probe: legacy call without --variant; variants are run by run_thor_v1_next.sh',flush=True);sys.exit(0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS=['euler:1:1.0','euler:2:1.0','euler:4:1.0','euler:6:1.0','euler:10:1.0']+[f'heun:32:{t}' for t in ['1.0','0.9','0.8','0.7','0.6','0.5','0.4']]
SEEDS=list(range(9201,9207))
DATA=Path(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/contexts/physicalai_heldout_contexts_v3_20261027')
FLAG=.05
FP8_MAX=448.
E2M1=[0.,.5,1.,1.5,2.,3.,4.,6.]


# ------------------------------------------------------------------------------------------------ FP8 e4m3 per tensor
class FP8Linear(nn.Module):
    """y = x W^T + b with W and x in FP8 e4m3, per-tensor scales (weights static, activations dynamic), bf16 output."""
    def __init__(self,lin,device='cuda'):
        super().__init__();w=lin.weight.detach().to(device,torch.bfloat16);self.out_features,self.in_features=w.shape
        assert self.in_features%16==0,'K must be a multiple of 16 for _scaled_mm'
        self.pad_n=(-self.out_features)%16  # zero rows so that N % 16 == 0; output sliced back
        amax=torch.zeros((),device=device,dtype=torch.float32)
        for i in range(0,len(w),4096):amax=torch.maximum(amax,w[i:i+4096].abs().amax().float())
        scale=(amax.clamp(min=1e-12)/FP8_MAX).reshape(())
        q=torch.empty(self.out_features+self.pad_n,self.in_features,dtype=torch.float8_e4m3fn,device=device)
        for i in range(0,len(w),4096):q[i:min(i+4096,len(w))]=(w[i:i+4096].float()/scale).clamp(-FP8_MAX,FP8_MAX).to(torch.float8_e4m3fn)
        if self.pad_n:q[self.out_features:]=0
        self.register_buffer('weight_fp8',q);self.register_buffer('weight_scale',scale)
        self.register_buffer('bias',None if lin.bias is None else lin.bias.detach().to(device,torch.bfloat16))

    def forward(self,x):
        shape=x.shape;x2=x.reshape(-1,self.in_features).to(torch.bfloat16)
        sa=(x2.abs().amax().float().clamp(min=1e-12)/FP8_MAX).reshape(())
        xq=(x2.float()/sa).clamp(-FP8_MAX,FP8_MAX).to(torch.float8_e4m3fn)
        y=torch._scaled_mm(xq,self.weight_fp8.t(),scale_a=sa,scale_b=self.weight_scale,out_dtype=torch.bfloat16)
        if self.pad_n:y=y[:,:self.out_features]
        if self.bias is not None:y=y+self.bias
        return y.reshape(*shape[:-1],self.out_features)


# ------------------------------------------------------------------------------------------------ NVFP4 (e2m1, block 16)
def nvfp4_quantize(x,nibble):
    """x (R,K) -> packed float4_e2m1fn_x2 (R,K/2), block scales e4m3 (R,K/16), global fp32 scale, dequantized fp32 (R,K)."""
    R,K=x.shape;xf=x.float();grid=torch.tensor(E2M1,device=x.device);bnd=(grid[1:]+grid[:-1])/2
    g=(xf.abs().amax().clamp(min=1e-12)/(6*FP8_MAX)).reshape(())
    xb=xf.view(R,K//16,16);bs=(xb.abs().amax(-1)/(6*g)).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    eff=bs.float()*g;v=xb/torch.where(eff>0,eff,torch.ones_like(eff))[...,None]
    idx=torch.bucketize(v.abs().clamp(max=6.),bnd);neg=(v<0)&(idx>0)
    codes=(idx|(neg.long()<<3)).to(torch.uint8).view(R,K)
    packed=(codes[:,0::2]|(codes[:,1::2]<<4)) if nibble=='lo_first' else ((codes[:,0::2]<<4)|codes[:,1::2])
    deq=(torch.where(neg,-1.,1.)*grid[idx]*eff[...,None]).view(R,K)
    return packed.contiguous().view(torch.float4_e2m1fn_x2),bs,g,deq


def to_blocked(s):
    """Swizzle (rows, K/16) scales into the 128x4 tiled layout used by cuBLASLt block-scaled GEMMs (rows/cols zero-padded)."""
    r,c=s.shape;R=-(-r//128)*128;C=-(-c//4)*4
    p=torch.zeros(R,C,dtype=s.dtype,device=s.device);p[:r,:c]=s
    return p.view(R//128,128,C//4,4).permute(0,2,1,3).reshape(-1,4,32,4).transpose(1,2).reshape(-1,32,16).flatten()


def nvfp4_mm(a,b,layout):
    """a (M,K), b (N,K) high precision -> (y bf16 (M,N) from _scaled_mm, reference fp32 from the dequantized operands).
    M is zero-padded to a multiple of 128 so that both scale layouts have whole tiles."""
    nib,lay=layout;M=len(a);pad=(-M)%128
    if pad:a=torch.cat([a,a.new_zeros(pad,a.shape[1])])
    qa,sa,ga,da=nvfp4_quantize(a,nib);qb,sb,gb,db=nvfp4_quantize(b,nib)
    fa,fb=(to_blocked(sa),to_blocked(sb)) if lay=='blocked' else (sa.flatten(),sb.flatten())
    y=torch._scaled_mm(qa,qb.t(),scale_a=fa,scale_b=fb,out_dtype=torch.bfloat16)
    return (y.float()*(ga*gb))[:M].to(torch.bfloat16),(da@db.t())[:M]


LAYOUTS=[(n,l) for l in ['blocked','rowmajor'] for n in ['lo_first','hi_first']]
# Real expert shapes (hidden 2048, intermediate 8256, 16 q heads / kv 1024) at batch 1 (64 tokens) and batch 6 (384 tokens).
VALIDATION_SHAPES=[(m,n,k) for m in [64,384] for n,k in [(2048,2048),(1024,2048),(8256,2048),(2048,8256)]]


def nvfp4_layout_validation(device='cuda',tol=1e-2):
    """For each candidate (nibble order, scale layout): relative error of _scaled_mm against the matmul of the same
    dequantized operands, on random data at the real expert shapes. Exactly one layout must pass (rel err < tol)."""
    torch.manual_seed(20261029);res={'tol':tol,'shapes':VALIDATION_SHAPES,'layouts':{}}
    for layout in LAYOUTS:
        errs=[]
        try:
            for m,n,k in VALIDATION_SHAPES:
                a=torch.randn(m,k,device=device);b=torch.randn(n,k,device=device)*.02
                y,ref=nvfp4_mm(a,b,layout);errs.append(float((y.float()-ref).norm()/ref.norm()))
            res['layouts']['/'.join(layout)]={'rel_err':errs,'pass':bool(max(errs)<tol)}
        except Exception as e:res['layouts']['/'.join(layout)]={'error':f'{type(e).__name__}: {str(e)[:300]}','pass':False}
    passing=[k for k,v in res['layouts'].items() if v['pass']]
    res['passing']=passing;res['passed']=len(passing)==1;res['layout']=passing[0] if len(passing)==1 else None
    return res


class NVFP4Linear(nn.Module):
    def __init__(self,lin,layout,device='cuda'):
        super().__init__();self.layout=layout;w=lin.weight.detach().to(device,torch.bfloat16)
        self.out_features,self.in_features=w.shape;assert self.in_features%64==0 and self.out_features%16==0
        qb,sb,gb,_=nvfp4_quantize(w,layout[0])
        self.register_buffer('weight_fp4',qb);self.register_buffer('weight_scale',to_blocked(sb) if layout[1]=='blocked' else sb.flatten())
        self.register_buffer('weight_global',gb);self.register_buffer('bias',None if lin.bias is None else lin.bias.detach().to(device,torch.bfloat16))

    def forward(self,x):
        shape=x.shape;a=x.reshape(-1,self.in_features).to(torch.bfloat16);M=len(a);pad=(-M)%128
        if pad:a=torch.cat([a,a.new_zeros(pad,self.in_features)])
        qa,sa,ga,_=nvfp4_quantize(a,self.layout[0]);fa=to_blocked(sa) if self.layout[1]=='blocked' else sa.flatten()
        y=torch._scaled_mm(qa,self.weight_fp4.t(),scale_a=fa,scale_b=self.weight_scale,out_dtype=torch.bfloat16)
        y=(y.float()*(ga*self.weight_global))[:M].to(torch.bfloat16)
        if self.bias is not None:y=y+self.bias
        return y.reshape(*shape[:-1],self.out_features)


# ------------------------------------------------------------------------------------------------ model surgery
EXCLUDE=('action_in_proj','action_out_proj')


def targets(model,variant):
    roots=['expert']+(['vlm'] if variant=='P2' else [])
    return [(n,m) for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.split('.')[0] in roots and not any(e in n for e in EXCLUDE)]


def make_quant(lin,variant,layout,device='cuda'):
    if variant=='P3':return NVFP4Linear(lin,layout,device)
    return FP8Linear(lin,device)


def quantize(model,variant,layout=None,inputs=None,device='cuda'):
    """Replace the target linears in place. With inputs {name: (rows,K) bf16} also measure the per-linear relative error
    against the bf16 linear on those activations. Linears the kernel cannot take stay bf16 and are listed as skipped."""
    report={'variant':variant,'flag_threshold':FLAG,'linears':{},'skipped':{}}
    for name in [n for n,_ in targets(model,variant)]:  # names only: each bf16 linear is freed as soon as it is replaced
        parent=model.get_submodule(name.rsplit('.',1)[0]) if '.' in name else model;child=name.rsplit('.',1)[-1];lin=getattr(parent,child)
        try:q=make_quant(lin,variant,layout,device)
        except AssertionError as e:report['skipped'][name]=f'{tuple(lin.weight.shape)}: {e}';continue
        entry={'shape':list(lin.weight.shape)}
        if inputs is not None and name in inputs:
            with torch.inference_mode():
                x=inputs[name].to(device);ref=F.linear(x,lin.weight.to(device,torch.bfloat16),None if lin.bias is None else lin.bias.to(device,torch.bfloat16))
                y=q(x);entry['rel_err']=float((y.float()-ref.float()).norm()/ref.float().norm().clamp(min=1e-12));entry['rows']=len(x)
                entry['flagged']=entry['rel_err']>FLAG
        setattr(parent,child,q);report['linears'][name]=entry;del lin
    errs=[e['rel_err'] for e in report['linears'].values() if 'rel_err' in e]
    report['summary']={'n_quantized':len(report['linears']),'n_skipped':len(report['skipped']),'n_measured':len(errs),
        'n_flagged':int(sum(e>FLAG for e in errs)),'max_rel_err':max(errs) if errs else None,
        'median_rel_err':float(np.median(errs)) if errs else None,'variant_flagged':bool(any(e>FLAG for e in errs))}
    gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return report


def capture_inputs(model,helper,processor,variant,clip_path,rows=512,per_call=64):
    """Real bf16 activations entering every target linear while prepare_field runs context_000 (VLM prefill + reasoning
    rollout + the official E10 replay through the expert): up to `per_call` random rows per call, `rows` per linear."""
    from frost.probes.alpamayo_field import prepare_field
    store={};gen=torch.Generator().manual_seed(20261029);hooks=[]
    def hook_for(name):
        def hook(mod,args):
            x=args[0].detach();x2=x.reshape(-1,x.shape[-1]);have=sum(len(t) for t in store.get(name,[]));take=min(per_call,rows-have,len(x2))
            if take>0:store.setdefault(name,[]).append(x2[torch.randperm(len(x2),generator=gen)[:take].to(x2.device)].to('cpu',torch.bfloat16))
        return hook
    for name,lin in targets(model,variant):hooks.append(lin.register_forward_pre_hook(hook_for(name)))
    try:field,*_=prepare_field(model,helper,processor,clip_path);del field
    finally:
        for h in hooks:h.remove()
    gc.collect();torch.cuda.empty_cache()
    return {k:torch.cat(v) for k,v in store.items()}


def load_host():
    """Whole Alpamayo-1.5 in bf16 on the host (P2: quantized layer by layer onto the GPU instead of offloading)."""
    import thor_compat  # noqa: F401  (SDPA attention patch)
    import frost.probes.alpamayo_field as af
    sys.path.insert(0,str(Path(os.environ['ALPAMAYO15_SRC'])/'src'))  # checkout of NVlabs/alpamayo1.5
    from frost.probes.alpamayo_compat import patch_legacy_hydra_targets
    patch_legacy_hydra_targets()
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    model=Alpamayo1_5.from_pretrained(str(af.CHECKPOINT),dtype=torch.bfloat16,local_files_only=True).eval();model.requires_grad_(False)
    return model,helper,helper.get_processor(model.tokenizer)


def build_model(variant,layout=None,calib_clip=None,report_path=None):
    """Load and quantize one variant. With calib_clip, record per-linear errors on its real activations to report_path."""
    import thor_compat
    inputs=None
    if variant=='P0':return (*thor_compat.load_alpamayo_thor(),None)
    if calib_clip is not None:
        model,helper,processor=thor_compat.load_alpamayo_thor()  # bf16 reference, same as P0
        t=time.time();inputs=capture_inputs(model,helper,processor,variant,calib_clip);print(json.dumps({'captured_linears':len(inputs),'s':time.time()-t}),flush=True)
        if variant=='P2':del model;gc.collect();torch.cuda.empty_cache()
    if variant=='P2':model,helper,processor=load_host()
    elif inputs is None:model,helper,processor=thor_compat.load_alpamayo_thor()
    report=quantize(model,variant,layout,inputs)
    if variant=='P2':model.to('cuda')
    report['cuda_allocated_gb']=torch.cuda.memory_allocated()/1e9
    if report_path is not None:Path(report_path).write_text(json.dumps(report,indent=2))
    print(json.dumps({'variant':variant,**report['summary'],'cuda_allocated_gb':report['cuda_allocated_gb']}),flush=True)
    return model,helper,processor,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=['P1','P2','P3'],required=True)
    p.add_argument('--out',type=Path,required=True,help='base output dir (out/thor_v1); this variant writes to OUT/<variant>/')
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--range',type=int,nargs=2,default=[0,60])
    p.add_argument('--calib-clip',default='context_000');p.add_argument('--validate-only',action='store_true',help='P3: run only the layout test')
    a=p.parse_args();out=a.out/a.variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():print(f'{a.variant}: probe already complete ({out}/manifest.json)');return
    layout=None
    if a.variant=='P3':
        val=nvfp4_layout_validation();(out/'layout_validation.json').write_text(json.dumps(val,indent=2));print(json.dumps(val),flush=True)
        if not val['passed']:print('P3: NVFP4 layout validation FAILED: P3 is not run',flush=True);sys.exit(3)
        layout=tuple(val['layout'].split('/'))
        if a.validate_only:return
    errs=out/'linear_errors.json'
    model,helper,processor,_=build_model(a.variant,layout,None if errs.exists() else a.data/f'{a.calib_clip}.npz',None if errs.exists() else errs)
    # Reasoning text + official E10 replay per clip (T5 compares P2's text with the bf16 VLM of P1, identical to P0's).
    import frost.probes.temperature_probe as tp
    log=open(out/'reasoning.jsonl','a');orig_sample=model.sample_trajectories_from_data_with_vlm_rollout;last={}
    def sample(*args,**kw):
        r=orig_sample(*args,**kw)
        if kw.get('return_extra'):last['extra']={k:np.asarray(v).tolist() for k,v in r[2].items()}
        return r
    model.sample_trajectories_from_data_with_vlm_rollout=sample;orig_prepare=tp.prepare_field
    def prepare(model_,helper_,processor_,path):
        r=orig_prepare(model_,helper_,processor_,path)
        log.write(json.dumps({'clip':Path(path).stem,'extra':last.pop('extra',None),'e10_replay_final_xy':r[4][-1,:2].tolist()})+'\n');log.flush()
        return r
    tp.prepare_field=prepare;tp.load_alpamayo=lambda:(model,helper,processor)
    sys.argv=['temperature_probe','--data',str(a.data),'--out',str(out),'--range',*map(str,a.range),'--seeds',*map(str,SEEDS),'--configs',*CONFIGS,'--resume']
    (out/'variant.json').write_text(json.dumps({'variant':a.variant,'layout':layout,'configs':CONFIGS,'seeds':SEEDS,'range':a.range,
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0)},indent=2))
    tp.main()


if __name__=='__main__':main()
