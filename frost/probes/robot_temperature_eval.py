"""Closed-loop LIBERO evaluation of flow policies with a chosen integrator and noise temperature.

Run with the isolated robotics venv. The policy's euler_integrate (shared by
SmolVLA and pi0.5 in LeRobot 0.6.1) is replaced in-process; nothing installed is
modified. Environment variables select the integrator:
  FROST_METHOD=euler|heun  FROST_STEPS=int  FROST_TAU=float
  FROST_PROBE_EVERY=int    FROST_PROBE_OUT=path.npz  (optional offline probe)
The probe never changes the executed action: it integrates extra copies from
fresh noise and stores chunks for shrinkage / effective-temperature analysis.
Time runs from t=1 (noise) to t=0 (actions), as in openpi.
"""
import os
import sys
import numpy as np
import torch

PROBE_CONFIGS=[('euler',1,1.),('euler',2,1.),('euler',4,1.),('euler',10,1.)]+[('heun',16,tau) for tau in [1.,.8,.6,.4,.2]]


def integrate(denoise_fn,noise,steps,method,tau):
    x=tau*noise;h=1./steps;bsize=noise.shape[0]
    for step in range(steps):
        t=torch.full((bsize,),1.-step*h,dtype=torch.float32,device=noise.device);v=denoise_fn(x,t)
        if method=='heun':
            v2=denoise_fn(x-h*v,torch.full((bsize,),1.-(step+1)*h,dtype=torch.float32,device=noise.device));x=x-.5*h*(v+v2)
        else:x=x-h*v
    return x


def install():
    method=os.environ.get('FROST_METHOD','euler');steps=int(os.environ.get('FROST_STEPS','10'));tau=float(os.environ.get('FROST_TAU','1'))
    probe_every=int(os.environ.get('FROST_PROBE_EVERY','0'));probe_out=os.environ.get('FROST_PROBE_OUT');state={'calls':0,'probes':{}}
    def patched(denoise_fn,noise,num_steps,*,rtc_enabled=False,**kw):
        if rtc_enabled:raise RuntimeError('RTC guidance is not supported by the FROST integrator')
        state['calls']+=1
        if probe_every and probe_out and state['calls']%probe_every==1:
            gen=torch.Generator(device=noise.device).manual_seed(state['calls'])
            for seed in range(4):
                z=torch.randn(noise.shape,generator=gen,device=noise.device,dtype=noise.dtype)
                for m,k,t in PROBE_CONFIGS:
                    state['probes'][f"c{state['calls']}_s{seed}_{m}{k}_t{t}"]=integrate(denoise_fn,z,k,m,t).float().cpu().numpy()
            np.savez_compressed(probe_out,**state['probes'])
        return integrate(denoise_fn,noise,steps,method,tau)
    from lerobot.policies.smolvla import modeling_smolvla
    from lerobot.policies.pi05 import modeling_pi05
    modeling_smolvla.euler_integrate=patched;modeling_pi05.euler_integrate=patched
    print(f'[frost] integrator={method}{steps} tau={tau} probe_every={probe_every}',flush=True)


if __name__=='__main__':
    install()
    from lerobot.scripts import lerobot_eval
    sys.argv=[sys.argv[0]]+sys.argv[1:];lerobot_eval.main()
