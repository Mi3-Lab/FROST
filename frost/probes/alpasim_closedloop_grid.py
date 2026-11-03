"""Resumable AlpaSim closed-loop grid: Alpamayo 1.5 with several flow samplers on fixed scenes.

One wizard run per configuration. The sampler is selected inside the driver
container by FROST_* environment variables (installed by
patches/alpasim/frost_sampler.patch as alpasim_driver/models/frost_sampler.py); the official configuration passes none.
A configuration is skipped when its aggregate metrics already exist.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

CONFIGS={'euler10_t1.0':None,'euler4_t1.0':('euler',4,1.),'heun32_t1.0':('heun',32,1.),
    'heun32_t0.84':('heun',32,.84),'heun32_t0.65':('heun',32,.65)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenes',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--alpasim',type=Path,required=True)
    p.add_argument('--only',nargs='*',default=None);p.add_argument('--extra',nargs='*',default=[],help='Additional wizard overrides')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(args.scenes.read_text())['scene_ids']
    for name,sampler in CONFIGS.items():
        if args.only and name not in args.only:continue
        log_dir=(args.out/name).resolve()
        if (log_dir/'aggregate'/'metrics_unprocessed.parquet').exists():continue
        # A few scenes fail AlpaSim's route sanity checks in every configuration, which blocks aggregation; per-rollout metrics suffice.
        if len(list(log_dir.glob('rollouts/*/*/metrics.parquet')))>=.9*len(scenes):continue
        if log_dir.exists():
            # The wizard refuses to reuse a log directory; keep failed attempts for inspection.
            log_dir.rename(args.out/f'{name}.failed.{len(list(args.out.glob(name+".failed.*")))}')
        overrides=['deploy=local','topology=1gpu','driver=alpamayo1_5',f'wizard.log_dir={log_dir}',
            'scenes.scene_ids=['+','.join(scenes)+']',*args.extra]
        if sampler:
            method,steps,tau=sampler
            # Hydra list elements containing '=' must be quoted.
            overrides.append(f'+services.driver.environments=["FROST_METHOD={method}","FROST_STEPS={steps}","FROST_TAU={tau}"]')
        print(f'[closedloop] {name}',flush=True)
        with open(args.out/f'{name}.log','a') as log:
            code=subprocess.run(['uv','run','alpasim_wizard',*overrides],cwd=args.alpasim,env=os.environ,stdout=log,stderr=subprocess.STDOUT).returncode
        print(f'[closedloop] {name} rc={code}',flush=True)


if __name__=='__main__':main()
