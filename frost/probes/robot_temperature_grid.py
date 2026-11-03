"""Resumable LIBERO grid: integrator x temperature x suite for a frozen flow policy.

Each (config, suite) runs robot_temperature_eval in its own process and is
skipped when its eval_info.json already exists. Up to --parallel runs share the
GPU; a single driver owns the queue, so two workers never touch the same run.
Same eval seed for every config, so episode i of a task starts from the same
LIBERO initial state.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys

CONFIGS=[('euler',10,1.),('euler',4,1.),('euler',2,1.),('euler',1,1.)]+[('heun',16,t) for t in [1.,.9,.75,.5,.25]]
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10);p.add_argument('--batch',type=int,default=10);p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--parallel',type=int,default=1)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    pending=[(m,k,t,s) for m,k,t in CONFIGS for s in SUITES if not (args.out/f'{m}{k}_t{t}'/s/'eval_info.json').exists()]
    # Longest suite first so the slow runs do not end up alone at the tail.
    pending.sort(key=lambda r:r[3]!='libero_10')
    def run(item):
        method,steps,tau,suite=item;run_dir=args.out/f'{method}{steps}_t{tau}'/suite
        # LeRobot refuses an existing output dir; a partial run has no eval_info.json, so restart it.
        shutil.rmtree(run_dir,ignore_errors=True)
        env={**os.environ,'FROST_METHOD':method,'FROST_STEPS':str(steps),'FROST_TAU':str(tau)}
        cmd=[sys.executable,'-P','frost/probes/robot_temperature_eval.py',f'--policy.path={args.policy}','--env.type=libero',
            f'--env.task={suite}',f'--eval.n_episodes={args.episodes}',f'--eval.batch_size={args.batch}',
            f'--output_dir={run_dir}',f'--seed={args.seed}']
        print(f'[grid] start {method}{steps} tau={tau} {suite}',flush=True)
        with open(args.out/f'{method}{steps}_t{tau}_{suite}.log','w') as log:
            code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        print(f'[grid] done {method}{steps} tau={tau} {suite} rc={code}',flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:list(pool.map(run,pending))


if __name__=='__main__':main()
