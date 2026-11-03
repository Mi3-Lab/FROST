"""Exact SimWAM ego inputs for our PhysicalAI-AV contexts, from the official egomotion interpolator.

SimWAM's PhysicalAI preprocessing (not shipped in the repo; verified numerically against the released
physicalai_train.jsonl on three training windows) takes ego_velocity / ego_acceleration at the anchor from the
logged egomotion (world frame, rotated into the anchor rig frame) and future_traj_10hz = x, y, heading of the next 40
poses at 10 Hz in the anchor frame (x forward, y left). We apply the same function (ego_series of
qwendrive_temperature_probe, the official interpolator) to each context's (clip_id, t0_us). Run in the Qwen-Drive venv
(physical_ai_av) with the user's HF token.
"""
import argparse,json
from pathlib import Path
import numpy as np
from frost.probes.qwendrive_temperature_probe import ego_series


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--range',type=int,nargs=2,default=[0,400]);p.add_argument('--names',nargs='+',default=None);a=p.parse_args()
    import physical_ai_av
    avdi=physical_ai_av.PhysicalAIAVDatasetInterface();out={};check={}
    records=sorted((r for r in json.loads(a.records.read_text()) if 'error' not in r),key=lambda r:r['name'])[a.range[0]:a.range[1]]
    if a.names:records=[r for r in records if r['name'] in set(a.names)]
    for r in records:
        ego=avdi.get_clip_feature(r['clip_id'],avdi.features.LABELS.EGOMOTION,maybe_stream=True);t0=r['t0_us']
        fut,_,_=ego_series(ego,t0,np.arange(1,41)/10);_,v,acc=ego_series(ego,t0,np.array([0.]))
        out[f"{r['name']}_vel"]=v[0].astype(np.float32);out[f"{r['name']}_acc"]=acc[0].astype(np.float32);out[f"{r['name']}_gt_xyh"]=fut.astype(np.float32)
        with np.load(a.records.parent/f"{r['name']}.npz") as z:check[r['name']]=float(np.abs(z['ego_future_xyz'][:40,:2]-fut[:,:2]).max())
        print(json.dumps({'context':r['name'],'max_diff_vs_context_future_m':check[r['name']]}),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,**out)
    (a.out.with_suffix('.check.json')).write_text(json.dumps(check))


if __name__=='__main__':main()
