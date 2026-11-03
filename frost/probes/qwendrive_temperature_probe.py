"""Numerical-temperature probe for Qwen-Drive-1.0 on PhysicalAI-AV clips.

Run with the isolated Qwen-Drive venv. Scenes follow docs/data.md of the Qwen-Drive release:
three cameras (front wide, cross left, cross right) at t0-1.5/-1.0/-0.5/0 s, 1.5 s of
10 Hz history and 5 s of 10 Hz future in the current ego frame, with velocity and
acceleration from the official egomotion interpolator (world frame, rotated here).
The navigation command is derived from the logged future heading, identically for all
configurations. The VLM prefill is computed once per clip; the planner is then sampled
with the release's Euler sampler at several step counts, and with a many-step sampler at
several initial-noise scales (tau multiplies noise_init_std). Six samples share the
release's per-sample noise seeds across configurations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

CAMERAS={'<FRONT VIEW>':'CAMERA_FRONT_WIDE_120FOV','<FRONT LEFT VIEW>':'CAMERA_CROSS_LEFT_120FOV','<FRONT RIGHT VIEW>':'CAMERA_CROSS_RIGHT_120FOV'}
TAUS=[1.,.9,.8,.7,.6,.5,.4]
CONFIGS=[(k,1.) for k in [1,2,4,10]]+[(64,t) for t in TAUS]
TURN_RAD=.35


def ego_series(ego,t0,offsets_s):
    sample=ego((t0+np.round(np.asarray(offsets_s)*1e6)).astype(np.int64))
    now=ego(np.array([t0],dtype=np.int64));inverse=now.pose.rotation[0].inv();origin=now.pose.translation[0]
    xy=inverse.apply(sample.pose.translation-origin)[:,:2]
    heading=np.array([(inverse*r).as_euler('xyz')[2] for r in sample.pose.rotation])
    velocity=inverse.apply(sample.velocity)[:,:2];acceleration=inverse.apply(sample.acceleration)[:,:2]
    return np.c_[xy,heading],velocity,acceleration


def nav_command(future):
    heading=future[-1,2]
    if heading>TURN_RAD:return 1,[1,0,0,0]
    if heading<-TURN_RAD:return 2,[0,0,1,0]
    return 0,[0,1,0,0]


def build_scene(avdi,clip,t0):
    from qwen_drive import CameraFrame,DrivingScene
    ego=avdi.get_clip_feature(clip,avdi.features.LABELS.EGOMOTION,maybe_stream=True)
    history,hist_vel,hist_acc=ego_series(ego,t0,np.arange(-15,1)/10)
    future,_,_=ego_series(ego,t0,np.arange(1,51)/10)
    history[-1]=0.
    command,driving=nav_command(future)
    stamps=(t0+np.array([-1.5,-1.,-.5,0.])*1e6).astype(np.int64);views={}
    for tag,feature in CAMERAS.items():
        camera=avdi.get_clip_feature(clip,getattr(avdi.features.CAMERA,feature),maybe_stream=True)
        frames,_=camera.decode_images_from_timestamps(stamps)
        views[tag]=[CameraFrame(Image.fromarray(np.asarray(f))) for f in frames]
    scene=DrivingScene(views=views,history=history,history_velocity=hist_vel,history_acceleration=hist_acc,
        ego_velocity=tuple(hist_vel[-1]),ego_acceleration=tuple(hist_acc[-1]),driving_command=driving,nav_command=command,token=clip)
    return scene,future


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--planner',default='planner-rl');p.add_argument('--range',type=int,nargs=2,default=[100,200])
    p.add_argument('--samples',type=int,default=6);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=args.resume)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    import physical_ai_av
    from qwen_drive import QwenDriveForPlanning
    records=sorted((r for r in json.loads(args.records.read_text()) if 'error' not in r),key=lambda r:r['name'])[args.range[0]:args.range[1]]
    model=QwenDriveForPlanning.from_pretrained(args.model,planner=str(Path(args.model)/args.planner),dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').cuda().eval()
    base_std=model.config.noise_init_std;base_floor=model.config.min_one_minus_t;avdi=physical_ai_av.PhysicalAIAVDatasetInterface()
    plans={}
    if args.resume and (args.out/'plans.npz').exists():
        with np.load(args.out/'plans.npz') as z:plans={k:z[k] for k in z.files}
    done={k.split('_s')[0] for k in plans}
    for record in records:
        name=record['name']
        if name in done:continue
        scene,future=build_scene(avdi,record['clip_id'],record['t0_us'])
        inputs={k:v.to(model.device) if torch.is_tensor(v) else v for k,v in model.processor(scene,with_reasoning=False,device='cpu').items()}
        with torch.no_grad():
            cache,anchor=model._prefill(inputs)
            for steps,tau in CONFIGS:
                # The release floors 1-t at 0.1 so its 10-step sampler lands on the endpoint; that floor
                # stops k>10 integrations short of the endpoint. 1/k keeps k<=10 identical to the release.
                model.config.min_one_minus_t=min(base_floor,1./steps)
                model.config.noise_init_std=base_std*tau
                try:trajectories=model._plan_from_cache(cache,anchor,inputs,num_samples=args.samples,num_steps=steps,seed=model.config.noise_seed)
                finally:model.config.noise_init_std=base_std;model.config.min_one_minus_t=base_floor
                for i,trajectory in enumerate(trajectories):plans[f'{name}_s{9201+i}_euler{steps}_t{tau}']=trajectory
        plans[f'{name}_human']=future
        np.savez_compressed(args.out/'plans.npz',**plans);print(json.dumps({'context':name,'nav_command':scene.nav_command,'plans':len(plans)}),flush=True)
        torch.cuda.empty_cache()
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'model':args.model,'planner':args.planner,'noise_init_std':base_std,
        'min_one_minus_t':base_floor,'floor_rule':'min(release floor, 1/steps)','official_steps':model.config.num_inference_steps,'range':args.range,
        'seeds':[9201+i for i in range(args.samples)],'configs':CONFIGS,'reference':'euler64_t1.0','turn_threshold_rad':TURN_RAD,
        'source_sha256':hashlib.sha256(source).hexdigest(),
        'limitations':['Navigation command derived from the logged future heading (same for all configurations)',
            'Direct planning mode, no reasoning','Trajectories are (x, y, heading) at 10 Hz over 5 s']},indent=2))


if __name__=='__main__':main()
