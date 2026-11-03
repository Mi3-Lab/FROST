"""Convert WOD-E2E validation records to the native Alpamayo-1.5 context .npz the probes read.

Reads the raw TFRecord shards (E2EDFrame protobufs, proto at waymo-research/waymo-open-dataset,
protos/end_to_end_driving_data.proto) with a dependency-free wire-format parser: no TensorFlow,
no waymo-open-dataset package. Field numbers used (verified against the released shards):
E2EDFrame{1: Frame, 5: future_states, 6: past_states, 7: intent, 8: preference_trajectories};
Frame{1: Context{1: name, 2: CameraCalibration}, 2: timestamp_micros, 4: CameraImage{1: camera name, 2: JPEG}};
EgoTrajectoryStates{1,2,3: pos_x,y,z; 4,5: vel_x,y; 6,7: accel_x,y; 8: preference_score}.
Camera names: 1 FRONT, 2 FRONT_LEFT, 3 FRONT_RIGHT, 4 SIDE_LEFT, 5 SIDE_RIGHT, 6..8 rear.

Adapter to the Alpamayo input profile (4 cameras x 4 frames at 10 Hz, 16 history poses at 10 Hz):
* cameras [cross_left, front_wide, cross_right, front_tele] <- [FRONT_LEFT, FRONT, FRONT_RIGHT,
  centre crop of FRONT (``--tele-zoom``, default 2.0x) resized back]. The Alpamayo cross cameras are 120 deg
  FOV and WOD's front-side cameras are narrower and yawed ~45 deg, so this is an approximation.
* Each record is ONE camera timestamp (frames of a segment are shuffled over shards, named <segment>-<idx>, idx presumably
  10 Hz; timestamp_micros is 0 so the spacing is not verifiable from the file). With ``--history-index`` (a full `scan`
  index) the 4 temporal frames are idx-3..idx of the same segment; any missing neighbour falls back to repeating the
  nearest newer available frame, and ``n_real_history_frames`` is recorded per context. Without it all 4 are copies.
* ego history: WOD past_states are 4 Hz over (-4 s, 0], origin at the t=0 rear axle, axes of the t=0 vehicle.
  The last 1.5 s are resampled to 16 poses at 10 Hz (monotone cubic, PCHIP); headings come from the path tangent
  (0 where speed < 0.5 m/s), the last pose is identity, as in the PhysicalAI contexts.
* ego_future_xyz: logged future interpolated to 10 Hz for 5 s, NaN for the remaining 1.4 s of Alpamayo's 6.4 s.
Extra keys written for RFS: rater_xy (K,21,2) (first waypoint at t=0.25 s as the logged future's, 21 waypoints; checked on 3 records), rater_scores (K,), wod_future_xyz (20,3), wod_past_xyz (16,3), wod_speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

CAMERA_FRONT,CAMERA_FRONT_LEFT,CAMERA_FRONT_RIGHT=1,2,3
HISTORY_STEPS=16;HISTORY_DT=.1;FUTURE_STEPS=64


def _varint(b,i):
    r=0;s=0
    while True:
        c=b[i];i+=1;r|=(c&127)<<s;s+=7
        if c<128:return r,i


def wire_fields(b):
    """Yield (field number, wire type, value) of a protobuf message; length-delimited values are bytes."""
    i=0;n=len(b)
    while i<n:
        k,i=_varint(b,i);f,w=k>>3,k&7
        if w==0:v,i=_varint(b,i)
        elif w==1:v=bytes(b[i:i+8]);i+=8
        elif w==2:
            m,i=_varint(b,i);v=b[i:i+m];i+=m
        elif w==5:v=bytes(b[i:i+4]);i+=4
        else:raise ValueError(f'wire type {w}')
        yield f,w,v


def iter_tfrecord(path,offsets=None):
    """Yield (offset, payload) of a TFRecord file; CRCs are not checked. ``offsets`` restricts to those records."""
    with open(path,'rb') as fh:
        if offsets is not None:
            for off in offsets:
                fh.seek(off);n=struct.unpack('<Q',fh.read(12)[:8])[0];yield off,fh.read(n)
            return
        while True:
            off=fh.tell();h=fh.read(12)
            if len(h)<12:return
            n=struct.unpack('<Q',h[:8])[0];yield off,fh.read(n);fh.seek(4,1)


def _floats(b):return np.frombuffer(b,dtype='<f4').astype(np.float64)


def _states(b):
    out={}
    for f,w,v in wire_fields(b):
        if f==8:out['score']=struct.unpack('<f',v)[0]
        elif w==2 and f<=7:out[f]=_floats(v)
    return out


def parse_frame(record,images=True):
    """Parse an E2EDFrame. Returns dict; JPEG bytes in ``images`` (camera name -> bytes) if requested."""
    d={'raters':[],'scores':[],'images':{}}
    for f,w,v in wire_fields(record):
        if f==1:
            for f2,w2,v2 in wire_fields(v):
                if f2==1:
                    for a,_,c in wire_fields(v2):
                        if a==1:d['frame_name']=c.decode()
                elif f2==2:d['timestamp_us']=v2
                elif f2==4 and images:
                    cam=None;jpg=None
                    for a,_,c in wire_fields(v2):
                        if a==1:cam=c
                        elif a==2:jpg=c
                    d['images'][cam]=bytes(jpg)
        elif f==5:s=_states(v);d['future']=np.stack([s[1],s[2],s.get(3,np.zeros_like(s[1]))],1)
        elif f==6:
            s=_states(v);d['past']=np.stack([s[1],s[2],s.get(3,np.zeros_like(s[1]))],1);d['past_vel']=np.stack([s[4],s[5]],1)
        elif f==7:d['intent']=int(v)
        elif f==8:
            s=_states(v);d['scores'].append(s.get('score',-1.));d['raters'].append(np.stack([s[1],s[2]],1) if 1 in s else np.zeros((0,2)))
    d['segment']=d['frame_name'].rsplit('-',1)[0];d['frame_index']=int(d['frame_name'].rsplit('-',1)[1])
    return d


def n_rated(d):return sum(s>=0 for s in d['scores'])


def history_10hz(past):
    """(16,3) 4 Hz past states (t=-3.75..0) -> 16 poses at 10 Hz (t=-1.5..0): xyz (16,3), rot (16,3,3)."""
    t4=-.25*np.arange(len(past)-1,-1,-1);t10=-HISTORY_DT*np.arange(HISTORY_STEPS-1,-1,-1)
    try:
        from scipy.interpolate import PchipInterpolator
        xyz=PchipInterpolator(t4,past,axis=0)(t10)
    except ImportError:xyz=np.stack([np.interp(t10,t4,past[:,k]) for k in range(3)],1)
    xyz=xyz-xyz[-1]
    tangent=np.gradient(xyz[:,:2],HISTORY_DT,axis=0);speed=np.linalg.norm(tangent,axis=1)
    yaw=np.where(speed>.5,np.arctan2(tangent[:,1],tangent[:,0]),0.);yaw[-1]=0.
    c,s=np.cos(yaw),np.sin(yaw);rot=np.zeros((len(yaw),3,3));rot[:,0,0]=c;rot[:,0,1]=-s;rot[:,1,0]=s;rot[:,1,1]=c;rot[:,2,2]=1
    return xyz.astype(np.float32),rot.astype(np.float32)


def future_10hz(future):
    """(20,3) 4 Hz future -> (64,3) 10 Hz, NaN after 5 s; plus yaw rotations."""
    t4=np.concatenate([[0.],.25*np.arange(1,len(future)+1)]);f=np.vstack([np.zeros((1,3)),future]);t10=HISTORY_DT*np.arange(1,FUTURE_STEPS+1)
    xyz=np.full((FUTURE_STEPS,3),np.nan);m=t10<=t4[-1]+1e-9
    xyz[m]=np.stack([np.interp(t10[m],t4,f[:,k]) for k in range(3)],1)
    rot=np.full((FUTURE_STEPS,3,3),np.nan);tan=np.gradient(xyz[m][:,:2],HISTORY_DT,axis=0);speed=np.linalg.norm(tan,axis=1)
    yaw=np.where(speed>.5,np.arctan2(tan[:,1],tan[:,0]),0.);c,s=np.cos(yaw),np.sin(yaw)
    r=np.zeros((len(yaw),3,3));r[:,0,0]=c;r[:,0,1]=-s;r[:,1,0]=s;r[:,1,1]=c;r[:,2,2]=1;rot[m]=r
    return xyz.astype(np.float32),rot.astype(np.float32)


def alpamayo_images(history,height=640,tele_zoom=2.):
    """List of 4 JPEG dicts (WOD camera id -> bytes), oldest to newest -> uint8 (4,4,H,W,3) in [cross_left, front_wide, cross_right, front_tele] order."""
    import cv2
    def dec(jpgs,cam):
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(jpgs[cam],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    def four(jpgs):
        left,front,right=dec(jpgs,CAMERA_FRONT_LEFT),dec(jpgs,CAMERA_FRONT),dec(jpgs,CAMERA_FRONT_RIGHT)
        h,w=front.shape[:2];ch,cw=int(h/tele_zoom),int(w/tele_zoom);y0,x0=(h-ch)//2,(w-cw)//2
        width=int(round(height*w/h/2))*2
        return [cv2.resize(x,(width,height),interpolation=cv2.INTER_AREA) for x in (left,front,right,front[y0:y0+ch,x0:x0+cw])]
    per_time=[four(j) for j in history]
    return np.stack([np.stack([per_time[t][c] for t in range(len(history))]) for c in range(4)]).astype(np.uint8)


def pad_rater(xy,n=21):
    """Rater paths may be shorter than 5 s (e.g. 12 points = 3 s); pad to n points with NaN (protocol addendum 1)."""
    out=np.full((n,2),np.nan);xy=np.asarray(xy,float)[:n];out[:len(xy)]=xy;return out


def to_context(d,history=None,height=640,tele_zoom=2.):
    history=history or [d['images']]*4
    hx,hr=history_10hz(d['past']);fx,fr=future_10hz(d['future'])
    rel=np.array([[0,.1,.2,.3],[.018,.118,.218,.318]]*2,np.float32)  # same layout as PhysicalAI contexts
    rated=[k for k,s in enumerate(d['scores']) if s>=0]
    return dict(image_frames=alpamayo_images(history,height,tele_zoom),ego_history_xyz=hx,ego_history_rot=hr,ego_future_xyz=fx,ego_future_rot=fr,
        relative_timestamps=rel,rater_xy=np.stack([pad_rater(d['raters'][k]) for k in rated]).astype(np.float32) if rated else np.zeros((0,21,2),np.float32),
        rater_scores=np.array([d['scores'][k] for k in rated],np.float32),wod_future_xyz=d['future'].astype(np.float32),wod_past_xyz=d['past'].astype(np.float32),
        wod_speed=np.float32(np.linalg.norm(d['past_vel'][-1])))


def scan_shard(path):
    """Index a shard without decoding images: list of dicts (offset, frame name, n_rated)."""
    out=[]
    for off,rec in iter_tfrecord(path):
        d=parse_frame(rec,images=False);out.append({'shard':Path(path).name,'offset':off,'frame_name':d['frame_name'],'segment':d['segment'],'n_rated':n_rated(d)})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='dir with val/ shards and val_sequence_name_to_scenario_cluster.json')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('scan',help='index rated frames of shards (no image decoding)');a.add_argument('--shards',type=int,nargs='+',required=True);a.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('convert',help='convert records listed in a scan index');c.add_argument('--index',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    c.add_argument('--history-index',type=Path,default=None,help='full scan index (all shards) used to fetch frames idx-3..idx');c.add_argument('--limit',type=int,default=None);c.add_argument('--height',type=int,default=640);c.add_argument('--tele-zoom',type=float,default=2.)
    args=p.parse_args()
    shard=lambda i:args.data/'val'/f'val_202504211843.tfrecord-{i:05d}-of-00093'
    if args.cmd=='scan':
        rows=[r for i in args.shards for r in scan_shard(shard(i))];args.out.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        print(json.dumps({'records':len(rows),'rated':sum(r['n_rated']>0 for r in rows),'segments':len({r['segment'] for r in rows})}));return
    rows=[json.loads(l) for l in args.index.read_text().splitlines()][:args.limit];args.out.mkdir(parents=True,exist_ok=True)
    cluster=json.loads((args.data/'val_sequence_name_to_scenario_cluster.json').read_text());source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    lookup={r['frame_name']:r for r in map(json.loads,args.history_index.read_text().splitlines())} if args.history_index else None
    records=[]
    for pos,row in enumerate(rows):
        name=f'context_{pos:03d}'
        try:
            (_,rec),=iter_tfrecord(shard(int(row['shard'].split('-')[1])),[row['offset']]);d=parse_frame(rec)
            assert d['frame_name']==row['frame_name'] and n_rated(d)>0
            history,real=[d['images']],1
            for back in (1,2,3):
                key=f"{d['segment']}-{d['frame_index']-back}"
                if lookup and key in lookup:
                    (_,r2),=iter_tfrecord(shard(int(lookup[key]['shard'].split('-')[1])),[lookup[key]['offset']]);history.insert(0,parse_frame(r2)['images']);real+=1
                else:history.insert(0,history[0])
            z=to_context(d,history,args.height,args.tele_zoom);path=args.out/f'{name}.npz';np.savez_compressed(path,**z)
            records.append({'context':pos,'name':name,'frame_name':d['frame_name'],'segment':d['segment'],'scenario_cluster':cluster.get(d['segment'],{}).get('scenario_cluster'),
                'intent':d.get('intent'),'speed_mps':float(z['wod_speed']),'n_rated':int(len(z['rater_scores'])),'n_real_history_frames':real,'rater_scores':z['rater_scores'].tolist(),
                'image_shape':list(z['image_frames'].shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as error:records.append({'context':pos,'frame_name':row['frame_name'],'error':repr(error)})
        print(json.dumps({k:v for k,v in records[-1].items() if k!='rater_scores'}),flush=True)
    (args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'index':str(args.index),'height':args.height,'tele_zoom':args.tele_zoom,'source_sha256':hashlib.sha256(source).hexdigest()},indent=2))


if __name__=='__main__':main()
