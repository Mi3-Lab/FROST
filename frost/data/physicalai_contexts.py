"""Cache PhysicalAI-AV test-split clips in the native solver-probe format, with logged future.

One clip per dataset chunk, sampled with a fixed seed before any solver result.
The logged future is the human driver's trajectory, not a unique correct action.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd

from frost.paths import PHYSICALAI_AV, env_path, hf_snapshot

INDEX=env_path('FROST_PHYSICALAI_INDEX',hf_snapshot(*PHYSICALAI_AV,kind='datasets'))


def sample_clips(index,split,count,seed,exclude_chunks=(),allowed=None):
    rows=index[(index.split==split)&~index.chunk.isin(list(exclude_chunks))].reset_index()
    if allowed is not None:rows=rows[rows.clip_id.isin(allowed)]
    per_chunk=rows.groupby('chunk').sample(n=1,random_state=seed)
    return per_chunk.sample(n=min(count,len(per_chunk)),random_state=seed).clip_id.tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=200);p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--split',default='test');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--t0-us',type=int,default=5_100_000)
    p.add_argument('--aligned-split',type=Path,help='splits/alpamayo_aligned.parquet: keep only clips in its test partition (held out from Alpamayo training)')
    p.add_argument('--exclude-records',type=Path,nargs='*',default=[],help='records.json of earlier samples whose chunks must not be reused')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).read_bytes();(args.out/'source_snapshot.py').write_bytes(source)
    sys.path.insert(0,str(env_path('ALPAMAYO15_SRC')/'src'))
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    index=pd.read_parquet(INDEX/'clip_index.parquet');meta=pd.read_parquet(INDEX/'metadata/data_collection.parquet')
    used={r['chunk'] for path in args.exclude_records for r in json.loads(path.read_text()) if 'chunk' in r}
    allowed=None
    if args.aligned_split:
        aligned=pd.read_parquet(args.aligned_split).reset_index();allowed=set(aligned.clip_id[aligned.split=='test'])
    clips=sample_clips(index,args.split,args.count,args.seed,used,allowed)
    (args.out/'sampled_clips.json').write_text(json.dumps(clips,indent=2))
    def fetch(item):
        position,clip=item;path=args.out/f'context_{position:03d}.npz'
        if path.exists():return None
        start=time.time()
        try:
            x=load_physical_aiavdataset(clip,t0_us=args.t0_us)
            assert list(x['camera_indices'])==[0,1,2,6]
            images=x['image_frames'].permute(0,1,3,4,2).numpy()
            history=x['ego_history_xyz'][0,0].numpy()
            np.savez_compressed(path,image_frames=images,ego_history_xyz=history,
                ego_history_rot=x['ego_history_rot'][0,0].numpy(),ego_future_xyz=x['ego_future_xyz'][0,0].numpy(),
                ego_future_rot=x['ego_future_rot'][0,0].numpy(),relative_timestamps=x['relative_timestamps'].numpy())
            speed=float(np.linalg.norm(history[-1,:2]-history[-2,:2])/.1)
            record={'context':position,'name':path.stem,'clip_id':clip,'chunk':int(index.loc[clip,'chunk']),
                'speed_mps':speed,'t0_us':args.t0_us,**{k:(v.item() if hasattr(v,'item') else v) for k,v in meta.loc[clip].items()},
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seconds':time.time()-start}
        except Exception as error:
            record={'context':position,'clip_id':clip,'error':repr(error),'trace':traceback.format_exc()[-800:]}
        print(json.dumps({k:record[k] for k in record if k!='trace'}),flush=True);return record
    with ThreadPoolExecutor(args.workers) as pool:records=[r for r in pool.map(fetch,enumerate(clips)) if r]
    previous=json.loads((args.out/'records.json').read_text()) if (args.out/'records.json').exists() else []
    records=sorted(previous+records,key=lambda r:r['context'])
    (args.out/'records.json').write_text(json.dumps(records,indent=2))
    (args.out/'manifest.json').write_text(json.dumps({'scope':__doc__,'split':args.split,'seed':args.seed,'requested':args.count,'excluded_chunks':len(used),'aligned_split':str(args.aligned_split) if args.aligned_split else None,
        'contexts':sum('error' not in r for r in records),'t0_us':args.t0_us,'camera_indices':[0,1,2,6],
        'image_layout':'[camera,time,height,width,RGB]','source_sha256':hashlib.sha256(source).hexdigest(),
        'limitations':['Logged future is one human trajectory, not the unique correct action',
            'Split membership from clip_index.parquet; Alpamayo training exposure to test clips is not independently verified',
            'One t0 per clip']},indent=2))


if __name__=='__main__':main()
