"""Merge temperature-probe shards (same code, disjoint clip ranges) into the main probe directory.

The main directory's own plans are kept as plans_before_merge.npz; the merged manifest lists every shard.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True);p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--range',type=int,nargs=2,required=True)
    p.add_argument('--allow-source',nargs='*',default=[],help='other source hashes accepted after checking the diff (recorded in the manifest)');args=p.parse_args()
    with np.load(args.main/'plans.npz') as z:plans={k:z[k] for k in z.files}
    (args.main/'plans_before_merge.npz').write_bytes((args.main/'plans.npz').read_bytes())
    manifest=json.loads((args.main/'manifest.json').read_text());sources={'main':str(args.main)}
    for shard in args.shards:
        sm=json.loads((shard/'manifest.json').read_text())
        assert sm['source_sha256'] in [manifest['source_sha256'],*args.allow_source] and sm['seeds']==manifest['seeds'],shard
        with np.load(shard/'plans.npz') as z:
            for k in z.files:
                assert k not in plans,k
                plans[k]=z[k]
        sources[str(shard)]=sm['range']
    clips=sorted({k.split('_s')[0] for k in plans});expected=[f'context_{i:03d}' for i in range(*args.range)]
    assert clips==expected,(len(clips),set(expected)-set(clips))
    np.savez_compressed(args.main/'plans.npz',**plans)
    manifest.update(range=args.range,shards=sources,**({'allowed_source_sha256':args.allow_source} if args.allow_source else {}));(args.main/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'clips':len(clips),'plans':len(plans),'shards':sources}))


if __name__=='__main__':main()
