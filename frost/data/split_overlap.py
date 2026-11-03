"""Count how many clips of each context sample fall in each partition of the Alpamayo-aligned PhysicalAI-AV split."""
import argparse
import collections
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sample',nargs=4,action='append',metavar=('NAME','RECORDS','START','STOP'),required=True)
    args=p.parse_args();split=pd.read_parquet(args.split).reset_index();column='split'
    partition=dict(zip(split['clip_id'],split[column]));result={'split':str(args.split),'column':column,'samples':{}}
    for name,records,start,stop in args.sample:
        names=sorted((r for r in json.loads(Path(records).read_text()) if 'error' not in r),key=lambda r:r['name'])[int(start):int(stop)]
        result['samples'][name]={'records':records,'range':[int(start),int(stop)],'clips':len(names),
            'partition':dict(collections.Counter(str(partition.get(r['clip_id'],'absent')) for r in names))}
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'summary.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result['samples'],indent=1))


if __name__=='__main__':main()
