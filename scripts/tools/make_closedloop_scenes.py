"""Closed-loop scene sample: scenes of one AlpaSim test suite whose PhysicalAI-AV clip is in the dataset's test split,
shuffled with a fixed seed. Reproduces artifacts/alpasim/closedloop_scenes_v1.json (suite public_2604, seed 20261026, 60)."""
import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd

from frost.paths import PHYSICALAI_AV, env_path, hf_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', default='public_2604');p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--n', type=int, default=60);p.add_argument('--alpasim', type=Path, required=True, help='AlpaSim checkout (reads its scene suite table)')
    p.add_argument('--out', type=Path, required=True);a = p.parse_args()
    index = pd.read_parquet(env_path('FROST_PHYSICALAI_INDEX', hf_snapshot(*PHYSICALAI_AV, kind='datasets')) / 'clip_index.parquet')
    index = index.reset_index()
    clip_split = dict(zip(index.clip_id, index.split))
    rows = [r for r in csv.DictReader(open(Path(a.alpasim, 'data', 'scenes', 'sim_suites.csv'))) if r['test_suite_id'] == a.suite]
    # AlpaSim scene ids are 'clipgt-<PhysicalAI clip id>'.
    split = {r['scene_id']: clip_split.get(r['scene_id'].removeprefix('clipgt-')) for r in rows}
    test = sorted(s for s, v in split.items() if v == 'test')
    random.Random(a.seed).shuffle(test)
    chosen = test[:a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'suite': a.suite, 'filter': 'PhysicalAI clip_index split == test', 'seed': a.seed, 'scene_ids': chosen}, open(a.out, 'w'))
    print(len(chosen), 'scenes ->', a.out)


if __name__ == '__main__':
    main()
