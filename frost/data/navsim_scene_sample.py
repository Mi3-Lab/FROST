"""Draw the pre-registered 500 NAVSIM navtest scenes (protocols/navsim.md).

Stratified by log, proportional to the number of navtest tokens per log (largest-remainder allocation,
ties broken by log name), tokens drawn without replacement inside each log with numpy RandomState(seed).
Run with a NAVSIM venv, e.g.  source configs/navsim.env; PYTHONPATH=$DIFFUSIONDRIVE_SRC:. \
python -m frost.data.navsim_scene_sample
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('OMEGACONF_MAX_YAML_EXPANDED_NODES', 'none')  # trusted navtest.yaml with a long token list
ARTIFACTS = Path(os.environ.get('FROST_ARTIFACTS', 'artifacts'))
SEED = 20261028
N = 500


def allocate(counts, n):
    """Largest-remainder proportional allocation; counts: {log: tokens}."""
    logs = sorted(counts);total = sum(counts.values())
    exact = {l: n * counts[l] / total for l in logs}
    alloc = {l: int(np.floor(exact[l])) for l in logs}
    order = sorted(logs, key=lambda l: (-(exact[l] - alloc[l]), l))
    for l in order[:n - sum(alloc.values())]:alloc[l] += 1
    assert sum(alloc.values()) == n and all(alloc[l] <= counts[l] for l in logs)
    return alloc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ARTIFACTS / f'navsim_scenes_v1_{SEED}.json')
    p.add_argument('--seed', type=int, default=SEED);p.add_argument('--n', type=int, default=N)
    p.add_argument('--repo', type=Path, default=Path(os.environ.get('DIFFUSIONDRIVE_SRC', 'third_party/diffusiondrive')), help='NAVSIM repo providing the navtest scene filter')
    args = p.parse_args()
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    cfg = OmegaConf.load(args.repo / 'navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml')
    scene_filter = instantiate(cfg)
    data = Path(os.environ['OPENSCENE_DATA_ROOT'])
    loader = SceneLoader(sensor_blobs_path=data / 'sensor_blobs/test', data_path=data / 'navsim_logs/test', scene_filter=scene_filter)
    per_log = {l: sorted(t) for l, t in loader.get_tokens_list_per_log().items()}
    counts = {l: len(t) for l, t in per_log.items()};alloc = allocate(counts, args.n)
    rng = np.random.RandomState(args.seed);scenes = []
    for log in sorted(per_log):
        if alloc[log] == 0:continue
        pick = sorted(rng.choice(len(per_log[log]), alloc[log], replace=False).tolist())
        scenes += [{'token': per_log[log][i], 'log': log} for i in pick]
    out = {'protocol': 'protocols/navsim.md', 'seed': args.seed, 'n': len(scenes),
        'method': 'largest-remainder proportional allocation per log; RandomState(seed).choice without replacement within each log (sorted logs, sorted tokens); output sorted by log then token index',
        'navtest_tokens': sum(counts.values()), 'navtest_logs': len(counts), 'logs_sampled': len(set(s['log'] for s in scenes)),
        'allocation': alloc, 'scenes': scenes,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True);args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ['n', 'navtest_tokens', 'navtest_logs', 'logs_sampled']}))


if __name__ == '__main__':main()
