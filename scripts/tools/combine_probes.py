"""Union of several probe directories whose plan keys are disjoint (e.g. the pooled probe plus extra configurations
run separately on the same clips, seeds and noise). The first directory's manifest is kept; its config list is
extended and the inputs are recorded under 'combined_from'."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True);p.add_argument('dirs', type=Path, nargs='+');a = p.parse_args()
    plans, configs = {}, set()
    for d in a.dirs:
        with np.load(d / 'plans.npz') as z:
            dup = set(z.files) & set(plans)
            assert not dup, f'{d}: {len(dup)} duplicate keys'
            plans.update({k: z[k] for k in z.files})
        configs |= {tuple(c) for c in json.loads((d / 'manifest.json').read_text())['configs']}
    manifest = json.loads((a.dirs[0] / 'manifest.json').read_text())
    manifest['configs'] = [list(c) for c in sorted(configs)]
    manifest['combined_from'] = [str(d) for d in a.dirs]
    manifest['note'] = 'union of probe directories with disjoint keys'
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'plans.npz', **plans)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f'{len(plans)} plans from {len(a.dirs)} directories -> {a.out}')


if __name__ == '__main__':
    main()
