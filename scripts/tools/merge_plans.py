"""Union of the plans.npz of several runs (later runs overwrite earlier keys), as used to merge the three restarts of
the matched schedule evaluation."""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out', type=Path, required=True)
    p.add_argument('dirs', type=Path, nargs='+');a = p.parse_args()
    plans = {}
    for d in a.dirs:
        with np.load(d / 'plans.npz') as z:
            plans.update({k: z[k] for k in z.files})
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / 'plans.npz', **plans)
    print(len(plans), 'plans ->', a.out)


if __name__ == '__main__':
    main()
