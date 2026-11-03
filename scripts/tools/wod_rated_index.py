"""Keep the WOD-E2E frames that have rater scores, sorted by segment name (input of the context conversion)."""
import json
import sys

src, dst = sys.argv[1:3]
rows = [json.loads(line) for line in open(src)]
rated = sorted((r for r in rows if r['n_rated'] > 0), key=lambda r: r['segment'])
with open(dst, 'w') as f:
    f.write(''.join(json.dumps(r) + '\n' for r in rated))
print(f'{len(rated)} rated frames of {len(rows)}')
