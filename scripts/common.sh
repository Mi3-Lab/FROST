# Shared helpers for the study launchers. Sourced, not executed.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
ENV_FILE=${FROST_ENV:-$REPO/configs/paths.env}
if [ -f "$ENV_FILE" ]; then source "$ENV_FILE"; else source "$REPO/configs/paths.example.env"; fi
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1
A=$FROST_ARTIFACTS
D=$FROST_DATA
mkdir -p "$A"
# Several analyses default to paths under ./artifacts (as in the original runs); point it at $FROST_ARTIFACTS.
if [ ! -e "$REPO/artifacts" ]; then ln -s "$A" "$REPO/artifacts"; fi

# Run a step only when the selected stage matches (STAGE=all by default; e.g. STAGE=analysis bash scripts/01_*.sh).
STAGE=${STAGE:-all}
stage() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Run one probe command per "START STOP OUT" triple in parallel on the same GPU (as in the original runs), then wait.
# usage: shards "<command with {a} {b} {out}>" "a1 b1 out1" "a2 b2 out2" ...
shards() {
  local cmd=$1; shift
  for spec in "$@"; do
    set -- $spec
    local c=${cmd//\{a\}/$1}; c=${c//\{b\}/$2}; c=${c//\{out\}/$3}
    echo "+ $c"; bash -c "$c" &
  done
  wait
}
