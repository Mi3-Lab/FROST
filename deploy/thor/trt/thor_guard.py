"""Board-sharing guard for the TensorRT prototype scripts (Thor only).

/proc is mounted hidepid=invisible on the Thor: a process started from one ssh session cannot see the bash of the
registered runner (run_thor_v1_next.sh), only some of its python children. So "runner finished" is read from the
runner's own log instead of from /proc: GPU work starts only after the last line of out/thor_v1/run.log says
'run_thor_v1 finished', no known foreign GPU job is visible and GR3D load is <= 10 % over 5 s. Polls every 5 min.
Never touches any process. Bypass (e.g. when the user frees the board explicitly): FROST_TRT_NO_GUARD=1.
"""
import os, re, subprocess, sys, time
from datetime import datetime, timezone

RUNLOG = os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/out/thor_v1/run.log'
FOREIGN = 'llm_stream_video|joint_filter|trtexec|llm_inference|llm_build|thor_quant_probe|thor_latency|temperature_probe'


def _state():
    try:
        last = [l for l in open(RUNLOG).read().splitlines() if l.strip()][-1]
    except (OSError, IndexError): last = ''
    # also free when the user paused the runner for us (its log says so) or released the board explicitly
    finished = ('run_thor_v1 finished' in last or 'paused by user request' in last
                or os.path.exists(os.environ.get('FROST_THOR_ROOT', 'thor_work') + '/trt/BOARD_RELEASED'))
    foreign = subprocess.run(['pgrep', '-f', FOREIGN], capture_output=True, text=True).stdout.split()
    foreign = [p for p in foreign if int(p) != os.getpid()]
    try:
        out = subprocess.run(['timeout', '6', 'tegrastats', '--interval', '1000'], capture_output=True, text=True).stdout
        loads = [int(x) for x in re.findall(r'GR3D_FREQ (\d+)', out)]; load = sum(loads) / len(loads) if loads else 100
    except Exception: load = 100
    return finished, foreign, load


def wait_for_free_board(tag=''):
    if os.environ.get('FROST_TRT_NO_GUARD') == '1': return
    while True:
        finished, foreign, load = _state()
        stamp = datetime.now(timezone.utc).strftime('%FT%TZ')
        print(f'[{stamp}] guard {tag}: runner_finished={finished} foreign={foreign or "none"} gr3d={load:.0f}%', flush=True, file=sys.stderr)
        if finished and not foreign and load <= 10: return
        time.sleep(300)
