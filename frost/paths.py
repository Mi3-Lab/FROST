"""Locations of external code, checkpoints and data, read from environment variables.

Nothing machine-specific is hard-coded: every external location is an environment variable documented in
configs/paths.example.env. Hugging Face snapshots default to the standard cache layout under $HF_HOME.
"""
import os
from pathlib import Path

# Pinned revisions of the public checkpoints and datasets used in the paper.
ALPAMAYO_15 = ('nvidia/Alpamayo-1.5-10B', '7aba8293c09993f2e125c6819df05d7fa3e873ea')
ALPAMAYO_R1 = ('nvidia/Alpamayo-R1-10B', 'd72c6a2757b10cacb69b688ca66ccfad3dafbf37')
PHYSICALAI_AV = ('nvidia/PhysicalAI-Autonomous-Vehicles', 'dfcc35f941c38f050e9ce256a4c0aff9e33615b9')


def hf_home():
    return Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))


def hf_snapshot(repo, revision, kind='models'):
    """Path of a snapshot in the standard Hugging Face hub cache (download it first, e.g. with
    `huggingface-cli download <repo> --revision <revision>`)."""
    return hf_home() / 'hub' / f"{kind}--{repo.replace('/', '--')}" / 'snapshots' / revision


def env_path(name, default=None):
    """Path from an environment variable; raises a clear error if it is unset and has no default."""
    value = os.environ.get(name)
    if value:
        return Path(value)
    if default is not None:
        return Path(default)
    raise RuntimeError(f'Set the environment variable {name} (see configs/paths.example.env).')
