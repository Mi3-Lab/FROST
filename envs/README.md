# Environments

The analyses, table generators and tests need only the `frost` package with the `analysis` extra
(`pip install -e ".[analysis]"`, Python >= 3.10). Each model probe ran in its own isolated environment because the
upstream models pin incompatible versions of torch, transformers and numpy; the files here are frozen from those
environments (editable installs of local upstream checkouts removed).

| File | Python | Used by |
|---|---|---|
| `alpamayo.txt` | 3.12 | Alpamayo-1.5/R1/2 probes, PhysicalAI-AV and WOD-E2E data preparation |
| `qwendrive.txt` | 3.12 | Qwen-Drive probes, SimWAM ego inputs |
| `flashdrive.txt` | 3.12 | FlashDrive probe |
| `simwam.txt` | 3.11 | SimWAM probe |
| `cosmos.txt` | 3.12 | Cosmos 3 Nano probe |
| `navsim_recogdrive.txt` | 3.10 | ReCogDrive NAVSIM probe |
| `navsim_diffusiondrive.txt` | 3.9 | DiffusionDrive NAVSIM probe, metric cache, PDM scoring, NAVSIM analysis |
| `robot.txt` | 3.12 | SmolVLA / LIBERO |

AlpaSim is run through its own `uv` project (`uv sync` in `$ALPASIM_ROOT`, see its README); the closed-loop launcher
itself needs only the standard library. Thor software versions are listed in `deploy/thor/README.md`.

Point the launchers at the interpreters with `PY_ALPAMAYO`, `PY_QWENDRIVE`, ... in `configs/paths.env`.
