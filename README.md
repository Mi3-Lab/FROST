# FROST: matched-temperature tests for few-step flow samplers in driving policies

Code, released results and reproduction scripts for the paper *FROST* (anonymous submission).

Flow-matching driving policies are often run with fewer integration steps than their reference sampler. Fewer
Euler steps contract the sampled plans toward the conditional mean. FROST asks how much of a sampler's change in
driving metrics is **only** that contraction. For each sampler it estimates an effective temperature
τ<sub>eff</sub>: the noise scale at which the accurate reference sampler, started from τ·z, has the same
across-sample spread. It then tests, by clip bootstrap and two one-sided tests (TOST), whether the sampler's error
(ADE<sub>1</sub>), decision changes (flips of trajectory-level decision classes) and task scores equal those of the
reference at τ<sub>eff</sub> within pre-registered margins. If they do, the reduced-step sampler is equivalent to a
cooler version of the reference on these observables. If not, it does something else.

The repository covers every study in the paper:

- the headline test on Alpamayo-1.5, Alpamayo-R1 and Alpamayo-2 (400 held-out PhysicalAI-AV clips);
- controls: Heun samplers, low temperatures, scaled noise, convergence of the reference, and a Gaussian contraction
  model;
- other policies and benchmarks: Qwen-Drive-1.0, FlashDrive, SimWAM, Cosmos 3 Nano, NAVSIM (ReCogDrive,
  DiffusionDrive), WOD-E2E with the official rater feedback score, and SmolVLA on LIBERO;
- closed loop in AlpaSim;
- deployment on NVIDIA DRIVE AGX Thor.

All studies were pre-registered. The English protocols are in [`protocols/`](protocols/README.md).

## Repository layout

```
frost/            Python package
  core/           decision classes, matched-temperature statistics, step schedules
  data/           context caches (PhysicalAI-AV, WOD-E2E, NAVSIM scenes), probe-shard merging, split overlap
  probes/         model probes (one per policy family; need the model's own environment and a GPU)
  analysis/       per-study analyses (CPU): tau_eff, residuals, bootstrap, TOST
results/          released summaries of every analysis the paper reports (about 12 MB)
reproduce/        generators of every table, macro file and figure in the paper (read only results/)
scripts/          one launcher per study (00-18) with the exact arguments used; Slurm templates; helper tools
deploy/thor/      DRIVE AGX Thor scripts (TensorRT engines, compiled VLM, latency) and guide
patches/alpasim/  opt-in sampler hook for the AlpaSim Alpamayo-1.5 driver
protocols/        pre-registration protocols with all addenda
configs/          environment-variable template for paths and interpreters
envs/             frozen requirements of each environment
docs/             provenance of the released results
tests/            unit tests and the end-to-end reproduction test
```

## Installation

Python ≥ 3.10. Tables, analyses and tests:

```bash
pip install -e ".[analysis,figures]"      # or: uv pip install -r envs/analysis.txt && pip install -e .
python -m pytest                          # about 10 s (figure test skipped without matplotlib)
```

The model probes need GPUs and model-specific environments (`envs/*.txt`, one per policy family). They are not
needed to regenerate the paper's tables and figures.

## Quick start: regenerate every table and figure from `results/` (CPU, under a minute)

```bash
python reproduce/make_all.py --check
# ok reproduce/tables/make_tables.py ... ok reproduce/figures/make_fig_wod.py
# 36/36 tables byte-identical to the paper
```

The outputs go to `reproduce/output/tables/*.tex` (LaTeX tables and `\newcommand` macro files) and
`reproduce/output/figures/*.pdf`. `--check` compares every generated table with the SHA-256 digest of the file the
paper was built from (`tests/golden/tables_sha256.json`). `tests/test_reproduce.py` runs the same check under
pytest. Several outputs have descriptive names in this release. The names used by the paper sources are:

| Release name | Paper name | Content |
|---|---|---|
| `numbers_controls.tex` | `numbers_v533.tex` | A2 Blackwell sensitivity, extension sample, scaled-noise control, minADE6 support |
| `numbers_reference_floor.tex` | `numbers_v542.tex` | two independent τ=1 references, original-sample A2 flips, schedule confirmation |
| `numbers_wod.tex` | `numbers_v543.tex` | WOD-E2E primary sample |
| `numbers_baselines.tex` | `numbers_v61.tex` | mean-of-reference baseline, DiffusionDrive decisions, shrinkage separation |
| `numbers_wod_secondary.tex` | `numbers_v62.tex` | WOD-E2E all 479 rated frames; FlashDrive released stack |
| `numbers_wod_official.tex` | `numbers_v63.tex` | official rater feedback score agreement; flip agreement |
| `numbers_crossmodel.tex` | `numbers_v7.tex` | Qwen-Drive v3, convergence extension, Thor addendum 3 |
| `tab_navsim_merged.tex` | `tab_navsim_v60.tex` | NAVSIM appendix table |
| `tab_onetwo.tex` | `tab_onetwo_v60.tex` | one- and two-step Euler residuals |

The figures (teaser, matched-temperature schematic, mechanism, curves, verdict atlas and summary, WOD-E2E) are
recomputed from the same summaries. Per-clip metrics come from `results/figure_cache/`, or from the raw plans when
`$FROST_ARTIFACTS` holds them. See `docs/PROVENANCE.md` for the byte-level comparison of the figures.

## Models and data

All models and datasets are public. Several are gated and need their terms to be accepted on Hugging Face or with the
data provider. Authenticate with `huggingface-cli login` (or `export HF_TOKEN=...`). `THIRD_PARTY.md` lists the exact
revisions, commits and licenses.

| Policy / data | Source | Used in |
|---|---|---|
| Alpamayo-1.5, Alpamayo-R1, Alpamayo-2 | `nvidia/Alpamayo-1.5-10B`, `nvidia/Alpamayo-R1-10B`, `nvidia/Alpamayo2-Super` | studies 1-6, 11-16 |
| Qwen-Drive-1.0 | `Qwen/Qwen-Drive-1.0-4B` + QwenLM/Qwen-Drive-1.0 | 5, 9, 12 |
| FlashDrive | `z-lab/Alpamayo-1.5-10B` + z-lab/flashdrive | 10 |
| SimWAM, Cosmos 3 Nano | `H-EmbodVis/SimWAM`, `nvidia/Cosmos3-Nano` | 5 |
| ReCogDrive, DiffusionDrive | upstream checkpoints (NAVSIM) | 8 |
| SmolVLA | `HuggingFaceVLA/smolvla_libero` | 17 |
| PhysicalAI-AV | `nvidia/PhysicalAI-Autonomous-Vehicles` (Alpamayo-aligned test split) | 1-6, 9-16 |
| WOD-E2E | Waymo Open Dataset, end-to-end driving validation set with rater scores | 7 |
| NAVSIM navtest | OpenScene / nuPlan | 8 |
| AlpaSim | NVlabs/alpasim public scene suite | 11 |
| LIBERO | via LeRobot | 17 |

## Full re-run

1. `cp configs/paths.example.env configs/paths.env` and edit it. It holds data and artifact roots, upstream
   checkouts, checkpoints, and one Python interpreter per environment (`PY_ALPAMAYO`, `PY_QWENDRIVE`, ...). Clone the
   upstream repositories at the commits in `THIRD_PARTY.md`.
2. Run the launchers in order. Each one is resumable and split into stages, so `STAGE=probe|merge|analysis|data`
   runs a single part. Steps that ran on a cluster are printed as `sbatch` commands; they use the generic templates in
   `scripts/slurm/` (fill in `<ACCOUNT>`, `<PARTITION>`, `<GPU>`).
3. Copy the analysis summaries you want to compare into a results directory and point `FROST_RESULTS` at it. Then
   `python reproduce/make_all.py`.

| Script | Study | Main outputs (artifact names as in `results/`) | Compute (original) |
|---|---|---|---|
| `00_data.sh` | context caches: PhysicalAI-AV test samples and 400 held-out clips, splits, WOD-E2E, SimWAM ego inputs | `physicalai_*_contexts_*`, `wod_e2e_val_contexts_v2_20261029` | CPU / 1 GPU node, hours; ~17 GB |
| `01_headline.sh` | matched temperature, Alpamayo-1.5/R1/2 | `temperature_probe_*`, `paper_numbers_heldout_pooled400_v2_20261028`, `paper_numbers_v4_20261026` | 1 GPU ≥ 80 GB (A2), ~2-3 GPU-days |
| `02_heun_controls.sh` | Heun-3/5 controls | `heun_controls_analysis_*` | 1-2 GPUs, ~1 day |
| `03_lowtau.sh` | τ = 0.3/0.2/0.1 reference, one/two-step Euler | `lowtau_*` | 1-2 GPUs, ~1 day |
| `04_scaled_noise.sh` | scaled-noise control | `scaled_noise_control_a15_v1_20261029` | CPU (A1.5); cluster for R1/A2 |
| `05_reference_convergence.sh` | Heun-64/128, RK45; Qwen-Drive, SimWAM, Cosmos 3 Nano | `convergence_*` | 1 GPU, ~2 days; cluster jobs |
| `06_gaussian_contraction.sh` | Gaussian contraction prediction | `gaussian_contraction_v1_20261026` | 1 GPU, hours |
| `07_wod_e2e.sh` | WOD-E2E long tail, RFS proxy and official RFS | `wod_e2e_analysis_*` | 1 GPU, ~1 day |
| `08_navsim.sh` | NAVSIM: ReCogDrive IL/RL, DiffusionDrive, PDM score | `navsim_analysis_*` | 1 GPU + 16-22 CPU workers |
| `09_qwen_drive.sh` | Qwen-Drive v2 and v3 (+ Heun, E6) | `qwendrive_*` | 2 GPUs (cluster), ~2 days |
| `10_flashdrive.sh` | FlashDrive velocity caching | `flashdrive_*` | 1 GPU, hours |
| `11_alpasim_closed_loop.sh` | AlpaSim closed loop (58 scenes × 5 configurations) | `closedloop_*` | 1 GPU + Docker, ~1 day |
| `12_thor.sh` | DRIVE AGX Thor (device steps in `deploy/thor/README.md`) | `thor_*` | DRIVE AGX Thor |
| `13_robustness.sh` | exploratory robustness of the verdicts | `robustness_v1_20261102` | CPU, ~1 h |
| `14_flip_agreement.sh` | exploratory flip agreement | `flip_agreement_v1_20261030` | CPU |
| `15_split_overlap.sh` | sample vs. Alpamayo training split | `split_overlap_*` | CPU |
| `16_toy_and_schedules.sh` | toy mode-collapse model; equal-NFE schedules | `mode_collapse_theory_*`, `schedule_*` | 1 GPU, ~1 day |
| `17_smolvla_libero.sh` | SmolVLA / LIBERO (manipulation) | `robot_grid_smolvla_v1_20261022` | 1 GPU, ~1 day |
| `18_reproduce_tables_figures.sh` | all tables and figures from `results/` | `reproduce/output/` | CPU, < 1 min |

The analyses (`STAGE=analysis`) need only the `analysis` extra and the probe outputs. They are deterministic: fixed
seeds and 2000 clip-bootstrap resamples, 10 000 for WOD-E2E. Re-running them on the original probe outputs
reproduces the released summaries. `docs/PROVENANCE.md` records the few artifacts written by an earlier,
backward-compatible version of a module.

## Large artifacts

The released summaries are enough to check every reported number and to regenerate all tables and figures. The
per-clip plans of the probes (about 0.1 GB for the headline study, about 1 GB in total), the context caches (camera
frames re-derived from the gated datasets) and the AlpaSim rollouts are not in this repository. Plans will be
released on Zenodo with the camera-ready version; until then they are available on request. Context caches cannot be
redistributed: `scripts/00_data.sh` rebuilds them from the original datasets.

## Hardware notes

- Probes ran on an NVIDIA RTX PRO 6000 Blackwell workstation (96 GB) and on cluster nodes with H200 NVL, A100 or
  Blackwell GPUs. Alpamayo-2 needs about 72 GB of GPU memory.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` and fixed seeds (language 123, action noise 9201-9206) make plans repeatable on
  the same GPU and software stack. Across hardware, plans differ slightly; the paper quantifies this (reference floor,
  Blackwell-only sensitivity, Thor hardware check).
- Thor latencies are those of the action expert (flow sampler), not of the full VLA, unless stated otherwise.

## Citation

```bibtex
@misc{frost2026,
  title  = {FROST: Matched-Temperature Tests for Few-Step Flow Samplers in Driving Policies},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT (`LICENSE`). Third-party code, models and data keep their own licenses and terms (`THIRD_PARTY.md`).
