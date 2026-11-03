# Provenance of the released results

Every directory in `results/` is a scrubbed copy of a stored artifact (see `results/README.md` for what was changed).
This note records how the stored artifacts relate to the code in this repository.

## Code version behind each artifact

Each analysis wrote a copy of its own source file next to its output. Comparing those copies with the modules in
`frost/` (after the package rename and the removal of machine-specific paths):

| Artifacts | Stored code vs. this repository |
|---|---|
| all `paper_numbers_heldout_*`, `heun_controls_analysis_*`, `lowtau_*`, `scaled_noise_control_*`, `convergence_a19/a20_*`, `gaussian_contraction_*`, `mode_collapse_theory_*`, `split_overlap*`, `flashdrive_*`, `flip_agreement_*`, `qwendrive_*`, `closedloop_*`, `thor_*`, `robustness_*`, `wod_e2e_analysis_all479_*`, `wod_e2e_analysis_official_rfs_*`, `navsim_analysis_diffusiondrive_v3_*`, `physicalai_heldout_contexts_v3_*` | identical |
| `paper_numbers_v4_20261026` | earlier version without the `--probes` option (added for the held-out runs); same computation with the default models |
| `temperature_probe_v1_20261022`, `temperature_probe_a15_heldout_pooled400_v3_20261028` | earlier version without `--configs` (default grid); same computation |
| `temperature_probe_{r1,a2}_*` (multimodel probe) | earlier version without `--configs` and with fixed checkpoint paths; same computation |
| `physicalai_test_contexts_v1_20261022` | earlier version without `--aligned-split` / `--exclude-records`; same sampling for the defaults |
| `schedule_eval_physicalai_v2_20261022` | earlier version without `--save-plans`; same computation |
| `wod_e2e_analysis_primary_converged10k_v2_20261029` | earlier version with the primary sample hard-coded (280 segments, 4 probe shards); the current defaults `--n 280 --shards 4` reproduce it |
| `navsim_analysis_recogdrive_{rl,il}_addendum2_v1_20261028` | the current version adds two keys (`tau_eff_censored`, `tau_eff_ci95_censored`, introduced by a later audit: out-of-range bootstrap resamples are kept as censored values); every stored key is unchanged |
| `navsim_analysis_recogdrive_rl_v2_20261028` | as above, and without the `recogdrive_lead` family (addendum 2) |
| `navsim_analysis_recogdrive_rl_v1_20261028` | the registered version of the protocol: flips were computed on the raw 8-point plan (0.5 s spacing). Addendum 2 of `protocols/navsim.md` resamples plans to 0.1 s before classifying (`flip`) and keeps the registered value as `flip_registered`. Re-running the current module writes both; the stored v1 `flip` equals the current `flip_registered` |

Stored artifacts without a code copy: `convergence_ext_summary_*` (`frost.analysis.convergence_ext_analysis`),
`lowtau_best_reference_*` (`frost.analysis.lowtau_best_reference`), `schedule_matched_report_v1_20261022`
(`frost.analysis.schedule_matched_report`), `robot_grid_smolvla_v1_20261022` and `robot_smoke_20261022`
(`frost.probes.robot_temperature_grid` / `robot_temperature_eval`), and the device outputs `thor_*` (`deploy/thor`).

## Commands

`scripts/NN_*.sh` contain the command lines that produced each artifact, recovered from the run logs and launch
scripts. They are exact except for the following, which were reconstructed from manifests, argument defaults and the
stored outputs:

- `native_solver_physicalai_v1_20261022` (study 6): the solver list was read from the stored prediction keys.
- `navsim_analysis_recogdrive_rl_v{1,2}_20261028` (study 8): the module version differs (table above).
- `qwendrive_v3_e6_rl_heldout_20261102` (study 9): the configuration string of the Slurm submission.
- `alpasim/closedloop_v1` (study 11): the grid was resumed several times; `--only` was used for individual configurations.
- `thor_trt_bench_20261031`, `thor_vlm_trt_20261031`, `thor_qwendrive_20261101` (study 12): repeated launches on a
  shared device; the arguments of each run are stored in its JSON output.
- `robot_grid_smolvla_v1_20261022` (study 17): a second worker ran the tau = 0.25 Heun configurations in parallel.

Several intermediate steps were short inline scripts in the original runs; they are provided as
`scripts/tools/{wod_rated_index,combine_probes,merge_plans,make_closedloop_scenes}.py`. `make_closedloop_scenes.py`
was checked to reproduce the stored scene list exactly.

## Hardware

Probes ran on one workstation GPU (NVIDIA RTX PRO 6000 Blackwell, 96 GB) or on cluster nodes (H200 NVL, A100,
Blackwell; Slurm templates in `scripts/slurm/`). The Alpamayo-2 held-out probe mixed H200 NVL and Blackwell nodes; the
paper reports a Blackwell-only sensitivity analysis (`paper_numbers_heldout_ext_a2blackwell_v1_20261028`). All runs set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; plans are reproducible on the same GPU type and software stack, and small
cross-hardware differences are quantified in the paper (reference floor, Thor hardware check). Analyses are CPU-only
and deterministic (fixed bootstrap seeds).

## Tables and figures

`reproduce/make_all.py` regenerates all 36 table and macro files; `tests/test_reproduce.py` checks them byte for byte
against the SHA-256 digests of the files the paper was built from (`tests/golden/tables_sha256.json`). The figure
scripts read the same summaries plus `results/figure_cache/`. With the Debian build of matplotlib 3.10.7 and the system
Times/STIX fonts used for the paper, five of the seven figure PDFs are byte-identical; `fig_frost_atlas` and
`fig_frost_summary` render identically but differ at the byte level, also when the original scripts are re-run on the
original artifacts. With the PyPI matplotlib wheel the bundled fonts differ slightly, so the PDFs are visually equal
but not byte-identical.
