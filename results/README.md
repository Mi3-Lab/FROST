# Released results

Small, scrubbed copies of every stored analysis output that the table and figure generators in `reproduce/` read
(total about 12 MB). Each directory keeps the name of the artifact it was copied from (study, version, date), so it can
be matched to the commands in `scripts/` and the provenance notes in `docs/PROVENANCE.md`.

What was changed relative to the original artifacts:

- Local paths, host-specific locations and internal document references in string fields were rewritten
  (`<FROST_THOR_ROOT>/...`, `protocols/<name>.md`, `frost.*` module names). No numeric field was changed; the
  end-to-end test regenerates all 36 table/macro files byte-identically from these files.
- The Thor addendum-3 analysis directory and one of its keys were given English names
  (`thor_addendum3_analysis_v1_20261101`, key `addendum2_reproduced`).
- `paper_numbers_heldout_pooled400_v2_20261028.macros.txt` is the `--macros` output of the pooled analysis
  (`.macros.tex` in the original run), renamed so that the repository contains no TeX files.
- `schedule_eval_physicalai_v2_20261022/rows.json` keeps only the rows (methods `uniform`, `rescaled_entropic`,
  `decision_info_plus_uniform`) and fields that the schedule-confirmation macros use; row order is unchanged.
- `robot_smoke_20261022/contraction.json` caches the SmolVLA contraction statistics computed from the 16 MB
  `probe.npz` (the generator recomputes them when `$FROST_ARTIFACTS/robot_smoke_20261022/probe.npz` exists).
- `closedloop_exploratory.json` caches the exploratory closed-loop McNemar tests computed from the AlpaSim rollouts
  (`reproduce/tables/make_closedloop_table.py --recompute`).
- `figure_cache/` holds the per-clip metrics (spread, ADE1, minADE6, flip share; float64) of the 400 held-out clips
  for Alpamayo-1.5/R1/2 and the six-seed plans of the page-one example clip, written by
  `reproduce/figures/build_figure_cache.py` from the raw plans. The figure scripts use the raw plans instead when
  `$FROST_ARTIFACTS` contains them.

Not included (large): per-clip plans (`plans.npz`, about 0.2 GB for the headline study), context caches
(about 17 GB of camera frames from PhysicalAI-AV), NAVSIM/WOD-E2E per-scene plans, AlpaSim rollouts, Thor engines and
LIBERO videos. See the README section "Large artifacts".
