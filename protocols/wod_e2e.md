# WOD-E2E Protocol v1: Alpamayo-1.5 on the Long Tail, with the Rater Score (RFS)

**Summary.** This protocol governs the test of whether the FROST effect ("few steps = lower temperature") holds on long-tail scenarios of the Waymo Open Dataset end-to-end driving benchmark (WOD-E2E, validation split) when the metric is the human rater score (an RFS-proxy reimplemented from the benchmark paper), and not only ADE against the log. It fixes the data, the camera adaptation, the RFS-proxy definition, the samplers, the tau_eff analysis, the equivalence margins and the predictions W1-W5 before any Alpamayo output on these data was generated or opened.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

**REGISTERED on 29/10/2026 ~01h30 (PDT), approved by the authors.** (See the correction of times at the end: the valid time is that of the commit, 28/10 23:01.) No Alpamayo output on the WOD-E2E data was generated or opened up to here; there was only a conversion of 3 records on CPU, to test the code. The full `scan` (CPU-only job) found 479 segments with a rater score in 106,360 frames. Since N > 300, the primary analysis uses the first 280 in name order, as foreseen in Section 3.

**Question (item 4 of the schedule).** Does the "few steps = lower temperature" effect (abstract of the paper; addenda 13-16 of the master protocol) hold in long-tail scenarios and when the metric is the rater score (RFS), and not only ADE against the log?

## 1. Data and what RFS is
- **Data:** WOD-E2E, *validation* split, 93 shards on the workstation's data disk. Each record is a frame (`<segment>-<idx>`), shuffled across shards. Only ~1 frame per segment has 3 rater trajectories (score 0-10; -1 = no label). In 3 shards: 9 labeled frames out of 3439.
- **RFS:** the WOD-E2E RFS module, a reimplementation from the paper (Xu et al., arXiv 2510.26125, Sec. 3.5). The official code is not public. We follow: rectangular trust region at t = 3 and 5 s (lateral 1.0 and 1.8 m; longitudinal 4x); thresholds x scale(v) (0.5 if v < 1.4 m/s; linear up to 1 at 11 m/s); score per rater = s*0.1^max(max(delta_lng/tau_lng, delta_lat/tau_lat) - 1, 0); maximum over raters; mean of 3 s and 5 s; floor 4.
- **Choices the paper does not fix (declared):** (i) delta_lng/delta_lat measured along/across the heading of the rater trajectory at time t; (ii) v = rater speed at the first waypoint; (iii) 10 Hz prediction interpolated at t = 3 and 5 s. Sensitivity: v = ego speed at t = 0; thresholds x 0.5 and x 2. **In the paper the number is called "RFS-proxy"**, not comparable to the leaderboard.
- Test: a unit-test file for the RFS module (synthetic: inside the region = rater score; exponential decay; floor; maximum across raters; rotation invariance). On the 3 real records, each rater trajectory used as a prediction receives its own score (10, 8, 6.5 etc.).

## 2. Camera and history adaptation (the WOD-E2E contexts module)
- Alpamayo cameras [cross_left, front_wide, cross_right, front_tele] <- WOD [FRONT_LEFT, FRONT, FRONT_RIGHT, central 2x crop of FRONT]. The WOD cameras have ~47 deg HFOV (Alpamayo expects 120 deg) and 972x1079 images: **the model is out of the camera domain**; comparisons hold between samplers on the same context, not at an absolute level.
- 4 frames = idx-3..idx of the same segment (index from `scan` over all shards). The 0.1 s spacing cannot be verified (`timestamp_micros` = 0); it is an assumption. Segments without the 4 frames leave the sample and are listed.
- Ego history: `past_states` (4 Hz) -> 16 poses at 10 Hz (PCHIP); heading = tangent (0 if v < 0.5 m/s); last pose = identity. No navigation text (`intent` is kept).
- Log future and rater trajectories stay in the npz (`wod_future_xyz`, `rater_xy`, `rater_scores`). Raters have 21 waypoints (first at t = 0.25 s, checked on 3 records).

## 3. Sample
- **All** labeled frames of validation (expected ~280-480, one per segment; the exact number comes from `scan`) with the 10 categories of the scenario-cluster mapping file (Intersections [sic: spelled "Interections" in the file], Foreign Object Debris, Cyclist, Pedestrian, Multi-Lane, Single-Lane, Special Vehicles, Others, Cut-ins, Construction). No draw and no selection by result.
- Independent unit = **segment** (one labeled frame per segment). Bootstrap by segment, also stratified by category in the secondary analysis.
- Execution order: segment name order (as in addenda 13-14), in case the GPU runs out. Primary analysis on the first 280 (or on all, if N <= 300).

## 4. Samplers and run
Model: Alpamayo-1.5 (the temperature probe, with `--configs`), seeds 9201-9206, same initial noise across configs. Configs:
- E10, E4, E1 (Euler, tau = 1);
- Heun-32 at tau in {1; 0.9; 0.8; 0.7; 0.6; 0.5; 0.4; 0.3; 0.2; 0.1} (the extended grid of addendum 16 already in the first run).
Optional, only if GPU time is left: Heun-3/5 (controls of addendum 15).

## 5. Metrics (per segment: mean of the 6 seeds, then bootstrap)
1. **RFS-proxy** (primary): 5 s, mean of 3 and 5 s, Alpamayo 10 Hz plan.
2. **ADE_1** against the log future at 5 s at 4 Hz (as in the paper). Exploratory secondary: ADE to the best rater trajectory.
3. **Flips:** change of decision class (`decisions` function of the trajectory-decision-gate module, same definition as addenda 13-16, on the full 6.4 s plan) between config and reference (Heun-32 at tau_eff), seed by seed.
4. Reference levels for the paper: RFS of the log future, of the best rater and of the Euler-10 tau = 1 plan.

## 6. Analysis
- **tau_eff** by matching the across-seed spread (as in the paper-numbers script), tau curve averaged per segment; bootstrap by segment (1000 resamples) with tau_eff recomputed in each.
- **Residual** = metric(sampler) - metric(Heun-32 at the sampler's tau_eff). 90% TOST.
- **Margins:** RFS +-0.08 (smallest gain the paper attributes to a component: 7.14 -> 7.22 -> 7.30 -> 7.39 in its Tab. 3); ADE_1 +-0.05 m and flips +-2 pt (equal to those of v1; sensitivity ADE_1 +-0.10 m, since the WOD horizon and scale are larger). The margins are fixed before any output.
- Raw effect: RFS(sampler) - RFS(Heun-32, tau = 1), same bootstrap.
- Exploratory, no hypothesis test: effect per category (n per category reported, 15-116 segments); RFS as a function of tau; sensitivity of RFS to the thresholds (x 0.5, x 2) and to the ego v.

## 7. Predictions (from addenda 13-16; any failure is reported)
- **W1 (primary):** the RFS residual of E10 and of E4 passes the TOST of +-0.08 (90% CI).
- **W2:** tau_eff of E10 in 0.75-0.92, of E4 in 0.50-0.75, of E1 <= 0.45 (outside the PhysicalAI pattern because of the domain shift, the interval is wider than that of addendum 13).
- **W3:** ADE_1 residual of E10 and of E4 passes the TOST of +-0.05 m.
- **W4:** flip residual of E10 passes at +-2 pt; that of E4 and E1 may fail (extrapolation of addendum 16).
- **W5 (direction of the metric):** RFS of Heun-32 at tau = 0.4 is >= 0.05 above that of tau = 1 (less temperature brings the plan closer to the safe mode). If there is no difference, "low temperature helps the score" does not hold.
- **Failure criteria:** a variation of RFS across configs < 0.1 makes the metric uninformative (report it so). If the per-segment CI of the raw effect includes 0, do not claim an effect. If the tau_eff of E10 leaves the grid (< 0.1), the result is inconclusive.

## 8. Limitations (for the paper)
RFS-proxy, not official; cameras out of domain (FOV 47 deg against 120 deg, cross cameras approximated by front-lateral); assumed frame spacing; one labeled frame per segment; raters only up to 5 s (Alpamayo plans 6.4 s); possible exposure of Alpamayo to Waymo data is unknown.

## 9. Execution and cost
1. **CPU (cluster):** the contexts module's `scan` command over shards 0..92, writing the full frame index (reads ~226 GB; in the test, 4 s per shard; ~6-7 min) -> labeled frames + frame index. It was not run on the login node; it should go as a CPU job.
2. `convert` with the labeled index and the history index (~4 s per context; ~12 MB per npz, ~5 GB in total). Produces `records.json` with category and `n_real_history_frames`.
3. **GPU:** the temperature probe with `--configs ...` on 1 H200 (or 4 shards). Estimate **~3-6 GPU-hours** for ~280-480 segments x 6 seeds x 13 configs (not measured: extrapolated from the cost of Heun-32 x 10 tau, ~4.5k expert evaluations per segment, plus the VLM prefill with 16 images). A pilot of 10 segments fixes the estimate before the full run.

## Addendum 1 (29/10, before any Alpamayo output on the WOD-E2E data)

- **Bug.** `convert` failed on 65 of the 479 rated frames. In all of them, a rater trajectory was shorter (for example 12 points, i.e. 3 s, against 21 points, 5 s) and could not be stacked with the others. Excluding those frames would be selection by bug, so we fixed it.
- **Converter:** rater trajectories are padded with NaN to 21 points.
- **RFS:**
  - a rater whose trajectory ends before t does not score at t (no extrapolation of the last point);
  - the score of each horizon is the maximum across the raters that cover it;
  - the frame score is the mean of the horizons covered by some rater, with floor 4.
  - New unit test in the RFS unit-test file.
- **Execution:** the conversion is redone for the 479 frames (CPU job) before the GPU run. The rest of the protocol does not change.


**Correction of times (note of 28/10, 23:10 PDT by the machine clock).** The times written by hand in the addenda of the evening of 28/10 were ahead by between 1 h and 2.5 h. The valid times are those of the commits (git, PDT):
- addendum 16 of the master protocol: commit recorded in the private repository, 28/10 22:29;
- addendum 17 of the master protocol: commit recorded in the private repository, 22:49, with the result in a later commit, 22:50;
- note to addendum 16: commit recorded in the private repository, 23:03;
- registration of WOD-E2E: commit recorded in the private repository, 28/10 23:01 (not 29/10 01h30);
- addendum 1 of WOD: commit recorded in the private repository.

The order registration -> result does not change.
