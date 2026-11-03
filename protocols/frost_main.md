# FROST Master Protocol: Numerical Error to Command Gate, Multi-Episode, with Addenda 1-22

**Summary.** This is the master pre-registration for the FROST study. It governs whether cheap ODE solvers in flow-matching driving VLAs (Alpamayo-1.5, Alpamayo-R1, Alpamayo-2, Qwen-Drive) change the downstream control command or the decision class read from the planned trajectory, whether the effect of few-step solvers is equivalent to an explicit noise-temperature reduction (matched by across-seed spread, the "effective temperature" tau_eff), and which per-input signals predict decision flips. It starts with a CARLA multi-episode gate and is extended by dated addenda (real PhysicalAI-AV clips, schedules, multi-model replication, held-out clips, converged references, scaled-noise controls, heated samplers). Each addendum states its predictions before the corresponding analysis; failed predictions are kept in the text.

*Translated from the original registration; timestamps are the original registration times. Dates are written DD/MM(/YYYY) as in the original; times are local to the registering machine (PDT/PST as noted where given). Addendum numbering is unchanged. Identifiers of machines, people, paths and private-repository commits were removed or generalized for the anonymous release.*

---

## Numerical-error-to-command gate, multi-episode (registered before collection)

Objective: test whether the finding from the two windows (a cheap solver changes the PID command) repeats in independent CARLA episodes, and whether a simple predictor (norm/cosine of the initial field, branch margin) already predicts this error. Without this result, the learned estimator of the internal candidate note on controller-state-aware inference is not justified.

Collection: own CARLA 0.9.16 server (ports 2200+), maps Town01, Town03, Town04, Town05, Town10HD_Opt; 4 spawns per map; target speed cycling through 3/6/9/12 m/s; 20 static NPCs; 2 contexts per episode (episode = independent unit; contexts of the same episode are correlated). Collection and spawn seeds fixed before seeing results. Target: 40 contexts / 20 episodes.

Inference: frozen Alpamayo1.5, BF16, language seed 123, action seeds 9201 and 9202 (both reported; no post-hoc selection). Solvers: Euler4/10, Heun5, RK4x2, initial dense Euler 10, reference Heun160. Controller: OfficialPIDFollower with empty buffers and real speed; previous steering 0.

Pre-declared metrics: fraction of contexts in which Euler10 and Heun160 differ in mode (accelerate/brake) or in |delta steer| > 0.05; Spearman correlation between the command error and (a) initial cosine, (b) field norm, (c) trajectory XY error. Report per episode, with CI by episode bootstrap. No retrospective filtering of contexts.

Limits: synthetic domain, scripted lane following, numerical reference is not the correct action, no closed loop and no safety claim.

## Collection addendum (before any inference)

Town03 (4/4 episodes) and Town05 (2/2 attempted) caused a segfault of the CARLA server at load time or during the episode; a fresh server per episode did not solve it. These episodes were discarded for infrastructure failure (folders kept under a `failed/` directory of the collection) and replaced by four Town02 episodes (spawns 7/31/55/83, indices 20-23). No solver result existed at the time of replacement. Expected total: 16 episodes / 32 contexts in 4 maps.

## Addendum 2: real PhysicalAI-AV data and final solver list (before any inference)

We add the `test` split of PhysicalAI-Autonomous-Vehicles (NVIDIA), the native format of Alpamayo1.5 (same 4 cameras, 4 frames at 10 Hz, 16 poses), with no simulator domain shift. Sample: 200 clips, one per chunk (seed 20261022), t0 = 5.1 s. We did not independently verify that these clips were unseen in Alpamayo's training; the split comes from `clip_index.parquet`.

Each clip carries the driver's real future. It becomes a third reference (`human`), in addition to Heun160 (converged flow) and Euler10 (published policy). The human future is not the only correct action; it serves to check whether solver changes bring the command closer to or further from observed behavior.

Solvers, the same in both sets: euler:4, euler:8, euler:10, heun:5, rk4:2, early_euler:10, cached_euler:8 (FlashDrive-style: fresh steps 0, 1, 6, 7 and reuse of the last velocity in the middle steps, 4 evaluations; our approximation, not their code) and heun:160 as reference.

Main question for the paper: do mean trajectory metrics (minADE/ADE, as used to call FlashDrive "near-lossless") hide command flips? And do per-input signals already used in prior work (ProbeFlow/FlashDrive cosine, AdaVLA curvature, norm) predict those flips? Competitors added after a new search: [AdaVLA](https://arxiv.org/abs/2608.29208), [ELASTIC](https://arxiv.org/abs/2606.31132), [Coda / Fewer Steps, Better Actions](https://arxiv.org/abs/2609.21216), [Adaptive time step FM for planning](https://arxiv.org/abs/2602.10285), [AdaFlow](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa9cfdb49f7de3dee54007f84c0745b1-Paper-Conference.pdf).

## Addendum 3: decisions read from the trajectory (registered before the PhysicalAI results)

The PID gate uses empty buffers; near equilibrium, accelerate <-> brake can flip from minimal differences. We add decision classes read from the plan, independent of the controller: longitudinal = stop (extent <= 1.5 m in 6.4 s) / decelerate / keep / accelerate (plan speed at 2-3 s versus current, band max(1 m/s, 15%)); lateral = turn left/right (final heading > 20 deg), deviate left/right (final offset > 1.75 m), straight. "Branch margin" baseline: normalized distance of the Euler10 plan to the class boundaries, available at inference. Signals compared: initial and minimum cosine (ProbeFlow/FlashDrive), accumulated field curvature (approximation of the AdaVLA idea), branch margin. The CARLA contexts already analyzed with the PID were seen before this addendum; PhysicalAI was not.

## Addendum 4: commitment time and per-window susceptibility (before running on PhysicalAI)

Hypothesis inspired by speciation in diffusion models (Biroli, Bonnaire, de Bortoli, Mezard 2024): the decision class of the plan is fixed at a time t_c of the flow, and numerical errors matter more near t_c. Measures per context and seed (6 seeds, 9201-9206): class of the predicted endpoint x+(1-t)v along Heun32; t_c = first t from which the predicted class always coincides with the final one. Causal interventions on Euler20: freeze the velocity in a window of 4 steps ([0,.2),...,[.8,1)), as a cache, and perturb the state at the start of the window (epsilon = 0.05*||x||, direction fixed per seed). Pre-declared predictions: (P1) susceptibility is not uniform in t; (P2) flips concentrate in contexts with late t_c; (P3) decision entropy across seeds predicts numerical flips. Falsification: if susceptibility is uniform or unrelated to t_c/entropy, the "commitment" line is abandoned.

## Addendum 5: redesign of the commitment probe (for cost, before interpretable results)

The Euler20 version with perturbations cost ~16 h for 200 clips; it was aborted after 12 cases (no flips; folder kept as an aborted run). New version: base = published Euler10 sampler; freezing of 2 steps per window ([0,.2) only freezes step 1, since step 0 needs an evaluation); Heun16 reference with class tracking of the predicted endpoint; no perturbations (in the normalized solvable model they did not localize commitment); FlashDrive-style cache added (Euler8, steps 2-5 reused). Same 6 seeds.

## Addendum 6: at-scale confirmation of the schedules (before downloading the new data)

Result on holdout 100-199 (2 seeds): decision information + uniform had fewer flips and lower ADE to the reference than uniform at all NFEs, but only one flip CI excluded zero. We also record a later analysis with leakage: the signal "class at t=0 != final class" used the class of the full Euler10 (AUROC 0.88); in the leak-free version (against the class of the cache itself) AUROC dropped to 0.48-0.51 and was discarded.

Confirmation: 436 new `test` clips, one per chunk not yet used (seed 20261023). Calibration curves frozen in a schedule-curves file for PhysicalAI v1 (sha256 in the evaluation manifest), without recalibration. Seeds 9201-9204; inline Heun80 reference with the same noise. Primary hypothesis: decision_info_plus_uniform has fewer flips than uniform at the same NFE, aggregating NFE 4/6/8 (mean paired difference, clip bootstrap CI excluding 0). Secondary: the same against rescaled_entropic, and ADE to the reference.

## Addendum 7: explicit vs implicit temperature (before running)

Exploratory finding (section 5 of the report): Euler-type solvers shrink 100% of the samples toward the conditional mean, and this trades ADE_1 for coverage (minADE_k) and for decision entropy. Test: on the v1 holdout clips 100-199, with 6 seeds, compare implicit shrinkage (Euler 1/2/4/6/10 with tau = 1) against explicit temperature (initial noise x tau, tau in {1, ..., 0.4}) with Heun3/Heun5 (6/10 NFE), Euler10 and Heun32 (reference). Hypothesis: at the same NFE, explicit temperature + Heun dominates or matches the frontier (ADE_1, minADE_6, decision entropy) of implicit Euler. If it does not dominate, "temperature control" is not a method; the analysis remains.

## Addendum 8: schedules at the same effective temperature (before seeing the confirmation of addendum 6)

Addendum 7 showed that Euler with few steps behaves as reduced temperature. Therefore a schedule may "reduce flips" simply because it shrinks less. Complement: run the same schedules on v1 100-199 with seeds 9201-9206 and an inline Heun32 reference, saving the plans. For each schedule, measure the across-seed dispersion and compare flips, decision entropy and ADE_1 with the Heun32*tau curve (temperature_probe_v1) interpolated at the same dispersion. Criterion: decision information only counts as a method if the residual (schedule - temperature curve) is smaller than that of uniform at the same NFE.

## Addendum 9: multi-model replication of numerical temperature (before running)

Same probe (Euler 1/2/4/6/10, Heun32*tau) on Alpamayo-R1 (same 4-camera cache, v1 clips 100-159, 6 seeds) and, when the GPU allows, on Alpamayo2-Super (official 7-camera profile, streaming of the same clip_ids/t0). Predictions: (i) Euler samples shrunk relative to Heun32*tau=1; (ii) Euler-k on the temperature curve (small flip residual compared with Heun-k); (iii) ADE_1 monotone in the shrinkage. The integrators run inside each model's own `diffusion.sample`, with that model's own `step_fn` (and CFG, if any).

## Addendum 10: Qwen-Drive-1.0 (before the at-scale run)

Qwen-Drive-1.0-4B (flow-matching planner with x-prediction, 10-step Euler, floor 1-t >= 0.1). PhysicalAI scenes built according to the release's data documentation (3 cameras at 2 Hz, 1.5 s history and 5 s future at 10 Hz, speed/acceleration from the official interpolator rotated from the world frame to the ego frame). The navigation command is derived from the recorded future heading (|delta psi| > 0.35 rad), identical for all configurations. The release's floor prevents integrations with k > 10 from reaching the endpoint (verified in the smoke test: Euler64 with spurious dispersion of 6.5 m); we use floor = min(0.1; 1/k), identical to the release for k <= 10. Smoke test (1 clip, RL): dispersion between samples of ~0.09 m at the precise solver, insensitive to tau. Execution: planner-rl and planner-sft, v1 100-159, 6 samples, Euler 1/2/4/10 and Euler64*tau. Question: did RL collapse stochasticity? Thesis prediction: without dispersion, temperature and solver have a small effect.

## Addendum 11: official FlashDrive at the same temperature (26/10, before the at-scale run)

Official code and checkpoints (z-lab/flashdrive, MIT; `z-lab/Alpamayo-1.5-10B` with W4A8 + DFlash + streaming; original `nvidia/Alpamayo-1.5-10B`), in eager mode. Per window, from the same VLM reasoning: FlashDrive cache (Euler 8 with steps 3-6 cached), Euler 8/10/4 and Heun32*tau (tau in {1; 0.9; ...; 0.5}), 6 seeds. PhysicalAI v1 clips 100-159, 4 consecutive windows at 10 Hz ending at the clip's t0 (in the z-lab stack the first only does prefill). Smoke test on 1 clip: the cache behaves as tau ~ 0.5-0.55 in both checkpoints, with ADE_1 equal to that of the temperature curve. Analysis: decompose the reported minADE_1 improvement into (a) the difference between checkpoints at the same temperature and (b) the effect of the cache measured against the Heun32*tau curve.

### Addendum 11b: prediction and analysis (recorded at 20:48:53, in the same command as the analysis; see correction below)

The nvidia checkpoint finished (60 clips x 4 windows). No metric had been computed yet. Unit = clip (mean over windows); reference = Heun32*tau.
- tau_eff of each configuration = tau at which the mean across-seed spread of the Heun32 curve (linear interpolation in tau in [0.5, 1]) equals that of the configuration. If it falls outside [0.5, 1], it is marked out of range and there is no match.
- **Prediction:** the FlashDrive cache (flashcache8) and Euler 4/8/10 match the curve. The ADE_1 residual (config - Heun32 at tau_eff) stays within +-0.05 m, with a 90% interval (clip bootstrap, 1000 resamples, tau_eff recomputed in each) contained in the margin. For flashcache, tau_eff ~ 0.5-0.6 is expected.
- Correction in the z-lab stack: the probe did not reset the streaming state between clips (crash at clip 101). The state is now reset per clip; clip 100 had already run with a fresh state and is kept.
- **Correction (20:53):** the time "20:55" of addendum 11b is wrong. The file was saved at 20:48:53, 3 s before summary.json, in the same command. No FlashDrive metric had been computed before that, but the pre-registration is from the same minute as the analysis, not before it. Clips 100-159 are the same 60 as the main sample; Euler 10 at 0.84 is a rerun, not a replication.

## Addendum 12: Qwen-Drive v2, with the reference inside the trained time range (27/10, recorded before the at-scale run)

**Reason.** The v1 reference (Euler-64) queried t > 0.9, a range in which the planner was never trained (x-prediction, t <= 0.9, floor 1-t >= 0.1). v1 also ran without reasoning, in bf16 and with wrong seed labels. See the internal review of cited papers.

**v2** (the Qwen-Drive temperature probe, version 2):
- reference = Heun-32 on [0; 0.9] plus the final jump to the endpoint at t = 0.9;
- tau in {1; 0.9; ...; 0.3}, with Heun-64 at tau=1 as a convergence check;
- E1/2/4/10 = the official sampler;
- reasoning_planning mode (greedy reasoning, shared across samplers);
- expert in fp32 with SDPA;
- seeds 42-47;
- v1 clips 100-159;
- RL and SFT planners.

**Already seen before this registration.** Smoke test of clip 100 (RL):
- E10 spread 0.149 versus 0.152 of the reference;
- E4 0.069;
- Heun-32 and Heun-64 differ by at most 0.6 mm.

**Prediction.** Same protocol of spread-matched tau_eff and residuals with clip bootstrap:
- E10 with tau_eff >= 0.9;
- E4 with tau_eff between 0.35 and 0.65;
- ADE_1 residuals within +-0.05 m and flips within +-2 points, with the caveat that flips are rare in a model with such low spread.

**If it fails.** Qwen-Drive is reported as a model in which the rule does not hold, with the reason.

## Addendum 13: replication on clips outside Alpamayo's training set (27/10 02:48, recorded before any run on these clips)

Reason: 57/60 clips of the main sample (context_100-159 of physicalai_test_contexts_v1) and 391/436 of the confirmation are in the *train* partition of the alpamayo_aligned split file. The new sample (held-out contexts v3, 2026-10-27) has 400 clips, one per chunk, all in the *test* partition (checked 400/400), seed 20261027.

Design, identical to v1 except for the clips: context_000-119 (120 clips, first in name order), same configurations, same seeds 9201-9206, same code (the temperature probe for A1.5; the multi-model temperature probe for R1 and A2), analysis with the paper-numbers script (range 0-120, clip bootstrap re-resolving tau_eff). Order: A1.5, R1, A2. If GPU time allows, context_120-399 enter as an extension, reported separately; the primary analysis is the 120-clip one.

Predictions (from v1):
1. tau_eff of Euler-10 between 0.78 and 0.90 in the three models; Euler-4 between 0.55 and 0.72; Euler-1/2 at the grid floor (<= 0.45).
2. ADE_1 residual of Euler-10 (sampler - reference at the same temperature) with point estimate |.| < 0.05 m in the three models; the raw effect of Euler-10 on ADE_1 is negative (lower error than Heun-32 tau=1) in all three.
3. Flip residual of Euler-10 within +-2 pt (90% CI) in A1.5 and R1.
4. Heun-3/5 (A1.5) with tau_eff >= 0.85.
Any prediction that fails is reported as a failure; if the numbers change materially, the paper uses the out-of-training clips as the main result and v1 becomes a secondary analysis.

### Addendum 13b (27/10 04:53, before opening any output of the out-of-training clips; no plans.npz of these clips had been read)

Equivalence criteria added at the suggestion of the review, in addition to the point predictions of addendum 13:
- 90% TOST with the same margins as v1 (ADE_1 +-0.05 m; flips +-2 pt) for the E4 and E10 residuals in the three models.
- Prediction: E10 passes both criteria in at least 2 of 3 models; E4 need not pass.

**Seeds.** Seeds 9201-9206 are the same as in v1, on purpose, to isolate the effect of swapping the clips. Since the clips are new, the noise samples do not repeat over the same inputs.

**Margins.** The margins were fixed after seeing the v1 clips; in this replication they are pre-specified.

### Execution (27/10 12:12)
R1 and A2 on the out-of-training clips now run as 3 processes of the same code, on disjoint clip ranges, merged afterwards (the shard-merge script, which checks the hash of the code and the seeds). No code or numerical change: each clip runs as before, only in parallel.

### Execution of A2 on the cluster (27/10)

**Where it runs.** A2 on the 120 out-of-training clips runs entirely on H200 GPUs (a GPU cluster, test partition), in 4 shards of 30 clips, with the same code (paths configurable through environment variables) and the same checkpoint snapshot.

**Hardware check.** On clip context_100 of v1:
- the trajectories differ from those of the local workstation by ~0.09 m (median of the per-trajectory maximum), due to bf16 rounding;
- the across-seed spread matches: E10 0.823 vs 0.822 m; Heun-32 1.288 vs 1.290 m;
- no model mixes hardware within the replication.

## Addendum 14: extension context_120-399 (27/10 ~21h, before running any clip of this range)

**Context.** In the 120-clip primary analysis, 13b failed: E10 passed the ADE_1 and flip TOSTs only in A1.5. In R1 and A2, the flip TOST failed because the interval was wide: the point residuals were +1.0 and +0.9 pt, with an upper limit around 2.5 pt. **The primary analysis does not change and remains reported as a failure.**

**Extension (already foreseen in addendum 13).** context_120-399, i.e. 280 clips, one per chunk, all from the test partition. Same code, same configurations and same seeds 9201-9206.

**Hardware.** Each model runs on the same hardware as its primary analysis:
- A2 on H200, on the cluster (4 shards of 70);
- R1 on the local workstation, when the GPU is free from closed-loop work;
- A1.5, if there is time, on the local workstation.

**Analyses, in this order:**
1. The extension alone (280 clips), with the same predictions 1-2 and 90% TOST with margins of +-0.05 m and +-2 pt.
2. The 400 clips together, as a sensitivity analysis, with clip bootstrap.

The result is reported as an extension: it does not convert 13b into a success.

**Prediction for the extension:** E10 passes the flip TOST in R1 and A2.

### Addendum 14b (28/10 00:35): hardware of the A2 extension

**Reason.** The cluster's 8 H200 GPUs are busy, and the A2 job would only start on 29/10 at 23h. A node with 2 Blackwell GPUs is free.

**Change.**
- Clips context_120-140, already run on H200, stay as they are.
- The rest of the A2 extension runs on Blackwell, with the same code, the same seeds and the same checkpoint.

**Effect on the analysis.**
- The extension analysis records the hardware of each shard. The hardware check of addendum 13 showed that across-seed spreads match to 0.1-0.2%, with trajectories that differ by ~9 cm due to bf16 rounding.
- As a sensitivity analysis, the extension is also analyzed using only the clips run on Blackwell.
- The 120-clip primary analysis of A2 remains entirely on H200.

## Addendum 15 (28/10 07:40, commit time; the draft said "~09h" by mistake): Heun-3/5 controls of R1 and A2 on the out-of-training clips

Registered before running. Request from the three reviewers of round 9: the controls that discriminate the test (Heun-3/5) exist only for A1.5.

**Design.**
- R1 and A2, context_000-399, seeds 9201-9206.
- Configurations: Heun-3 tau=1, Heun-5 tau=1 and Heun-32 tau=1 (reference).
- The code is the same, with a `--configs` option that only swaps the list of configurations. The initial noise per seed and the language branch do not change.
- Runs on the local workstation (Blackwell), one model at a time, after the A1.5 extension.

**Analysis.**
- Heun-k flips are measured against the Heun-32 tau=1 reference **of the same run**. This way, differences in hardware or run do not become spurious flips: part of A2 ran on H200.
- tau_eff and the curve value at tau_eff come from the temperature curve of the main run (per-clip means).
- I also report the maximum difference between the reference of this run and that of the main run.
- Clip bootstrap that re-resolves tau_eff, as in the paper-numbers script.

**Prediction** (from A1.5): Heun-3/5 with tau_eff >= 0.85 and flip residual > +2 pt (outside the margin) in R1 and A2.

**Addendum 15b (28/10 08:20, before running):** to parallelize, the A2 Heun controls run on the cluster, on the Blackwell node (4 shards of 100 clips). R1 stays on the local workstation. Since each Heun-k is compared with the reference of its own run, hardware does not enter the flips.

### Addendum 11c (28/10 ~08:55): FlashDrive flips and minADE_6, **exploratory**

Registered before computing flips or minADE_6 on the FlashDrive probes. The ADE_1 results of these probes were already known, so the analysis is **exploratory**, not confirmatory.

**Definition.**
- Flips are measured per window and seed, against Heun-32 tau=1 with the same seed and window. They use the decision function of the trajectory-decision-gate module with the clip's current speed taken from the records file of the PhysicalAI test contexts v1. This is the speed at t0, and the windows end up to 0.3 s earlier, which is an approximation.
- minADE_6 is the minimum over the 6 seeds, against the window's human trajectory.
- Mean per clip, tau_eff by spread (as in the ADE_1 analysis), residual against the Heun-32 tau curve at tau_eff, bootstrap of 1000 clip resamples with seed 20261026, re-resolving tau_eff.
- nvidia and z-lab checkpoints.

**Note to addendum 15b (28/10, 15h45).** The A2 Heun job on the cluster's H200 partition never left pending because of priority. It was cancelled and resubmitted on the test partition, on 2 Blackwell GPUs (one job). Nothing in the protocol changes: each Heun-k is compared with its own Heun-32, which runs in the same job and on the same hardware. If the 1 h job does not finish, I will resubmit manually, one job at a time and only after the previous one ends.

## Addendum 16 (28/10 ~21h30, before running): low-tau reference on the 400 held-out clips

**Reason.** E_1 and E_2 fall below the measured range (tau_eff <= 0.4). This is the TODO of the "one-step limit" in the paper.

**Run.** Heun-32 configs at tau in {0.3, 0.2, 0.1}, same seeds (9201-9206) and same initial noise.
- **Models and clips:** A1.5, R1 and A2, on held-out clips 0-399.
- **Machine:** the workstation, in 4 processes per model, in the order A1.5 -> R1 -> A2.
- **Code:** the same probe. A1.5 gained only the `--configs` flag, without changing the sampling.

**Analysis.** The tau curve becomes {1, ..., 0.4} of the main run joined with {0.3, 0.2, 0.1} of this run. From it come tau_eff, residuals and clip bootstrap, as in addendum 13.

**Predictions:**
- **L1:** E_2 falls within the extended range, with tau_eff between 0.2 and 0.4 in the 3 models.
- **L2:** E_1 falls within the extended range (tau_eff >= 0.1) in at least 2 of the 3 models.
- **L3:** at the tau_eff of E_2, the flip residual of E_2 is above +2 pt in at least 2 of the 3 models. That is, very few steps stop being temperature in decisions. This is an extrapolation of the growing residual of E_4 and E_10, and is therefore exploratory.

The result will be reported in the paper whatever it is.

## Addendum 17 (28/10 ~23h45, before analyzing): scaled-noise control with Euler-10 (the test of Yang et al. 2610.32102)

**Reason.** Yang et al. scale the initial noise and keep integrating with Euler-10. Since this did not reproduce the gain of their Euler-2, they conclude that this gain comes from a "structured contraction", not from temperature. Reviewers ask for this control.

**Data.** The 400 held-out clips of A1.5 already have Euler-10 at tau in {0.9, ..., 0.4} (`euler10_t*` configs of the main probe, run before this addendum, but never analyzed for this question). CPU only; no new data.

**Analysis.** Two temperature curves, each with spread, ADE_1 and flips per tau:
- **(A) ours:** Heun-32 tau.
- **(B) Yang's:** Euler-10 tau.

For each sampler S in {E_2, E_4, E_6}:
- tau_eff and residuals of S against curve A and against curve B, matching the spread in each curve;
- clip bootstrap (2000 resamples, seed 20261029), re-resolving tau_eff;
- protocol margins; flips measured against the Heun-32 tau=1 reference at the same seed.

**Predictions:**
- **Y1:** curve B lies below curve A in spread at the same tau, because Euler-10 already contracts. Hence the tau_eff of S measured on curve B is larger than on curve A.
- **Y2:** the residuals of S against B are of magnitude similar to the residuals against A: both curves are temperatures, and what matters is matching the spread.
- **Y3 (against Yang's reading):** if the noise is scaled *without* matching the spread, for example using tau = 1/k itself or a fixed tau, the residuals are large. This would explain their conclusion without temperature failing. This part is exploratory and depends on the tau Yang actually used, which we will verify in their methods section.

The result will be reported whatever it is.

**Note to addendum 16 (29/10 ~02h, before any analysis of low tau).** Only A1.5 runs on the workstation. The low-tau runs of R1 and A2 move to the cluster (test partition, 1 h jobs, one at a time, shards c0-c3), to free the workstation for WOD-E2E and Bench2Drive.
- **Hardware:** the main curve of these two models already mixes machines, and addendum 13 showed spreads equal to 0.1-0.2%. The low-tau points enter the curve as per-clip means.
- **Record:** the hardware of each shard is recorded.
- **Sensitivity:** the analysis is repeated only on the clips whose main run was made on the same type of GPU.


**Correction of times (note of 28/10, 23:10 PDT by the machine clock).** The times written by hand in the addenda of the evening of 28/10 were ahead by between 1 h and 2.5 h. The valid times are those of the commits (git, PDT):
- addendum 16: commit recorded in the private repository, 28/10 22:29;
- addendum 17: commit recorded in the private repository, 22:49, with the result in a later commit, 22:50;
- note to addendum 16: commit recorded in the private repository, 23:03;
- registration of WOD-E2E: commit recorded in the private repository, 28/10 23:01 (not 29/10 01h30);
- addendum 1 of WOD: commit recorded in the private repository.

The order registration -> result does not change.

## Addendum 17b (28/10 23:10 PDT, before running): replication of the scaled-noise control on R1 and A2

**Reason.** Addendum 17 was post hoc and used a single model (A1.5). This is the pre-registered replication.

**Run.** Euler-10 at tau in {0.9, 0.8, 0.7, 0.6, 0.5, 0.4}, same seeds and same initial noise, on the 400 held-out clips.
- **Where:** cluster, test partition, Blackwell GPUs, 1 h jobs, after the low-tau run of R1.
- **Order:** R1, then A2.
- **Code:** the cluster probe batch script with `--configs euler:10:tau`.

**Analysis.** The scaled-noise-control analysis script without changes, applied to each model (bootstrap seed 20261029).

**Predictions:**
- **Y1:** the Euler-10 curve lies below the Heun-32 curve in spread, for every tau.
- **Y2:** against both curves, E_4 is equivalent in ADE_1 (+-0.05 m) in both models, and E_6 is equivalent in ADE_1 and flips (+-2 pt).
- **Y2b:** the flip residual of E_4 against the Euler-10 curve is smaller than against Heun-32 in at least one of the two models. Against Heun-32, E_4 left +1.6 and +1.4 pt on the 400 clips.

The result will be reported whatever it is.

**Provenance of addenda 13 and 13b (note of 28/10 23:17 PDT).** The git history of the private repository begins on 27/10 12:42 PDT, after the held-out analysis of A1.5 (12:12). The evidence that these addenda came before is the log of the coding-assistant session (a local file), which has a UTC timestamp on every tool call:
- addendum 13 written on 27/10 09:48:11Z (02:48 PDT);
- held-out probe (`run_heldout_v3`) launched at 09:48:41Z, 30 s later;
- edits of addendum 13b at 11:47Z and 11:53Z (04:47 and 04:53 PDT);
- A1.5 analysis at 12:12 PDT.

The log is local, not a public record. The paper must say so.

## Addendum 18 (29/10/2026 14:43 PDT, before running): convergence of the reference and resolution of the tau grid

**Reason.** The three independent reviewers of the round of 29/10 ask that the Heun-32 reference and the interpolation on the tau grid (step 0.1) not decide the verdict on E_10.

**Sample.** Alpamayo-1.5, held-out clips context_000-099 (the first 100), seeds 9201-9206, same initial noise.

**Run.** Workstation, after the Bench2Drive pilot.
- **Refined reference:** Heun-64 and Heun-128 at tau in {1; 0.9; 0.85; 0.8; 0.75; 0.7}.
- **Dense grid:** Heun-32 at tau in {0.875; 0.85; 0.825; 0.775; 0.75}, which are added to the main grid.

**Analysis.** With the Heun-32 (main), Heun-64 and Heun-128 curves on the 100 clips:
1. Discretization error of the reference in the observables: Heun-32 minus Heun-128 difference in spread, ADE_1 and flips (against each solver's own tau=1 reference) at each tau, with clip CI.
2. tau_eff and residuals of E_10 and E_4 (ADE_1, flips) against each curve. Clip bootstrap with 2000 resamples, seed 20261030, redoing tau_eff.
3. Grid resolution: the same residuals with the dense grid (step 0.025 near tau_eff) against the grid with step 0.1.

**Predictions:**
- **C1:** |Heun-32 - Heun-128| < 0.01 m in ADE_1 and < 0.5 pt in flips for every tau of the grid.
- **C2:** the tau_eff of E_10 changes by less than 0.02 between the Heun-32, Heun-64 and Heun-128 curves.
- **C3:** the equivalence verdicts of E_10 (ADE_1 +-0.05 m, flips +-2 pt) do not change between the three curves nor between the two grids.

The result will be reported whatever it is.

## Addendum 19 (30/10/2026 00:33 PDT, before running): flip equivalence of E_10 against the converged reference on the 400 clips

**Reason.** In addendum 18 (100 clips), C2 held: the tau_eff of E_10 stayed between 0.853 and 0.857. C1 and C3 failed on flips: against Heun-64/128, the 90% TOST of the E_10 flip residual reaches 2.6-2.7 pt. Heun-64 and Heun-128 differ by at most 0.17 pt, so Heun-64 is already converged. The paper's verdict uses the pooled 400 held-out clips of Alpamayo-1.5, where the converged reference has not yet been tested.

**Sample.** Alpamayo-1.5, the 400 held-out clips of the pooled400 set (context_000-399 of the held-out contexts v3, 2026-10-27), seeds 9201-9206, same initial noise.

**Run.** Workstation, unchanged probe. Heun-64 at tau in {1; 0.9; 0.85; 0.8; 0.75; 0.7} on clips 100-399; clips 0-99 come from addendum 18. E_10 comes from the pooled400 probe (A1.5 held-out pooled400 v3, 2026-10-28).

**Analysis.** Same as addendum 18 (the convergence-a18 analysis script, same logic as paper_numbers), with the Heun-64 curve on the grid registered above. Flips against the solver's own tau=1 reference. Clip bootstrap: 2000 resamples, seed 20261031, redoing tau_eff. 90% TOST with margins of 0.05 m (ADE_1) and 2 pt (flips).

**Predictions:**
- **D1:** the flip residual of E_10 against Heun-64 is equivalent: 90% TOST within +-2 pt.
- **D2:** the ADE_1 residual of E_10 against Heun-64 is equivalent (+-0.05 m).
- **D3:** the tau_eff of E_10 against Heun-64 is within 0.02 of the tau_eff against Heun-32 on the same 400 clips (0.850).

**Recorded consequence.** If D1 fails, the paper says that the flip equivalence of E_10 does not hold against the converged reference. The thesis then rests on tau_eff and error.

**Exploratory.** The same analysis with Heun-64 only at tau in {1; 0.9; 0.8; 0.7}, to separate the effect of the grid.

The result will be reported whatever it is.

## Addendum 20 (30/10/2026 09:10 PDT, before running): converged reference on Alpamayo-R1 and Alpamayo-2

**Reason.** In addendum 19, against Heun-64, the flip residual of E_10 in A1.5 went from +0.14 to +0.96 pt, with 90% TOST [0.35; 1.63]: still equivalent, but the interval moved away from zero. The R2 review of v6.3 points out that R1 and A2 were not rechecked against the converged reference, and that the paper's "flips 3/3" depends on it.

**Sample.** Alpamayo-R1 and Alpamayo-2, the 400 held-out clips of pooled400 (context_000-399 of the held-out contexts v3, 2026-10-27), seeds 9201-9206, same initial noise.

**Run.** Workstation (RTX PRO 6000 Blackwell), unchanged multi-model temperature probe. Configurations: Heun-64 at tau in {1; 0.9; 0.85; 0.8; 0.75; 0.7} (the grid of addendum 19) and E_10 at tau=1. E_10 runs again because, in A2, part of the pooled400 clips ran on H200 and part on Blackwell. Running here, sampler and reference are on the same hardware. First R1, then A2; one job at a time, 4 processes on disjoint ranges, then the merge.

**Analysis.** The same as addendum 19 (the convergence-a19 analysis script), per model. tau_eff by dispersion matching on the Heun-64 curve. Flips against the solver's own tau=1 reference. Clip bootstrap: 2000 resamples, seed 20261101, redoing tau_eff. 90% TOST with margins of 0.05 m (ADE_1) and 2 pt (flips). The main E_10 is the one from this run.

**Predictions (per model, R1 and A2):**
- **F1:** the flip residual of E_10 against Heun-64 is equivalent: 90% TOST within +-2 pt.
- **F2:** the ADE_1 residual of E_10 against Heun-64 is equivalent (+-0.05 m).
- **F3:** the tau_eff of E_10 against Heun-64 is within 0.02 of the tau_eff against Heun-32 on the same 400 clips (pooled400).

**Recorded consequence.** The paper reports the number of models in which E_10 is equivalent in flips against the converged reference (A1.5, from addendum 19, plus R1 and A2), whatever it is. If F1 fails in any model, the phrase "flips 3/3" leaves the paper, and the thesis on flips is restricted to the models in which F1 holds. The thesis on tau_eff and error continues to depend on F2 and F3.

**Sensitivity.** The E_10 of pooled400 in place of the E_10 of this run. This measures the effect of hardware.

**Exploratory.** The grid {1; 0.9; 0.8; 0.7}, as in addendum 19. E_4 does not enter, because it falls below 0.7.

The result will be reported whatever it is.

**Execution note of addendum 20 (31/10/2026 02:13 PDT, before any analysis of A2):** Alpamayo-2 Super occupies ~72 GB per process, so only one shard fits on the workstation. Clips 0-199 run on the workstation (RTX PRO 6000); clips 200-399 run on the cluster (H200, cluster probe batch script). Each clip runs entirely on a single GPU, so the E10 and the Heun-64 of a clip are always on the same hardware. The probe, configurations, seeds and analysis do not change. As an unregistered sensitivity analysis, I also report the verdicts split by GPU type.

> Execution note, addendum 20 (31/10 06:47 PDT, before any analysis of A2): the cluster job for clips 200-299 has an estimated start only at 11:19; clips 300-399 now run on the workstation (same probe, configs and seeds), and the cluster keeps only 200-299. The unregistered sensitivity analysis by GPU type now compares 0-199 and 300-399 (workstation) with 200-299 (H200).

> Execution note, addendum 20 (31/10 10:47 PDT, before any analysis of A2): the cluster runs about 9 clips per GPU in each 1 h job; clips 250-299 move to the workstation (after 300-399). The cluster keeps 200-249 (shards c0, c1); any cluster output for 250-299 does not enter the merge. The sensitivity by GPU type compares 200-249 (H200) with the rest.

> Execution note, addendum 20 (31/10 12:47 PDT, before any analysis of A2): the cluster job ended at the time limit with 26 clips; the next one had an estimated start on 1/11 23:07 and was cancelled. The partial cluster shards do not enter the merge, and all 400 clips run on the workstation (200-249 in shard s5). The sensitivity by GPU type becomes moot. The a20d run (300-399) stopped at clip 367 because of a transient connection error to Hugging Face and is resumed, with up to 3 attempts per range.

> Execution note, addendum 17b (31/10 14:53 PDT, before any analysis of 17b): R1 runs entirely on the cluster's gpu partition (2 A100, jobs of up to 2 days, a dedicated batch script), because the queue of the test partition is slow. A2 finishes on the test partition (H200) the shards c2 and c3 already started there. Same configs (Euler-10 at tau 0.9-0.4), seeds and clips.

## Addendum 21 (31/10/2026 15:09 PDT, before running): E_4 against the converged reference and the low tau of A2

**Reason.** The internal general findings report (section "failures that may be wrong", items 1 and 5) points to two verdicts that depend on missing data:
- The E_4 flips in R1 and A2 (+1.6 and +1.4 pt, not equivalent) were measured only against Heun-32, which does not converge in flips at the ~1 pt scale (addendum 18). Against Heun-64, the E_4 spread falls below tau=0.7, the end of the grid (addenda 19 and 20), and the residual was never measured.
- L3 of addendum 16 was judged without A2, which never ran at low tau.

**21a: Run.** Heun-64 at tau in {0.6; 0.5}, same seeds (9201-9206), same noise and the 400 held-out clips.
- A1.5 and A2: workstation, after the Bench2Drive grid. This is the same hardware as the existing Heun-64 curves.
- R1: cluster, gpu partition (A100, dedicated batch script). It also runs tau=0.7, which overlaps with the workstation curve for the G0 check.

**21a: Analysis.** The E_4 of pooled400 (each model) against the Heun-64 curve at tau in {1; 0.9; 0.85; 0.8; 0.75; 0.7; 0.6; 0.5}, with the same interpolation, the same clip bootstrap (2000, seed 20261029, tau_eff re-resolved in each resample) and FROST 90% TOST. In R1, points 1-0.7 are those of the workstation, and 0.6 and 0.5 come from the cluster.

**21a: Predictions:**
- **G0 (R1 only, hardware check):** the cluster's Heun-64 tau=0.7 against the workstation's, under the same noise, changes <= 3% of the decision classes, and the ratio of mean spreads stays at 1 +- 0.01. If it fails, the R1 analysis is redone with the cluster curve at tau <= 0.7, and both versions are reported.
- **G1:** the ADE_1 residual of E_4 against Heun-64 is equivalent (+-0.05 m) in each of the three models.
- **G2:** the flip residual of E_4 against Heun-64 is equivalent (+-2 pt) in each of the three models. There is no prediction of direction: against Heun-32 it passed in A1.5 and failed in R1 and A2, and the E_10 residual grew from Heun-32 to Heun-64 (addendum 19).
- **G3:** |tau_eff(E_4) against Heun-64 - tau_eff(E_4) against Heun-32| < 0.02 in each model.

**Consequence fixed now:** the scorecard row "E_4 equivalent in error and flips" uses the verdict against Heun-64, whatever it is, and the verdict against Heun-32 goes to the supplement.

**21b: Run.** A2, Heun-32 at tau in {0.3; 0.2; 0.1}, 400 held-out clips, on the workstation (the same script as addendum 16, the held-out low-tau run script with MODELS=a2), after 21a.

**21b: Analysis.** That of addendum 16, without changes, now with the three models. L1-L3 are judged as registered ("in at least 2 of the 3 models").

**Outside this addendum.** Item 4 of the report (NAVSIM, DDIM-100 against DDIM-200) is not executable: ReCogDrive has 100 training timesteps, so DDIM-100 is already the complete discrete chain, and addendum 2 of the NAVSIM protocol already uses this reference.

The results will be reported whatever they are.

> Execution note, addendum 21 (31/10 22:18 PDT, before any analysis of 21a or 21b): to use the cluster, the A2 parts leave the workstation and go to the cluster's H200 partition.
> - **21a A2:** Heun-64 at tau {0.7; 0.6; 0.5}. tau=0.7 overlaps with the workstation curve and enters the G0 check, as in R1.
> - **21b A2:** Heun-32 at tau {0.3; 0.2; 0.1}. The workstation points (tau >= 0.4) and the cluster points are declared as mixed hardware. The sensitivity analysis uses the tau=0.4 point, which exists only on the workstation, as an anchor.
> - **A1.5:** continues on the workstation, after Bench2Drive.

> Execution note, addendum 21 (02/11 05:02 PST, before any analysis of 21a or 21b): the cluster's H200 partition only forecast a start on 04/11. The 21a A2 and 21b A2 jobs move to the gpu partition (2 A100 80 GB each), with the same arguments, configs, seeds and clips. The cluster hardware changes from H200 to A100. G0 and the declaration of mixed hardware remain valid, now with the A100 as the cluster hardware.

> Execution note, addendum 21 (02/11 06:49 PST, before any analysis of 21a or 21b): the change to the gpu partition failed. The cluster's A100s have 40 GB, and A2 needs ~72 GB, so both jobs died with CUDA OOM when loading the model, without writing any plan. The jobs return to the H200 partition, with the same arguments (21a: `--configs heun:64:0.7 heun:64:0.6 heun:64:0.5`; 21b: `--configs heun:32:0.3 heun:32:0.2 heun:32:0.1`), as new jobs. The note of 02/11 05:02 is void: the cluster hardware remains H200.

> Execution note, addendum 21 (02/11 21:26 PST, before any analysis of 21a or 21b): the A2 parts move to blocks of <= 1 h on the cluster's `test` partition (2 x H200 NVL per job, one process per GPU), chained by a chain driver script (run as a system service on the workstation, with its own log). Same type of hardware as before (H200); same scripts (the cluster probe batch script), arguments, configs (21a: `--configs heun:64:0.7 heun:64:0.6 heun:64:0.5`; 21b: `--configs heun:32:0.3 heun:32:0.2 heun:32:0.1`), seeds, initial noise and clips (0-400, 4 shards), with per-clip `--resume`; each block is cancelled only right after a clip is saved. The two jobs on the H200 partition were put on hold (`scontrol hold`) so as not to write to the same outputs, and will be cancelled when the `test` runs finish. Nothing was analyzed.

> Execution note, addendum 21 (02/11 21:28 PST, before any analysis of 21a or 21b), correcting the previous one: the `test` blocks request 1 x H200 NVL (not 2), because the 2 were not free on the same node; the probe batch script then runs the 4 shards in sequence (it now skips shards that have a `manifest.json`). Only H200 is used: other GPUs of `test` (H100, a single node) were not used because the peak memory of A2 on 80 GB has not been measured (~72 GB is the model size), so the hardware remains H200 for all clips. The two held jobs were released from hold; the chain driver is mutually exclusive with them (it does not submit while one of them is RUNNING and cancels the `test` block, right after a saved clip, if one of them starts; it cancels the H200-partition job of each part when the part finishes). Both use `--resume`. Nothing was analyzed.

## Addendum 22 (02/11/2026 15:20 PST, before running): inverse validation with "heated" samplers (tau > 1)

**Reason.** Reviewer request: FROST was only tested with samplers that contract (tau_eff < 1) and with the scaled-noise control downward (addenda 17 and 17b). The inverse test is missing: a sampler that spreads *more* than the tau=1 reference. If dispersion matching is a measure of temperature, and not an artifact of cold samplers, it must return the right tau_eff above 1.

**Run.** Same 400 held-out clips (held-out contexts v3, context_000-399), same seeds (9201-9206), same initial noise and same reasoning (language seed 123), probes unchanged (the temperature probe for A1.5, the multi-model temperature probe for R1):
- **Extended curve:** Heun-32 starting from tau*z, tau in {1.1; 1.2; 1.3}.
- **Heated samplers:** Euler-10 starting from tau'*z, tau' in {1.2; 1.4}.
- **Models:** Alpamayo-1.5 and Alpamayo-R1, one after the other. A2 does not fit on the GPU of this machine and is left out.
- **Machine (declared hardware):** a secondary PC (RTX 5090 32 GB, Blackwell sm_120, driver 595.99; 33 GB of RAM), with the environment recreated from the pinned environment freeze file (Python 3.12, torch 2.8.0 cu128, transformers 4.57.1; same versions as the workstation). One process at a time. Launcher: a dedicated shell script for the heated A22 run. See the internal note on this machine.
- **Smoke test (structure only):** 2 clips (context_000-001) with Euler-4 tau=1 and Heun-32 tau=0.7, to measure memory and time (a dedicated smoke script). The A1.5 one has already run (peak RSS 5.8 GiB; peak GPU 23.1 GiB; ~8 s per clip in those configs). The R1 one runs automatically before the R1 run, when the weights finish copying. These plans are not compared with anything and do not enter the analysis.

**Step 1: reproduction check (structural, before the main run).** For each model, Euler-10 tau=1 on clips 0-4 (30 plans), compared bit by bit with the stored plans of pooled400 (A1.5 and R1 held-out pooled400 v3 probes, 2026-10-28, run on the workstation, RTX PRO 6000 Blackwell). Script: the heated-A22 reproduction check (only a count of identical plans and the maximum difference; no FROST metric).

**Hardware rule (fixed now).**
- If the 30 plans are bit-identical, the extended curve is the stored curve (tau <= 1, Heun-32, pooled400) joined with the points tau in {1.1; 1.2; 1.3} from the secondary PC, and E_10 is the one from pooled400.
- If *any* plan differs, every comparison becomes same-hardware: the secondary PC also runs Heun-32 at tau in {1.0; 0.9; 0.8} and Euler-10 tau=1, and the analysis uses only plans from the secondary PC (curve {0.8; 0.9; 1.0; 1.1; 1.2; 1.3}, E_10 and the Heun-32 tau=1 flip reference of this hardware). tau=0.8 enters in addition to 1.0 and 0.9 because the tau_eff of E_10 (0.83-0.85 in both models) is below 0.9; without it, E_10 would fall outside the same-hardware curve. The stored tau <= 1 run is reported only as a sensitivity analysis.
- The rule is applied per model, automatically by the launcher (`repro_check.json`, field `all_bit_identical`).

**Analysis.** That of q6b of the FROST robustness script (dispersion matching by linear interpolation on the Heun-32 curve of the same run; ADE_1 and flip residuals at tau_eff; flips against the Heun-32 tau=1 reference at the same seed), with the extended curve above. Clip bootstrap: 2000 resamples drawn as in the paper-numbers script (seed 20261026, the same draws for both models), re-resolving tau_eff in each resample. 90% TOST with margins of 0.05 m (ADE_1) and 2 pt (flips). A version of the analysis for the extended curve will be written before opening any tau > 1 plan and committed with this addendum cited.

**Predictions:**
- **H1:** for each tau' in {1.2; 1.4} and each model, tau_eff(Euler-10 from tau'*z) = tau' * tau_eff(E_10) +- 0.03, with tau_eff(E_10) measured on the same curve. Same criterion as the scaled-noise control.
- **H2:** the residuals of the heated samplers are equivalent in both TOSTs (+-0.05 m in ADE_1, +-2 pt in flips), in each model.
- **H3:** the matched tau_eff of E_10 on the extended curve changes by less than 0.03 relative to the tau_eff matched on the tau <= 1 curve of the same hardware.
- If tau' * tau_eff(E_10) falls outside the curve (above the spread of tau=1.3), that point counts as out of range and H1/H2 are not judged on it.

**Outside this addendum.** Alpamayo-2 (does not fit in 32 GB); Heun-64 above tau=1.

The results will be reported whatever they are.

> Execution note, addendum 22 (02/11 15:26 PST, before any analysis): structural check of A1.5 on the secondary PC (RTX 5090): 0 of 30 Euler-10 tau=1 plans (clips 0-4) bit-identical to those of pooled400 (RTX PRO 6000), maximum difference 6.9 cm. By the registered rule, the A1.5 run includes Euler-10 tau=1 and Heun-32 at tau {1.0; 0.9; 0.8}, and the analysis uses only plans from the secondary PC. The R1 check runs before the R1 run, with the same rule.
