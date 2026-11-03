# FROST Protocol for Qwen-Drive v1 (v3 data, 400 held-out clips)

**Summary.** This protocol governs the FROST analysis (spread-matched effective temperature tau_eff, equivalence tests on ADE_1 and decision flips) of Qwen-Drive-1.0-4B, a flow-matching driving planner, on the 400 held-out PhysicalAI-AV clips. It fixes the inputs, the factory sampler, the reference, the estimator, the margins and the predictions before any plan of the v3 probe was analyzed. Addendum 1 (Heun-3/Heun-5 control) and Addendum 2 (Euler-6) were added after the main analysis and are registered before their own runs.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

Recorded on 02/11/2026 04:53 PST (from `date`), **before any analysis of the v3 plans**. The authors approved this analysis the same day. The commit of this file precedes any computation of tau_eff, residual, spread, ADE or flips on the four v3 Qwen-Drive RL held-out run folders (c0 to c3).

## 1. State of the data at the time of registration

**The data already exist and have not yet been analyzed.** The v3 probe (the Qwen-Drive temperature probe v3, sha256 3d3e7d9a..., run by a cluster batch script) ran on 01/11 in four disjoint ranges of 100 clips (c0: 0-99, c1: 100-199, c2: 200-299, c3: 300-399).

**Already seen before this registration:**
- **Structure** (checked today, without looking at metrics): 4 x 7900 keys; 400 distinct clips (context_000-399), 400 distinct clip_id, no duplicate between ranges, no missing clip, all arrays with shape (50, 3) and finite; 13 configurations x 6 seeds x 400 clips = 2400 plans per configuration; `expert_fp32 = true`, `command_rule = simwam` and seeds 9201-9206 in the four ranges; `reasoning.json` with 100 traces per range.
- **Heun-32 against Heun-64 at tau=1 on these same data** (internal general findings report, update 01/11 ~17:30): difference of 0.2 mm in median and 4.8 mm at most over the 2400 clip x seed pairs; classes 100% equal. This is a check of the reference, not a FROST result.
- **ADE5 of 1.167 m** on the 400 held-out, but from the reproduction run (a Qwen-Drive reproduction run on the held-out clips, 2026-11-01), not from these plans.
- **v2** (addendum 12 of the master protocol, 60 clips context_100-159, seeds 42-47, legacy command): E10 with tau_eff 0.98, E4 with 0.42 (RL), E1 and E2 below the grid, ADE_1 almost constant in tau, ADE_1 residual of E4 +0.041 [0.022; 0.062] (TOST fails), flip residual of E4 +2.6 pt [-1.4; 7.2] (fails). The predictions below start from these numbers.

No statistic of tau_eff, spread, ADE or flips was computed on the v3 plans.

## 2. Fair comparison (see fair_comparison.md)

**Rule 1, official mode.**
- **Inputs:** release scene (Qwen-Drive's data documentation): 3 cameras x 4 frames (-1.5; -1; -0.5; 0 s), 1.5 s history, ego speed and acceleration, t0 = 5.1 s. The reproduction on the 644 gold clips gives ADE3 0.422 m and ADE5 1.125 m against 0.42 and 1.11 in the paper; the inputs reproduce the model.
- **Mode:** reasoning_planning with greedy reasoning, the documented evaluation mode; the same trace is shared by all samplers of a clip.
- **Factory sampler:** Euler-10 ending at the predicted endpoint at t = 0.9 (floor 1-t >= 0.1). E1, E2 and E4 are the same sampler with fewer steps.
- **Declared deviation: expert in fp32 (the release runs in bf16).** On the DRIVE AGX Thor, on the same 40 clips, the bf16 expert inflates the spread ratio to 1.18-1.23 against fp32 (internal note on other VLAs on Thor), and fp32 on the board reproduces the reference (0.997). In bf16, the expert's precision acts as a temperature above 1 and mixes with the integrator effect. fp32 isolates the integrator, which is what FROST measures; the reference also runs in fp32.
- **bf16 release variant: outside this registration.** There are no bf16 plans with the Heun curve under the v3 conditions (the only bf16 plans are the 40 Thor clips, E10 only, other hardware). The variant "E10 bf16 against the fp32 curve" stays as a future item, to be registered when there are data; nothing here depends on it.

**Rule 2, same information.**
- Qwen-Drive receives a navigation command by construction. Here it comes from the recorded future by the SimWAM rule (heading of the last segment of the 4 s future, threshold 0.31 rad), the same one used in SimWAM. All samplers of a clip receive the same command (2a).
- **2b:** flips are also reported separated into longitudinal and lateral. The verdict comparable across models uses the **longitudinal flips**, which the command does not fix. Alpamayo runs without navigation.
- **2c ("straight" command in all clips): not executed.** There are no data; it remains declared as pending.

**Rule 3, converged reference.** The reference is the model's own ODE (velocity = (endpoint - x)/(1 - t)) on [0; 0.9], with Heun, plus the final jump to the endpoint at t = 0.9, starting from tau*z. Convergence (convergence extension summary for Qwen, 2026-11-01): 20 clips x 6 seeds, tau in {1; 0.6}: Heun-256 against Heun-512 <= 1.7e-4 m; Heun-32 against Heun-64 <= 3.1e-3 m; decision class equal to that of Heun-512 in 120/120 pairs per tau at all levels; RK45 at 1e-7 completes 120/120 and stays within 1.1e-3 m of Heun-512. The curve used is the Heun-32 of the probe (tau in {1; 0.9; ...; 0.3}); the Heun-64 at tau=1 of the 400 clips enters as a check (section 5, R0).

**Rule 4, same metrics.**
- **Clips and seeds:** the 400 held-out clips of PhysicalAI-AV (the records file of held-out contexts v3, 2026-10-27) and seeds 9201-9206, equal to those of Alpamayo.
- **Training:** Qwen-Drive's training list is not published, so it cannot be verified that the 400 clips were left out of it. Declared as a limitation.
- **Horizon:** Qwen's native one is 5 s (50 points at 10 Hz) and is the primary analysis. The sensitivity uses the common 4 s horizon (the first 40 points of the plans and of the recorded future), with the same classifier.
- **Classifier:** the `decisions` function of the trajectory-decision-gate module, with the thresholds of addendum 3 (master protocol) unchanged, current speed from the records file.
- **Lateral sign:** checked against the recorded future (section 5, R1).
- **Tests:** margins, bootstrap and TOST of the paper's FROST analysis (section 4).

## 3. Estimator

That of the Qwen-Drive analysis script (addendum 12 of the master protocol), reused without change in logic:
- **Unit:** the clip. Metrics per clip over the 6 seeds: spread (mean of pairwise distances between seeds, averaged over the horizon), ADE_1 (mean of the 6 ADEs against the recorded future), minADE_6, flip rate.
- **Flip:** the class (longitudinal, lateral) of the plan differs from the class of the Heun-32 tau=1 reference with the same noise. Longitudinal flip and lateral flip: only the corresponding component differs.
- **tau_eff:** the tau at which the mean spread of the Heun-32 curve equals that of the sampler, by linear interpolation on the grid tau in {1; 0.9; ...; 0.3}.
- **Residual:** sampler metric minus that of the curve interpolated at the same spread. **Effect:** sampler minus tau=1 reference.
- **Bootstrap:** 2000 clip resamples (with replacement, the same resamples for all samplers, horizons and metrics), seed **20261102**, redoing spread, tau_eff and residuals in each resample.
- **Intervals:** percentile 95% CI for estimates; percentile 90% CI for the TOST.
- The CIs condition on the 6 seeds and on the fixed reasoning traces, as in the paper.

## 4. Margins and tests (those of the Alpamayo FROST analysis, unchanged)

- **ADE_1:** equivalent if the 90% CI of the residual stays within +-0.05 m.
- **Flips** (total, longitudinal and lateral): 90% CI of the residual within +-2 pt.
- **Delta tau_eff** between horizons (margin of the Thor protocol): 90% CI of tau_eff(4 s) - tau_eff(5 s), paired on the same resample, within +-0.03.
- **minADE_6:** reported without a test, as in the paper.

**tau_eff outside the grid.**
- If the point spread of the sampler falls outside [spread(tau=0.3); spread(tau=1)], tau_eff is reported as "< 0.3" or "> 1", and residuals are not tested (no extrapolation).
- If the point falls inside, but fewer than 95% of the resamples fall inside, tau_eff and residuals are reported (with the fraction) and the TOST is marked "not interpretable" (np.interp truncates at the edges).

**Exclusions.** Only by structure, never by result: clips with `error` in the records file, missing keys or non-finite values. Duplicates by clip_id: the first range is kept (c0 < c1 < c2 < c3). In the structural check: 0 exclusions, 0 duplicates, N = 400.

**Distinct units.** For each flip residual, I report how many distinct clips have at least one flip in the sampler and in the reference at the neighboring temperature, so that an effect carried by few clips is visible.

## 5. Predictions

**Checks (before the tests):**
- **R0 (reference):** Heun-64 against Heun-32 at tau=1 on 400 x 6: max <= 1 cm and equal classes in >= 99.5% of pairs. If it fails, the analysis proceeds and the report says that the Heun-32 curve is not converged on these data.
- **R1 (lateral sign):** in clips with |y of the recorded future at 4 s| > 1.75 m, the sign of y of the mean of the 6 E10 plans at 4 s agrees with that of the future in >= 80%. If it fails, the analysis stops and the axis is investigated before any verdict.

**Primary (E10, factory sampler, native 5 s horizon):**
- **Q1a:** tau_eff(E10) >= 0.90, inside the grid.
- **Q1b:** ADE_1 residual of E10 equivalent (+-0.05 m).
- **Q1c:** flip residual (total) of E10 equivalent (+-2 pt).
- **Q1d (comparable across models, rule 2b):** longitudinal flip residual of E10 equivalent (+-2 pt).

Qwen-Drive enters the paper's count "E10 passes FROST" only if Q1b and Q1c hold. The comparison with Alpamayo uses Q1b and Q1d.

**Secondary:**
- **Q2a:** tau_eff(E4) between 0.35 and 0.65, inside the grid.
- **Q2b:** ADE_1 residual of E4 positive, with 95% CI above zero (E4 pays discretization error; v2: +0.041).
- **Q2c:** flip residual (total) of E4 positive (point estimate > 0).
- **The E4 TOSTs** (ADE_1, total, longitudinal and lateral flips) have no prediction of direction. They are reported, and the verdict enters the scorecard whatever it is.
- **Q3:** E1 and E2 with spread below that of the curve at tau=0.3: tau_eff "< 0.3", no residual tested.
- **Q4 (cooling does not change the error):** the effect of cooling on the curve, ADE_1(tau_eff(E4)) - ADE_1(tau=1), has a 90% CI within +-0.05 m.
- **Q5 (lateral flips of E10):** residual equivalent (+-2 pt). Reported separately because the command fixes the lateral intent.

**Horizon sensitivity (4 s, the first 40 points):**
- **H1:** |tau_eff(4 s) - tau_eff(5 s)| equivalent within +-0.03 for E10 and E4.
- **H2:** the verdicts of Q1b-Q1d are the same at the 4 s horizon.

## 6. What counts as passing or failing

- Each prediction is judged as written above, without reinterpretation after seeing the numbers.
- A prediction fails when the criterion is not met. A "not interpretable" TOST (tau_eff outside the grid in more than 5% of resamples) counts as **not passed**.
- Failures are reported in the results document and in the general findings report (section 5, if the failure may be wrong).
- Analyses made after this registration and not listed here are marked as exploratory.

**Very low flip rate.** Qwen-Drive has a small spread (v2: 0.136 m), so flips may be rare, and equivalence at +-2 pt may be of little information. This does not change the verdict, but the report includes the flip rate of the reference at the matched temperature and the number of distinct clips with flips.

## 7. Execution

- **Script:** the Qwen-Drive v3 analysis script, which imports `clip_table` and `matched` from the Qwen-Drive analysis module and adds the horizon, the separated flips and the R0/R1 checks.
- **Output:** a `summary.json` and a source snapshot in the Qwen-Drive v3 analysis output folder (2026-11-02).
- **Results:** the internal results document on decision and information, and the general findings report.

The result will be reported whatever it is.

## Addendum 1 (02/11/2026 10:31 PST, from `date` and from the commit): Qwen-Drive Heun-3/Heun-5 control

Registered **before running**. No Heun-3 or Heun-5 plan of Qwen-Drive exists; nothing was computed on them.

**Declaration.** This is a control added **after** the main v3 analysis (sections 1-7, Qwen-Drive v3 analysis output of 2026-11-02), whose numbers I have already seen: E10 tau_eff 0.98 and residuals ~ 0; E4 tau_eff 0.40 with flip residual +1.7 pt; flips of the reference curve between 0 (tau=1) and 1.4% (tau=0.3); ADE_1 of the curve almost constant (1.167-1.173 m); reference spread 0.151 m. The reason is to fill the Heun-5 cell of Qwen-Drive in table 1 of the paper with a registered control, like those of Alpamayo (addendum 15 of the master protocol). The predictions below start from these numbers.

**Run.**
- Script: the Qwen-Drive Heun-control v3 script (imports `build_scene` from the v3 probe and `plan` from v2, unchanged). Job: a cluster batch script, `gpu` partition (A100, as v3), 4 ranges of 100 clips equal to those of v3.
- Same 400 clips, scenes, command (SimWAM rule), reasoning_planning mode with greedy reasoning, fp32 expert, seeds 9201-9206 and initial noise (`_initial_noise` per seed, equal in every sampler).
- Configurations, all at tau=1: Heun-3 (6+1 evaluations), Heun-5 (10+1), and again E10 and Heun-32 (reference), for the R2 check and to have a reference from the same run.
- The reasoning is regenerated (greedy). The trace stored in `reasoning.json` cannot be used to rebuild the cache: it loses the thinking block, and the cache of `generate` is not that of a prefill.

**R2 check (reproduction, before the tests).** On the 400 clips: (a) reasoning trace equal to the stored one in >= 99% of clips; (b) regenerated E10 and Heun-32 tau=1 against the stored ones over the 2400 clip x seed pairs: median difference <= 1 mm and equal classes in >= 99.5% of pairs. I will look at (a) and (b) on the first clips of each range as soon as they come out; if they do not match, I cancel and investigate before proceeding.
- **R2 passes:** the Heun-k plans enter the v3 tables as one more sampler. tau_eff, spread and residuals against the stored Heun-32 curve, and flips against the stored Heun-32 tau=1 reference (the estimator of section 3 unchanged).
- **R2 fails:** the Heun-k flips are measured against the Heun-32 tau=1 of the same run, the convention of the Alpamayo controls (addendum 15). tau_eff and ADE_1 residual remain against the stored curve. The report says that R2 failed and by how much.

**Estimator, bootstrap and margins.** Those of sections 3-4 unchanged: clip as unit, 6 seeds, 2000 resamples with seed 20261102 (the same resamples as the main analysis), tau_eff re-resolved in each, 95% CI for estimates, 90% CI for the TOST, margins +-0.05 m (ADE_1) and +-2 pt (flips). The native 5 s horizon is the main one; 4 s is reported without prediction. Distinct units: clips with >= 1 flip in Heun-k and in the neighboring curve.

**tau_eff above the grid.** A second-order integrator may not shrink, and the Heun-k spread may fall above that of the curve at tau=1 (in Alpamayo-1.5 Heun-5 fell outside the grid from above). Two readings, both reported:
- **Rule of section 4 (the registered one):** tau_eff "> 1", TOST "not interpretable", counts as not passed.
- **Table 1 cell:** uses the same convention as Alpamayo-1.5 in the table: tau_eff truncated at 1.00 and residual against the curve at tau=1 (= effect), with the same bold criterion (90% CI of the residual outside the margin). The fraction of resamples inside the grid goes along.

**Predictions.** The Alpamayo prediction ("tau_eff >= 0.85 and flip residual above 2 pt") is not what I expect here. In Qwen, the whole flip curve goes from 0 to 1.4 pt between tau=1 and 0.3, and even E4, with tau_eff 0.40, leaves only +1.7 pt of residual. In Alpamayo, the raw flips of Heun-5 are about 0.6-0.7 times those of E4 (8.5-11% against ~15%). By the same ratio, Qwen's Heun-5 would be near 2% raw flips, with the reference at tau ~ 1 near 0. This is the edge of the margin, so I register as prediction the side I find most likely, with that caveat.
- **QH1:** tau_eff(Heun-5) >= 0.85, counting "> 1" as met. The same for Heun-3. (Alpamayo: 0.96-1.00; Heun is second order and barely shrinks.)
- **QH2:** flip residual (total) of Heun-5 greater than that of E10, with 95% CI of the paired difference (same resamples) above zero. The Heun-5 error is not a temperature error, and E10 has residual ~ 0.
- **QH3:** flip residual (total) of Heun-5 with point estimate between 0 and +2 pt (below the margin). By the bold criterion, I predict that the cell is **not** bold, but with low confidence: the 90% CI may cross +2 pt even with the point below.
- **QH4:** ADE_1 residual of Heun-5 equivalent (90% CI within +-0.05 m). Qwen's spread (0.15 m) is small next to ADE_1 (1.17 m), so the non-thermal error of Heun moves ADE_1 little.
- **QH5:** flip residual of Heun-3 greater than that of Heun-5 (point estimate). No margin prediction for Heun-3.
- The longitudinal and lateral flip TOSTs and the 4 s horizon are reported without prediction.

**Analysis.** The Qwen-Drive v3 Heun analysis script, a wrapper that imports the functions of the v3 analysis script without changing the original. Output: a `summary.json` in a Heun-analysis folder named by date. Results in the internal results document on decision and information and in the general findings report (E67). Reported whatever the result.

## Addendum 2 (02/11/2026 14:02 PST, from `date`): Qwen-Drive Euler-6

Registered **before running**. No Euler-6 plan of Qwen-Drive exists; nothing was computed on it.

**Declaration.** Another control added **after** the main v3 analysis and addendum 1, whose numbers I have already seen: E4 tau_eff 0.40 [0.38; 0.41], spread 0.064 m, ADE_1 residual +0.025 m (90% CI 0.018-0.032, equivalent) and flip residual +1.7 pt (90% CI 0.9-2.7, **not** equivalent); E10 tau_eff 0.98, residuals ~ 0; reference spread at tau=1 0.151 m; Heun-3/5 above the grid. The reason is to have in the Qwen-Drive panel of fig. 4 the same Euler columns as Alpamayo (E2/E4/E6/E10; Qwen's E2 falls below the grid). The predictions below start from these numbers.

**Run.**
- Script: the Qwen-Drive Heun-control v3 script without code change, only `--configs euler:6:1,euler:10:1`. Job: the same cluster batch script with an optional third argument (the list of configurations; without it, the default of addendum 1). `gpu` partition (A100) or `test` (A100, <= 1 h, resuming through `--resume`), whichever starts first; 4 ranges of 100 clips equal to those of v3.
- Same 400 clips, scenes, command (SimWAM rule), reasoning_planning mode with regenerated greedy reasoning, fp32 expert, seeds 9201-9206 and initial noise (`_initial_noise` per seed, equal in every sampler). Configurations, all at tau=1: Euler-6 and again Euler-10 (R2 check). Heun-32 is not regenerated (65 evaluations per plan); in addendum 1 the regenerated E10 and Heun-32 came out bit-identical to the stored ones.

**R2 check (reproduction, before the tests).** That of addendum 1, with E10 only: (a) reasoning trace equal to the stored one in >= 99% of clips; (b) regenerated E10 tau=1 against the stored one over the 2400 clip x seed pairs: median difference <= 1 mm and equal classes in >= 99.5%. I look at the first clips of each range as soon as they come out.
- **R2 passes:** E6 enters the v3 tables as one more sampler (stored curve, stored Heun-32 tau=1 reference, estimator of section 3 unchanged).
- **R2 fails:** I stop, report by how much it failed, and run again with the Heun-32 tau=1 in the same run, to measure the E6 flips against the reference of the same run (convention of addendum 1). tau_eff and ADE_1 remain against the stored curve.

**Estimator, bootstrap and margins.** Those of sections 3-4 unchanged: clip as unit, 6 seeds, 2000 resamples with seed 20261102 (the same as the main analysis), tau_eff re-resolved in each, 95% CI for estimates, 90% CI for the TOST, margins +-0.05 m (ADE_1) and +-2 pt (flips). The native 5 s horizon is the main one; 4 s is reported without prediction. Distinct units: clips with >= 1 flip in E6 and in the neighboring curve.

**Predictions.** Qwen's spread rises from 0.064 m (E4) to 0.149 m (E10). I do not know where E6 falls in this interval, because with only two Euler points the shape is not determined; in Alpamayo, the tau_eff of E6 falls near the middle between E4 and E10. Therefore the tau_eff prediction is only of order.
- **QE1:** tau_eff(E6) inside the grid and strictly between those of E4 and E10 (points), i.e. 0.40 < tau_eff < 0.98. High confidence. A guess, which does not count as a prediction: 0.6-0.85.
- **QE2:** ADE_1 residual of E6 equivalent (90% CI within +-0.05 m). High confidence: E4, farther away, is already at +0.025 m.
- **QE3:** flip residual (total) of E6 with point estimate between that of E10 and that of E4 (+0.03 to +1.75 pt). Medium confidence.
- **QE4:** flip residual (total) of E6 equivalent (90% CI within +-2 pt). Medium confidence: E4 exceeded the margin only through the upper limit of the 90% CI (2.7 pt), and I expect E6 to be smaller. But Qwen's flips are rare (<= 1.4% on the curve), and the CI is wide near the margin.
- The longitudinal and lateral flip TOSTs, minADE_6 and the 4 s horizon are reported without prediction.

**Analysis.** The Qwen-Drive v3 E6 analysis script, a wrapper that imports the v3 analysis script and the R2 check of the addendum 1 wrapper, without changing the originals. Output: a `summary.json` in an E6-analysis folder named by date. Results in the internal results document on decision and information and in the general findings report (E67). Reported whatever the result.
