# Protocol: Numerical Temperature in a Second Family of Planners (NAVSIM navtest)

**Summary.** This protocol governs the replication of the FROST "few steps = lower temperature" effect in a second family of diffusion-based driving planners (ReCogDrive-2B RL and IL, and DiffusionDrive v1 as a limit case) on 500 NAVSIM navtest scenes. It fixes the models, scenes, seeds, deterministic DDIM sweeps, spread-matched tau_eff analysis, equivalence margins (PDMS, ADE_1, decision flips) and predictions N1-N3. Addenda record two grid defects found in smoke tests, a bug in the decision classifier for 8-point plans, and an additional pre-registered run.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

Registered on 28/10/2026, before any inference of these models on GPU. The setup is described in an internal setup note for the second family.

## Question

In diffusion planners of another family (Xiaomi, HUST), does the deterministic few-step integrator also act as a lower temperature? And does temperature explain what few steps change in PDMS, ADE_1 and decisions?

This was the main criticism in internal review: the positive replication comes from a single family.

## Models

1. **ReCogDrive-2B RL** (primary; the paper reports PDMS 90.8). It is a VLM with a DiT diffusion planner and predicts epsilon. The release uses DDIM with eta=1 (stochastic), 5 steps.
2. **ReCogDrive-2B IL** (secondary; PDMS 86.5).
3. **DiffusionDrive v1** (limit case). Truncated diffusion from anchors, predicts x_0, DDIM with 2 steps and eta=0, and the initial noise has standard deviation 0.03. I predict that the temperature knob is almost inert; the case enters as exploratory.

DiffusionDriveV2 (with a learned selector over 800 candidates) is outside this protocol.

## Data and units

- **Scenes:** 500 from navtest, drawn with seed 20261028 and stratified by log (proportional to the number of tokens per log).
- **Noise seeds:** 6 per scene (9201-9206). In ReCogDrive, the VLM runs once per scene, with the same hidden state for all configurations.
- **Statistical unit:** the scene. Scene bootstrap with 2000 resamples and seed 20261028, re-resolving tau_eff in each resample, as in the paper-numbers script.

## Configurations (ReCogDrive)

- **Reference (temperature curve):** DDIM with eta=0 (deterministic probability-flow) and 50 steps, starting from tau*z, with tau in {1; 0.9; 0.8; 0.7; 0.6; 0.5; 0.4; 0.3}. Convergence is checked against 100 steps at tau=1 (maximum difference reported).
- **Deterministic samplers:** DDIM with eta=0 and k in {1, 2, 5, 10} steps at tau=1.
- **Release sampler:** DDIM with eta=1 and 5 steps at tau=1. Enters as exploratory, because the noise injected at each step is not covered by the paper's theory.
- **Control:** shrink_k, which moves each sample of the tau=1 reference linearly to the spread of E_k, as in the paper.

For DiffusionDrive v1, the release (2 steps) and the reference with 20 steps on the same tau grid, with the anchor noise scaled by tau.

## Metrics

- **tau_eff:** the tau at which the mean across-seed spread of the reference curve equals that of the sampler (linear interpolation).
- **PDMS** per sample (official NAVSIM scorer, navtest metric cache), **ADE_1** against the human trajectory (NAVSIM horizon, 4 s) and decision **flips** against the tau=1 reference under the same noise. The longitudinal and lateral classes come from the `decisions` function of the trajectory-decision-gate module, with the same thresholds.
- **Residual:** sampler minus the reference at tau_eff. Effect: sampler minus the reference at tau=1.

## Predictions (ReCogDrive RL and IL)

- **N1.** DDIM with eta=0 contracts: tau_eff(E_1) < tau_eff(E_2) < tau_eff(E_5) < tau_eff(E_10) <= 1, with tau_eff(E_5) between 0.6 and 0.95.
- **N2.** Residuals of E_5 and E_10 equivalent to zero (90% TOST) with margins of ADE_1 +-0.05 m, flips +-2 pt and **PDMS +-1.0 point**.
- **N3.** The effect of temperature on PDMS (reference at tau_eff against tau=1) has the same sign as the effect of E_5 on PDMS.

**Falsification:** if E_5 or E_10 fail N2 through a residual outside the margin (and not through a wide interval), the temperature reading does not transfer to this family, and the paper says so.

## Order and hardware

Local workstation (Blackwell), one model at a time: ReCogDrive RL, then IL, then DiffusionDrive v1. Starts after the Heun controls of R1 (addendum 15 of the master protocol). PDMS scoring runs on CPU (ray) and the metric cache is shared.

## What does not change after seeing data

Scenes, seeds, configurations, margins and predictions. Any change becomes a dated addendum before the analysis it affects.

## Addendum 1 (28/10 ~09:30, after the 1-scene CPU smoke test and before any GPU run)

The smoke test showed two defects in the step grid inherited from the original code.

**1. ReCogDrive: "leading" DDIM spacing.**
- The original uses t_i = i*(100//k), i = 0...k-1. With k=1 the only step is t=0 from pure noise: in the smoke test, PDMS 0 and ADE_1 ~ 33 m. With k=2 the integration starts at t=50 and not at maximum noise.
- This is a grid defect, not few integration steps, and does not test the question.
- **Decision:** the deterministic sweep (eta=0) uses *trailing* spacing, t_i = round(T - i*T/k) - 1, so that every k starts at t = T-1. This is the usual correction for DDIM with few steps.
- The 50-step reference and the 100-step check also use trailing.
- Two configurations are reported separately and marked: (a) the release (eta=1, 5 steps, leading), exploratory as before; (b) DDIM eta=0 with 5 steps on the release's leading grid, to isolate the effect of eta.
- Predictions N1-N3 hold for the trailing sweep.

**2. DiffusionDrive: reference.**
- The noise is truncated at t=8 and the release does 2 steps (t = 8 -> 0). The `step_num` rule of the original with 20 steps traverses t = 19...0, which does not refine the same interval.
- **Decision:** the reference is DDIM eta=0 over all integer timesteps from 8 to 0 (9 steps), the finest the discrete scheduler allows. Convergence is checked against 5 steps in the same interval, with the maximum difference reported.
- Since the reference is necessarily coarse, DiffusionDrive remains an exploratory limit case.

No other element changes: scenes, seeds, metrics and margins.

## Addendum 1b (28/10, after the second CPU smoke test and before any GPU run)

With the trailing grid, the E_1 of ReCogDrive remains degenerate: ADE_1 31.9 m and PDMS 0 in the smoke test. The cause is no longer the grid. The schedule is cosine, with alpha_bar(t=99) = 2.4*10^-7, and in an epsilon parametrization the inversion epsilon -> x_0 in one step divides by sqrt(alpha_bar) ~ 5*10^-4 and saturates at the clip +-1. In a model that predicts epsilon, one step does not return the conditional mean, unlike the velocity models of the paper.

**Decision:**
- E_1 leaves prediction N1, which becomes tau_eff(E_2) < tau_eff(E_5) < tau_eff(E_10) <= 1, with tau_eff(E_5) between 0.6 and 0.95.
- E_1 continues to be run and reported as degenerate in this parametrization.
- Nothing else changes.

## Order note (28/10, 13h, before any GPU run)

NAVSIM now runs before the Heun controls of R1 (addendum 15 of the master protocol), right after the A1.5 extension. Only the execution order changes; nothing in the protocol changes.

## Addendum 2 (28/10, ~15h; after the first RL analysis and the investigation recorded in an internal NAVSIM RL investigation note)

**Situation.** This addendum comes after seeing the RL results and is not blind. We have already seen the registered flips and an exploratory corrected variant produced by the investigation. The registered analysis (v1) continues to be reported without change.

**1. Bug in the decisions.**
- The `decisions` function of the trajectory-decision-gate module assumes plans of 64 points at 0.1 s, as in Alpamayo. NAVSIM plans have 8 points at 0.5 s.
- As a result, `plan_speed` slices an empty interval, the speed becomes NaN and the longitudinal class never leaves "keep" or "stop".
- **Fix:** prepend the origin (0, 0) at t = 0 and linearly interpolate x and y to the 0.1 s grid up to 4 s (40 points), before calling `decisions` without any change. With this, the speed window (2-3 s) and the heading (last 0.5 s) have the same meaning as in Alpamayo. The remaining difference is the horizon: lateral displacement is measured at 4 s, against 6.4 s in Alpamayo.
- **Report:** the registered flips (v1) and the corrected ones side by side. N2 is judged with the corrected ones; the registered ones remain as reported.

**2. Sensitivities (exploratory):**
- tau_eff per axis (longitudinal x and lateral y spread separately).
- tau_eff against the 100-step reference. It is available only at tau = 1, so we assume the spread is linear in tau, as the 50-step curve shows (ratio within 1%).

**3. New GPU run (pre-registered before running):**
- **Reason:** the trailing grid starts at t = 99, where alpha_bar = 2.4*10^-7. In addition, the RL may have been tuned only at the timesteps of the release grid (80, 60, 40, 20, 0); this is a hypothesis, not verified.
- **Sweep:** upstream *leading* grid (t_i = i*floor(100/k)) with eta = 0, k in {2, 4, 5, 10, 20, 50}, at tau = 1.
- **Reference:** 100 steps (all timesteps; here leading and trailing coincide) on the same tau grid as the protocol.
- **Models and data:** RL and IL, same scenes and seeds; `--verify-release` on.
- **Predictions:**
  - **N1':** tau_eff grows with k on the leading grid, with k >= 4 inside the range of the curve.
  - **N2':** leading E5 and leading E10 are equivalent to the corresponding temperature in PDMS and ADE_1 (protocol margins). On the corrected flips, with a 2 pt margin, we do **not** predict equivalence: the two configurations already seen were above.
  - **H-RL (exploratory):** outside the release timesteps, the RL departs more from the curve than the IL.

Scenes, seeds, metrics and margins do not change.
