# Protocol: Numerical Temperature in Closed Loop (AlpaSim, Alpamayo 1.5)

**Summary.** This protocol governs a closed-loop test of the FROST claim in the AlpaSim simulator with the official Alpamayo-1.5 driver: whether the safety effect seen in non-reactive replay (fewer Euler steps -> fewer collisions, explained by temperature) also appears when simulated agents react and the ego actually drives. It fixes 60 scenes, five sampler configurations, the AlpaSim metrics, the predictions C1-C3 and the paired analysis. Addenda record an infrastructure fix, two invalid scenes, additional comfort/lane metrics registered before computing them, and a validity criterion for the lane-departure metric.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

Registered on 25/10/2026, before any rollout of the grid. Smoke test (1 scene, official sampler): full pipeline working (NuRec renderer, physics, controller, runtime, driver), ~0.8 s per decision, ~200 decisions per scene.

## Question

Does the safety effect observed in non-reactive replay (section 9 of the internal results document on decision and information: fewer Euler steps => fewer collisions, explained by temperature) appear in closed loop, with simulated agents and the ego actually driving?

## Design

- Simulator: AlpaSim (a local copy of the public repository, commit affc2ea), renderer `nre-ga:26.04`, 1-GPU topology, official `alpamayo1_5` driver (4 cameras, no CFG), 1 rollout per scene.
- Scenes: 60 from the `public_2604` suite, only clips of the `test` split of PhysicalAI-AV (459 eligible), drawn with seed 20261026 (a closed-loop scenes file, v1).
- Configurations (injection through environment variables in a sampler module added to the driver; no variables = official sampler):
  1. Euler10, tau=1 (official)
  2. Euler4, tau=1
  3. Heun32, tau=1 (precise flow)
  4. Heun32, tau=0.84 (tau_eff of Euler10 measured open-loop)
  5. Heun32, tau=0.65 (tau_eff of Euler4 measured open-loop)
- AlpaSim metrics: `collision_at_fault`, `offroad`, `offroad_or_collision_at_fault`, `dist_to_gt_trajectory`, `dist_traveled_m`, `min_distance_to_obstacle_m`.

## Predictions

- **C1 (temperature reduces incidents):** incidents(Heun32 tau=0.65) <= incidents(Heun32 tau=1).
- **C2 (Euler4 ~ temperature):** |Euler4 - Heun32 tau=0.65| < |Euler4 - Heun32 tau=1| in `offroad_or_collision_at_fault`.
- **C3:** the same for Euler10 with Heun32 tau=0.84.

Analysis paired by scene (McNemar for binary rates; scene bootstrap for continuous ones). With 60 scenes, power is limited; null results must be reported as such.

**Falsification:** if Euler4 differs from Heun32 at tau_eff as much as from tau=1, the temperature explanation does not hold in closed loop.

## Addendum 1 (26/10, before any result of the grid)

The first execution of the grid failed in all configurations without producing rollouts: GPU JPEG decoding in the driver (`image_decode_device: cuda`, the default of the Alpamayo 1.5 config in AlpaSim) produced CUDA error 710 with 4 concurrent rollouts. We switched to `driver.model.image_decode_device=cpu` (default of the driver schema), which produces the same decoded pixels. Test with 4 concurrent scenes: 4/4 rollouts completed, without errors. The failed attempts are kept in `*.failed.0` folders of the run.

## Addendum 2 (26/10, before looking at metrics of any configuration)

Two scenes of the draw have an invalid route in the NuRec data and are rejected by the runtime in every configuration: `clipgt-1661c375-...` (initial waypoint 27.4 m from the original trajectory, limit 10 m) and `clipgt-964ebe15-...` (route folds over itself, 180 deg). With one scene failing, AlpaSim does not aggregate; the analysis uses the per-rollout metrics (`rollouts/*/*/metrics.parquet`) of the **58 scenes common** to all configurations. A data defect, independent of the sampler.

## Addendum 3 (28/10 ~05h): hard braking and lane departure (request from a collaborator, 27/10)

Registered before computing these metrics in any configuration. To learn the file schema, I opened the metrics summary of a single rollout (scene clipgt-0026d903, E4). The 58 x 5 rollouts already exist; nothing is run again.

**Metrics per rollout** (steps with `eval_relevant`):
- **Hard braking (binary):** an event occurs when the longitudinal acceleration of the ego stays <= -3.0 m/s^2 (~0.3 g) for at least 0.3 s.
  - Speed comes from the ego poses (`rollout.asl`, `actor_poses`), by finite difference of the xy position. Speed and acceleration are smoothed with a 0.5 s moving average.
  - Associated continuous measures: maximum deceleration and RMS of jerk (comfort).
- **Lane departure:**
  - `lane_departure_frac`: fraction of steps with `min_distance_to_lane_boundary_m` = 0, i.e. ego polygon outside the union of possible lanes (continuous);
  - `lane_departure_any`: any sequence of at least 1 s with value 0 (binary).
  - `wrong_lane` is reported only as descriptive, because AlpaSim itself says it has many false positives at intersections.

**Tests:** the same pairs as C1-C3.
- **C1:** tau=0.65 vs tau=1.
- **C2:** E4 vs tau=0.65 and vs tau=1.
- **C3:** E10 vs tau=0.84 and vs tau=1.
- Binary: exact McNemar paired by scene. Continuous: mean difference with 95% CI by scene bootstrap (2000 resamples, seed 20261028).

**Predictions:**
- **C1':** lower temperature has less than or equal hard braking and lane departure.
- **C2' and C3':** Euler is closer to its tau_eff than to tau=1 in each metric.

With 58 scenes, null results are reported as null.

**Addendum 3b (28/10, before running the 290 rollouts):** in the test of the script with the only rollout already opened (E4, clipgt-0026d903), `lane_departure_frac` = 1.0 with no off-road at all. This indicates that the AlpaSim metric may be degenerate when the scene has no lane polygons.

**Validity criterion**, fixed before comparing configurations: if `lane_departure_frac` >= 0.99 in more than half of the Heun-32 tau=1 rollouts, lane departure measured this way is declared invalid and is not interpreted. The numbers are reported as a diagnostic.

**Erratum (28/10):** addendum 3b says "before running the 290 rollouts". The rollouts already existed since addendum 3, and nothing was run again. The correct wording is "before computing the new metrics over the 290 rollouts". Commits (private repository): addendum 3 at 04:47, 3b at 04:48 and the analysis after that.
