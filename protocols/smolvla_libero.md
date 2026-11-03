# Protocol: Numerical Temperature in Manipulation Policies (Closed Loop)

**Summary.** This protocol governs a test of the numerical-temperature thesis outside the Alpamayo family, with closed-loop task success on the LIBERO manipulation benchmark (SmolVLA fine-tuned on LIBERO; pi0.5 fine-tuned on LIBERO). It fixes the policies, the sampler configurations (few-step Euler vs precise Heun at explicit temperature), the measures and the predictions P1-P3, and a falsification criterion. Addendum 1 fixes the grid after a smoke test and before the at-scale rollouts. Manipulation is a secondary domain of the paper.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

Registered on 22/10/2026, before any rollout. Objective: test outside the Alpamayo family, and with closed-loop success rate, the thesis of the internal draft outline of the numerical-temperature paper. Euler-type solvers with few steps are equivalent to reducing the temperature of the initial noise; "fewer steps" gains (Coda, arXiv 2609.21216: SmolVLA with 2 steps raising success) would be a temperature effect.

## Policies and environment

- SmolVLA fine-tuned on LIBERO (`HuggingFaceVLA/smolvla_libero`) and pi0.5 fine-tuned on LIBERO (official LeRobot checkpoint, recorded in the manifest), both frozen, through LeRobot 0.6.1 in an isolated virtual environment.
- LIBERO with the suites recorded in the manifest before execution; same environment seeds for all configurations (paired comparison per episode).

## Configurations (same episode, same environment seed)

1. Official sampler (Euler with the default number of steps) with tau=1.
2. Euler with few steps: k in {1, 2, 4}, tau=1.
3. Explicit temperature with a precise solver: Heun with enough steps to converge (defined by an offline convergence test before the rollouts), tau in {1, 0.8, 0.6, 0.4, 0.2, 0}.
4. The noise temperature is the `temperature` parameter already exposed by the sampler, when it exists; otherwise, the initial noise is scaled by tau.

## Measures

- Success per episode (primary), with CI by task/episode bootstrap.
- Offline, on recorded observations of the official-sampler rollouts: shrinkage ratio (distance to the conditional mean, Euler vs precise) and effective tau by across-seed dispersion, as in Alpamayo.

## Predictions (falsifiable)

- **P1 (shrinkage):** Euler-k samples are closer to the conditional mean than those of the precise solver, with tau_eff increasing in k.
- **P2 (temperature explains success):** the success(tau) curve of the precise solver, evaluated at the tau_eff of each Euler-k, predicts the success of Euler-k. Paired difference |success(Euler-k) - success(precise, tau_eff)| smaller than |success(Euler-k) - success(precise, tau=1)|.
- **P3:** if success increases with fewer steps, then tau<1 with the precise solver reproduces this increase.

**Falsification:** if Euler-k differs from the precise solver at the corresponding tau_eff as much as it differs from tau=1, the temperature thesis does not hold in manipulation and must be restricted to Alpamayo.

## Addendum 1: grid fixed before the at-scale rollouts (22/10, after smoke test)

Smoke test (SmolVLA, LIBERO-Spatial task 0): working pipeline, 9.3 s/episode with 10 environments. Offline probe of the smoke test (72 calls x 4 noises): 100% of the Euler samples shrunk toward the conditional mean; ratios 0.91 (Euler10), 0.77 (Euler4), 0.52 (Euler2), with tau_eff by dispersion of ~0.92, 0.78 and 0.53. P1 confirmed in manipulation (exploratory, one task).

Grid: policy `HuggingFaceVLA/smolvla_libero`; suites libero_spatial, libero_object, libero_goal and libero_10 (all 10 tasks of each); 10 episodes per task (400 per configuration); evaluation seed 1000 for all. Configurations: Euler 10/4/2/1 with tau=1 and Heun16 with tau in {1; 0.9; 0.75; 0.5; 0.25}. Action executed as in the checkpoint (replanning at every step, `n_action_steps=1`). Primary analysis (P2): fit success(tau) of Heun16 per suite and compare the success of each Euler-k with the curve at tau_eff (measured by a separate offline probe) against the success of Heun16 with tau=1. Resumable driver: the robot-temperature grid script.
