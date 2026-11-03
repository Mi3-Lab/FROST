# Fair Comparison Across Models in FROST

**Summary.** This document fixes the rules that make cross-model comparisons in FROST fair: each model runs in its own official way, all models receive the same information, each model has a converged reference, and metrics, clips, seeds, horizons and tests are the same. A shortcut that an agent finds does not count if it is not fair to the model being compared with the others. The rules apply to Alpamayo, SimWAM, Cosmos 3 Edge, Qwen-Drive and any new model, and must be met before any main-run pre-registration. The Qwen-Drive protocol (qwen_drive.md) cites it item by item.

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY.*

---

Written on 31/10/2026, at the request of the authors. The comparison must be fair to all models: a shortcut that an agent found does not count if it is not fair to the model being compared with the others. These rules hold for Alpamayo, SimWAM, Cosmos 3 Edge and any new model, and must be met before any main-run pre-registration.

## 1. Each model runs the way its authors run it
- **Inputs:** cameras, number of frames, resolution, history and ego state in the format of the model's own official evaluation or inference pipeline. If the official one uses N frames, we use N. We do not reduce the input for convenience.
- **Ego state:** speed, acceleration and history computed by the same preprocessing function of the model, over the recorded egomotion. We do not use our own approximations (for example, a quadratic fit) when the official computation exists.
- **Factory sampler:** the model's default (steps, solver, grid, shift) defines the "release". Configurations that the group itself uses in deployment, such as Cosmos with 8 steps, enter as additional samplers.

## 2. The same information for all
- **Future information:** no model receives information derived from the recorded future that the others do not receive.
- **Current situation:** Alpamayo runs without navigation. SimWAM, by construction (official training and evaluation, inherited from DriveWAM), receives a one-hot command taken from the intent of the future path. The "UNKNOWN" position appears in only 0.4% of training, so it does not serve as neutral.
- **Rule for SimWAM:**
  - **(a) Main analysis:** official command (the model used as it was made). FROST compares samplers within the model, with all receiving the same command.
  - **(b) Comparison across models:** since the command fixes the lateral intent, decision flips are reported separated into longitudinal and lateral for **all** models. The verdict comparable across models uses the **longitudinal flips**, which no command fixes.
  - **(c) Sensitivity without future information:** "straight" command, the training mode (83%), in all clips. Report that this biases the lateral decision toward straight.
- **Published benchmarks:** SimWAM's own table compares with Alpamayo-1.5 in this unequal condition. We do not reproduce that comparison without the caveat.

## 3. Converged reference for each model
- **What the reference is:** the solution of the model's own ODE, in its own parametrization (for example, the sigma grid with shift of Cosmos), starting from tau*z.
- **Convergence:** shown in the decisions (class agreement under the same noise) and in the distance of the plan, with increasing steps (Heun-32/64/128/256 or an adaptive solver with tight tolerance). Only after that is the reference used.
- **Grid change:** the reference grid is not changed because it "converged worse" without demonstrating the convergence of the chosen one.
- **No convergence:** if the reference does not converge at some tau, the tau_eff of the model is declared ill-defined in that range. This is reported; it is not worked around.

## 4. Same metrics, same units
- **Clips and seeds:** the same 400 held-out clips of PhysicalAI-AV and the same 6 seeds, when the model runs on those data. Beforehand, check that the clips are not in the model's training set (SimWAM: 0 of 400).
- **Horizon:** the main analysis uses the native horizon of each model (Alpamayo 6.4 s, Cosmos 6 s, SimWAM 4 s). As a sensitivity analysis, all models at the **common 4 s horizon**, with the same decision classifier and the same conversion of units and axes. The lateral sign is checked against the recorded future.
- **Tests:** same margins, same clip bootstrap and same TOST.

## 5. Before the main run
The pre-registration of each model cites this document and says how it met each item. Deviations are written down with the justification.

## 6. Findings from reading the papers (31/10 22:11 PDT; internal paper-reading note on the models)
- **Cosmos 3 Edge:** it has no documented AV policy; the report evaluates AV only as inverse dynamics, and the NIM documentation says "AV policy cases need a general-purpose nano variant". An AV policy on Edge is an extrapolation, and the pilot on the Thor was stopped. The fair version is **Cosmos 3 Nano** (16B, ~35 GB in bf16), with the official inputs: tier 480, image conditioning and the maneuver prompt, which is navigation information and falls under rule 2.
- **Alpamayo:** the release accepts navigation (`nav_text`), and we run without it. In R1, minADE_6 is 0.794 m with route and 0.955 m without. Equality of information with SimWAM, which receives a command taken from the future, requires one of two things: (i) run both without future information, or (ii) give Alpamayo a navigation taken from the future by the same rule. While this is not decided, the comparable verdict uses the longitudinal flips (rule 2b).
- **SimWAM:**
  - Official speed and acceleration are the fields recorded in the rig, rotated to the anchor frame. Our reconstruction reproduces its jsonl to the 4th decimal.
  - The official command is the heading of the last segment of the future, with threshold 0.31 rad; it matches the training labels in ~94% of cases.
  - The official sampler is Euler-10 with shift 5, no CFG.
  - The official PhysicalAI test uses 1000 clips curated for rare scenes. In its table, Alpamayo runs without command.
- **Qwen-Drive:**
  - The v2 sampler and reference are correct.
  - The 60 clips used are 57/60 in the Alpamayo training partition, and Qwen's training list is not published.
  - The ADE_1 of our reference (1.55 m) does not match that of the paper (1.11-1.27 m), without explanation.
  - For the comparison table: redo on the 400 held-out, with the 6 seeds and the 4 s horizon, and first reproduce the paper's ADE.
