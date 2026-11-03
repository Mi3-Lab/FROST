# Numerical-Error Gate in Generative Policies

**Summary.** This is the earliest protocol of the study, registered before the first native-solver probe (the Alpamayo1.5 solver probe on two CARLA windows). It governs the question of whether numerical error in the ODE solver of a frozen flow-matching driving policy affects the quantities used for decisions (terminal advance, lateral displacement, distance traveled, speed profile), and under which conditions a learned error estimator would be justified. It fixes the solver grid, the references, the controls to run before any training, and the conditions for continuing. The follow-up multi-episode gate is in frost_main.md.

*Translated from the original registration; timestamps are the original registration times (the original carries no explicit date; the registration precedes the first native-solver probe of 2026-10-22).*

---

Registered before the execution of the first native-solver probe (2026-10-22). Candidate to investigate: learn/estimate the error in the quantities used for deciding, instead of choosing numerical effort only by the norm/cosine of the velocity field. Dual-weighted residuals and goal-oriented adaptivity are classical techniques; ProbeFlow, FlashDrive and FlowCast already accelerate samplers. Applying them or adding a loss does not establish novelty.

First test: frozen Alpamayo1.5, images from four cameras with four real synchronous frames at 10 Hz, 16 past poses consistently converted to a right-handed system. Two windows of a single CARLA recording; not two independent scenes. Simulated data have domain shift. Language state and initial noise of the action expert equal across solvers. No camera zeroed.

Euler with 4, 8, 10, 20, 40, 80 steps; Heun with 4, 10, 20, 40 steps, counting two evaluations per step. Two action seeds per context. Eighty steps are a numerical reference, not the correct action; measure 40 vs 80 to assess convergence. The published ten-step configuration is also kept explicit as a distinct behavioral reference. No fine-tuning, safety or speedup is established at this stage.

Store actions, trajectories, measured cost, states and fields of the Euler10 integration. Initial quantities: terminal advance, terminal lateral displacement, distance traveled and speed profile. Examine whether scalar error and global trajectory error have different orderings. Do not retrospectively select a favorable indicator and call it a decision gain.

Only continue to a learned method if there is relevant error that same-cost controls do not solve and if the cost of estimating/correcting the error can be lower than simply solving the ODE better. Compare with Euler/Heun, fixed schedules, ProbeFlow/cosine and caching before claiming advance. A numerical guarantee preserves the specified reference behavior; it does not certify the physical safety of the policy.

## Addendum after v2: reference not yet converged, and precision

Euler40 still diverges from Euler80, especially at seed 9202/context 1. Therefore, 80 steps cannot be assumed to be a converged solution. We add Euler40/80/160/320 and Heun20/40/80/160, first in BF16 and then with the expert/projections/cache promoted to FP32, keeping the VLM frozen in BF16. Using seed 9202 in both contexts is a post hoc diagnostic selection, not an independent replication. The precision promotion changes the implemented field and must not be confused with changing only the solver. Compare convergence within each precision and then the sensitivity across precisions.

## Addendum of controls before any training

The convergence extension showed that Heun helps, and promoting the expert to FP32 does not eliminate the hard case. We add classical controls: Euler with grid t=u^2, Euler with t=(1-cos(pi*u))/2, with 4/8/10 evaluations; RK4 with 1/2/5/10 steps (4/8/20/40 evaluations); Heun5 to compare exactly with Euler10. These grids are simple fixed controls, not reproductions of ProbeFlow or FlashDrive. Seed 9202 remains diagnostic. Measure the costs without competing with other new jobs.
