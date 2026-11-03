# FROST Pre-Registration Protocols

English translations of the pre-registration protocols used by the FROST paper. Each file keeps the original structure, addendum numbering, predictions, margins and registration timestamps; only the language was changed, and personal names, machine/cluster names, addresses, local paths and private-repository commit hashes were removed or generalized. Dates are DD/MM/YYYY; times are the original registration times (PDT/PST as noted).

| File | Study | Registration dates | Paper results governed |
|---|---|---|---|
| `frost_main.md` | Master protocol: numerical error to command gate (CARLA), PhysicalAI-AV real clips, decision classes, commitment time, schedules, explicit vs implicit temperature, FlashDrive, Qwen-Drive v2, held-out replication, converged references, scaled-noise control, low-tau and heated (tau > 1) samplers; Addenda 2-22 | 22/10/2026 to 02/11/2026 | Core FROST results: tau_eff of few-step samplers, ADE_1 and decision-flip equivalence (TOST) for Alpamayo-1.5, R1 and A2, Heun-3/5 controls, low-tau extension, converged-reference checks, scaled-noise control, heated-sampler validation |
| `qwen_drive.md` | FROST on Qwen-Drive-1.0-4B, 400 held-out clips (v3), plus Heun-3/5 (Addendum 1) and Euler-6 (Addendum 2) controls | 02/11/2026 (04:53, 10:31 and 14:02 PST) | Qwen-Drive column of the cross-model scorecard and Table 1; Qwen-Drive panel of the E_k figure |
| `navsim.md` | Second planner family (ReCogDrive RL/IL, DiffusionDrive v1) on 500 NAVSIM navtest scenes; PDMS, ADE_1, flips | 28/10/2026 (Addenda 1, 1b, 2 the same day) | Second-family replication (PDMS-based temperature test) |
| `wod_e2e.md` | Alpamayo-1.5 on the WOD-E2E long tail with the rater-feedback score (RFS-proxy) | 28/10/2026 23:01 PDT (header says 29/10 ~01h30; corrected in the text); Addendum 1 on 29/10 | Long-tail / rater-score replication |
| `closed_loop_alpasim.md` | Closed-loop test in AlpaSim (60 scenes, 5 sampler configurations); hard braking and lane-departure metrics | 25/10/2026; Addenda 26/10 and 28/10 | Closed-loop safety/temperature result |
| `thor.md` | DRIVE AGX Thor: latency per sampler, effect of FP8/NVFP4, TensorRT engines and compiled VLM on tau_eff | 29/10/2026 14:59 PDT; deviation 30/10; Addenda 2-3 on 30/10 and 31/10 | Deployment results (latency; quantization does or does not shift tau_eff) |
| `smolvla_libero.md` | Manipulation policies (SmolVLA, pi0.5) on LIBERO: temperature vs success rate | 22/10/2026 | Secondary-domain (manipulation) result |
| `fair_comparison.md` | Rules for fair cross-model comparison (official inputs, same information, converged reference, same metrics) | 31/10/2026 | Cross-model comparison design; cited by `qwen_drive.md` |
| `decision_flip_gate.md` | Earliest protocol: numerical-error gate in generative policies (native-solver probe, convergence and control addenda) | before 22/10/2026 (first native-solver probe) | Motivation and solver-convergence controls for the decision-flip analysis |

## How pre-registration was used

1. Each protocol was written and committed before the data collection or analysis it governs; predictions, margins (e.g. ADE_1 +-0.05 m, flips +-2 pt, delta tau_eff +-0.03) and seeds are fixed in the text.
2. Deviations, infrastructure changes and post hoc extensions are recorded as dated addenda or execution notes (including corrections of mis-stated registration times, and explicit labels such as "exploratory" or "post hoc"), never by editing earlier text.
3. Predictions that failed are reported as failures in the paper; the addenda state in advance what the paper will say if a prediction fails.
