# Addendum 19 analysis (E10 flip equivalence against the converged reference, 400 clips)

Registered in protocols/frost_main.md, Addendum 19 (30/10/2026 00:33 PDT).
Run 30/10/2026 08:48 PDT, CPU only:

    python -m frost.analysis.convergence_a19_analysis \
        --out artifacts/convergence_a19_analysis_v1_20261030

Inputs: artifacts/temperature_probe_a15_heldout_pooled400_v3_20261028 (E10, E4, Heun-32; clips 0-399),
artifacts/convergence_a18_a15_heldout_v3_20261029 (Heun-64, clips 0-99) and
artifacts/convergence_a19_a15_heun64_heldout_v3_20261030 (Heun-64, clips 100-399). 400 clips, seeds 9201-9206,
2000 clip resamples, seed 20261031 (joint across curves). Console output in ../convergence_a19_analysis_v1_20261030.log.

Verdicts: D1 holds (E10 flip residual vs Heun-64 +0.96 pt, TOST90 [0.35, 1.63]).
D2 holds (ADE1 residual +0.007 m, TOST90 [-0.001, 0.015]).
D3 holds (tau_eff 0.853 vs Heun-64, 0.850 vs Heun-32; difference 0.003).
Exploratory: Heun-64 on {1,.9,.8,.7}: flips +0.74 pt, TOST90 [0.09, 1.41]; same verdicts.
