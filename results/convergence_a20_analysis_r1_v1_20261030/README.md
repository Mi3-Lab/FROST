# Addendum 20 analysis, Alpamayo-R1 (E10 against the converged Heun-64 reference, 400 clips)

Registered in protocols/frost_main.md, Addendum 20 (30/10/2026 09:10 PDT).
Run 30/10/2026 20:48 PDT, CPU only (A2 run still on the GPU):

    python -m frost.analysis.convergence_a20_analysis \
        --model r1 --out artifacts/convergence_a20_analysis_r1_v1_20261030

Inputs: artifacts/convergence_a20_r1_heun64_heldout_v3_20261030 (merged A20 run: Heun-64 tau {1,.9,.85,.8,.75,.7}
and E10 tau=1, clips 0-399) and artifacts/temperature_probe_r1_heldout_pooled400_v3_20261028 (Heun-32 grid, pooled400
E10). 400 distinct clips, seeds 9201-9206, 2000 clip resamples, seed 20261101 (joint across curves and samplers).
Console output in ../convergence_a20_analysis_r1_v1_20261030.log.

The A20 E10 and the pooled400 E10 are bit-identical on all 400 clips (R1 pooled400 ran on the same workstation),
so primary and sensitivity coincide.

Verdicts: F1 holds (E10 flip residual vs Heun-64 +0.90 pt, TOST90 [0.13, 1.65]).
F2 holds (ADE1 residual +0.0002 m, TOST90 [-0.013, 0.015]).
F3 holds (tau_eff 0.827 vs Heun-64, 0.830 vs Heun-32; difference -0.003, 90% CI [-0.004, -0.003]).
Exploratory: Heun-64 on {1,.9,.8,.7}: flips +0.87 pt, TOST90 [0.12, 1.59]; same verdicts.
