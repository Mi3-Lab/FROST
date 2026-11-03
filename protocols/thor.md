# Thor Protocol v1: Latency per Sampler and Effect of Deploy Precision on tau_eff (DRIVE AGX Thor)

**Summary.** This protocol governs the on-vehicle-computer evaluation of FROST on an NVIDIA DRIVE AGX Thor board: (a) the latency of each sampler (E_1, E_2, E_4, E_10, Heun-32) per field call and per plan, and (b) whether deploy precision (FP8 and NVFP4 quantization of the action expert, later TensorRT engines and a compiled VLM) shifts the effective temperature tau_eff or opens residuals in the matched-temperature test. It fixes the board environment and forced deviations, the precision variants, the 60-clip sample, the analysis, margins and predictions T1-T5. Addenda extend the study to TensorRT engines on the 400 held-out clips (Addendum 2) and to a compiled VLM prefix (Addendum 3).

*Translated from the original registration; timestamps are the original registration times. Dates are DD/MM/YYYY. The board is a shared device; machine names, addresses and local paths were removed.*

---

**REGISTERED on 29/10/2026 14:59 PDT, approved by the authors** (instruction: keep monitoring the Thor to start the experiments there). Up to here, the only Alpamayo-1.5 output on the Thor is the smoke test described in section 0. Sections 1-8 are fixed from this registration. The scripts marked "to be written after registration" (latency, quantization P1-P3 and analysis) implement what is described here, without changing predictions, margins or sample.

## 0. Environment (prepared on 29/10)

**Software:**
- own virtual environment on the board's SSD, with the Thor wheels `torch-2.8.0+thor101` and `torchvision-0.23.0+thor101`, both from the board's local build directory. The shared environment was not touched.
- transformers 4.57.1, accelerate 1.13.0, safetensors 0.7.0, tokenizers 0.22.2, einops 0.8.2, hydra-core 1.3.2 and numpy 2.2.6, in the same versions as the workstation.

**Model and data:**
- Weights: `models/Alpamayo-1.5-10B`, copy of snapshot `7aba829`. Sizes and sha256 of config and index match the workstation.
- HF offline, with only the config, tokenizer and processor of Cosmos-Reason2-8B, Qwen3-VL-8B and Qwen3-VL-2B. There is no HF token on the board.
- Data: context_000-059 of held-out v3, plus `records.json`.

**Deviations forced by the board (declared):**
1. **Attention:** the config asks for `flash_attention_2`. On the workstation, this becomes the `kernels-community/flash-attn` kernel, which only has an x86_64 build. On the Thor, a compatibility shim swaps it for SDPA.
2. **Memory:** DriveOS 7.0.3 exposes only **14.7 GiB** to CUDA (`total_memory=15020MB`), although the board has 58 GB. Alpamayo-1.5 in bf16 occupies 20.6 GiB, as follows:
   - language model: 14.1 GiB;
   - action expert: 4.25 GiB;
   - lm_head: 1.19 GiB;
   - vision: 1.07 GiB.

   The vision tower, the embeddings, the lm_head, the expert and 2 of the 36 language layers stay resident on the GPU. The other 34 go to host memory and accelerate runs them on the GPU, with the same bf16 arithmetic.

   **Consequence:** the VLM prefill and reasoning are slow (41.7 s per clip, with no value as latency). The velocity field, on the other hand, only calls the expert, which is resident, over the prefix cache, so the per-call latency is valid.

   **Alternative to ask of the board owner (not to do on our own):** increase the iGPU memory in DriveOS.

**Smoke test** (29/10, ~14:55 PDT; idle GPU, GR3D 0%):

| | NFE | per field call (median) | per solve (median of 5) | max \|Thor - workstation\| in the plan | endpoint error |
|---|---|---|---|---|---|
| E_10, tau=1 | 10 | 69.5 ms | 0.70 s | 1.6 cm | 1.7 cm (plan of ~181 m) |
| Heun-32, tau=1 | 64 | 69.5 ms | 4.47 s | 2.0 cm | 2.0 cm |

Both plans use the same initial noise of the seed. Differences of centimeters are compatible with SDPA vs FA2 and with different hardware; the decision is the same. Peak memory: 12.9 GB.

**Low-precision support in the Thor torch (a low-precision check script):**
- FP8 e4m3 with per-tensor scale (`torch._scaled_mm`): runs, with relative error 1.7*10^-3.
- FP8 with per-row scale: **fails** (the CUTLASS kernel falls into an sm90 path and gives *unspecified launch failure*).
- NVFP4 (e2m1 with e4m3 block scale): runs; only the trivial case was checked, the layout of the scales remains to be validated.
- INT8 (`torch._int_mm`): fails in cuBLASLt.

## 1. Questions

**(a) Latency on the vehicle computer.** How much does each sampler cost per field call and per plan, on the Thor, in PyTorch eager, with batch 1:
- E_1, E_2, E_4, E_10 and Heun-32;
- E_6 enters only as a point of the curve.

What matters is what fits in a 100 ms (10 Hz) control cycle.

**(b) Does deploy precision shift the effective temperature?** Quantizing the field (FP8 and, if validated, NVFP4) against the bf16 reference can do two things:
- change the tau_eff of E_k;
- open residuals in the matched-temperature test.

The measure is the paper's matched-temperature test, on held-out clips, entirely on the Thor, with each precision against its own curve.

## 2. Precision variants

| | VLM (prefill + reasoning) | action expert (the field) | role |
|---|---|---|---|
| **P0 bf16** | bf16 (partial offload) | bf16 | reference |
| **P1 FP8-expert** | bf16 (partial offload) | linears in FP8 e4m3: weights with per-tensor scale, activations with dynamic per-tensor scale (`_scaled_mm`, bf16 output) | **primary**: isolates the field, with a prompt cache identical to P0 |
| **P2 FP8-full** | FP8 linears as in P1 (~10.3 GiB, fits without offload) | FP8 | secondary, the realistic deploy; the reasoning may diverge from P0 |
| **P3 NVFP4-expert** | bf16 | NVFP4 with block-16 scale | exploratory, only if the layout test passes before the run |

**Outside quantization:** embeddings, normalizations, `action_in_proj` and `action_out_proj` stay in bf16 in all variants.

**Validation before the run:** in each variant, the relative error of each linear against bf16 on a batch of real activations of context_000 is recorded. If it exceeds 5%, the variant is flagged.

## 3. Sample, samplers and seeds

**Clips:** the first 60 of held-out v3 (context_000-059). They are in the workstation's pooled400, so there is a hardware cross-reference. The execution order is by name. Independent unit: the clip.

**Seeds:** 9201-9206, with the same initial noise across configs and precisions. `torch.randn` on the Thor GPU gave the same noise as the workstation: the E_10 plan matched to 1.6 cm.

**Configs** (the temperature probe's `--configs`, code unchanged):
- E_1, E_2, E_4, E_6, E_10 at tau = 1;
- Heun-32 at tau in {1; 0.9; 0.8; 0.7; 0.6; 0.5; 0.4}.

This gives 471 NFE per seed.

**Sample size, from the measured time:**
- Cost per clip and precision: 6 x 471 x 69.5 ms ~ 196 s, plus ~42 s of prefill with offload, ~ **4.0 min**.
- 60 clips: ~ 4 h per precision. P0 + P1 + P2 ~ 12 h of board time; P3 adds 4 h.
- **Expected precision with N = 60:** in the paper, with N = 400, the CI of the ADE_1 residual of E_10 has a half-width of ~0.01 m. With N = 60 it is ~0.025 m, inside the 0.05 m margin. Flips are ~+-2 pt: on the boundary, so flips are secondary (below).
- **If the board only gives a short window:** the first 30 clips of P0 and P1, with everything marked as a pilot.

## 4. Latency (question a)

**Measure:** the Thor latency script, to be written after registration. On an already prepared clip (the prefix cache stays fixed):
- 20 warm-up calls, then 200 timed calls of the field, with `cuda.synchronize` before and after;
- 30 complete solves per sampler (includes `action_to_traj`);
- on 10 clips (context_000-009), batch 1 and batch 6 (the 6 seeds together).

**Report:** median and p90 per call and per plan, for P0, P1 and P2 (P3 if present). Also the GPU clock (devfreq `performance`, 1575 MHz) and the tegrastats temperature at the start and end. There is no power rail: tegrastats does not report it.

**What does not enter:** the VLM prefill with offload. It is only reported in P2, where the model fits entirely on the GPU.

## 5. Analysis (question b)

Same as the paper (the matched-temperature statistics module / the paper-numbers script), applied to each precision against its own Heun-32 curve:

- **tau_eff(E_k, P):** the tau at which the mean across-seed spread of the Heun-32 curve of precision P equals the spread of E_k in P (linear interpolation).
- **Residual:** metric(E_k, P) - metric(Heun-32 at tau_eff, P).
- **Shift by precision:** delta tau_eff(E_k) = tau_eff(E_k, P1) - tau_eff(E_k, P0). Also delta of Heun-32 tau=1 itself between precisions: the spread at tau=1 gives the "native" temperature of the precision.
- **Metrics:** ADE_1 against the log (horizon and rate of the paper), decision flips (the `decisions` function of the trajectory-decision-gate module), entropy, stop, decelerate.
- **Bootstrap:** by clip, 2000 resamples, seed 20261029, re-resolving tau_eff in each. P0 and P1 are paired (same clips resampled together). TOST with 90% CI.
- **Paper margins** (`MARGINS` in the matched-temperature statistics module, declared on 26/10): ADE_1 +-0.05 m; minADE_6 +-0.05 m; flips +-2 pt; entropy +-0.05; stop +-1 pt; decelerate +-2 pt.
- **New margin, for delta tau_eff:** +-0.03. It is about 1/3 of the E_10-E_6 distance in the paper (0.85 against 0.76) and larger than the difference in tau_eff of E_10 among the three Alpamayos (0.83-0.85).
- **Hardware check, before everything:** P0 on the Thor against the workstation (pooled400), clip by clip:
  - median max |delta plan|;
  - tau_eff(E_10) and tau_eff(E_4) on the Thor against those of the workstation on the same 60 clips.

## 6. Predictions

- **T1 (latency):** in bf16 eager, the field call stays at 65-75 ms, with no difference between E_k and Heun (it is the same network), and the time per plan is NFE x call +-5%. **Only E_1 fits in 100 ms.** E_4 takes ~0.28 s, E_10 ~0.70 s and Heun-32 ~4.5 s. P1 reduces the call by >= 25%. If this fails, it is because the per-tensor `_scaled_mm` does not pay off at batch 1, and we report it so.
- **T2 (primary, b):** |delta tau_eff(E_10)| and |delta tau_eff(E_4)| between P1 and P0 pass the TOST of +-0.03. This means that FP8 quantization of the field does not act as temperature.
- **T3:** in P1, the ADE_1 residual of E_10 and of E_4 against its own curve passes at +-0.05 m, as in P0, and the tau_eff of E_10 is at 0.80-0.90, as in the paper (0.85). Flips are secondary, because of the power with N = 60.
- **T4 (hardware):** in P0, the median max |delta plan| against the workstation is < 0.10 m. The tau_eff of E_10 on the Thor differs by < 0.02 from that of the workstation on the same 60 clips.
- **T5 (secondary, P2):** with the VLM also in FP8, delta tau_eff(E_10) passes at +-0.03. It may fail if the reasoning diverges. In that case, we report the fraction of clips with reasoning text different from P0, and P2 is no longer read as a field effect.
- **Falsification:** if the delta tau_eff of P1 leaves the margin with CI away from zero, quantization shifts the effective temperature. This is a finding that the paper reports (deploy changes the implicit temperature), not a caveat. If the CI is too wide to conclude, the result is inconclusive with N = 60.

## 7. Execution
- **The run script on the board (prepared, not started):**
  - waits for the board to become free: no other GPU process besides the desktop, no `llm_stream_video` or `joint_filter`, GR3D <= 10% over 5 s and >= 32 GB available;
  - checks again every 5 min;
  - then runs P0 with the temperature probe's `--configs ... --range 0 60 --resume`, which resumes if interrupted, and then the latency and quantization scripts, if they exist;
  - log in the run's output folder on the board's SSD.
- **Board policy:** never kills or signals other users' processes. If another user starts a heavy job in the middle, there is no automatic pause yet. This needs to be agreed beforehand; the proposal is to run at night, with notice to the board's other users.
- **To be written after registration:**
  - the latency script;
  - the quantization probe: P1/P2/P3 as `nn.Linear` wrappers applied after the Thor model loader, calling the same temperature probe;
  - the P0xP1 analysis (reuse of the matched-temperature statistics, with paired delta tau_eff).

## 8. Limitations (for the paper)

- PyTorch eager, not TensorRT-Edge-LLM: latency is a ceiling, not the production number.
- 15 GB visible to CUDA force offload in bf16, so the VLM prefill is not measured in bf16.
- SDPA attention instead of FA2.
- Per-tensor FP8, not per-row (the per-row kernel fails in this torch).
- N = 60 clips, a subset of the held-out.

## Deviation 1 (30/10/2026 00:51 PDT): whole model on the GPU from the P0 latency on

At the authors' request, the board's owner increased the hugetlbfs pool from 7510 to 13312 pages of 2 MB (about 26 GiB visible to CUDA). With this, the run was restarted with Alpamayo-1.5 bf16 entirely resident on the GPU (the GPU-layers environment variable set to 999). Before, only 2 language layers stayed resident and the others were brought from host memory.

- **P0 probe (60 clips):** done with offload, stays as it is. In accelerate's offload the weights are copied to the GPU and the computation runs there, in the same bf16 arithmetic, so the plans do not depend on it. An equality check of the plans of 1 clip between the two modes will be made and reported.
- **P0 latency:** the partial run with offload (1 clip) was moved to a separate partial-run folder, without deleting anything. The latencies of P0, P1 and P2 are all measured with the resident model from now on.
- **VLM prefill in bf16:** can now be measured. Enters as exploratory.
- **Free-RAM threshold of the runner:** lowered from 32 to 16 GB, because the model now stays in the GPU pool.

## Addendum 2 (30/10/2026 17:35 PDT, before running): FROST with TensorRT engines on the 400 clips of the paper

**Reason.**
- The registered P0xP1 analysis gave T2 and T4: eager FP8 does not change tau_eff, and the Thor reproduces the workstation.
- T3 failed only through power in E_4, with N=60, and bf16 fails equally.
- Eager FP8 does not speed up. TensorRT engines do: on the field call, 27 ms in bf16, 18.5 ms in FP8 and 16.6 ms in NVFP4, against 69 ms in eager (internal TensorRT plan note, pilot on one clip).
- The paper's question is whether these deploy speedups change the effective temperature or leave residuals. The authors asked for the 400 clips of the paper.

**Sample.** context_000-399 of held-out v3, the same 400 of the paper's pooled400. Seeds 9201-9206, same initial noise. Independent unit: the clip.

**Run** (the TensorRT probe script of the Thor deployment code):
- **Prefix phase:** the VLM in PyTorch bf16 (resident model) runs once per clip, and the prefix is reused by the three variants. On context_000, this prefix reproduces P0 bit by bit.
- **Field phase:** batch-6 engines with L=3456, on E_1, E_2, E_4, E_6 and E_10 at tau=1, and on the Heun-32 curve at tau in {1; 0.9; ...; 0.4}.
- Clips with prefix larger than L are skipped and listed.

**Variants:**

| | engine | calibration | role |
|---|---|---|---|
| **V0** | TRT-bf16 | n/a | deploy reference |
| **V1** | TRT-FP8 (per-tensor Q/DQ) | ModelOpt, 8 clips of the dev v2 split (outside the 400) | primary |
| **V2** | TRT-NVFP4 "no_ends" (layers 0, 1, 34 and 35 of the expert in bf16; 448 of 504 quantizers in FP4) | same | secondary |

The all-layers NVFP4 variant does not enter: its only engine was calibrated on evaluation clips.

**Analysis.**
- The Thor analysis script, unchanged, with the variants in place of P0/P1/P3.
- Each variant is tested against its own Heun-32 curve.
- Delta tau_eff of V1 and V2 is measured against V0, with the clips resampled together.
- Bootstrap by clip: 2000 resamples, seed 20261031, redoing tau_eff. 90% TOST.
- Margins:
  - of the paper: ADE_1 +-0.05 m and flips +-2 pt;
  - of the Thor protocol: delta tau_eff +-0.03.
- Engine check: V0 against the workstation (pooled400) on the 400 clips.

**Predictions:**
- **U1 (engine):** V0 reproduces the workstation. The median of max |delta plan| is < 0.10 m, and the tau_eff of E_10 differs by < 0.02 from that of the workstation (0.850).
- **U2 (primary):** in V1 (FP8), |delta tau_eff| of E_10 and E_4 against V0 passes the TOST of +-0.03.
- **U3:** in V0 and in V1, E_10 against its own curve is equivalent in ADE_1 and in flips, and E_4 is equivalent in ADE_1.
- **U4 (secondary):** in V2 (NVFP4), |delta tau_eff(E_10)| against V0 passes at +-0.03, and E_10 is equivalent in ADE_1 against its own curve. No prediction for E_1, E_2 and E_4 in V2, where the plan deviation reaches tens of cm.

**Reading.**
- If delta tau_eff leaves the margin with CI away from zero, quantization acts as temperature, and the paper reports it.
- If delta tau_eff stays within the margin but the residual against V0 at the same tau is not equivalent, quantization is error and not temperature, and FROST distinguishes the two cases.

**Latency (exploratory).** Time per call and per plan at batch 1 and batch 6, on 10 clips, after the run.

**Order on the board.**
1. This run, about 9.6 h.
2. Then, the paused P2 (a continuation script, resumes from clip 7).
3. The prefix files (~184 GB) stay on the board's SSD until the analysis.

The result will be reported whatever it is.

## Addendum 3 (31/10/2026 23:37 PDT, before running): tau_eff with the compiled VLM (FP8 and FP8-prefill + NVFP4-AWQ-decode) on the 400 clips

Time read from `date` (Sat Oct 31 11:37 PM PDT 2026 = 01/11 06:37 UTC; commit recorded in the private repository at 23:37:06; the header time was corrected from 23:38 to 23:37 in the following commit, still before running; workstation clock = board clock). Nothing of this addendum was run on 400 clips before the commit. Code: the compiled-VLM FROST script of the Thor deployment code.

**Reason.** Addendum 2 measured tau_eff with the prefix in PyTorch bf16 and only the field in TensorRT. The compiled VLM (internal note on the compiled Alpamayo) changes the prefix: on 40 clips the KV differs 7-9% from PyTorch, and the reasoning is equal in 38/40. The question is whether the compiled prefix shifts tau_eff.

**Sample, seeds, configurations.** Identical to those of addendum 2 and of the TensorRT FROST-400 artifact (2026-10-31): context_000-399 of held-out v3; seeds 9201-9206; initial noise `torch.manual_seed(seed); randn(1, 64, 2)`; E_1, E_2, E_4, E_6, E_10 at tau=1 and Heun-32 at tau in {1; 0.9; ...; 0.4}. Reasoning with seed 123, top-p 0.98, T 0.6, max 256 tokens (as the prefix of addendum 2). Independent unit: the clip.

**Variants** (one pass per clip, no prefix files: the VLM cache is the field's `past_kv`, zero copy):

| | compiled VLM | field (batch 6, L=3456) | preprocessing |
|---|---|---|---|
| **V3** | FP8 in prefill and decode (FP8 engine, attention in bf16 with materialized output) | FP8 (the batch-6 FP8 probe engine) | HF on CPU, identical to PyTorch |
| **V4** | FP8 prefill (same engine) + NVFP4-AWQ-full decode (8 calibration clips of the dev v2 split, outside the 400) | FP8 (the same) | same |

Already existing references: `trt_fp8` (PyTorch bf16 prefix + same FP8 field) is the V1 of addendum 2; `trt_bf16` is V0. Since V3 and V4 use the same field as V1, the difference against V1 isolates the compiled VLM. GPU preprocessing does not enter this run: it changes the pixels by up to 1/64 and is not part of the question.

**Choice of NVFP4 (made before, on 40 clips, against the FP8 field on the PyTorch prefixes; these 40 are among the 400):**

| decode | same text | delta plan E_10 median / p90 / max (cm) | E_10 spread ratio [95% CI] | ms/token |
|---|---|---|---|---|
| FP8 (V3) | 39/40 | 9.3 / 30 / 70 | 0.992 [0.985; 0.998] | 46.4 |
| NVFP4-AWQ-full, FP8 prefill (V4) | 38/40 | 10.6 / 26 / 54 | 0.992 [0.986; 0.999] | 36.1 |
| NVFP4-AWQ-lite, FP8 prefill | 37/40 | 13.7 / 30 / 130 | 1.002 [0.994; 1.010] | 36.0 |
| AWQ-lite with layers 9-17 in FP8 | 36/40 | 12.5 / 42 / 117 | 1.001 [0.994; 1.008] | 38.6 |
| AWQ-lite with layers 0-17 in FP8 | 38/40 | 11.7 / 30 / 52 | 0.994 [0.988; 1.001] | 41.2 |

V4 is the one with the smallest delta plan among those that decode at <= 38 ms/token. The numbers above inform the predictions; they are not independent of them.

**Predictions:**
- **W1 (V3 against V1):** |delta tau_eff| of E_10 and E_4 passes the TOST of +-0.03. The E_10 of V3 has tau_eff in 0.80-0.90.
- **W2 (V4 against V1 and against V3):** the same, |delta tau_eff| of E_10 and E_4 within +-0.03. If V4 fails and V3 does not, NVFP4 in decode acts as temperature or as error, and the paper reports it.
- **W3 (plans):** the median max |delta plan| of E_10 against the V1 plans on the same clip and seed is < 0.15 m in V3 and < 0.20 m in V4.
- **W4 (text):** the reasoning is identical to that of PyTorch in >= 90% of clips, in V3 and in V4.
- **W5 (spread):** the spread ratio (Heun-32 tau=1 and E_10, V/V1) is within 0.98-1.02 in V3 and in V4.
- **W6:** no skipped clip (prefix <= 3456 and prompt + 256 tokens <= 3456). If any, the list goes in `timing.jsonl`.
- **Falsification:** as in addendum 2. If delta tau_eff leaves the margin with CI away from zero, the compiled prefix shifts the effective temperature.

**Analysis:** the Thor analysis script, unchanged, run by the requester. The run only writes the plans (`plans.npz` in the probe format, plus `timing.jsonl` with the reasoning text, P and offset per clip). No statistic is computed in the run.

**Order on the board.** V3, then V4, sequential. Written only to a dedicated folder on the board's SSD (plans of ~22 MB each); copied to the artifacts folder on the workstation at the end. Each variant takes about 400 x (1.7 s of VLM + field) ~ 20-40 min.
