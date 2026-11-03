# Third-party code, models and data

This repository contains only original code (MIT, see `LICENSE`), with two small exceptions noted below. None of the
models, datasets or upstream repositories listed here are redistributed. Obtain each from its source and accept its
terms. The license column gives the license file or model card as it was at the stated revision; check the source for
the current terms.

## Code included in this repository

| File | Origin | License |
|---|---|---|
| `frost/probes/alpamayo_compat.py` | adapted from the authors' Alpamayo integration code (normalizes legacy Hydra targets of the released configs) | Apache-2.0 |
| `patches/alpasim/frost_sampler.patch` | patch against NVlabs/alpasim (adds an opt-in sampler hook to the Alpamayo-1.5 driver) | Apache-2.0, as AlpaSim |

## Upstream source code (cloned, not redistributed)

| Variable | Repository | Commit used | License |
|---|---|---|---|
| `ALPAMAYO15_SRC` | https://github.com/NVlabs/alpamayo1.5 | `2eff7037e47afb96a578b3d1bca453a373cd781e` | Apache-2.0 |
| `ALPAMAYO2_SRC` | https://github.com/NVlabs/alpamayo2 | `beb2977d9a7e9d66837d4a3ad5144ff59de37519` | Apache-2.0 |
| `ALPAMAYO_R1_SRC` | https://github.com/NVlabs/alpamayo | `e6aefbc8d7a93480f045147bd36fcddeb9c829a2` (the checkout carried local diagnostic hooks, which the probe disables at runtime) | Apache-2.0 |
| `QWENDRIVE_SRC` | https://github.com/QwenLM/Qwen-Drive-1.0 | `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0` | Apache-2.0 |
| `FLASHDRIVE_SRC` | https://github.com/z-lab/flashdrive | `26265174f0413db77c8a16d99993dbeecacf458f` | MIT |
| `SIMWAM_SRC` | https://github.com/H-EmbodVis/SimWAM | `1433ee7e06a62bd8b89da74455df0d82bda7b726` | MIT |
| `RECOGDRIVE_SRC` | https://github.com/xiaomi-research/recogdrive | `5831ee83c7a6a40bf6919715b96b0e9ba2b48335` | Apache-2.0 |
| `DIFFUSIONDRIVE_SRC` | https://github.com/hustvl/DiffusionDrive | `9b52ed0ec06b073d82d6f392ab084c7b301c8681` | MIT |
| `ALPASIM_ROOT` | https://github.com/NVlabs/alpasim | `affc2eab209fa43bdfa2f26c0f8d437922d78a68` + `patches/alpasim/frost_sampler.patch` | Apache-2.0 |
| `WAYMO_RFS_OFFICIAL` | `src/waymo_open_dataset/metrics/python/rater_feedback_utils.py` from https://github.com/waymo-research/waymo-open-dataset, last changed in commit `e378e5ba5bb8460184e5c878c82c9e593bdb2927`; sha256 `fe994a096141957b8f936d613b08876bee60ec5b746749b1a79331217f4edee9` (checked before loading) | Apache-2.0 |
| (LIBERO, LeRobot) | https://github.com/huggingface/lerobot, https://github.com/Lifelong-Robot-Learning/LIBERO (installed through `envs/robot.txt`) | Apache-2.0 / MIT |

## Models (Hugging Face)

| Model | Revision | Terms |
|---|---|---|
| `nvidia/Alpamayo-1.5-10B` | `7aba8293c09993f2e125c6819df05d7fa3e873ea` | OpenMDW-1.1 (model card), gated |
| `nvidia/Alpamayo-R1-10B` | `d72c6a2757b10cacb69b688ca66ccfad3dafbf37` | NVIDIA model license on the card, gated |
| `nvidia/Alpamayo2-Super` | `00554695e729a6ff0b6281fd2c81b18d06e33dbe` | OpenMDW-1.1 (model card), gated |
| `z-lab/Alpamayo-1.5-10B` (FlashDrive stack) | `03637f1f3acac01d8bf5a094147d6352570e6c46` | see model card |
| `Qwen/Qwen-Drive-1.0-4B` | `28484089a7cc8c335cf5089fb0745cf7c49b6eaa` | Apache-2.0 |
| `H-EmbodVis/SimWAM` | `5d4492937e357e6ca64de128cdccbd25c2b34f30` | "other" on the card; see the card |
| `nvidia/Cosmos3-Nano` | `e59a53c25979a090fa8706c9acc0c254a6e89b92` | NVIDIA open model license on the card |
| `HuggingFaceVLA/smolvla_libero` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | see model card |
| ReCogDrive-2B-IL/RL, ReCogDrive-VLM-2B, DiffusionDrive (navsim 88.1 PDMS), `timm/resnet34.a1_in1k` | as linked from the upstream READMEs | see upstream |

## Datasets

| Dataset | Version | Terms |
|---|---|---|
| NVIDIA PhysicalAI-Autonomous-Vehicles (`nvidia/PhysicalAI-Autonomous-Vehicles`) | revision `dfcc35f941c38f050e9ce256a4c0aff9e33615b9`; split files `splits/alpamayo_aligned.parquet` | NVIDIA dataset license, gated on Hugging Face |
| Waymo Open Dataset, end-to-end driving (WOD-E2E), validation split with rater scores | as released at the time of the study | Waymo Open Dataset license (non-commercial) |
| NAVSIM navtest (OpenScene) and nuPlan maps | NAVSIM v1 data as required by the upstream repos | nuPlan / OpenScene terms (non-commercial) |
| AlpaSim public scene suite `public_2604` | shipped with AlpaSim at the commit above | see AlpaSim |
| LIBERO benchmark | as installed by LeRobot | MIT |

The released `results/` contain only aggregate statistics and scene/clip identifiers of these datasets, plus
per-clip/per-segment summary metrics (no images, trajectories or sensor data).
