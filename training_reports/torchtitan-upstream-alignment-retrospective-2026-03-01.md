# Experiment: torchtitan-upstream-alignment-retrospective

- **Date**: 2026-03-01
- **Author**: Codex-assisted
- **Goal**: Restore TorchTitan shared runtime behavior close to upstream while preserving nanoVLM checkpoint compatibility and validating the three hard targets: faster runtime, lower VRAM, and loss parity versus nanoVLM_main.
- **General description**: This cycle removed nanoVLM-specific forks from shared TorchTitan runtime code and moved required checkpoint-load behavior into the nanoVLM adapter. We then ran short paired vanilla checks and a soft-gating smoke run.
- **Models**: nanoVLM 230m vanilla, nanoVLM 230m soft-gating
- **Datasets**: HuggingFaceM4/FineVisionMax (streaming)

---

## 1. Setup

### 1.1 Model & task

- Model: `nanoVLM` in `torchtitan` (`--module nanoVLM`)
- Task: compare TorchTitan training behavior against `nanoVLM_main` baseline
- Architecture note: TorchTitan active loaded shape was `lm_hidden_dim=576`, `lm_n_blocks=30` from checkpoint path used by this port, not the `960/32` paper YAML shape.

### 1.2 Data

- Dataset: `HuggingFaceM4/FineVisionMax`
- Split: `train` streaming
- Filtering: relevance/image_correspondence/visual_dependency/formatting minimum rating `1`
- Sequence settings: `max_sample_length=2048`, no packing

### 1.3 Base hyperparameters

- Local batch size `8`, global batch size `64`, gradient accumulation `8`
- LR groups: `lm=1e-4`, `vision=0`, `projector=1e-5`
- Compile: enabled for model and loss
- Hardware: `1x A100-SXM4-40GB`

---

## 2. Runs

### 2.1 Run table

| run_id | config_name / label | key_param_changed | job_id / link | logs_file |
|--------|---------------------|-------------------|---------------|-----------|
| r1 | torchtitan vanilla (5 steps) | upstream-aligned shared runtime + adapter hook | local foreground | `/tmp/torchtitan_align5_mem2.log` |
| r2 | nanovlm_main vanilla (5 steps) | baseline reference run | local foreground | `/tmp/nanovlm_align5_mem2.log` |
| r3 | torchtitan soft-gating (2 steps) | adapter remap smoke (`_orig_mod` keys) | local foreground | terminal output (session) |

### 2.2 Notes per run

- r1 and r2 used external `nvidia-smi` sampling for comparable peak VRAM.
- r3 validated that compiled-checkpoint key remap still loads after moving logic out of shared checkpoint manager.

---

## 3. Results

### 3.1 Metrics

| run_id | main_metric_name | main_metric_value | other_metrics | notes |
|--------|------------------|-------------------|--------------|-------|
| r1 | avg tps (steps 2-5) | `37620.75` | loss=`[0.67954, 0.57038, 0.60279, 0.52181, 0.69893]`, peak VRAM=`21767 MiB` | TorchTitan |
| r2 | avg tokens/s (steps 2-5) | `15650.78` | step_loss=`[0.6465, 0.5982, 0.5755, 0.6096, 0.5698]`, peak VRAM=`27443 MiB` | nanoVLM_main baseline |
| r3 | soft-gating HF load | pass | `HF key adaptation: remapped=270, dropped=0, kept=422` | confirms remap hook still works |

### 3.2 Plots / qualitative observations

- Speed improved strongly in TorchTitan under this setup (`~2.40x` vs baseline throughput metric).
- VRAM reduced by `5676 MiB` (`~20.68%` lower peak).
- Exact short-run loss parity not reached.

---

## 4. Analysis

- Upstream alignment of shared runtime succeeded and reduced long-term drift risk.
- Moving checkpoint key adaptation into the model adapter preserved required nanoVLM compatibility without keeping custom logic in shared `checkpoint.py`.
- Exact loss matching remains open. In short runs, LR schedule semantics and warmup handling differences are a likely confounder and need controlled parity harness runs for final conclusion.

---

## 5. Lessons learned -> candidate skills

- Candidate skill 1: keep shared TorchTitan runtime close to upstream and isolate model quirks in adapters/hooks.
- Candidate skill 2: enforce loss-parity triangulation protocol (fixed-batch replay, LR alignment, token-accounting checks) before making runtime-level changes.

Mention relevant files:
- This report: `training_reports/torchtitan-upstream-alignment-retrospective-2026-03-01.md`
- Logs: `/tmp/torchtitan_align5_mem2.log`, `/tmp/nanovlm_align5_mem2.log`
