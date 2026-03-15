---
name: torchtitan-downstream-eval-raw-first-fallback
description: >
  Run TorchTitan nanoVLM downstream evaluation with `torchtitan_nanovlm` as the
  default backend and an optional explicit raw-HF compatibility path.
  Use when: evaluating checkpoints on mmstar/docvqa/vqav2 while preserving
  reproducible JSON artifacts and explicit backend provenance.
metadata:
  short-description: "Reliable downstream eval path for TorchTitan nanoVLM"
  tags:
    - research
    - torchtitan
    - evaluation
    - lmms-eval
    - nanovlm
  domain: research
  created: 2026-03-10
  author: codex
---

# TorchTitan Downstream Eval Default Backend

## General Description

This skill captures the stable downstream evaluation workflow for TorchTitan
nanoVLM checkpoints. The default path is the repo-local
`torchtitan_nanovlm` lmms-eval backend. Raw `huggingface` loading is retained
only as an explicit compatibility check when requested.

It also enforces deterministic local outputs per run:
- `summary.json`
- `per_task.json`
- `metadata.json`

## When to Apply

Use this knowledge when:
- You need downstream quality signals after TorchTitan training/fine-tuning.
- Your checkpoint source can be either HF folder or TorchTitan DCP folder.
- You need explicit evidence of which eval backend actually produced scores.

Do NOT use when:
- You require only raw HF backend runs without fallback allowance.
- You are running integrated in-training eval hooks (this skill is manual-run focused).

## Results Summary

Reference runs:
- baseline output: `eval_results/torchtitan/mmstar-full-20260310`
- cached-runtime output: `eval_results/torchtitan/mmstar-full-kvcache-regression-20260310`

| Metric | Value | Notes |
|--------|-------|-------|
| task | `mmstar` | full set (`limit=null`) |
| average | `0.3221761082` | identical in both runs |
| backend used | `torchtitan_nanovlm` | raw `huggingface` failed, fallback succeeded |
| baseline fallback duration | `2230.87s` | from `mmstar-full-20260310/metadata.json` |
| cached-runtime fallback duration | `1205.92s` | from `mmstar-full-kvcache-regression-20260310/metadata.json` |
| cached-runtime speedup | `~45.9%` faster | same score, substantially lower fallback duration |

## Recommended Practice

### Step 1: Use the canonical runner

- Preferred script: `scripts/nanovlm_downstream_eval.py`
- Convenience wrapper: `evaluation_torchtitan.sh`

### Step 2: Use `torchtitan_nanovlm` as the default local path

- primary backend: `torchtitan_nanovlm`
- fallback backend: `none`

This is the one clear path for local TorchTitan nanoVLM checkpoints and merged
exports.

### Step 3: Use raw HF only as an explicit compatibility check

When you want to probe generic lmms-eval Hugging Face loading, pass:
- `--model_backend huggingface`
- `--fallback_backend none`

Only enable `--fallback_backend torchtitan_plugin` when you deliberately want a
secondary retry path after a raw-HF failure.

### Step 4: Preserve artifact provenance

Always retain:
- primary attempt outcome (`metadata.primary_attempt`)
- fallback attempt outcome (`metadata.fallback_attempt`)
- final backend used (`metadata.backend_used`)

These are required for fair result interpretation.

### Step 5: Check backend provenance before reporting scores

For this checkpoint class, score interpretation is invalid without checking
`metadata.backend_used`. A completed JSON artifact does not imply that a raw-HF
compatibility attempt worked.

### Step 6: Prefer the cached TorchTitan runtime

The standalone TorchTitan backend uses KV-cache decode support. It preserves
`mmstar` scores while avoiding the older full-sequence recompute path.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| Raw `huggingface` backend load | Local checkpoint is not a reliable generic HF lmms-eval target for this model family | Keep `torchtitan_nanovlm` as the default local backend |
| Standalone fallback took impractically long | Decode path recomputed full sequence and vision path each generated token | Keep KV-cache runtime enabled for fallback eval and regression-check score parity after any runtime rewrite |
| Mixed concurrent eval processes with same run name | Two runs write into same output dir and make completion status ambiguous | Keep one canonical run process and avoid duplicate launches per run name |
| Silent backend ambiguity | Score file exists but backend origin unclear | Enforce `metadata.json` backend provenance checks before reporting numbers |

## Configuration

```bash
source ../nanoVLM_main/.venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path /abs/path/to/checkpoint \
  --checkpoint_format auto \
  --tasks mmstar \
  --batch_size 16 \
  --device cuda \
  --output_dir eval_results/torchtitan
```

## References

- Report: `training_reports/torchtitan-downstream-mmstar-eval-2026-03-10.md`
- Report: `training_reports/torchtitan-downstream-mmstar-kvcache-regression-2026-03-10.md`
- Artifact dir: `eval_results/torchtitan/mmstar-full-20260310`
- Artifact dir: `eval_results/torchtitan/mmstar-full-kvcache-regression-20260310`
- Related skill: `.codex/skills/torchtitan-upstream-alignment-guardrail/SKILL.md`
