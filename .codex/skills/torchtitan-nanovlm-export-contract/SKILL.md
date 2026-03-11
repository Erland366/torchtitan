---
name: torchtitan-nanovlm-export-contract
description: >
  Export TorchTitan nanoVLM checkpoints in a format that loads directly in
  `nanoVLM_main` without manual patching.
  Use when: converting TorchTitan DCP checkpoints for direct nanoVLM eval or
  cross-framework validation.
metadata:
  short-description: "Direct-load export contract for TorchTitan nanoVLM"
  tags:
    - research
    - torchtitan
    - nanovlm
    - checkpoint
    - export
  domain: research
  created: 2026-03-11
  author: codex
---

# TorchTitan nanoVLM Export Contract

## General Description

This skill captures the checkpoint export contract required for TorchTitan
nanoVLM weights to load directly in `nanoVLM_main`. The key point is that
cross-framework compatibility is not just about tensor values; it also depends
on structural checkpoint keys and the exact files emitted into the export
folder.

Use this when validating TorchTitan checkpoints through `nanoVLM_main`
evaluation or when debugging export/load mismatches between TorchTitan and the
original nanoVLM codebase.

## When to Apply

Use this knowledge when:
- You are converting a TorchTitan DCP checkpoint for use outside TorchTitan.
- You want `nanoVLM_main/evaluation.py --mode nanovlm --model <export_dir>` to
  load directly.
- You see checkpoint load failures involving RoPE buffers or tied embedding
  aliases.

Do NOT use when:
- You are staying entirely inside TorchTitan and loading native DCP checkpoints.
- You only need generic HF shard output and do not care about direct
  `nanoVLM_main` compatibility.

## Results Summary

Reference validation:

| Metric | Value | Notes |
|--------|-------|-------|
| exported `config.json` | present | emitted automatically by `convert_to_hf.py --model_name nanoVLM` |
| exported `model.safetensors` | present | emitted automatically alongside consolidated shard |
| `decoder.rotary_embd.inv_freq` | present | required by strict `nanoVLM_main` loading |
| tied LM embedding alias duplication | avoided | export keeps canonical `decoder.head.weight` only |
| `mmstar average,none` | `0.4375` | matched between TorchTitan eval and nanoVLM eval on `limit=20` |

## Recommended Practice

### Step 1: Use the canonical converter

Use:

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
  <dcp_checkpoint_dir> \
  <output_dir> \
  --model_name nanoVLM \
  --model_flavor <flavor> \
  --export_dtype bfloat16
```

### Step 2: Preserve the compatibility keys

For nanoVLM exports:
- include `decoder.rotary_embd.inv_freq`
- keep tied LM embeddings canonical (`decoder.head.weight`)
- do not materialize `decoder.token_embedding.weight` separately in tied mode

### Step 3: Validate from the consumer side

After export, validate with the actual consumer:

```bash
python nanoVLM_main/evaluation.py \
  --mode nanovlm \
  --model <output_dir> \
  --tasks mmstar \
  --limit 20
```

If this succeeds, the export contract is correct for the original nanoVLM load
path.

## Failure Modes

| What Failed | Why | Lesson Learned |
|-------------|-----|----------------|
| `Missing key(s) in state_dict: "decoder.rotary_embd.inv_freq"` | TorchTitan export dropped a deterministic RoPE buffer that nanoVLM strict loading still expects | Export the RoPE buffer explicitly for direct nanoVLM compatibility |
| `Unexpected key(s) in state_dict: "decoder.token_embedding.weight"` | Tied embedding alias was materialized redundantly in export | Keep one canonical LM weight key in tied mode |
| Export folder has shards but no direct-load bundle files | Generic HF export does not automatically imply nanoVLM loader compatibility | Emit `config.json` and `model.safetensors` directly when exporting `nanoVLM` |

## Configuration

```bash
source ../nanoVLM_main/.venv/bin/activate
cd /home/coder/edd/nanoVLM_root/torchtitan

python scripts/checkpoint_conversion/convert_to_hf.py \
  /abs/path/to/dcp_checkpoint \
  /abs/path/to/export_dir \
  --model_name nanoVLM \
  --model_flavor 230m_vanilla \
  --export_dtype bfloat16
```

## References

- Report: `training_reports/torchtitan-downstream-mmstar-kvcache-regression-2026-03-10.md`
- Code: `scripts/checkpoint_conversion/convert_to_hf.py`
- Code: `torchtitan/models/nanoVLM/state_dict_adapter.py`
