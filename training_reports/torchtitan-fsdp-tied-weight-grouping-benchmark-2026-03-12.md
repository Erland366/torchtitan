# Report: torchtitan-fsdp-tied-weight-grouping-benchmark

**Date:** 2026-03-12
**Author:** codex
**Status:** Completed

## Objective

Test whether keeping tied nanoVLM LM weights inside the same FSDP unit improves
correctness and preserves training quality under 2-GPU FSDP, while measuring any
impact on speed and VRAM.

## Setup

### Environment
- Hardware: `2x A100-SXM4-40GB`
- Runtime: `torchrun --nproc_per_node=2 -m torchtitan.train --module nanoVLM`
- Dataset: `HuggingFaceM4/FineVisionMax` (streaming)
- W&B entity/project: `patrickirawan-mbzuai/momh`

### Change Under Test

- File: `torchtitan/models/nanoVLM/parallelize.py`
- Old behavior:
  - `tok_embeddings` sharded separately
  - `[norm, output]` sharded together
- New behavior when `lm_tie_weights=True`:
  - `[tok_embeddings, norm, output]` sharded together

This change keeps the tied parameter relationship
`output.weight = tok_embeddings.weight` inside one FSDP wrapper.

### Common Benchmark Configuration

- Steps: `100`
- `dp_shard=2`, `dp_replicate=1`
- `training.local_batch_size=16`
- `training.global_batch_size=64`
- `activation-checkpoint.mode=none`
- Checkpointing disabled
- Metrics log frequency: `1`

## Experiments

### Run 1: `nanovlm_230m_vanilla_finevisionmax_nopack`

| Variant | W&B Run | Final Loss | Median TPS excl step1 | Peak VRAM (MiB, any GPU) | Elapsed (s) |
|---------|---------|-----------:|----------------------:|--------------------------:|------------:|
| baseline | `utriabh7` | `5.81250` | `15207` | `59696` | `523` |
| tied-grouped | `g3rn91jj` | `5.81250` | `15557` | `58474` | `513` |

Delta (`tied-grouped - baseline`):
- final loss: `0.00000`
- median TPS: `+350`
- peak VRAM: `-1222 MiB`
- elapsed: `-10s`

Step-level loss comparison:
- compared steps: `100`
- exact parity: `no`
- mismatches:
  - step `64`: `6.71875` vs `6.68750`
  - step `79`: `5.96875` vs `5.93750`
  - step `87`: `6.65625` vs `6.62500`
- max abs diff: `0.03125`

### Run 2: `nanovlm_230m_momh_soft_gating_b5_tttv_nopack`

| Variant | W&B Run | Final Loss | Median TPS excl step1 | Peak VRAM (MiB, any GPU) | Elapsed (s) |
|---------|---------|-----------:|----------------------:|--------------------------:|------------:|
| baseline | `fws2j2m1` | `5.43750` | `22440` | `59696` | `358` |
| tied-grouped | `xqbxfkgh` | `5.43750` | `22176` | `58780` | `357` |

Delta (`tied-grouped - baseline`):
- final loss: `0.00000`
- median TPS: `-264`
- peak VRAM: `-916 MiB`
- elapsed: `-1s`

Step-level loss comparison:
- compared steps: `100`
- exact parity: `no`
- mismatches:
  - step `29`: `6.46875` vs `6.43750`
- max abs diff: `0.03125`

## Analysis

### What Worked
- The tied-weight grouping fix keeps shared LM weights inside one FSDP unit.
- Both vanilla and soft-gating completed cleanly for `100` steps on `2` GPUs.
- Peak VRAM dropped on both configs.
- Vanilla showed a small wall-clock and throughput improvement.
- Final loss matched in both configs.

### What Failed
- The change did not preserve exact step-level parity across all `100` steps.
- Soft-gating did not gain speed in a meaningful way from this change.

### Key Insights
1. This is a correctness-oriented FSDP change, not a parity-preserving refactor.
2. Shared-parameter FSDP grouping can improve memory behavior without changing end-of-run loss.
3. Final-loss equality is not sufficient evidence for exact parity; step-level comparison remains necessary.
4. For soft-gating, this change is essentially neutral on wall clock and slightly worse on median TPS in this sample.

## Decision

- Treat same-group FSDP wrapping for tied LM weights as the correct structural implementation.
- Do not use this change as evidence of exact parity preservation.
- Keep parity claims gated on explicit step-level comparisons.

## Appendix

- Output directories:
  - `outputs/fsdp_tie_group_bench_wandb/baseline_vanilla_100`
  - `outputs/fsdp_tie_group_bench_wandb/patched_vanilla_100`
  - `outputs/fsdp_tie_group_bench_wandb/baseline_softgating_100`
  - `outputs/fsdp_tie_group_bench_wandb/patched_softgating_100`
- Code touched:
  - `torchtitan/models/nanoVLM/parallelize.py`
  - `tests/unit_tests/test_nanovlm_parallelize.py`
  - `torchtitan/models/nanoVLM/README.md`
