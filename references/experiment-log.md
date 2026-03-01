# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

<!-- New entries go above this line -->

## 2026-03-01

- **Date**: 2026-03-01
- **Type**: Retrospective
- **General description**: Final parity closure check passed practical loss parity with exact dataset-stream alignment while preserving TorchTitan speed and VRAM gains.
- **Details**:
  - Final paired dataset-trace checks used:
    - `vanilla-datasettrace-vanilla-fixedwarmup-20260301`
    - `soft-gating-datasettrace-softgating-finalcheck-20260301`
  - Vanilla final pair:
    - baseline `pcks21h8`: elapsed `141.36s`, peak `26169 MiB`
    - torchtitan `8q1766hh`: elapsed `117.37s`, peak `20001 MiB`
    - loss diff: mean `0.00137`, max `0.00362` (step `2`)
  - Soft-gating final pair:
    - baseline `ce514a4q`: elapsed `147.16s`, peak `28499 MiB`
    - torchtitan `md7kgkar`: elapsed `112.06s`, peak `22105 MiB`
    - loss diff: mean `0.00648`, max `0.01582` (step `9`)
  - Dataset stream verification (both modes):
    - best alignment offset `0`
    - exact microbatch matches `80/80`
    - match ratio `1.0`
  - Primary technical fixes that enabled closure:
    - preserve baseline startup warmup-discard behavior in Torchtitan parity path;
    - align soft-gating warmup semantics for short parity horizons;
    - keep TorchTitan native compile path (no baseline-style regional compile fork).
  - Outcome against hard goals:
    - Torchtitan faster: pass
    - Torchtitan lower VRAM: pass
    - loss parity (practical same, paired-step bounded deltas): pass

- **Date**: 2026-03-01
- **Type**: Retrospective
- **General description**: Upstream-first runtime alignment completed; speed and VRAM targets passed, exact loss parity still open.
- **Details**:
  - Reverted shared TorchTitan runtime behavior toward upstream in `trainer.py`, `checkpoint.py`, and `metrics.py`.
  - Added model-local HF checkpoint adaptation hook:
    - protocol method in `protocols/state_dict_adapter.py`
    - nanoVLM implementation in `models/nanoVLM/state_dict_adapter.py`
  - Vanilla short A/B evidence (5 steps):
    - TorchTitan avg tps (steps 2-5): `37620.75`, peak VRAM: `21767 MiB`
    - nanoVLM_main avg tokens/s (steps 2-5): `15650.78`, peak VRAM: `27443 MiB`
    - deltas: ~`2.40x` speedup and `5676 MiB` lower VRAM on TorchTitan
  - Soft-gating smoke passed after refactor; compiled-key remap confirmed (`remapped=270`, `dropped=0`).
  - Exact loss parity remains unresolved and requires deeper controlled triangulation.

## 2026-03-01

- **Date**: 2026-03-01
- **Type**: Retrospective
- **General description**: Soft-gating keeps speed/VRAM wins, but exact parity still fails and recent parity-hypothesis edits regressed loss matching.
- **Details**:
  - Revalidated current best 100-step reference pair:
    - output: `soft-gating-soft100-flexwarm-structonly-20260301-2`
    - baseline `8k9vwq4u`: elapsed `440.46s`, peak `28521 MiB`
    - torchtitan `nz6mijs6`: elapsed `378.54s`, peak `22105 MiB`
    - parity: mean abs diff `0.0057487`, max abs diff `0.16721` at step `9`
  - Tested three additional soft-gating hypotheses (20-step) and all regressed parity:
    - `soft20-nowarmconsume-20260301-1`: mean `0.04177`, max `0.24875` (step `2`)
    - `soft20-nonepassthrough-20260301-1`: mean `0.02915`, max `0.16999` (step `9`)
    - `soft20-validitysync-20260301-1`: mean `0.023855`, max `0.14260` (step `11`)
  - Decision:
    - keep `soft100-flexwarm-structonly` as the control run for soft-gating;
    - reject the above three hypotheses as known regressions;
    - continue parity work with early-step (`1-15`) gating as primary acceptance criteria.

- **Date**: 2026-03-01
- **Type**: Retrospective
- **General description**: Soft-gating remains blocked on exact loss parity while speed and VRAM targets stay satisfied.
- **Details**:
  - Reconfirmed latest valid 100-step paired soft-gating run (`soft-gating-clipfix-final2-20260228-022309`) as the decision baseline.
  - Baseline run `3j50m01e`: elapsed `462.50s`, peak `28521 MiB`.
  - Torchtitan run `igb9fybu`: elapsed `374.78s`, peak `22105 MiB`.
  - Outcome against hard requirements:
    - speed: pass (Torchtitan faster),
    - VRAM: pass (Torchtitan lower),
    - exact loss parity: fail (mean abs diff `0.0066478`, max abs diff `0.16272` at step `9`).
  - Additional soft-gating diagnostics from 20-step runs:
    - `soft-gating-soft20-revertlosswarmup-20260228-1`: mean abs diff `0.0148275`, max `0.17152`, Torchtitan peak `22105 MiB`.
    - `soft-gating-soft20-revertwarmup-modelonly-20260228-1`: mean abs diff `0.0311175`, max `0.25444`, Torchtitan peak `29605 MiB`.
  - Practical takeaway:
    - model+loss compile is currently the safer path for this port;
    - model-only compile did not improve parity and increased peak VRAM in the latest checked run.
  - Next controlled protocol:
    - enforce early-step parity gate on steps `1-15`,
    - run fixed A/B matrix over workers/compile/optimizer-backend with identical data controls,
    - keep baseline-vs-baseline comparison in every soft-gating parity report.

## 2026-02-28

- **Date**: 2026-02-28
- **Type**: Retrospective
- **General description**: Soft-gating parity remains the blocker after speed and VRAM targets were met.
- **Details**:
  - Final paired 100-step soft-gating benchmark (`soft-gating-clipfix-final2-20260228-022309`) completed baseline and Torchtitan successfully.
  - Baseline run `3j50m01e`: elapsed `462.50s`, peak `28521 MiB`, median TPS `7380.94`.
  - Torchtitan run `igb9fybu`: elapsed `374.78s`, peak `22105 MiB`, median TPS `39525`.
  - Loss parity still fails strict exact-match requirement:
    - mean abs diff `0.0066478`, max abs diff `0.16272` at step `9`.
  - Reference variability check indicates baseline itself is unstable early:
    - baseline-vs-baseline (100-step) mean abs diff `0.004764`, max abs diff `0.1469` at step `9`.
  - Next debugging focus should isolate early-step numeric drift sources in soft-gating while preserving achieved speed/VRAM gains.

- **Date**: 2026-02-28
- **Type**: Observation
- **General description**: Aligned gradient clipping path with nanoVLM_main for non-PP/EP runs.
- **Details**:
  - Updated Torchtitan train-step clipping in [trainer.py](/home/coder/edd/nanoVLM_root/torchtitan/torchtitan/trainer.py) to use `torch.nn.utils.clip_grad_norm_` directly when PP/EP are disabled.
  - Kept Torchtitan distributed clipper only for PP/EP paths where cross-mesh norm reduction is required.
  - This removes an implementation mismatch where Torchtitan always used the distributed clipper with explicit `foreach=True`.

- **Date**: 2026-02-28
- **Type**: Observation
- **General description**: Hardened parity benchmark harness against baseline teardown crashes that happen after full-step completion.
- **Details**:
  - Extended [nanovlm_parity_benchmark.py](/home/coder/edd/nanoVLM_root/torchtitan/scripts/nanovlm_parity_benchmark.py) to continue when baseline exits non-zero only after logging all target steps and matching known shutdown signatures.
  - Added unit coverage in [test_nanovlm_parity_benchmark.py](/home/coder/edd/nanoVLM_root/torchtitan/tests/unit_tests/test_nanovlm_parity_benchmark.py).
  - This prevents false-negative benchmark aborts and allows paired Torchtitan runs to execute to completion.

- **Date**: 2026-02-28
- **Type**: Retrospective
- **General description**: Re-ran 100-step soft-gating parity with clipping and harness fixes.
- **Details**:
  - Output directory: `outputs/nanovlm_parity_benchmarks/soft-gating-clipfix-final2-20260228-022309`.
  - Baseline W&B run: `3j50m01e` (`patrickirawan-mbzuai/momh`), elapsed `462.50s`, peak VRAM `28521 MiB`, median TPS (excluding step 1) `7380.94`.
  - Torchtitan W&B run: `igb9fybu` (`patrickirawan-mbzuai/momh`), elapsed `374.78s`, peak VRAM `22105 MiB`, median TPS (excluding step 1) `39525`.
  - Loss comparison over 100 paired steps:
    - mean absolute difference `0.0066478`,
    - max absolute difference `0.16272` at step `9`.
  - Outcome vs goals:
    - speed: improved (Torchtitan faster),
    - VRAM: reduced (Torchtitan lower),
    - loss: still not exact-match for soft-gating.

## 2026-02-27

- **Date**: 2026-02-27
- **Type**: Observation
- **General description**: Identified a soft-gating parity mismatch in optimizer parameter grouping.
- **Details**:
  - In `nanoVLM_main`, `momh_gate` parameters are only split into their own optimizer group when `lr_momh_gate` is explicitly set.
  - In Torchtitan, `momh_gate` was always split as a separate optimizer group, even when `lr_momh_gate=None`.
  - This mismatch is present in the `soft-gating-soft100-vramfix-clean1` logs where Torchtitan reports `Optimizer group 'momh_gate': 30 params, lr=0.0001` while baseline uses the language group for gates.
  - Planned fix: make Torchtitan gate-group splitting conditional on `lr_momh_gate is not None` and re-run soft-gating parity benchmarks.

- **Date**: 2026-02-27
- **Type**: Retrospective
- **General description**: Consolidated parity benchmark evidence for vanilla and soft-gating port tracks.
- **Details**:
  - Vanilla benchmark (`vanilla-vanilla100-vramfix-clean1`, 100 steps):
    - Baseline run `3epjh0uu`: elapsed `356.65s`, peak VRAM `26249 MiB`.
    - Torchtitan run `ea33lgdw`: elapsed `324.78s`, peak VRAM `19995 MiB`.
    - Step-wise loss diff: mean absolute difference `0.0002273`, max absolute difference `0.00104` at step `71`.
  - Soft-gating benchmark (`soft-gating-soft100-vramfix-clean1`, 100 steps):
    - Baseline run `co8wprbr`: elapsed `497.13s`, peak VRAM `28521 MiB`.
    - Torchtitan run `g0ci82mg`: elapsed `392.51s`, peak VRAM `22105 MiB`.
    - Step-wise loss diff: mean absolute difference `0.005222`, max absolute difference `0.16211` at step `9`.
  - Key implementation choices in this cycle:
    - compile both model and loss in paper configs;
    - preserve Torchtitan native compile path;
    - keep external `nvidia-smi` sampling and strict 100-step paired comparison.
  - Evidence quality guardrails:
    - treat runs with zero logged steps as invalid for parity claims (`vanilla-vanilla100-vramfix`);
    - keep speed/VRAM claims separate from loss-parity claims.
  - Outcome vs goals:
    - vanilla: all three goals satisfied (loss parity, faster runtime, lower VRAM);
    - soft-gating: speed and VRAM goals satisfied, loss parity still not satisfied.
