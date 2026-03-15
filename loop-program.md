# Autonomous Loop Program

## Task
Fix the 2-GPU FSDP aux-loss soft-gating WSM run so it no longer hangs or aborts
before the first logged training step.

## Success Criteria
Command-based:

```bash
source ../nanoVLM_main/.venv/bin/activate && \
timeout 600 torchrun --standalone --max-restarts=0 --nproc_per_node=2 \
  -m torchtitan.train \
  --module nanoVLM \
  --config nanovlm_230m_momh_soft_gating_b5_tttv_nopack_balance_aux_wsm \
  --dump_folder /tmp/aux_sigabrt_autoloop \
  --training.steps 2 \
  --training.global-batch-size 64 \
  --training.local-batch-size 32 \
  --dataloader.num_workers 2 \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.data_parallel_shard_degree 2 \
  --comm.init-timeout-seconds 300 \
  --comm.train-timeout-seconds 100 \
  --metrics.log_freq 1 \
  --metrics.no-enable-wandb \
  --activation-checkpoint.mode full
```

Exit code `0` means success. Timeout or non-zero exit means failure.

## Files in Scope
- torchtitan/models/nanoVLM/
- torchtitan/trainer.py
- tests/unit_tests/
- loop-program.md
- results.tsv

## The Experiment Loop

The experiment runs on a dedicated branch.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
   Read `results.tsv` to understand history. Decide what to try next.
   Consider: What has been tried? What nearly worked? What hasn't been explored?
   If <advise> skills are available in .codex/skills/, consult them.
   Prefer one focused hypothesis per iteration.

2. Modify only in-scope files. Make a focused, single-idea change.

3. `git add <in-scope files> && git commit -m "autoloop #N: <description>"`

4. Run the success criteria command / assessment (see Evaluation below).

5. Record the results in `results.tsv` (tab-separated).

6. If the result is an improvement: you "advance" the branch, keeping the
   git commit.
   If the result is equal or worse: `git reset --hard HEAD~1` to revert
   back to where you started.

7. Every 5 iterations:
   - Run <retrospective> — summarize learnings, update result skills
   - Run $code-simplifier on kept changes
   - Commit any simplifier/retrospective artifacts

8. Re-read this file (loop-program.md) to restore full context after
   compaction. Also re-read results.tsv to know current state.

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep. If they don't, discard. And you're advancing the branch so
that you can iterate. If you feel like you're getting stuck in some way, you can
rewind but you should probably do this very very sparingly (if ever).

## Evaluation

**Command-based criteria:**
- Run the success criteria command. Exit code 0 = keep, non-zero = discard.
- Capture stdout/stderr summary for the results.tsv description.

## Error Handling

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment:
If it's something dumb and easy to fix (e.g. a typo, a missing import), fix
it and re-run. If the idea itself is fundamentally broken, just skip it, log
it as discarded in the tsv, and move on.

**Timeout**: If a run exceeds a reasonable time (2x the expected duration),
kill it and treat it as a failure (discard and revert).

**Git conflicts**: Should not happen (single branch, linear history). If they
do, resolve by taking the current branch version.

**Tool failures**: Log the error in results.tsv as a discard, continue.

## Rules

**NEVER STOP**: Once the experiment loop has begun (after the initial setup),
do NOT pause to ask the human if you should continue. Do NOT ask "should I
keep going?" or "is this a good stopping point?". The human might be asleep,
or gone from a computer and expects you to continue working *indefinitely*
until you are manually stopped. You are autonomous. If you run out of ideas,
think harder — re-read the in-scope files for new angles, try combining
previous near-misses, try more radical changes, try the opposite of what
failed. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The
user then wakes up to experimental results, all completed by you while they
slept!

- **NEVER ask questions** during the loop. Make your best judgment call.
- Log EVERYTHING to results.tsv.
- Keep each iteration focused — one idea per commit.
- Always commit before evaluating. This ensures clean keep/discard via git.
- Always re-read loop-program.md at the start of each iteration.
  Context compaction will erase your memory — this file is your source of truth.
