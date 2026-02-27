# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| (Template) | Describe the error message or behavior | Root cause analysis | Step-by-step fix |
| Torchtitan CLI boolean flag mismatch | Training command fails with `Unrecognized options: false` | Torchtitan CLI expects negation flags (for example `--metrics.no-track_peak_memory`) rather than passing string booleans like `false` | Use the correct negation flags (`--*.no-*`) and verify option names with `python -m torchtitan.train --help` before long runs |
| HF streaming fetch transient failure | Run fails during dataset startup with errors like `Temporary failure in name resolution` or `Cannot send a request, as the client has been closed` | Temporary network/DNS instability while reading parquet shards from remote Hugging Face streaming datasets | Re-run after network recovers; avoid treating failed startup runs as parity/performance evidence; keep a successful foreground validation run before launching long tmux jobs |
| Loss or memory regression after moving image flattening into model forward | Step loss drifts from baseline and peak memory rises after collation changes | Changing image container semantics (list-based collation + in-forward flattening) altered runtime behavior and shape flow | Keep images flattened to tensor in dataloader collation for this port, and validate parity with paired 100-step A/B runs |
