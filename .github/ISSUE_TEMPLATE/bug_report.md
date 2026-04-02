---
name: Bug Report
about: Report incorrect metrics, crashes, or unexpected benchmark behavior
title: "[Bug] "
labels: bug
assignees: ''
---

## Description
<!-- What went wrong? -->

## Environment
- Hardware: <!-- e.g. MacBook Pro 14 M5 Max -->
- Backend: <!-- mlx / llamacpp / ollama / vllm -->
- Backend version:
- Python / uv version:

## Track & Model
- Track ID: <!-- e.g. gen-512, prefill-16k -->
- Model: <!-- e.g. Qwen3.5-9B -->
- Quantization: <!-- e.g. Q4_K_M -->

## Expected vs Actual
**Expected:**

**Actual:**

## Reproduction
```bash
uv run python -m src.runner --config config.yaml ...
```

## Logs / Output
<!-- Paste relevant console output or result CSV rows -->

## Additional Context
