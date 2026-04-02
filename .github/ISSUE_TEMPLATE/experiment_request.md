---
name: Experiment Request
about: Request benchmarking of a new model, backend, or hardware platform
title: "[Experiment] "
labels: experiment-request
assignees: ''
---

> [!WARNING]
> Experiment requests from **unverified individuals or organizations** may be ignored without notice.
> To be considered, the model should be published by a recognized research lab, company, or well-established open-source maintainer on Hugging Face.

## Model

**Hugging Face Model ID:**
<!-- e.g. `Qwen/Qwen3.5-9B` -->

## Target Device

- [ ] MacBook Pro 14 (M5 Max, 128GB)
- [ ] DGX Spark (GB10, 128GB)
- [ ] Ryzen AI MAX+ 395 (128GB)
- [ ] Other: 

## Requested Backends
- [ ] MLX
- [ ] llama.cpp
- [ ] Ollama
- [ ] vLLM
- [ ] Other: 

## Requested Tracks

**Generation** — short input (64 tok), fixed output length, measures decode speed

- [ ] `gen-512` — 512 output tokens
- [ ] `gen-2048` — 2,048 output tokens
- [ ] `gen-4096` — 4,096 output tokens
- [ ] `gen-8192` — 8,192 output tokens

**Prefill** — minimal output (10 tok), varying input length, measures KV cache fill speed

- [ ] `prefill-1k` — 1,024 input tokens
- [ ] `prefill-4k` — 4,096 input tokens
- [ ] `prefill-16k` — 16,384 input tokens
- [ ] `prefill-64k` — 65,536 input tokens
- [ ] `prefill-128k` — 131,072 input tokens

**Concurrency** — closed-loop parallel requests, measures aggregate throughput & latency under load

- [ ] `Pass 1` — Coarse scan · levels 1 / 2 / 4 / 8 · tracks gen-512, gen-2048
- [ ] `Pass 2` — Boundary refinement · higher levels (model-dependent) · tracks gen-512, gen-2048
- [ ] `Pass 3` — Long response stress · levels from Pass 2 boundary · track gen-8192
- [ ] `Pass 4` — Optional extensions (Q8_0, prefill concurrency, open-loop arrival rate)

## Additional Context
