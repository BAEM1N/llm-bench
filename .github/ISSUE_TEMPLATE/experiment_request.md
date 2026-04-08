---
name: Experiment Request
about: Request benchmarking of a new model, backend, or hardware platform
title: "[Experiment] "
labels: experiment-request
assignees: ''
---

> [!NOTE]
> This benchmark measures **single-stream inference speed** (generation tok/s, prefill tok/s, TTFT) under controlled conditions — not model quality, accuracy, or multi-user throughput.
> Models should be available on Hugging Face (GGUF, GPTQ, or BF16).

## Model

**Hugging Face Model ID:**
<!-- e.g. `Qwen/Qwen3.5-9B` -->

## Target Device

- [ ] MacBook Pro 14 (M5 Max, 128GB unified)
- [ ] RTX 3090×2 (Ryzen 9 5950X, 48GB VRAM)
- [ ] DGX Spark (GB10, 128GB unified)
- [ ] Ryzen AI MAX+ 395 (HP Z2 Mini G1a, 96GB VRAM)
- [ ] All available
- [ ] Other: 

## Requested Backends
- [ ] llama.cpp
- [ ] MLX (Mac only)
- [ ] Ollama
- [ ] vLLM
- [ ] Lemonade (Ryzen AI only)
- [ ] All available for target device

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
