# LLM Bench — Cross-Platform Local LLM Inference Benchmark

Controlled benchmark of Qwen3.5 models (9B / 27B / 35B-A3B MoE / 122B-A10B MoE) across **4 hardware platforms × 5 inference engines**.

> **Methodology**: Cold prefill (`--no-cache-prompt`), per-run random nonce, server restart between prefill tracks, randomized execution order. 5 runs per combo, median. CV<0.3 outlier filter. **~5,100 total measurements**.

---

## Generation Speed (Track B — same llama.cpp, same GGUF)

### Q4_K_M (gen-512, tok/s)

| Model | M5 Max (128GB) | RTX 3090×2 (48GB) | DGX Spark GB10 (128GB) | Ryzen AI MAX 395 (96GB) |
|-------|:-:|:-:|:-:|:-:|
| **9B** Dense | 75.9 | **117.6** | 36.8 | 32.6 |
| **27B** Dense | 24.8 | **41.4** | 11.5 | 10.3 |
| **35B-A3B** MoE (3B active) | 94.1 | **138.9** | 59.6 | 58.0 |
| **122B-A10B** MoE (10B active) | 42.9 | OOM | 21.7 | 22.9 |

### Q8_0 (gen-512, tok/s)

| Model | M5 Max | RTX 3090×2 | DGX Spark | Ryzen AI |
|-------|:-:|:-:|:-:|:-:|
| **9B** | 50.8 | **82.2** | 24.3 | 21.7 |
| **27B** | 16.9 | **27.5** | 7.6 | 7.1 |
| **35B-A3B** MoE | 88.4 | **130.3** | 52.6 | 50.8 |

---

## Prefill Throughput (Track B — llama.cpp, Q4_K_M, tok/s)

### 9B

| Input | M5 Max | 3090×2 | DGX Spark | Ryzen AI |
|-------|------:|------:|------:|------:|
| 1K | 1,705 | **3,258** | 2,217 | 205 |
| 4K | 1,844 | **5,317** | 2,490 | 278 |
| 16K | 1,590 | **6,244** | 2,239 | 915 |
| 64K | 955 | **5,827** | 1,093 | 159 |
| 128K | 711 | **4,952** | 986 | 56 |

### 35B-A3B MoE

| Input | M5 Max | 3090×2 | DGX Spark | Ryzen AI |
|-------|------:|------:|------:|------:|
| 1K | 2,302 | **3,372** | 1,602 | 732 |
| 4K | 2,798 | **5,302** | 1,949 | 924 |
| 16K | 2,417 | **6,131** | 1,696 | 960 |
| 64K | 1,214 | **3,726** | 1,180 | 767 |
| 128K | 732 | **3,142** | 856 | 582 |

### 122B-A10B MoE

| Input | M5 Max | 3090×2 | DGX Spark | Ryzen AI |
|-------|------:|------:|------:|------:|
| 1K | **815** | OOM | 536 | 215 |
| 4K | **980** | OOM | 663 | 275 |
| 16K | **722** | OOM | 614 | 312 |
| 64K | 439 | OOM | **445** | 258 |
| 128K | 296 | OOM | **341** | 205 |

---

## Engine Comparison (Track A — gen-512, Q4_K_M, tok/s)

> Same hardware, different engines. Cross-platform comparison uses Track B above.

### M5 Max — MLX vs llama.cpp

| Model | MLX | llama.cpp | Ollama |
|-------|----:|----------:|-------:|
| 9B | **102.4** | 75.4 | 29.2 |
| 27B | **28.8** | — | — |
| 35B-A3B | **138.3** | 91.0 | — |
| 122B | **66.8** | 38.5 | — |

### RTX 3090×2 — vLLM vs llama.cpp vs Ollama

| Model | vLLM GPTQ | llama.cpp | Ollama |
|-------|----------:|----------:|-------:|
| 9B | 83.6 | **117.3** | 100.5 |
| 27B | 19.3 | **41.5** | 36.7 |
| 35B-A3B | **156.3** | 138.6 | 101.7 |
| 122B | N/A | OOM | 4.7 |

### DGX Spark — llama.cpp vs Ollama vs vLLM (Docker)

| Model | llama.cpp | Ollama | vLLM Docker |
|-------|----------:|-------:|------------:|
| 9B | **35.7** | 35.1 | 12.9 |
| 27B | **11.5** | 11.4 | 8.5 |
| 35B-A3B | **61.2** | 59.2 | 34.8 |
| 122B | **22.0** | 6.6 | N/A |

### Ryzen AI MAX 395 — llama.cpp vs Ollama vs Lemonade

| Model | llama.cpp | Ollama | Lemonade |
|-------|----------:|-------:|---------:|
| 9B | **36.2** | 31.9 | 33.2 |
| 27B | **12.3** | 11.1 | 11.3 |
| 35B-A3B | **58.4** | 43.9 | 48.0 |
| 122B | **22.8** | 4.6 | N/A |

---

## Prefill Engine Comparison (prefill-16k, Q4_K_M, tok/s)

| Engine × Hardware | 9B | 27B | 35B MoE | 122B |
|-------------------|---:|----:|--------:|-----:|
| **3090 vLLM** | 8,398 | 2,845 | **13,146** | N/A |
| DGX vLLM Docker | 6,773 | 1,614 | 4,331 | N/A |
| 3090 llama.cpp | 6,236 | 1,799 | 4,186 | OOM |
| Mac MLX | 3,011 | 784 | 3,774 | 1,281 |
| 3090 Ollama | 3,101 | 998 | 2,239 | 141 |
| DGX llama.cpp | 2,236 | 625 | 1,694 | 623 |
| Mac llama.cpp | 1,291 | 352 | 2,412 | 658 |
| Ryzen llama.cpp | 915 | 298 | 960 | 313 |

---

## MoE Efficiency

35B-A3B MoE (3B active) is **faster than 9B Dense on all platforms**:

| Hardware | 9B Dense | 35B MoE | Speedup |
|---------|----------|---------|--------:|
| M5 Max | 75.9 | **94.1** | +24% |
| 3090×2 | 117.6 | **138.9** | +18% |
| DGX Spark | 36.8 | **59.6** | +62% |
| Ryzen AI | 32.6 | **58.0** | +78% |

The lower the memory bandwidth, the bigger the MoE advantage — loading 3B weights per token instead of 9B makes a massive difference on bandwidth-limited platforms.

---

## Hardware

| | M5 Max | RTX 3090×2 | DGX Spark GB10 | Ryzen AI MAX 395 |
|--|:--:|:--:|:--:|:--:|
| GPU | Apple GPU 40C | RTX 3090 ×2 (Ampere) | GB10 Blackwell | Radeon 8060S (RDNA 3.5) |
| Memory | 128GB unified | 128GB DDR4 + 48GB VRAM | 128GB unified | 128GB unified (96GB VRAM) |
| Bandwidth | **546 GB/s** | ~936 GB/s GDDR6X | 273 GB/s | 256 GB/s |

## Models

| Model | Type | Total | Active | Context |
|-------|------|------:|-------:|--------:|
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Dense | 9B | 9B | 256K |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Dense | 27B | 27B | 256K |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | MoE | 35B | ~3B | 256K |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | MoE | 122B | ~10B | 256K |

Quantization: [unsloth](https://huggingface.co/unsloth) Dynamic 2.0 GGUF (Q4_K_M, Q8_0)

## Engines

| Engine | Platforms | Notes |
|--------|-----------|-------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | All 4 | Universal baseline for Track B |
| [MLX](https://github.com/ml-explore/mlx) | Mac only | Apple Silicon optimized |
| [Ollama](https://ollama.com/) | All 4 | Easy-to-use wrapper |
| [vLLM](https://github.com/vllm-project/vllm) | 3090, DGX | GPTQ/BF16, Docker on DGX |
| [Lemonade](https://lemonade-server.ai/) | Ryzen AI | AMD inference server |

---

## Key Findings

1. **Memory bandwidth determines generation speed.** RTX 3090×2 at 936 GB/s dominates everything that fits in 48GB VRAM.
2. **Unified memory runs everything.** Mac, DGX Spark, and Ryzen AI all run 122B. The 3090 can't even load it.
3. **MoE flips the rankings.** 35B-A3B (3B active) is 18-78% faster than 9B Dense — active parameter count matters more than total params.
4. **DGX Spark ≈ Ryzen AI MAX 395.** Despite different architectures (Blackwell vs RDNA 3.5), within 10-15% on most models. Both bandwidth-limited at ~256-273 GB/s.
5. **vLLM GPTQ-Marlin sets the record.** 35B MoE at **156.3 tok/s** on 3090 — fastest single result.
6. **llama.cpp is the universal winner.** Best generation speed on 3 of 4 platforms (MLX wins on Mac).

---

## Experiment Design (v3)

- **Track B**: Same llama.cpp + same GGUF across all hardware → pure hardware comparison
- **Track A**: All available backends per platform → engine comparison (within-platform only)
- `--no-cache-prompt` + `--slot-prompt-similarity 0` + `--no-enable-prefix-caching`
- Per-run random nonce prefix (no prompt cache hits possible)
- Server restart between prefill tracks (cold KV cache)
- Randomized backend / model / track execution order
- Warmup 1 (separate prompt) + Measure 5, median
- Thermal guard: 85°C → 60s cooldown
- OOM/failures recorded as skip rows in CSV

---

## Quick Start

```bash
uv sync

# Track B — hardware comparison (llama.cpp only)
uv run python -m src.runner --config config.yaml --backends llamacpp

# Track A — engine comparison (all backends)
uv run python -m src.runner --config config.yaml --backends llamacpp ollama mlx

# Specific models/tracks
uv run python -m src.runner --models qwen3.5-35b-a3b --tracks gen-512 prefill-16k
```

## Request a Benchmark

Want a specific model/quantization tested on this hardware? [Open an issue](https://github.com/baem1n/llm-bench/issues/new?template=experiment_request.md).

This tool measures **single-stream inference throughput** under controlled conditions — not model quality, accuracy, or multi-user serving performance.

---

## Blog

- [Part 1: Experiment Design](https://baem1n.dev/posts/llm-bench-01-methodology/)
- [Part 2: Detailed Analysis](https://baem1n.dev/posts/llm-bench-02-results-analysis/)
- [Part 3: Results Tables](https://baem1n.dev/posts/llm-bench-03-results-tables/)

## License

MIT
