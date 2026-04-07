# LLM Bench — Qwen3.5 Cross-Platform Inference Benchmark

4대 하드웨어 × 5개 엔진에서 Qwen3.5 모델의 **생성 속도(Gen TPS)**와 **프리필 속도(Prefill TPS)**를 동일 조건으로 측정.

> **v3 실험 설계**: prompt cache 차단 (`--no-cache-prompt`), cold prefill (서버 재시작), 실행 순서 랜덤화, run별 nonce prefix.  
> 각 조합 5회 측정 중앙값, CV<0.3 필터 적용. 총 **~4,700회** 측정.

---

## 생성 속도 (Track B — 동일 llama.cpp + 동일 GGUF)

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

## 프리필 속도 (Track B — llama.cpp, Q4_K_M, tok/s)

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

## 엔진 비교 (Track A — gen-512, Q4_K_M, tok/s)

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
| 122B | — | — | 4.7 🚫 |

### DGX Spark — llama.cpp vs Ollama vs vLLM Docker

| Model | llama.cpp | Ollama | vLLM Docker |
|-------|----------:|-------:|------------:|
| 9B | **35.7** | 35.1 | 12.9 |
| 27B | **11.5** | 11.4 | 8.5 |
| 35B-A3B | **61.2** | 59.2 | 34.8 |
| 122B | **22.0** | 6.6 | — |

### Ryzen AI MAX 395 — llama.cpp vs Ollama vs Lemonade

| Model | llama.cpp | Ollama | Lemonade |
|-------|----------:|-------:|---------:|
| 9B | **36.2** | 31.9 | 6.5 |
| 27B | **12.3** | 11.1 | 11.4 |
| 35B-A3B | **58.4** | 43.9 | 48.0 |
| 122B | **22.8** | 4.6 🚫 | — |

---

## 프리필 엔진 비교 (prefill-16k, Q4_K_M, tok/s)

| Engine × Hardware | 9B | 27B | 35B MoE | 122B |
|-------------------|---:|----:|--------:|-----:|
| **3090 vLLM** | 8,398 | 2,845 | **13,146** | — |
| DGX vLLM Docker | 6,773 | 1,614 | 4,331 | — |
| 3090 llama.cpp | 6,236 | 1,799 | 4,186 | — |
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
| Qwen3.5-9B | Dense | 9B | 9B | 256K |
| Qwen3.5-27B | Dense | 27B | 27B | 256K |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 256K |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 256K |

---

## Key Findings

1. **3090×2 absolute speed king** — 936 GB/s GDDR6X bandwidth. 35B MoE at 139 tok/s.
2. **M5 Max best for daily use** — stable TTFT (120ms), MLX 35B MoE at 138 tok/s.
3. **vLLM GPTQ-Marlin top record** — 35B MoE at **156.3 tok/s** on 3090.
4. **DGX Spark bandwidth-limited** — 273 GB/s = half of Mac's 546.
5. **Ryzen AI runs 122B** — $2K mini PC at 22.9 tok/s.
6. **MoE universally efficient** — 35B-A3B (3B active) > 9B Dense, +18~78% across all platforms.

---

## Experiment Design (v3)

- **Track B**: Same llama.cpp + same GGUF → hardware comparison
- **Track A**: All available backends per platform → engine comparison (within platform only)
- `--no-cache-prompt` + `--slot-prompt-similarity 0` + `--no-enable-prefix-caching`
- Per-run random nonce prefix prompt (cold prefill)
- Server restart between prefill tracks
- Randomized backend / model / track order
- Warmup 1 (separate prompt) + Measure 5, median
- OOM/failures recorded as skip rows in CSV

---

## Quick Start

```bash
uv sync

# Track B (hardware comparison)
uv run python -m src.runner --config config.yaml --backends llamacpp

# Track A (engine comparison)
uv run python -m src.runner --config config.yaml --backends llamacpp ollama mlx

# Specific models/tracks
uv run python -m src.runner --models qwen3.5-35b-a3b --tracks gen-512 prefill-16k
```

---

## Blog

- [Part 1: 실험 방법론](https://baem1n.github.io/posts/llm-bench-01-methodology)
- [Part 2: 상세 분석](https://baem1n.github.io/posts/llm-bench-02-results-analysis)
- [Part 3: 결과표](https://baem1n.github.io/posts/llm-bench-03-results-tables)
