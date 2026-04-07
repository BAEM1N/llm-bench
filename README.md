# LLM Bench — Qwen3.5 Cross-Platform Inference Benchmark

4대 하드웨어에서 Qwen3.5 모델의 **생성 속도(Gen TPS)**와 **프리필 속도(Prefill TPS)**를 동일 조건으로 측정.

> **v3 실험 설계**: prompt cache 차단, cold prefill, 실행 순서 랜덤화, run별 프롬프트 재생성.  
> 상세 방법론 → [블로그: 실험 설계편](https://baem1n.github.io/posts/llm-bench-01-methodology)

---

## 하드웨어 비교 (Track B — 동일 llama.cpp + 동일 GGUF)

> 변수: **하드웨어만**. 엔진·가중치·설정 모두 동일.

### Generation TPS (gen-512, Q4_K_M, 중앙값)

| Model | 🍎 Mac M5 Max | 🖥️ 3090 ×2 | 🔷 DGX Spark | 🔶 Ryzen AI 395 |
|-------|:------------:|:----------:|:-----------:|:--------------:|
| **9B** Dense | 75.9 | **117.5** | 36.8 | 36.2¹ |
| **27B** Dense | 24.4 | **41.4** | 11.5 | 26.9² |
| **35B-A3B** MoE | 94.1 | **135.0** | 59.6 | 58.0 |
| **122B-A10B** MoE | 42.9 | **130.7** | — | 22.9 |

> ¹ Ryzen 9B Track B 데이터 일부 이상 (config 문제 가능성). Track A에서 36.2 tok/s.  
> ² Ryzen 27B Track B 불완전 (1/4 gen). Track A 기준 12.3 tok/s.  
> DGX 122B: GGUF 파일 다운로드 실패 → 미측정.

### Prefill TPS (prefill-1k, Q4_K_M, 중앙값)

| Model | 🍎 Mac | 🖥️ 3090 | 🔷 DGX | 🔶 Ryzen |
|-------|:------:|:-------:|:------:|:-------:|
| **9B** | 1,705 | **2,774** | 2,217 | — |
| **27B** | 504 | **961** | 674 | — |
| **35B-A3B** | 2,302 | **2,819** | 1,602 | 732 |
| **122B** | **815** | — | — | 215 |

---

## 엔진 비교 (Track A — 플랫폼 내부)

> 같은 플랫폼 안에서 백엔드별 성능 차이. **다른 플랫폼 간 비교 불가.**

### 🍎 Mac M5 Max (gen-512, Q4_K_M)

| Backend | 9B | 27B | 35B-A3B | 122B |
|---------|---:|----:|--------:|-----:|
| **MLX** | 102.4 | 28.8 | **138.3** | **66.8** |
| **llama.cpp** | 75.9 | 24.4 | 94.1 | 42.9 |
| **Ollama** | 29.2 | — | — | — |

### 🖥️ 3090 2-WAY (gen-512)

| Backend | 9B Q4 | 9B Q8 | 27B Q4 | 35B Q4 |
|---------|------:|------:|-------:|-------:|
| **vLLM** | 83.6 | 83.5 | — | **156.3** |
| **llama.cpp** | 117.3 | 130.6 | 41.5 | 138.6 |
| **Ollama** | 100.5 | 73.5 | 36.7 | 101.7 |

### 🔷 DGX Spark (gen-512)

| Backend | 9B Q4 | 27B Q4 | 35B Q4 | 122B Q4 |
|---------|------:|-------:|-------:|--------:|
| **llama.cpp** | 35.7 | 11.5 | **61.2** | — |
| **Ollama** | 35.1 | 11.4 | 59.2 | 6.6 |

### 🔶 Ryzen AI MAX 395 (gen-512)

| Backend | 9B Q4 | 27B Q4 | 35B Q4 | 122B Q4 |
|---------|------:|-------:|-------:|--------:|
| **llama.cpp** | **36.2** | **12.3** | **58.4** | **22.8** |
| **Ollama** | 31.9 | 11.1 | 43.9 | 4.6 |

---

## Key Findings

1. **MoE가 Dense를 압도**: 35B-A3B (3B active) > 9B Dense, 전 플랫폼 공통
2. **3090 2-WAY 절대 속도 1위**: GDDR6X 936 GB/s 대역폭 이점 (122B에서 131 tok/s)
3. **Mac M5 Max 균형 최강**: TTFT 120ms 안정, MLX 35B 138 tok/s
4. **vLLM 35B GPTQ-Marlin = 156 tok/s**: 3090에서 llamacpp(139) 대비 12% 빠름
5. **DGX Spark 대역폭 병목**: 273 GB/s로 Mac(546)의 절반 속도
6. **Ryzen AI 122B 실행 가능**: 96GB VRAM으로 65GB 모델 풀 GPU, 22.9 tok/s
7. **Ollama TTFT 구조적 문제**: 256K KV pre-allocation으로 TTFT 5~112초

---

## Hardware Specs

| | 🍎 Mac M5 Max | 🖥️ 3090 ×2 | 🔷 DGX Spark | 🔶 Ryzen AI 395 |
|--|:--:|:--:|:--:|:--:|
| GPU | Apple GPU 40C | RTX 3090 ×2 | GB10 Blackwell | Radeon 8060S |
| Memory | 128GB unified | 128GB + 48GB VRAM | 128GB unified | 128GB (96GB VRAM) |
| Bandwidth | **546 GB/s** | ~936 GB/s (GDDR6X) | 273 GB/s | 256 GB/s |

---

## Experiment Design (v3)

- **Track B**: 동일 llama.cpp + 동일 GGUF → 하드웨어 비교
- **Track A**: 플랫폼별 가용 백엔드 전부 → 엔진 비교 (플랫폼 내부만)
- `--no-cache-prompt` + `--slot-prompt-similarity 0` (cache 차단)
- `--no-enable-prefix-caching` (vLLM prefix cache 차단)
- Run별 랜덤 nonce prefix 프롬프트 (cold prefill 보장)
- Prefill track마다 서버 재시작
- Backend / model / track 순서 랜덤화
- Warmup 1회 (별도 프롬프트) + Measure 5회, 중앙값
- OOM/실패 → CSV에 skip row 기록

---

## Data

| Platform | Total Rows | OK |
|----------|----------:|---:|
| 🖥️ 3090 | 1,491 | 1,366 |
| 🔷 DGX | 1,279 | 1,189 |
| 🔶 Ryzen | 1,105 | 1,041 |
| 🍎 Mac | 1,106 | 1,028 |

---

## Quick Start

```bash
uv sync

# Track B (하드웨어 비교)
uv run python -m src.runner --config config.yaml --backends llamacpp

# Track A (엔진 비교)
uv run python -m src.runner --config config.yaml --backends llamacpp ollama mlx

# 특정 모델/트랙
uv run python -m src.runner --models qwen3.5-35b-a3b --tracks gen-512 prefill-1k
```

---

## Status

- [x] MacBook Pro 14 (M5 Max) — llamacpp ✅ ollama 🔄 mlx ✅
- [x] RTX 3090 ×2 — llamacpp ✅ ollama ✅ vllm ✅
- [x] DGX Spark — llamacpp ✅ ollama ✅ vllm 🔄 (Docker 빌드 중)
- [x] Ryzen AI MAX 395 — llamacpp ✅ ollama ✅ lemonade ⏳ vllm ⏳

---

## Blog

- [Part 1: 실험 방법론](https://baem1n.github.io/posts/llm-bench-01-methodology)
- [Part 2: 성능 비교 결과](https://baem1n.github.io/posts/llm-bench-02-results)
