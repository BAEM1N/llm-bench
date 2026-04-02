# LLM Bench — Qwen3.5 Cross-Platform Benchmark

Benchmarking the **Qwen3.5 model family** (9B / 27B / 35B-A3B MoE / 122B-A10B MoE) across hardware platforms and inference backends.

Measured: Generation TPS, Prefill TPS, TTFT, Peak Memory, E2E Latency.

---

## Hardware Platforms

| Platform | Memory | Status | Backends |
|----------|--------|--------|----------|
| **MacBook Pro (Apple Silicon)** | 128GB Unified | ✅ Complete | MLX · llama.cpp · Ollama |
| **NVIDIA DGX Spark (GB10)** | 128GB Unified | 🔜 Planned | vLLM · llama.cpp · Ollama |
| **AMD Ryzen AI MAX+ 395** | 128GB DDR5 | 🔜 Planned | llama.cpp (ROCm) · Ollama · vLLM (ROCm) |

---

## Models

| Model | Type | Total Params | Active Params | Context |
|-------|------|-------------|---------------|---------|
| Qwen3.5-9B | Dense | 9B | 9B | 256K |
| Qwen3.5-27B | Dense | 27B | 27B | 256K |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 256K |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 256K |

All models use **Linear Attention + Full Attention (3:1 ratio)** hybrid architecture.

---

## Results: MacBook Pro (Apple Silicon, 128GB)

> Warmup 1 run + median of 4 runs. Thermal guard at ~88°C → 60s cooldown.
>
> **해석 주의**: MLX는 자체 4-bit 가중치, llama.cpp/Ollama는 unsloth GGUF 사용 → 엔진+가중치 패키지 비교에 가까움. 상세 caveat → [REPORT.md](REPORT.md#측정-한계-및-해석-주의사항)

### Generation TPS — Q4_K_M

| Model | MLX | llama.cpp | Ollama |
|-------|----:|----------:|-------:|
| 9B | **101.6** | 73.8 | 56.4 |
| 27B | **30.9** | 20.4 | 17.9 |
| 35B-A3B (MoE) | **142.2** | 93.6 | 60.4 |
| 122B-A10B (MoE) | **66.8** | 39.4 | 28.9 |

### Generation TPS — Q8_0

| Model | MLX | llama.cpp | Ollama |
|-------|----:|----------:|-------:|
| 9B | **59.0** | 51.1 | 41.1 |
| 27B | **17.2** | 14.9 | 12.8 |
| 35B-A3B (MoE) | **102.9** | 88.2 | 53.5 |
| 122B-A10B (MoE) | N/A | N/A | N/A |

> 122B Q8_0: 128GB 유니파이드 메모리 한계 초과 — 전 백엔드 측정 불가

### Prefill TPS — 128k context, Q4_K_M

| Model | llama.cpp | Ollama | MLX |
|-------|----------:|-------:|----:|
| 9B | **574,324** (229ms) | 65,011 (2s) | 2,821 (46s) |
| 27B | **441,661** (297ms) | OOM | 797 (162s) |
| 35B-A3B (MoE) | **571,342** (231ms) | 61,548 | 2,980 |
| 122B-A10B (MoE) | **488,660** (269ms) | 27,748 | 1,135 |

> llama.cpp Flash Attention 압도. 128k 처리를 229ms에 완료.  
> Ollama 27B 128k: num_ctx=262144 전체 KV 캐시 pre-allocate → OOM

### Memory Usage — Q4_K_M (GB)

| Model | MLX | llama.cpp | Ollama |
|-------|----:|----------:|-------:|
| 9B | **4.7** | 6.7 | 19.9 |
| 27B | **14.1** | 17.8 | 40.4 |
| 35B-A3B (MoE) | **18.2** | 21.5 | 33.3 |
| 122B-A10B (MoE) | **64.0** | 72.8 | 92.0 |

> Ollama는 262K KV 캐시 사전 할당으로 2–3× 높은 메모리 — 알고리즘 설정 차이이며 불리한 비교 조건임

### Key Insights

- **35B MoE는 9B보다 빠르다** — hidden_size 2,048 (9B의 절반), active params 3B. MLX 기준 142 vs 101 tok/s, 메모리는 18.2 vs 4.7GB
- **llama.cpp Flash Attention이 prefill 압도**: 128k 기준 MLX 대비 204×, Ollama 대비 8.8× 빠름 (229ms vs 46s vs 2s)
- **MoE prefill 동치**: 35B MoE(571K) ≈ 9B Dense(574K) — sparse 계산이 active params 수준으로 수렴
- **MLX가 generation에서 우위** (Metal 4-bit 커널), prefill은 16k 이후 TPS 감소 (Flash Attention 미적용)
- **Q4 vs Q8 속도 비**: MLX 1.72× (9B), llama.cpp 1.06× (35B) — llama.cpp 35B는 Q4에서도 이미 메모리 대역폭 한계 해소

### E2E Latency — gen-8192, Q4_K_M (seconds)

| Model | MLX | llama.cpp | Ollama |
|-------|----:|----------:|-------:|
| 9B | 51.7 | 55.6 | 102.7 |
| 27B | 165.6 | 161.2 | 337.0 |
| **35B-A3B (MoE)** | **21.8** | 55.3 | 85.8 |
| 122B-A10B (MoE) | 50.8 | 123.0 | 176.9 |

---

## Backend Summary

| Metric | Winner | Notes |
|--------|--------|-------|
| **Generation TPS** | MLX | Metal 4-bit kernel, 35B MoE 142 tok/s |
| **Prefill TPS** | llama.cpp | Flash Attention, 128k at 574K tok/s |
| **TTFT** | llama.cpp | 75ms (9B Q4) vs MLX 248ms |
| **Long-context TTFT** | llama.cpp | 229ms for 128k vs Ollama 2s vs MLX 46s |
| **Memory Efficiency** | MLX | 3–4× less than Ollama |
| **E2E Latency (35B)** | MLX | 21.8s for 8192 tokens |

---

## Quick Start

```bash
# Install
uv sync

# Run full benchmark
uv run python -m src.runner --config config.yaml

# Select backends/models/tracks
uv run python -m src.runner \
  --backends mlx llamacpp ollama \
  --models qwen3.5-9b qwen3.5-35b-a3b \
  --tracks gen-512 gen-2048 gen-4096 gen-8192

# Prefill tracks
uv run python -m src.runner \
  --tracks prefill-1k prefill-4k prefill-16k prefill-64k prefill-128k
```

Hardware-specific config: set `hardware.id` in `config.yaml` (`macbook-m-series` / `dgx-spark` / `ryzen-ai-395`).

→ 온도/메모리 모니터링 방법: [docs/setup.md](docs/setup.md)

---

## Docs

| 문서 | 내용 |
|------|------|
| [REPORT.md](REPORT.md) | Apple Silicon 전체 상세 보고서 (모든 수치, caveat 포함) |
| [docs/mac-apple-silicon.md](docs/mac-apple-silicon.md) | Mac 실험 환경 및 결과 요약 |
| [docs/methodology.md](docs/methodology.md) | 측정 방법론, 지표 정의 |
| [docs/setup.md](docs/setup.md) | 설치 가이드, 온도/메모리 모니터링 명령어 |
| [docs/dgx-spark.md](docs/dgx-spark.md) | DGX Spark 실험 (예정) |
| [docs/ryzen-ai-395.md](docs/ryzen-ai-395.md) | Ryzen AI MAX+ 395 실험 (예정) |

---

## Pending

- [x] ~~MacBook Pro Apple Silicon (MLX / llama.cpp / Ollama)~~
- [ ] DGX Spark — vLLM + llama.cpp CUDA + Ollama
- [ ] Ryzen AI MAX+ 395 — llama.cpp ROCm + Ollama
- [ ] Cross-platform comparison report
