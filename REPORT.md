# Qwen3.5 Family — MacBook Pro 14 (M5 Max) Benchmark Report

**하드웨어**: MacBook Pro 14 (M5 Max, 128GB Unified Memory)  
**실험 기간**: 2026-03-31 17:44 ~ 2026-04-02 04:11  
**상태**: ✅ 완료

---

## 측정 한계 및 해석 주의사항

| # | 항목 | 내용 |
|---|------|------|
| 1 | **모델 가중치 소스 차이** | MLX는 mlx-community 변환 가중치(자체 4-bit 포맷), llama.cpp/Ollama는 unsloth GGUF. 동일 quant 레벨이어도 양자화 구현이 달라 "엔진만" 비교가 아닌 "엔진+가중치 패키지" 비교에 가까움 |
| 2 | **Ollama KV 캐시 pre-allocation** | `num_ctx=262144` 설정으로 모델 로드 시 KV 캐시 전체(30~40GB) 사전 할당 → llama.cpp/MLX는 실제 입력 길이만큼만 on-demand 할당. 메모리 비교에서 Ollama가 구조적으로 불리 |
| 3 | **Ollama gen 트랙 TTFT 무효** | Ollama TTFT가 max_tokens에 비례 증가하는 이상 동작 → gen 트랙 Prefill TPS 수치 신뢰 불가. **Prefill 트랙 결과(Section 5)만 신뢰** |
| 4 | **Flash Attention 비대칭** | llama.cpp: `flash-attn=on` (명시적 활성화) / MLX: 미적용 (16k 이후 TPS 감소로 확인) / Ollama: 추정만 가능. Prefill 비교는 알고리즘 수준 차이도 포함됨 |
| 5 | **122B Q8_0 미측정** | 128GB 유니파이드 메모리 한계. 122B Q4_K_M 기준 MLX 64GB, llama.cpp 72.8GB, Ollama 92GB → Q8은 어떤 백엔드도 적재 불가 |

---

## 0. 모델 아키텍처 상세

| 항목 | 9B | 27B | 35B-A3B (MoE) | 122B-A10B (MoE) |
|------|---:|---:|---:|---:|
| **타입** | Dense | Dense | MoE | MoE |
| **레이어 수** | 32 | 64 | 40 | 48 |
| **Hidden size** | 4,096 | 5,120 | 2,048 | 3,072 |
| **Attention heads** | 16 | 24 | 16 | 32 |
| **KV heads (GQA)** | 4 | 4 | 2 | 2 |
| **Head dim** | 256 | 256 | 256 | 256 |
| **FFN intermediate** | 12,288 | 17,408 | — | — |
| **MoE experts (total)** | — | — | 256 | 256 |
| **Active experts/tok** | — | — | 8 | 8 |
| **MoE intermediate** | — | — | 512 | 1,024 |
| **Vocab size** | 248,320 | 248,320 | 248,320 | 248,320 |
| **Context window** | 262,144 | 262,144 | 262,144 | 262,144 |
| **RoPE theta** | 10M | 10M | 10M | 10M |
| **Attention 패턴** | Linear×3+Full×1 반복 | Linear×3+Full×1 반복 | Linear×3+Full×1 반복 | Linear×3+Full×1 반복 |
| **Vision 지원** | ✓ | ✓ | ✓ | ✓ |

> Qwen3.5는 순수 Self-Attention이 아닌 **Linear Attention + Full Attention 혼합 (3:1 비율)**. 35B MoE의 hidden_size가 9B(4,096)보다 작은 2,048인데 속도가 더 빠른 핵심 이유.

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| 하드웨어 | MacBook Pro 14 (M5 Max), 128GB Unified Memory |
| mlx-lm | 0.31.2 |
| llama.cpp | b8500 (342d6125b) |
| Ollama | 0.18.2 |
| 측정 방식 | warmup 1회 + measure 4회 (run 2~5), 중앙값 |
| 온도 가드 | Heavy pressure (~88°C) → 60s 쿨다운 대기 |

### 백엔드 설정

| 백엔드 | 주요 옵션 |
|--------|----------|
| MLX | stream_generate, temperature=0, Metal GPU |
| llama.cpp | n-gpu-layers=99, flash-attn=on, batch=512, no-mmap |
| Ollama | temperature=0, num_ctx=262144 |

---

## 1. 모델 로드 메모리 (GB)

> MLX = Metal peak memory, llama.cpp = 프로세스 RSS, Ollama = 프로세스 RSS 델타

| 모델 | Quant | MLX | llama.cpp | Ollama |
|------|-------|----:|----------:|-------:|
| 9B | Q4_K_M | **4.7** | 6.7 | 19.9 |
| 9B | Q8_0 | **8.9** | 10.5 | 23.7 |
| 27B | Q4_K_M | **14.1** | 17.8 | 40.4 |
| 27B | Q8_0 | **26.6** | 26.7 | 52.1 |
| 35B-A3B (MoE) | Q4_K_M | **18.2** | 21.5 | 33.3 |
| 35B-A3B (MoE) | Q8_0 | **34.3** | 35.4 | 47.1 |
| 122B-A10B (MoE) | Q4_K_M | **64.0** | 72.8 | 92.0 |
| 122B-A10B (MoE) | Q8_0 | OOM | OOM | OOM |

> Ollama가 llama.cpp 대비 **2~3배 많은 메모리** 사용 → 262K KV 캐시 전체 pre-allocate 포함  
> 122B Q8_0: 예상 ~128GB (MLX) / ~145GB (llama.cpp) / ~180GB (Ollama) → 128GB 한계 초과

---

## 2. Generation TPS 중앙값 (tok/s)

### Q4_K_M

| 모델 | gen-512 | gen-2048 | gen-4096 | gen-8192 |
|------|--------:|---------:|---------:|---------:|
| **MLX 9B** | 101.6 | 87.4 | 97.4 | 96.9 |
| llama.cpp 9B | 73.8 | 67.1 | 67.9 | 68.0 |
| Ollama 9B | 56.4 | 52.0 | 52.3 | 51.3 |
| **MLX 27B** | 30.9 | 32.5 | 32.5 | 31.9 |
| llama.cpp 27B | 20.4 | 23.8 | 23.4 | 23.4 |
| Ollama 27B | 17.9 | 18.1 | 17.9 | 17.5 |
| **MLX 35B-A3B** | **142.2** | **139.1** | **137.9** | **136.0** |
| llama.cpp 35B-A3B | 93.6 | 83.2 | 88.2 | 87.8 |
| Ollama 35B-A3B | 60.4 | 58.8 | 59.9 | 59.9 |
| **MLX 122B-A10B** | 66.8 | 58.7 | 63.3 | 63.4 |
| llama.cpp 122B-A10B | 39.4 | 39.2 | 38.8 | 38.8 |
| Ollama 122B-A10B | 28.9 | 27.1 | 27.9 | 27.9 |

### Q8_0

| 모델 | gen-512 | gen-2048 | gen-4096 | gen-8192 |
|------|--------:|---------:|---------:|---------:|
| **MLX 9B** | 59.0 | 57.2 | 57.8 | 58.7 |
| llama.cpp 9B | 51.1 | 49.6 | 49.3 | 49.2 |
| Ollama 9B | 41.1 | 39.6 | 39.6 | 39.5 |
| **MLX 27B** | 17.2 | 18.5 | 18.4 | 18.4 |
| llama.cpp 27B | 14.9 | 16.8 | 16.2 | 16.0 |
| Ollama 27B | 12.8 | 13.6 | 13.3 | 13.1 |
| **MLX 35B-A3B** | **102.9** | **93.4** | **98.2** | **100.2** |
| llama.cpp 35B-A3B | 88.2 | 76.5 | 83.4 | 83.0 |
| Ollama 35B-A3B | 53.5 | 46.5 | 53.1 | 53.1 |
| MLX 122B-A10B | OOM | OOM | OOM | OOM |
| llama.cpp 122B-A10B | OOM | OOM | OOM | OOM |
| Ollama 122B-A10B | OOM | OOM | OOM | OOM |

> 122B Q8_0: 128GB 유니파이드 메모리 초과 — 전 백엔드 측정 불가

---

## 3. TTFT 중앙값 (ms, gen-512 기준)

> ⚠️ Ollama TTFT는 max_tokens에 비례 증가 → 진짜 First Token Latency가 아님

| 모델 | Quant | llama.cpp | MLX | Ollama* |
|------|-------|----------:|----:|--------:|
| 9B | Q4 | **75** | 248 | ~9,619 |
| 9B | Q8 | **77** | 240 | ~12,872 |
| 27B | Q4 | **176** | 351 | ~29,268 |
| 27B | Q8 | **183** | 492 | ~41,305 |
| 35B-A3B | Q4 | **90** | 272 | ~9,174 |
| 35B-A3B | Q8 | **101** | — | — |
| 122B-A10B | Q4 | **167** | 987 | ~19,001 |

---

## 4. Prefill TPS 중앙값 (tok/s, gen-512 기준)

> ⚠️ Ollama Prefill TPS = input_tokens / TTFT. TTFT 이슈로 신뢰 불가 (gen 트랙 기준)

| 모델 | Quant | llama.cpp | MLX | Ollama* |
|------|-------|----------:|----:|--------:|
| 9B | Q4 | **853** | 258 | ~7 |
| 9B | Q8 | **835** | 267 | ~5 |
| 27B | Q4 | **365** | 182 | ~2 |
| 27B | Q8 | **350** | 130 | ~2 |
| 35B-A3B | Q4 | **713** | 235 | ~7 |
| 35B-A3B | Q8 | **634** | — | — |
| 122B-A10B | Q4 | **383** | 65 | ~2 |

---

## 5. Prefill TPS 중앙값 — Prefill 트랙 (tok/s)

> 각 백엔드의 실제 prefill 처리 능력. 입력 길이별 스케일링 특성 비교.

### llama.cpp (Flash Attention, Q4_K_M)

| 모델 | 1k | 4k | 16k | 64k | 128k |
|------|---:|---:|----:|----:|-----:|
| 9B | 18,790 | 66,064 | 187,321 | 434,305 | **574,324** |
| 27B | 10,426 | 38,212 | 123,770 | 317,147 | **441,661** |
| 35B-A3B | 19,293 | 67,089 | 191,502 | 438,034 | **571,342** |
| 122B-A10B | 13,913 | 50,159 | 152,190 | 358,702 | **488,660** |

### llama.cpp (Flash Attention, Q8_0)

| 모델 | 1k | 4k | 16k | 64k | 128k |
|------|---:|---:|----:|----:|-----:|
| 9B | 19,457 | 66,140 | 188,227 | 427,724 | **565,341** |
| 27B | 10,118 | 37,026 | 119,515 | 310,444 | **434,610** |
| 35B-A3B | 18,628 | 64,905 | 186,278 | 431,571 | **567,943** |
| 122B-A10B | OOM | OOM | OOM | OOM | OOM |

> 122B Q8_0: 128GB 메모리 한계 초과

### MLX (Q4_K_M)

| 모델 | 1k | 4k | 16k | 64k | 128k |
|------|---:|---:|----:|----:|-----:|
| 9B | 2,242 | 4,078 | 4,657 | 3,277 | 2,821 |
| 27B | 1,040 | 1,399 | 1,112 | 1,023 | 797 |
| 35B-A3B | 2,294 | 4,745 | **6,336** | 4,025 | 2,980 |
| 122B-A10B | 700 | 1,548 | 1,867 | 1,553 | 1,135 |

### Ollama (Q4_K_M)

| 모델 | 1k | 4k | 16k | 64k | 128k |
|------|---:|---:|----:|----:|-----:|
| 9B | 1,642 | 1,379 | 6,801 | 26,870 | **65,011** |
| 27B | 603 | 377 | 1,979 | 8,864 | timeout |
| 35B-A3B | 1,258 | 1,745 | 7,952 | 28,166 | **61,548** |
| 122B-A10B | 528 | 642 | 2,981 | 11,478 | **27,748** |

> Ollama 27B prefill-128k: KV 캐시 + 모델 메모리 합산 128GB 초과로 타임아웃

### 백엔드별 Prefill 특성 요약

| 특성 | llama.cpp | MLX | Ollama |
|------|-----------|-----|--------|
| 128k TTFT (9B Q4) | **229ms** | 46,536ms | 2,017ms |
| 128k TPS (9B Q4) | **574,324** | 2,821 | 65,011 |
| 스케일링 패턴 | 길수록 TPS 급증 | 16k 피크 후 감소 | 16k 이후 급증 |
| Flash Attention | ✓ | ✗ | ✓ (추정) |

---

## 6. E2E 레이턴시 중앙값 (gen-8192, Q4_K_M, 초)

| 모델 | MLX | llama.cpp | Ollama |
|------|----:|----------:|-------:|
| 9B | 51.7 | 55.6 | 102.7 |
| 27B | 165.6 | 161.2 | 337.0 |
| **35B-A3B** | **21.8** | 55.3 | 85.8 |
| 122B-A10B | 50.8 | 123.0 | 176.9 |

> llama.cpp 35B E2E = gen-8192에서 8192/(87.8 tok/s) ≈ 93초 추론 + 쿨다운 포함

---

## 7. 핵심 인사이트

### MoE 아키텍처 효율

| 모델 | 메모리 (MLX Q4) | Gen TPS (MLX Q4) | Prefill TPS (llama.cpp 128k) |
|------|---------------:|----------------:|-----------------------------:|
| 27B Dense | 14.1 GB | 31 tok/s | 441,661 |
| **35B-A3B MoE** | **18.2 GB** | **142 tok/s** | **571,342** |
| 122B-A10B MoE | 64.0 GB | 66 tok/s | 488,660 |

- 35B MoE가 27B Dense보다 메모리 29% 추가로 Gen TPS **4.6배**, Prefill TPS **29% 빠름**
- Active params 3B만 실제 계산 → 9B보다도 빠른 이유 (Gen 기준)
- 122B MoE도 64GB로 66 tok/s — 실용적 운영 가능

### Prefill: llama.cpp Flash Attention의 압도

| 비교 | llama.cpp | MLX | Ollama |
|------|----------:|----:|-------:|
| 9B 128k TPS | **574,324** | 2,821 | 65,011 |
| MLX 대비 | 204x | — | — |
| Ollama 대비 | 8.8x | — | — |
| 128k TTFT | **229ms** | 46초 | 2초 |

- llama.cpp는 입력이 길수록 TPS가 계속 증가 (1k→128k: 18,790→574,324)
- MLX는 16k에서 피크 후 감소 (Flash Attention 미적용 추정)
- Ollama는 16k부터 급증하지만 llama.cpp에 8.8배 차이

### MoE Prefill 동치 현상

| 모델 | llama.cpp 128k TPS |
|------|------------------:|
| 9B Dense | 574,324 |
| 35B-A3B MoE | 571,342 |
| 122B-A10B MoE | 488,660 |

- 35B MoE의 prefill이 9B Dense와 **거의 동일** — active params 3B가 prefill 속도도 결정
- 122B MoE도 488K tok/s — 모델 크기와 무관하게 MoE sparse 계산이 병렬화

### 백엔드별 강점

| 지표 | 1위 | 비고 |
|------|-----|------|
| **Gen TPS** | MLX | Metal 4bit 커널 최적화, 35B MoE 142 tok/s |
| **Prefill TPS** | llama.cpp | Flash Attention, 128k 기준 9B 574K tok/s |
| **TTFT (gen)** | llama.cpp | 9B Q4 75ms, 35B Q4 90ms |
| **TTFT (128k prefill)** | llama.cpp | 229ms vs Ollama 2초 vs MLX 46초 |
| **메모리 효율** | MLX | Ollama 대비 3~4배 절약 |

### Q4 vs Q8 속도 비율

| 백엔드 | 9B Q4/Q8 (Gen) | 35B Q4/Q8 (Gen) |
|--------|---------------:|----------------:|
| MLX | **1.72x** | **1.38x** |
| llama.cpp | 1.44x | 1.06x |
| Ollama | 1.37x | 1.13x |

MLX Metal 커널이 4bit 가속에 가장 최적화. llama.cpp는 35B에서 Q4/Q8 차이 거의 없음 (memory-bound 해소).

---

## 8. 실험 소요 시간 및 메모리

> **Wall**: 실제 경과 시간 (thermal 쿨다운 + sleep 포함)  
> **Pure**: `total_latency_s` 합산 (순수 추론 시간, warmup 1회 제외)  
> **Overhead**: Wall - Pure (쿨다운 + inter-run/track sleep + 기타)

| 백엔드 | 모델 | Quant | 메모리 | Wall | Pure | Overhead |
|--------|------|-------|-------:|-----:|-----:|---------:|
| MLX | 35B-A3B | Q4 | 18.2 GB | 10.5분 | 5.2분 | 5.3분 |
| MLX | 35B-A3B | Q8 | 34.3 GB | 15.8분 | 8.8분 | 7.0분 |
| MLX | 9B | Q4 | 4.7 GB | 18.5분 | 10.3분 | 8.3분 |
| MLX | 122B-A10B | Q4 | 64.0 GB | 22.1분 | 12.5분 | 9.6분 |
| MLX | 9B | Q8 | 8.9 GB | 24.6분 | 14.6분 | 10.0분 |
| MLX | 27B | Q4 | 14.1 GB | 50.6분 | 30.6분 | 20.0분 |
| MLX | 27B | Q8 | 26.6 GB | 72.9분 | 47.9분 | 25.0분 |
| llama.cpp | 9B | Q4 | 6.7 GB | 24.1분 | 12.5분 | 11.6분 |
| llama.cpp | 9B | Q8 | 10.5 GB | 28.6분 | 16.3분 | 12.3분 |
| llama.cpp | 27B | Q4 | 17.8 GB | 57.8분 | 35.1분 | 22.7분 |
| llama.cpp | 27B | Q8 | 26.7 GB | 20.5분 *(gen 전체)* | 13.3분 | 7.2분 |
| llama.cpp | 35B-A3B | Q4 | 21.5 GB | ~15분 | ~8분 | ~7분 |
| llama.cpp | 35B-A3B | Q8 | 35.4 GB | ~18분 | ~10분 | ~8분 |
| llama.cpp | 122B-A10B | Q4 | 72.8 GB | ~45분 | ~25분 | ~20분 |
| Ollama | 35B-A3B | Q4 | 33.3 GB | 27.5분 | 17.0분 | 10.5분 |
| Ollama | 35B-A3B | Q8 | 47.1 GB | 31.3분 | 19.4분 | 11.9분 |
| Ollama | 9B | Q4 | 19.9 GB | 32.5분 | 19.5분 | 13.0분 |
| Ollama | 9B | Q8 | 23.7 GB | 44.7분 | 25.6분 | 19.1분 |
| Ollama | 122B-A10B | Q4 | 92.0 GB | 60.2분 | 35.3분 | 24.9분 |
| Ollama | 27B | Q4 | 40.4 GB | 92.7분 | 59.3분 | 33.4분 |
| Ollama | 27B | Q8 | 52.1 GB | 111.3분 | 75.1분 | 36.2분 |

---

## 9. 알려진 이슈 / 측정 한계

| 항목 | 내용 |
|------|------|
| Ollama TTFT (gen 트랙) | max_tokens에 비례 증가 → gen 트랙 Prefill TPS 신뢰 불가 |
| Ollama 27B prefill-128k | KV 캐시 + 모델 메모리 합산 128GB 초과 → 타임아웃 |
| MLX 128k TTFT | 9B 기준 46초, 27B 기준 162초 — 실용 한계 수준 |
| llama.cpp 35B/122B Wall time | gen 트랙 rough estimate (별도 측정 미실시) |
