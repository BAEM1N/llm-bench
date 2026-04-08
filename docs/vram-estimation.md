# VRAM Estimation vs Actual Measurement

> 작성일: 2026-04-08
> 상태: 이론값 계산 완료, 실측 대기

## 계산 공식

```
Total VRAM = Model Weights (GGUF file size) + KV Cache

KV Cache = 2 (K+V) × n_layers × n_kv_heads × head_dim × context_length × 2 bytes (FP16)
```

## Qwen3.5 모델 파라미터

| Model | n_layers | n_kv_heads | head_dim | GQA ratio | Context (max) |
|-------|:--------:|:----------:|:--------:|:---------:|:-------------:|
| 9B Dense | 36 | 8 | 128 | 4:1 | 262,144 (256K) |
| 27B Dense | 48 | 8 | 128 | 4:1 | 262,144 (256K) |
| 35B-A3B MoE | 64 | 4 | 128 | 8:1 | 262,144 (256K) |
| 122B-A10B MoE | 96 | 8 | 128 | 4:1 | 262,144 (256K) |

> MoE 35B-A3B는 kv_heads=4로 Dense 모델 대비 KV cache 절반

---

## Context = 256K (모델 최대)

| Model | Weights | KV Cache | **Total** | 3090 (48GB) | DGX (128GB) | Ryzen (96GB) | Mac (128GB) |
|-------|--------:|---------:|----------:|:-----------:|:-----------:|:------------:|:-----------:|
| 9B Q4_K_M | 5.5G | 36.0G | **41.5G** | OK | OK | OK | OK |
| 9B Q8_0 | 9.5G | 36.0G | **45.5G** | OK | OK | OK | OK |
| 27B Q4_K_M | 15.5G | 48.0G | **63.5G** | OOM +16G | OK | OK | OK |
| 27B Q8_0 | 28.0G | 48.0G | **76.0G** | OOM +28G | OK | OK | OK |
| 35B-A3B Q4_K_M | 20.5G | 32.0G | **52.5G** | OOM +4G | OK | OK | OK |
| 35B-A3B Q8_0 | 37.0G | 32.0G | **69.0G** | OOM +21G | OK | OK | OK |
| 122B Q4_K_M | 72.0G | 96.0G | **168.0G** | OOM +120G | OOM +40G | OOM +72G | OOM +40G |

> 256K 풀 컨텍스트에서는 122B를 어디서든 올릴 수 없음

---

## Context = 64K (DGX/Ryzen 벤치마크 설정)

| Model | Weights | KV Cache | **Total** | 3090 (48GB) | DGX (128GB) | Ryzen (96GB) |
|-------|--------:|---------:|----------:|:-----------:|:-----------:|:------------:|
| 9B Q4_K_M | 5.5G | 9.0G | **14.5G** | OK | OK | OK |
| 9B Q8_0 | 9.5G | 9.0G | **18.5G** | OK | OK | OK |
| 27B Q4_K_M | 15.5G | 12.0G | **27.5G** | OK | OK | OK |
| 27B Q8_0 | 28.0G | 12.0G | **40.0G** | OK | OK | OK |
| 35B-A3B Q4_K_M | 20.5G | 8.0G | **28.5G** | OK | OK | OK |
| 35B-A3B Q8_0 | 37.0G | 8.0G | **45.0G** | OK | OK | OK |
| 122B Q4_K_M | 72.0G | 24.0G | **96.0G** | OOM +48G | OK | OK (barely) |

---

## Context = 8K (gen 트랙 실제 사용)

| Model | Weights | KV Cache | **Total** |
|-------|--------:|---------:|----------:|
| 9B Q4_K_M | 5.5G | 1.1G | **6.6G** |
| 9B Q8_0 | 9.5G | 1.1G | **10.6G** |
| 27B Q4_K_M | 15.5G | 1.5G | **17.0G** |
| 27B Q8_0 | 28.0G | 1.5G | **29.5G** |
| 35B-A3B Q4_K_M | 20.5G | 1.0G | **21.5G** |
| 35B-A3B Q8_0 | 37.0G | 1.0G | **38.0G** |
| 122B Q4_K_M | 72.0G | 3.0G | **75.0G** |

---

## 인사이트

1. **KV cache가 모델 웨이트를 압도** — 256K에서 9B 모델도 KV cache 36GB (웨이트의 6.5배)
2. **MoE KV cache 절반** — 35B-A3B (kv_heads=4) vs 27B (kv_heads=8): 동일 context에서 KV 32G vs 48G
3. **Gen 트랙에서는 KV cache 무시 가능** — 8K context면 최대 3GB. 모델 웨이트가 지배
4. **Prefill 트랙이 VRAM 킬러** — prefill-128k는 context 131,072 → KV cache만 27B에서 24GB
5. **3090 OOM의 진짜 원인** — 122B가 72GB 웨이트 + 24GB KV = 96GB. 48GB VRAM으로 절대 불가

---

## 실측 계획

### 측정 방법

| 디바이스 | 측정 방법 | 비고 |
|---------|----------|------|
| Mac M5 Max | `process_rss` (unified memory) | 현재 작동 중, 기존 데이터 있음 |
| RTX 3090×2 | `nvidia-smi --query-compute-apps` | GPU 0에 drone-detection 점유 중 → 해제 후 |
| DGX Spark | `nvidia-smi --query-compute-apps` (Processes 섹션) | 학습 중 → 완료 후. Memory-Usage "Not Supported"이나 프로세스별은 잡힘 |
| Ryzen AI 395 | `rocm-smi --showmeminfo vram` | 서버 운용 중 → 여유 시 |

### 측정 절차

각 모델별로:
1. 다른 프로세스 없는 상태 확인
2. llama-server 시작 (특정 context size)
3. 모델 로드 완료 대기
4. `nvidia-smi` / `rocm-smi` 로 VRAM 기록
5. 짧은 추론 1회 실행 (KV cache 할당 유발)
6. 다시 VRAM 기록 (KV cache 포함)
7. 서버 종료, 다음 모델

### 측정 대상

- [ ] 9B Q4_K_M @ 8K context → 이론: 6.6G
- [ ] 9B Q4_K_M @ 64K context → 이론: 14.5G
- [ ] 27B Q4_K_M @ 8K context → 이론: 17.0G
- [ ] 27B Q4_K_M @ 64K context → 이론: 27.5G
- [ ] 35B-A3B Q4_K_M @ 8K context → 이론: 21.5G
- [ ] 35B-A3B Q4_K_M @ 64K context → 이론: 28.5G
- [ ] 122B Q4_K_M @ 8K context → 이론: 75.0G
- [ ] 122B Q4_K_M @ 64K context → 이론: 96.0G

### 기존 Mac 실측 데이터 (process_rss, gen 트랙)

| Model | 이론 (8K) | 실측 (Mac RSS) | 차이 |
|-------|----------:|---------------:|-----:|
| 9B Q4_K_M | 6.6G | 14.4G | +7.8G (llama.cpp 오버헤드) |
| 27B Q4_K_M | 17.0G | 33.3G | +16.3G |
| 35B-A3B Q4_K_M | 21.5G | 26.3G | +4.8G |
| 122B Q4_K_M | 75.0G | 78.6G | +3.6G |

> **이론값이 실측보다 크다** — 이론 KV cache는 FP16(2 bytes) 기준이나, 벤치마크는
> `--flash-attn on`으로 실행. Flash Attention은 KV cache를 Q8_0/Q4_0으로 양자화 저장하여
> 실제 KV cache가 이론의 **1/2~1/4** 수준.
>
> `--no-mmap` 사용 중이므로 RSS = 실제 프로세스 메모리 (mmap 부풀림 없음).
> context_window=262144 (256K)로 서버를 띄우므로 gen-512에서도 256K 분량 KV buffer가 startup 시 할당됨.
>
> **실측이 이론보다 작은 이유**: Flash Attention KV quantization + llama.cpp 내부 최적화.

---

## 실측 결과 (TODO — 장비 여유 시 채울 것)

### DGX Spark (nvidia-smi --query-compute-apps)

| Model | Context | 이론 | 실측 | 차이 |
|-------|--------:|-----:|-----:|-----:|
| 9B Q4_K_M | 8K | 6.6G | — | — |
| 27B Q4_K_M | 8K | 17.0G | — | — |
| 35B-A3B Q4_K_M | 8K | 21.5G | — | — |
| 122B Q4_K_M | 8K | 75.0G | — | — |

### RTX 3090×2 (nvidia-smi)

| Model | Context | 이론 | 실측 | 차이 |
|-------|--------:|-----:|-----:|-----:|
| 9B Q4_K_M | 8K | 6.6G | — | — |
| 27B Q4_K_M | 8K | 17.0G | — | — |
| 35B-A3B Q4_K_M | 8K | 21.5G | — | — |

### Ryzen AI 395 (rocm-smi)

| Model | Context | 이론 | 실측 | 차이 |
|-------|--------:|-----:|-----:|-----:|
| 9B Q4_K_M | 8K | 6.6G | — | — |
| 27B Q4_K_M | 8K | 17.0G | — | — |
| 35B-A3B Q4_K_M | 8K | 21.5G | — | — |
| 122B Q4_K_M | 8K | 75.0G | — | — |
