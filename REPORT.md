# Qwen3.5 로컬 추론 벤치마크 보고서

> **실험 예정일**: 2026-03-29 (일)
> **디바이스**: Apple Silicon MacBook, 128GB unified memory
> **상태**: 🔲 실험 전 — 수치 미기입

---

## 1. 실험 개요

128GB Apple Silicon MacBook에서 로컬 LLM 추론 백엔드 4종의 성능을 정량 비교한다.
Qwen3.5 패밀리로 모델 변수를 통일하고, **백엔드 순수 성능 비교**와 **Dense vs MoE 아키텍처 비교**를 동시에 수행한다.

---

## 2. 실험 환경

### 2.1 하드웨어

| 항목 | 사양 |
|------|------|
| 기기 | Apple Silicon MacBook |
| Unified Memory | 128GB |
| CPU | — *(실험 전 `system_profiler SPHardwareDataType` 으로 기입)* |
| GPU | Apple Silicon 내장 GPU |
| macOS | — *(실험 전 기입)* |

### 2.2 소프트웨어 버전

| 백엔드 | 버전 | 비고 |
|--------|------|------|
| Ollama | — | `ollama --version` |
| LM Studio | — | 앱 내 확인 |
| llama.cpp (llama-server) | — | `llama-server --version` |
| MLX / mlx-lm | — | `uv run python -c "import mlx_lm; print(mlx_lm.__version__)"` |
| Python | — | `uv run python --version` |

### 2.3 llama.cpp 실행 옵션

```
--n-gpu-layers 99    # 전체 레이어 Metal 오프로드
--flash-attn         # prefill 성능 향상
--batch-size 512
--ubatch-size 512
--no-mmap            # 측정 일관성 확보
--ctx-size 262144    # 256K
```

### 2.4 공통 생성 파라미터

| 파라미터 | 값 |
|----------|-----|
| temperature | 0 |
| top_p | 1.0 |
| repeat_penalty | 1.0 |
| context_window | 262,144 (256K) |
| warmup_runs | 1 |
| measure_runs | 5 (중앙값 집계) |

---

## 3. 모델 정보

| 모델 | 아키텍처 | Total Params | Active Params | Context Window | GGUF 소스 | MLX 소스 |
|------|----------|-------------|---------------|----------------|-----------|----------|
| Qwen3.5-9B | Dense | 9B | 9B | 262,144 | unsloth | mlx-community |
| Qwen3.5-27B | Dense | 27B | 27B | 262,144 | unsloth | mlx-community |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 262,144 | unsloth | mlx-community |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 262,144 | unsloth | mlx-community |

### 양자화 매트릭스

| 모델 | GGUF Q4_K_M | GGUF Q8_0 | MLX 4bit | MLX 8bit |
|------|:-----------:|:---------:|:--------:|:--------:|
| 9B | ✅ | ✅ | ✅ | ✅ |
| 27B | ✅ | ✅ | ✅ | ✅ |
| 35B-A3B | ✅ | ✅ | ✅ | ✅ |
| 122B-A10B | ✅ | — | ✅ | — |

> GGUF 3개 백엔드(Ollama, LM Studio, llama.cpp)는 **동일 파일**을 공유. MLX는 별도 변환 모델.

---

## 4. 실험 설계

### 4.1 Generation Track — 짧은 입력(64 tokens), 출력 고정

| Track | 출력 토큰 |
|-------|---------|
| gen-512 | 512 |
| gen-2048 | 2,048 |
| gen-4096 | 4,096 |
| gen-8192 | 8,192 |

### 4.2 Prefill Track — 입력 길이 변화, 출력 10 tokens

| Track | 입력 토큰 |
|-------|---------|
| prefill-1k | 1,024 |
| prefill-4k | 4,096 |
| prefill-16k | 16,384 |
| prefill-64k | 65,536 |
| prefill-128k | 131,072 |

---

## 5. 결과

> 아래 표의 모든 수치는 **5회 측정 중앙값**.
> `—` = 해당 조합 해당 없음 또는 OOM.
> `※` = 주석 참고.

---

### 5.1 Generation TPS (tok/s) — gen-2048 기준

*입력 64 tokens, 출력 2,048 tokens 고정.*

#### Q4_K_M

| 모델 | Ollama | LM Studio | llama.cpp | MLX |
|------|:------:|:---------:|:---------:|:---:|
| 9B | — | — | — | — |
| 27B | — | — | — | — |
| 35B-A3B | — | — | — | — |
| 122B-A10B | — | — | — | — |

#### Q8_0

| 모델 | Ollama | LM Studio | llama.cpp | MLX |
|------|:------:|:---------:|:---------:|:---:|
| 9B | — | — | — | — |
| 27B | — | — | — | — |
| 35B-A3B | — | — | — | — |

---

### 5.2 Generation TPS — 출력 길이별 (9B Q4_K_M)

*9B를 기준으로 출력 길이에 따른 TPS 변화 확인.*

| Track | Ollama | LM Studio | llama.cpp | MLX |
|-------|:------:|:---------:|:---------:|:---:|
| gen-512 | — | — | — | — |
| gen-2048 | — | — | — | — |
| gen-4096 | — | — | — | — |
| gen-8192 | — | — | — | — |

---

### 5.3 Prefill TPS (tok/s) — 컨텍스트 길이별

*입력 토큰 처리 속도 (prompt_tps). 출력 10 tokens 고정.*

#### 9B Q4_K_M

| Track | 입력 토큰 | Ollama | LM Studio | llama.cpp | MLX |
|-------|---------|:------:|:---------:|:---------:|:---:|
| prefill-1k | 1,024 | — | — | — | — |
| prefill-4k | 4,096 | — | — | — | — |
| prefill-16k | 16,384 | — | — | — | — |
| prefill-64k | 65,536 | — | — | — | — |
| prefill-128k | 131,072 | — | — | — | — |

#### 27B Q4_K_M

| Track | 입력 토큰 | Ollama | LM Studio | llama.cpp | MLX |
|-------|---------|:------:|:---------:|:---------:|:---:|
| prefill-1k | 1,024 | — | — | — | — |
| prefill-4k | 4,096 | — | — | — | — |
| prefill-16k | 16,384 | — | — | — | — |
| prefill-64k | 65,536 | — | — | — | — |
| prefill-128k | 131,072 | — | — | — | — |

#### 35B-A3B Q4_K_M

| Track | 입력 토큰 | Ollama | LM Studio | llama.cpp | MLX |
|-------|---------|:------:|:---------:|:---------:|:---:|
| prefill-1k | 1,024 | — | — | — | — |
| prefill-4k | 4,096 | — | — | — | — |
| prefill-16k | 16,384 | — | — | — | — |
| prefill-64k | 65,536 | — | — | — | — |
| prefill-128k | 131,072 | — | — | — | — |

#### 122B-A10B Q4_K_M

| Track | 입력 토큰 | Ollama | LM Studio | llama.cpp | MLX |
|-------|---------|:------:|:---------:|:---------:|:---:|
| prefill-1k | 1,024 | — | — | — | — |
| prefill-4k | 4,096 | — | — | — | — |
| prefill-16k | 16,384 | — | — | — | — |
| prefill-64k | 65,536 | — | — | — | — |
| prefill-128k | 131,072 | — | — | — | — |

---

### 5.4 TTFT (ms) — gen-2048 기준

| 모델 / Q4_K_M | Ollama | LM Studio | llama.cpp | MLX |
|--------------|:------:|:---------:|:---------:|:---:|
| 9B | — | — | — | — |
| 27B | — | — | — | — |
| 35B-A3B | — | — | — | — |
| 122B-A10B | — | — | — | — |

---

### 5.5 Peak Memory (GB)

| 모델 | Q4 | Q8 | Q4 예측 | Q8 예측 |
|------|:--:|:--:|:-------:|:-------:|
| 9B | — | — | ~5.5G | ~9.5G |
| 27B | — | — | ~16G | ~28G |
| 35B-A3B | — | — | ~20G | ~35G |
| 122B-A10B | — | — | ~68G | — |

> 예측값은 파일 크기 기준 추정. 실제 런타임 메모리는 KV cache 포함으로 더 높을 수 있음.

---

### 5.6 양자화별 비교 — 9B (Q4 vs Q8)

*Gen TPS와 Prefill TPS에서 Q8이 Q4 대비 성능 변화율.*

| 백엔드 | Gen TPS Q4 | Gen TPS Q8 | 변화율 | Prefill TPS Q4 | Prefill TPS Q8 | 변화율 |
|--------|:----------:|:----------:|:------:|:--------------:|:--------------:|:------:|
| Ollama | — | — | — | — | — | — |
| LM Studio | — | — | — | — | — | — |
| llama.cpp | — | — | — | — | — | — |
| MLX | — | — | — | — | — | — |

---

## 6. 분석

### 6.1 백엔드 오버헤드 (같은 GGUF 파일 기준)

*Ollama, LM Studio, llama.cpp 3개는 동일 GGUF 파일을 사용. 성능 차이 = 백엔드 래핑 오버헤드.*

> **[실험 후 기입]**
> - Ollama vs llama.cpp 직접 오버헤드: —
> - LM Studio vs llama.cpp 오버헤드: —
> - Ollama의 Go 래퍼 비용 추정: —

---

### 6.2 MLX vs GGUF 백엔드

*Apple Silicon 네이티브 MLX와 llama.cpp Metal 백엔드 비교.*

> **[실험 후 기입]**
> - Generation TPS: MLX가 llama.cpp 대비 ±—%
> - Prefill TPS: MLX가 llama.cpp 대비 ±—%
> - 특이사항: —

---

### 6.3 Dense 스케일링 (9B → 27B)

*파라미터 3배 증가 시 TPS 변화.*

> **[실험 후 기입]**
> - TPS 감소율 (이상적: 1/3 = 33%): —
> - 실제 감소율: —
> - 해석: —

---

### 6.4 Dense vs MoE (27B Dense vs 35B-A3B MoE)

*유사 메모리 풋프린트, 다른 아키텍처.*

| 지표 | 27B Dense Q4 (~16G) | 35B-A3B Q4 (~20G) | 차이 |
|------|:-------------------:|:-----------------:|:----:|
| Gen TPS (Ollama) | — | — | — |
| Gen TPS (MLX) | — | — | — |
| Prefill TPS (Ollama) | — | — | — |
| Prefill TPS (MLX) | — | — | — |
| TTFT | — | — | — |

> **[실험 후 기입]**
> - MoE의 active params 이점이 실제 Gen TPS에 나타나는가: —
> - Prefill에서는 어느 아키텍처가 유리한가: —

---

### 6.5 MoE 스케일링 (35B-A3B → 122B-A10B)

*총 파라미터 3.5배, active 파라미터 3.3배 증가.*

> **[실험 후 기입]**
> - Gen TPS 감소율: —
> - active params 기준 효율: —

---

### 6.6 Prefill 스케일링 (컨텍스트 길이)

*1K → 128K 입력에서 Prefill TPS 변화. 선형 유지 시 어텐션 최적화 잘 된 것.*

> **[실험 후 기입]**
> - llama.cpp flash_attn 효과 (1K vs 64K TPS 비율): —
> - MLX의 long-context 스케일링: —
> - 128K에서 OOM 또는 성능 급감 여부: —

---

## 7. 종합 순위

> **[실험 후 기입]**

### Generation TPS (높을수록 좋음)

| 순위 | 백엔드 | 모델 | 양자화 | TPS |
|------|--------|------|--------|-----|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |

### Prefill TPS (높을수록 좋음)

| 순위 | 백엔드 | Track | TPS |
|------|--------|-------|-----|
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 4 | — | — | — |

---

## 8. 이상 징후 및 특이사항

> **[실험 후 기입]**

| 항목 | 내용 |
|------|------|
| Thermal throttling | — |
| OOM 발생 모델 | — |
| 비정상적 수치 | — |
| 재실험 필요 항목 | — |

---

## 9. 실험 실행 체크리스트

실험 당일 순서:

- [ ] 전원 어댑터 연결
- [ ] macOS 전력 설정 → "높은 성능" 모드
- [ ] 불필요한 앱 모두 종료
- [ ] `sudo powermetrics --samplers thermal -n 1` (sudo 캐시 활성화)
- [ ] `./scripts/check_models.sh` 로 전체 모델 완료 확인
- [ ] LM Studio 실행 후 각 모델 context 256K, Flash Attention ON 설정
- [ ] `uv run python -m src.runner --config config.yaml` 실행
- [ ] 실험 완료 후 `uv run python visualize.py results/<csv>` 시각화

---

## 10. 실험 후 업데이트 방법

```bash
# 벤치마크 실행 (결과 CSV 자동 저장)
uv run python -m src.runner --config config.yaml --output results/bench_20260329.csv

# 특정 백엔드/모델만
uv run python -m src.runner --backends llamacpp mlx --models qwen3.5-9b

# 시각화
uv run python visualize.py results/bench_20260329.csv
```

결과 CSV가 생성되면 이 보고서의 `—` 값들을 `visualize.py` 출력 또는 CSV에서 직접 채운다.
