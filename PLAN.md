# LLM Local Inference Benchmark Plan

## 목적

128GB Apple Silicon MacBook에서 로컬 LLM 서빙 백엔드 4종의 추론 성능을 정량 비교한다.
모델은 **Qwen3.5 패밀리**로 통일하여, 모델 변수를 제거하고 순수 백엔드 성능 비교에 집중한다.

---

## 1. 비교 대상 백엔드

| 백엔드 | 버전 확인 | API 방식 | 비고 |
|--------|-----------|----------|------|
| **Ollama** | `ollama --version` | REST `/v1/chat/completions` (streaming) | llama.cpp 기반, Metal 지원 |
| **LM Studio** | 앱 내 확인 | REST `/v1/chat/completions` (streaming) | llama.cpp fork, GUI |
| **llama.cpp** | `llama-server --version` | REST `/v1/chat/completions` 또는 CLI | Metal 직접 제어, 파라미터 튜닝 가능 |
| **MLX (mlx-lm)** | `pip show mlx-lm` | Python API (`mlx_lm.generate`) | Apple Silicon 네이티브, unified memory 최적 |

---

## 2. 테스트 모델 — Qwen3.5 패밀리

Qwen3.5 (2026.02.16 출시)는 Dense와 MoE 모두 포함하는 풀 라인업.
단일 모델 패밀리로 통일하여 **백엔드 간 공정 비교** + **Dense vs MoE 아키텍처 비교**를 동시에 수행한다.

### 선정 모델

| 모델 | 구조 | Total / Active Params | 양자화 | 예상 메모리 (Q4/Q8) |
|------|------|----------------------|--------|---------------------|
| **Qwen3.5-9B** | Dense | 9B / 9B | Q4_K_M, Q8_0, FP16 | ~5GB / ~10GB / ~18GB |
| **Qwen3.5-27B** | Dense | 27B / 27B | Q4_K_M, Q8_0 | ~15GB / ~28GB |
| **Qwen3.5-35B-A3B** | MoE | 35B / 3B active | Q4_K_M, Q8_0 | ~20GB / ~36GB |
| **Qwen3.5-122B-A10B** | MoE | 122B / 10B active | Q4_K_M | ~68GB |

### 비교 관점
- **Dense 스케일링**: 9B → 27B (파라미터 3배 증가 시 TPS 변화)
- **MoE 스케일링**: 35B-A3B → 122B-A10B (총 파라미터 대비 active 파라미터 영향)
- **Dense vs MoE**: 27B Dense vs 35B-A3B MoE (유사 메모리에서 속도 차이)
- **양자화 영향**: Q4 vs Q8 vs FP16 (9B에서 전 구간 테스트)

### 모델 소싱

| 백엔드 | 소스 | 형식 |
|--------|------|------|
| **Ollama** | `ollama pull qwen3.5:9b`, `qwen3.5:27b`, etc. | 자체 레지스트리 (GGUF 기반) |
| **LM Studio** | 앱 내 검색 "Qwen3.5" | GGUF (HuggingFace) |
| **llama.cpp** | `unsloth/Qwen3.5-*-GGUF` (HuggingFace) | GGUF 직접 다운로드 |
| **MLX** | `mlx-community/Qwen3.5-*-4bit` 등 (HuggingFace) | MLX 변환 모델 |

> **주의**: GGUF(Ollama/LM Studio/llama.cpp)와 MLX는 양자화 구현이 다르다.
> 동일 양자화 레벨(Q4)이라도 미세한 정밀도 차이가 있을 수 있으며, 결과 해석 시 명시한다.
> GGUF 3개 백엔드는 **동일 GGUF 파일**을 공유하여 완전한 공정 비교를 보장한다.

---

## 3. 측정 지표

| 지표 | 정의 | 단위 | 측정 방법 |
|------|------|------|-----------|
| **TTFT** | 요청 전송 ~ 첫 토큰 수신 | ms | streaming 첫 chunk 타임스탬프 |
| **Generation TPS** | 생성 토큰 수 / 생성 시간 | tok/s | (총 토큰 - 1) / (마지막 토큰 시간 - 첫 토큰 시간) |
| **Prompt TPS** | 프롬프트 토큰 처리 속도 | tok/s | API 응답의 prompt eval 지표 또는 TTFT 기반 추정 |
| **Peak Memory** | 추론 중 최대 메모리 사용 | GB | `memory_pressure` 또는 프로세스 RSS 모니터링 |
| **Total Latency** | 요청 전송 ~ 마지막 토큰 수신 | s | end-to-end 측정 |

---

## 4. 테스트 프롬프트

3가지 길이의 프롬프트로 입출력 크기에 따른 성능 변화를 측정한다.
모든 백엔드에 동일 프롬프트를 사용한다.

| ID | 유형 | 설명 | 예상 출력 길이 |
|----|------|------|---------------|
| `short` | 짧은 설명 | "Explain quicksort in 3 sentences." | ~50 tokens |
| `medium` | 코드 생성 | "Write a Python HTTP server with routing, error handling, and logging. Include complete code." | ~500 tokens |
| `long` | 장문 생성 | "Write a comprehensive guide to database indexing strategies covering B-tree, hash, GIN, GiST indexes with examples and when to use each." | ~1500 tokens |

### 고정 파라미터
```
temperature: 0
max_tokens: 2048
top_p: 1.0
repeat_penalty: 1.0
context_window: 262144   # Qwen3.5 최대 네이티브 컨텍스트 (256K)
```

> Qwen3.5는 네이티브 262,144 tokens, YaRN 확장 시 최대 1M tokens.
> 벤치마크에서는 네이티브 최대인 **262,144 (256K)** 로 통일한다.
> 단, 컨텍스트 윈도우가 클수록 KV cache 메모리가 증가하므로
> 122B-A10B에서 OOM 발생 시 128K → 64K로 단계적 축소하고 이를 결과에 명시한다.

---

## 5. 실험 프로토콜

### 5.1 사전 준비

#### 백엔드 설치 현황 (2026-03-27)
| 백엔드 | 상태 | 버전 | 비고 |
|--------|------|------|------|
| Ollama | 설치됨 | 0.18.2 | 모델 다운로드 필요 |
| LM Studio | 설치됨 | 앱 내 확인 | 모델 다운로드 필요 |
| llama.cpp | 설치됨 | b8500 (ggml 0.9.8) | `brew install llama.cpp`, 모델 다운로드 필요 |
| MLX (mlx-lm) | 설치됨 | 0.29.1 (mlx 0.29.3) | `pip install mlx-lm`, 모델 다운로드 필요 |

#### 환경 설정
1. macOS 전원 설정: "High Performance" 모드, 전원 어댑터 연결
2. 불필요한 앱 모두 종료 (브라우저, IDE 등)
3. 미설치 백엔드 설치 (llama.cpp, mlx-lm)
4. 각 백엔드 버전 기록
5. 모든 모델 사전 다운로드 완료 확인
6. 시스템 정보 기록: `sysctl hw.memsize`, `system_profiler SPHardwareDataType`

#### 모델 다운로드
```bash
# Ollama
ollama pull qwen3.5:9b
ollama pull qwen3.5:27b
ollama pull qwen3.5:35b-a3b
ollama pull qwen3.5:122b-a10b

# llama.cpp (GGUF — Ollama/LM Studio와 공유 가능)
# HuggingFace에서 다운로드:
#   unsloth/Qwen3.5-9B-GGUF
#   unsloth/Qwen3.5-27B-GGUF
#   unsloth/Qwen3.5-35B-A3B-GGUF
#   unsloth/Qwen3.5-122B-A10B-GGUF

# MLX
# HuggingFace에서 다운로드:
#   mlx-community/Qwen3.5-9B-4bit
#   mlx-community/Qwen3.5-27B-4bit
#   mlx-community/Qwen3.5-35B-A3B-4bit
#   mlx-community/Qwen3.5-122B-A10B-4bit
```

### 5.2 실행 순서
```
for backend in [ollama, lmstudio, llamacpp, mlx]:
    for model in [9B, 27B, 35B-A3B, 122B-A10B]:
        for quant in [Q4_K_M, Q8_0]:  # 9B는 FP16 추가, 122B는 Q4_K_M만
            1. 다른 백엔드 프로세스 종료 확인
            2. 백엔드 서버 시작 + 모델 로드
            3. 모델 로드 완료 대기 (모델 로드 시간도 기록)
            4. 워밍업 실행 1회 (결과 버림)
            5. for prompt in [short, medium, long]:
                   5회 반복 측정
                   각 실행 사이 3초 대기
            6. 메모리 사용량 기록
            7. 백엔드 종료
            8. 30초 대기 (메모리 해제)
```

### 5.3 전체 실험 규모
```
백엔드:  4개
모델:    4개 (9B, 27B, 35B-A3B, 122B-A10B)
양자화:  9B(3종) + 27B(2종) + 35B-A3B(2종) + 122B-A10B(1종) = 8 조합
프롬프트: 3개
반복:    5회
─────────────────────────────
총 실행:  4 × 8 × 3 × 5 = 480회 (+ 워밍업 96회)
```

### 5.4 통제 조건
- 각 측정은 5회 반복, 중앙값(median) 사용
- 실행 간 3초 간격
- 백엔드 교체 시 30초 쿨다운
- 동시 실행 금지 (한 번에 하나의 백엔드만)

---

## 6. 결과 수집 형식

### CSV 스키마
```csv
timestamp,backend,backend_version,model,architecture,total_params,active_params,quantization,prompt_id,run_number,ttft_ms,gen_tps,prompt_tps,total_latency_s,peak_memory_gb,output_tokens
```

### 예시
```csv
2026-03-27T10:00:00,ollama,0.8.1,qwen3.5-9b,dense,9B,9B,Q4_K_M,short,1,120.5,45.2,890.0,1.23,5.1,52
2026-03-27T10:05:00,mlx,0.22.0,qwen3.5-35b-a3b,moe,35B,3B,Q4_K_M,short,1,95.3,58.7,1100.0,0.98,20.3,51
```

---

## 7. 시각화 계획

### 차트 목록
1. **백엔드별 Generation TPS 비교** — 모델별 grouped bar chart
2. **백엔드별 TTFT 비교** — 모델별 grouped bar chart
3. **양자화 영향** — Q4 vs Q8 (vs FP16), 백엔드별 TPS 변화율
4. **Dense 스케일링 곡선** — 9B → 27B, 백엔드별 TPS 감소 추이
5. **Dense vs MoE** — 27B Dense vs 35B-A3B MoE, 백엔드별 TPS/메모리 비교
6. **MoE 스케일링** — 35B-A3B → 122B-A10B, active params 대비 TPS
7. **메모리 효율** — TPS per GB memory, scatter plot
8. **종합 히트맵** — 백엔드 × 모델 × 양자화, TPS 색상 매핑

### 도구
- `matplotlib` + `seaborn` 또는 `plotly`
- 결과 CSV → pandas DataFrame → 차트

---

## 8. 프로젝트 구조

```
llm-bench/
├── PLAN.md                 # 이 문서
├── README.md               # 사용법
├── pyproject.toml          # 의존성 관리
├── prompts.json            # 테스트 프롬프트 정의
├── config.yaml             # 모델/백엔드 설정
├── src/
│   ├── __init__.py
│   ├── runner.py           # 메인 벤치마크 실행기
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py         # 백엔드 공통 인터페이스
│   │   ├── ollama.py       # Ollama 측정 로직
│   │   ├── lmstudio.py     # LM Studio 측정 로직
│   │   ├── llamacpp.py     # llama.cpp 측정 로직
│   │   └── mlx.py          # MLX 측정 로직
│   ├── metrics.py          # 지표 계산 로직
│   └── memory.py           # 메모리 사용량 모니터링
├── scripts/
│   ├── setup_models.sh     # 모델 일괄 다운로드
│   └── system_info.sh      # 시스템 정보 수집
├── results/                # 측정 결과 CSV 저장
├── charts/                 # 생성된 차트 저장
└── visualize.py            # 차트 생성 스크립트
```

---

## 9. 백엔드별 측정 방식 상세

### Ollama
```bash
# 서버 시작 (보통 자동 실행)
ollama serve

# 컨텍스트 윈도우 최대 설정 (환경변수 또는 Modelfile)
# 방법 1: 환경변수
OLLAMA_NUM_CTX=262144 ollama serve

# 방법 2: API 호출 시 num_ctx 지정
POST http://localhost:11434/v1/chat/completions
{
  "model": "qwen3.5:9b",
  "options": { "num_ctx": 262144 },
  ...
}
# streaming으로 TTFT, TPS 측정
# /api/generate 엔드포인트의 eval_count, eval_duration 활용 가능
```

### LM Studio
```bash
# GUI에서 서버 시작 (포트 1234 기본)
# 모델은 앱 내에서 수동 로드

POST http://localhost:1234/v1/chat/completions
# Ollama와 동일한 OpenAI-compatible API
```

### llama.cpp
```bash
# 서버 직접 실행 (컨텍스트 256K)
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 262144 --port 8080

# API 호출
POST http://localhost:8080/v1/chat/completions
# 또는 /completion 엔드포인트에서 timings 직접 획득
```

### MLX
```python
# Python API 직접 호출
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-9B-4bit")
# generate() 전후 시간 측정
# 또는 mlx_lm.server로 OpenAI-compatible API 사용
```

---

## 10. 주의사항 및 한계

1. **GGUF vs MLX 양자화 차이**: GGUF(Ollama/LM Studio/llama.cpp)와 MLX는 양자화 구현이 다르다. 동일 Q4라도 미세한 정밀도 차이 존재. GGUF 3개 백엔드는 동일 파일을 공유하여 공정성 확보.
2. **Ollama 오버헤드**: Ollama는 llama.cpp 위에 Go 래퍼가 있어 순수 llama.cpp 대비 약간의 오버헤드 존재.
3. **LM Studio 자동화 한계**: GUI 기반이라 모델 로드는 수동. API 서버 모드로 측정하되, 자동화 범위에 한계.
4. **MoE 백엔드 지원**: MoE 모델(35B-A3B, 122B-A10B)의 백엔드별 지원 수준이 다를 수 있음. 사전에 각 백엔드에서 로드 가능 여부 확인 필요.
5. **메모리 측정**: unified memory 특성상 GPU/CPU 메모리 구분이 무의미. 프로세스 RSS 기준으로 통일.
6. **재현성**: 동일 조건에서도 ±5% 변동 가능. 5회 반복 + 중앙값으로 완화.
7. **thermal throttling**: 장시간 연속 실행 시 발열로 성능 저하 가능. 백엔드 교체 시 쿨다운 적용.
8. **122B-A10B 메모리**: Q4에서 ~68GB 예상. 128GB에서 가능하지만 다른 프로세스와 OS 메모리를 고려하면 여유가 크지 않음. swap 발생 시 결과에 명시.
