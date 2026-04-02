# llm-bench — Claude 컨텍스트

## 프로젝트 개요

Qwen3.5 패밀리(9B / 27B / 35B-A3B MoE / 122B-A10B MoE)를 여러 백엔드와 하드웨어에서 벤치마크하는 도구.
측정 지표: TTFT, Prefill TPS, Generation TPS, Peak Memory, CPU 온도.
크로스 플랫폼 비교 대상: Apple Silicon MacBook (128GB) → NVIDIA DGX Spark → AMD Ryzen AI MAX 395+ (HP Z2 Mini G1a, 128GB).

## 환경 설정 규칙

- Python 환경은 반드시 **uv** 기반. `uv run`, `uv add`만 사용.
- 패키지 추가 시 `uv add <package>` — pyproject.toml 직접 편집 금지.
- 실행: `uv run python -m src.runner --config config.yaml ...`

## 프로젝트 구조

```
llm-bench/
├── config.yaml              # 벤치마크 설정 (tracks, models, backends, thermal)
├── pyproject.toml           # uv 의존성
├── MODEL_STATUS.md          # 모델 다운로드 현황 (링크·명령어 포함)
├── PLAN.md                  # 전체 실험 계획
├── scripts/
│   ├── download.sh          # 디바이스별 모델 다운로드 (mac / dgx-spark / ryzen-ai)
│   └── check_models.sh      # GGUF / MLX / Ollama 완료 여부 체크
├── src/
│   ├── runner.py            # 메인 오케스트레이터
│   ├── metrics.py           # BenchmarkResult dataclass, CSV append
│   ├── memory.py            # macOS unified memory 모니터 (vm_stat)
│   ├── thermal.py           # CPU 온도 측정 (powermetrics / istats)
│   ├── prompt_gen.py        # 프롬프트 생성 (prefill / generation 트랙)
│   └── backends/
│       ├── base.py          # BaseBackend, GenerateResult
│       ├── ollama.py        # /api/generate 스트리밍
│       ├── lmstudio.py      # /v1/chat/completions SSE
│       ├── llamacpp.py      # llama-server 서브프로세스 + /completion
│       └── mlx_backend.py   # mlx_lm.stream_generate() 직접 호출
├── results/                 # 벤치마크 CSV 결과
└── visualize.py             # 결과 시각화 (7종 차트)
```

## 실험 설계

### Generation Track — 짧은 입력(64 tokens), 출력 고정
| Track ID | 출력 토큰 |
|----------|----------|
| gen-512 | 512 |
| gen-2048 | 2048 |
| gen-4096 | 4096 |
| gen-8192 | 8192 |

### Prefill Track — 입력 길이 변화, 출력 최소(10 tokens)
| Track ID | 입력 토큰 |
|----------|----------|
| prefill-1k | 1,024 |
| prefill-4k | 4,096 |
| prefill-16k | 16,384 |
| prefill-64k | 65,536 |
| prefill-128k | 131,072 |

- **출력 고정 이유**: 크로스 플랫폼 공정 비교, 출력 길이 편차 제거
- **컨텍스트 윈도우**: 262,144 (Qwen3.5 최대)
- **측정**: warmup 1회 + measure 5회, 중앙값 집계
- **온도 가드**: 85°C 초과 시 60초 쿨다운 대기

## 모델 정보

| 모델 | 아키텍처 | Total Params | Active Params | Context Window |
|------|----------|-------------|---------------|----------------|
| Qwen3.5-9B | Dense | 9B | 9B | 262,144 (256K) |
| Qwen3.5-27B | Dense | 27B | 27B | 262,144 (256K) |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 262,144 (256K) |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 262,144 (256K) |

- GGUF 소스: **unsloth** (Dynamic 2.0 양자화, lmstudio-community 대비 품질 우수)
- MLX 소스: **mlx-community** (4bit/8bit, 전 모델 제공)

## 모델 경로 (Mac 기준)

```
GGUF  : ~/.lmstudio/models/unsloth/<ModelName>-GGUF/<file>.gguf
MLX   : ~/.cache/mlx/<ModelName>-<quant>/
Ollama: ollama list 로 확인
```

## 백엔드별 모델 식별자 (config.yaml)

각 모델의 quantization 항목에 4개 키 존재:
- `gguf_path`      — llama.cpp / LM Studio (파일 경로)
- `lmstudio_model` — LM Studio API model ID (e.g. `qwen3.5-9b@q4_k_m`)
- `ollama_model`   — Ollama 태그 (e.g. `qwen3.5:9b-q4_K_M`)
- `mlx_model`      — MLX 디렉토리 경로 (비어있으면 해당 quant 스킵)

## 실행 명령

```bash
# 전체 실행
uv run python -m src.runner --config config.yaml

# 백엔드/모델 선택
uv run python -m src.runner --backends llamacpp mlx --models qwen3.5-9b

# 특정 트랙만
uv run python -m src.runner --tracks gen-512 prefill-4k

# 출력 경로 지정
uv run python -m src.runner --output results/my_run.csv

# 다운로드 상태 확인
./scripts/check_models.sh
```

## 디바이스별 다운로드

```bash
# Mac (MLX + GGUF + Ollama)
./scripts/download.sh mac

# DGX Spark (GGUF + vLLM AWQ)
./scripts/download.sh dgx-spark

# Ryzen AI 395 (GGUF + Ollama ROCm)
./scripts/download.sh ryzen-ai

# 필터 옵션
./scripts/download.sh mac --backends gguf --models 9b,27b
./scripts/download.sh dgx-spark --backends vllm
```

## 디바이스별 하드웨어 및 백엔드

| 디바이스 | 메모리 | 백엔드 | config hardware.id |
|---------|--------|--------|-------------------|
| Mac (Apple Silicon) | 128GB unified | llama.cpp, LM Studio, Ollama, MLX | `macbook-m-series` |
| DGX Spark (GB10) | 128GB unified | llama.cpp, vLLM (AWQ) | `dgx-spark` |
| Ryzen AI MAX 395+ (HP Z2 Mini G1a) | 128GB unified (Strix Halo) | llama.cpp (ROCm/Vulkan), Ollama (ROCm), vLLM (ROCm, 실험적) | `ryzen-ai-max-395` |
| Ryzen 9 5950X + RTX 3090 × 2 (Linux) | 128GB DDR4 + 48GB VRAM | llama.cpp (CUDA), Ollama (CUDA), vLLM (CUDA) | `linux-5950x-3090x2` |

> 크로스 플랫폼 비교 시 `hardware.id`를 반드시 변경한 후 실행할 것.

## 알려진 이슈 / 주의사항

### HuggingFace CLI stall
- `huggingface-cli download` 가 CDN URL에서 stall하는 케이스 발생
- 증상: 프로세스 살아있고 CPU 0%, 파일 크기 미증가
- 해결: `curl -L -C - <hf_resolve_url> -o <dest>` 로 직접 resume
- `download.sh`에 자동 curl fallback 내장됨

### 122B GGUF shard 1 크기
- `Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf`는 **10MB가 정상** (메타데이터/라우팅)
- shard 2: 47GB, shard 3: 25GB

### MLX 35B-A3B 8bit
- mlx-community에서 제공됨 (`mlx-community/Qwen3.5-35B-A3B-8bit`). 이전 판단 오류.

### LM Studio 모델 ID 형식
- `qwen3.5-9b@q4_k_m` (소문자, @로 양자화 구분)
- 35B/122B는 양자화 suffix 없음 (`qwen/qwen3.5-35b-a3b`, `qwen3.5-122b-a10b`)
- LM Studio에서 한 번에 하나의 양자화만 로드 → 35B 양자화 비교 시 수동 전환 필요

### Python 3.9 호환
- `float | None` 타입 힌트 불가 → `Optional[float]` 사용 (thermal.py 적용됨)

### macOS 온도 측정
- `powermetrics`는 sudo 필요. 실험 전 `sudo powermetrics --samplers thermal -n 1` 한 번 실행해 sudo 캐시 활성화 권장
- sudo 없을 경우 `istats` (gem install iStats) 사용
- 둘 다 없으면 온도 필드 -1로 기록됨 (실험 계속 진행)

### DGX Spark 설정
- GGUF 기본 경로: `/models/gguf` (환경변수 `GGUF_DIR`로 오버라이드 가능)
- vLLM AWQ 기본 경로: `/models/vllm` (환경변수 `VLLM_DIR`로 오버라이드 가능)
- `config.yaml`의 `hardware.id`를 `dgx-spark`로 변경 필수

### Ryzen 9 5950X + RTX 3090 × 2 설정 (Linux)
- **아키텍처**: 별도 VRAM 24GB × 2 = 48GB. CPU RAM 128GB DDR4
- llama.cpp: CUDA 빌드 (`cmake -DGGML_CUDA=ON ..`)
- vLLM: `tensor_parallel_size: 2`, `max_model_len: 32768` (KV cache 제약)
- MLX: 불가 (macOS 전용)
- `config.yaml`의 `hardware.id`를 `linux-5950x-3090x2`로 변경 필수

**모델별 실행 전략 (llama.cpp):**

| 모델 | VRAM 점유 | n_gpu_layers | 비고 |
|------|-----------|-------------|------|
| 9B Q4_K_M | ~5GB | 99 | 단일 GPU |
| 9B Q8_0 | ~9.5GB | 99 | 단일 GPU |
| 27B Q4_K_M | ~15GB | 99 | 단일 GPU |
| 27B Q8_0 | ~28GB | 99 | 양쪽 분산 |
| 35B-A3B Q4_K_M | ~20GB | 99 | 단일 GPU |
| 35B-A3B Q8_0 | ~37GB | 99 | 양쪽 분산 |
| 122B-A10B Q4_K_M | ~65GB | **60** | ~43GB GPU + ~22GB RAM 오프로드 |

- **122B 실행 시**: `config.yaml`의 `llamacpp.n_gpu_layers`를 **60**으로 변경 후 실행
- 오프로드 레이어는 CPU(5950X 16코어)에서 실행 → gen TPS 저하 예상
- 122B Q8_0 (~125GB): VRAM+RAM 합산(176GB)엔 들어가나 gen TPS 매우 낮을 것

### Ryzen AI MAX 395+ 설정 (HP Z2 Mini G1a, 128GB)
- **아키텍처**: Strix Halo — iGPU (Radeon 8060S, 40 CU, RDNA 3.5) + 128GB 유니파이드 메모리
- Apple Silicon과 동일 구조: CPU/GPU/NPU가 동일 메모리 풀 공유 → 전 모델 실행 가능
- llama.cpp: ROCm 빌드 (`cmake -DGGML_HIPBLAS=ON ..`) 또는 Vulkan 빌드 권장
- Ollama: ROCm 빌드 사용
- vLLM: ROCm 6.x 이상 필요. Strix Halo iGPU는 공식 tier-1 미지원이나 커뮤니티에서 동작 확인됨
- `config.yaml`의 `hardware.id`를 `ryzen-ai-max-395`로 변경 필수
