# Benchmark Methodology

---

## Measurement Protocol

| 항목 | 값 |
|------|-----|
| Warmup | 1회 (결과 제외, config `warmup_runs`) |
| Measure | 4회 (config `measure_runs: 5`에서 run_number 2~5) |
| 집계 | 중앙값 (median) |
| 온도 가드 | Heavy pressure (~88°C) → 60초 쿨다운 후 재개 |
| Inter-run sleep | 5초 (config `inter_run_sleep`) |
| Inter-track sleep | 60초 (config `inter_track_sleep`) |
| Inter-model sleep | 120초 (config `inter_model_sleep`) |

---

## Tracks

### Generation Track
짧은 입력, 출력 길이 고정으로 **순수 generation 속도** 측정.

> ⚠️ config의 `input_tokens: 64`는 CSV 기록용이며, 실제 프롬프트는 `build_generation_prompt(track_id)` 고정 문자열 사용.
> 실측 입력 길이: gen-512 ≈47 tok, gen-2048 ≈60 tok, gen-4096 ≈69 tok, gen-8192 ≈74 tok.
> generation TPS에 대한 영향은 미미(출력 512~8192 토큰 대비 입력 <75 토큰).

| Track ID | 실측 Input (tok) | Output |
|----------|-----------------:|-------:|
| gen-512 | ~47 | 512 |
| gen-2048 | ~60 | 2,048 |
| gen-4096 | ~69 | 4,096 |
| gen-8192 | ~74 | 8,192 |

### Prefill Track
출력 최소(10 tokens), 입력 길이 변화로 **KV cache 채우기 속도** 측정.

| Track ID | Input | Output |
|----------|------:|-------:|
| prefill-1k | 1,024 | 10 |
| prefill-4k | 4,096 | 10 |
| prefill-16k | 16,384 | 10 |
| prefill-64k | 65,536 | 10 |
| prefill-128k | 131,072 | 10 |

출력을 10토큰으로 고정하는 이유: prefill 이후 generation 오염 최소화.

---

## Metrics

### Generation TPS (tok/s)
- **llama.cpp**: `timings.predicted_per_second` (서버 내부 타이밍)
- **MLX**: `response.generation_tps` (mlx_lm GenerationResponse)
- **Ollama**: `eval_count / (eval_duration_ns / 1e9)`
- **vLLM** (예정): `usage.completion_tokens / generation_duration`

### TTFT (Time To First Token, ms)
- **llama.cpp / MLX**: `(t_first_chunk - t_request_start) * 1000`
- **Ollama**: `(t_first_response_chunk - t_request_start) * 1000`  
  ⚠️ Ollama TTFT는 max_tokens에 비례 증가하는 이상 동작 있음 — gen_tps만 신뢰 가능.

### Prefill TPS (tok/s)
백엔드 native 값 우선, 미제공 시 `input_tokens / TTFT` 폴백.

- **llama.cpp**: `timings.prompt_per_second` (native)
- **MLX**: `response.prompt_tps` (native)
- **Ollama**: `prompt_eval_count / (prompt_eval_duration_ns / 1e9)` (native)
  - gen 트랙에서는 TTFT 이상으로 신뢰 낮음 → Prefill 트랙 결과만 유효

### Peak Memory (GB)
- **MLX**: `mx.metal.get_peak_memory()` — 모델 로드 전 `reset_peak_memory()` 후 측정
- **llama.cpp**: 서버 프로세스 RSS (`ps -o rss=`)
- **Ollama**: 서버 프로세스 RSS 델타 (로드 전후 차이, 실제 메모리와 근사)
- **vLLM** (예정): `nvidia-smi` 또는 `torch.cuda.memory_allocated()`

---

## Backend Configuration

| 백엔드 | 주요 옵션 |
|--------|----------|
| MLX | temperature=0, Metal GPU |
| llama.cpp | n-gpu-layers=99, flash-attn=on, batch=512, ubatch=512, no-mmap |
| Ollama | temperature=0, num_ctx=262144 |
| vLLM | temperature=0, max-model-len=262144, tensor-parallel-size=auto |

---

## Thermal Guard

MacBook Pro M5 Max에서 thermal pressure level로 온도 추정:

| Pressure Level | 추정 온도 |
|----------------|----------|
| nominal | ~60°C |
| moderate | ~75°C |
| heavy | ~88°C |
| critical | ~98°C |

`heavy` 이상 감지 시 60초 쿨다운. `powermetrics`는 `sudo -n`으로 호출 (NOPASSWD 설정 필요).

---

## Reproducibility

동일 실험 재현 시:
1. `hardware.id`를 `config.yaml`에서 현재 하드웨어로 설정
2. 동일 모델 파일 (GGUF hash, MLX revision 동일)
3. 동일 백엔드 버전
4. 백그라운드 프로세스 최소화 (Chrome, Xcode 등 종료 권장)
5. `./scripts/check_models.sh`로 모델 파일 존재 확인 후 실행
