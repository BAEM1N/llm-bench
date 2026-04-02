# Benchmark Methodology

---

## Measurement Protocol

| 항목 | 값 |
|------|-----|
| Warmup | 1회 (결과 제외, config `warmup_runs`) |
| Measure | 5회 (config `measure_runs: 5`) |
| 집계 | 중앙값 (median) + p95 |
| 온도 가드 | 85°C 초과 → 60초 쿨다운 후 재개 |
| Inter-run sleep | 5초 (config `inter_run_sleep`) |
| Inter-track sleep | 60초 (config `inter_track_sleep`) |
| Inter-model sleep | 120초 (config `inter_model_sleep`) |

CSV의 `actual_runs` 컬럼: 5회 중 실제 성공한 run 수. 5 미만이면 통계 신뢰도가 낮음.

---

## Tracks

### Generation Track
짧은 입력, 출력 길이 고정으로 **순수 generation 속도** 측정.

모든 트랙은 동일한 numbered FAQ 꼬리 지시문(`Question 1:`)을 사용한다.
트랙 간 비교는 출력 길이만 다르며, 태스크 구조는 동일하다.

| Track ID | Input (tok, 근사) | Output |
|----------|------------------:|-------:|
| gen-512  | 64 | 512 |
| gen-2048 | 64 | 2,048 |
| gen-4096 | 64 | 4,096 |
| gen-8192 | 64 | 8,192 |

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
- **llama.cpp**: `timings.predicted_per_second` (서버 내부 타이밍, native)
- **MLX**: `response.generation_tps` (mlx_lm GenerationResponse, native)
- **Ollama**: `eval_count / (eval_duration_ns / 1e9)` (native)
- **vLLM**: `usage.completion_tokens / generation_duration` (클라이언트 측 타이밍)
- **LM Studio**: `output_tokens / generation_duration` (클라이언트 측 타이밍)

### TTFT (Time To First Token, ms)
모든 백엔드에서 클라이언트 측 측정(`t_first_chunk - t_request_start`).
네트워크 레이턴시 및 서버 큐 대기 포함.

⚠️ Ollama TTFT는 num_predict에 비례 증가하는 이상 동작이 보고된 바 있다.
   gen 트랙에서 Ollama TTFT는 gen_tps보다 신뢰도가 낮다.

### Prefill TPS (tok/s)
CSV의 `prefill_tps_source` 컬럼으로 신뢰도 확인:

| 값 | 의미 |
|----|------|
| `native` | 백엔드가 직접 측정한 값 (llama.cpp timings, MLX prompt_tps, Ollama prompt_eval_duration) |
| `ttft_estimate` | `input_tokens / TTFT` 폴백 — TTFT에 큐/네트워크 포함되어 과소 추정 가능 |

vLLM과 LM Studio는 항상 `ttft_estimate`. Prefill TPS를 비교할 때는 `native` 값만 사용 권장.

### Peak Memory (GB)
CSV의 `peak_memory_method` 컬럼으로 측정 방식 확인:

| 값 | 백엔드 | 의미 |
|----|--------|------|
| `metal_peak` | MLX | `mx.metal.get_peak_memory()` — 모델 로드 시 Metal 최대 사용량. 가장 정확. |
| `process_rss` | llama.cpp | llama-server PID의 RSS — 공유 라이브러리 포함, 약간 과대 추정 |
| `rss_delta` | Ollama | 로드 전후 RSS 차이 — 모델에 귀속된 증분만 측정, 실제보다 낮을 수 있음 |
| `gpu_smi` | vLLM | nvidia-smi / rocm-smi 전체 사용량 — 다른 GPU 프로세스 포함 가능 |
| `unknown` | LM Studio | GUI 앱 메모리 분리 불가 |

백엔드 간 절대 비교는 이 방식 차이를 감안해야 한다.

### Power & Efficiency
- `cpu_power_w`: CPU 전력 (W). macOS = powermetrics, Linux = RAPL 200ms 샘플.
- `gpu_power_w`: GPU 전력 (W). Linux = nvidia-smi.
- `efficiency_tps_per_w`: `gen_tps / (cpu_power_w + gpu_power_w)`. 측정 불가 시 -1.

⚠️ 전력 측정은 run 시작 직전 스냅샷이다. run 중 평균 전력이 아님.
   맥북 Apple Silicon에서는 CPU와 GPU가 unified memory를 공유하므로
   `cpu_power_w + gpu_power_w`가 시스템 전체 전력에 근사한다.

---

## Backend Configuration

| 백엔드 | 주요 옵션 |
|--------|----------|
| MLX | temperature=0, Metal GPU |
| llama.cpp | n-gpu-layers=99, flash-attn=on, batch=512, ubatch=512 |
| Ollama | temperature=0, num_ctx=262144 (full ctx) |
| vLLM | temperature=0, max-model-len=262144, tensor-parallel-size=auto |

---

## Thermal Guard

| Pressure Level | 추정 온도 |
|----------------|----------|
| nominal | ~60°C |
| moderate | ~75°C |
| heavy | ~88°C |
| critical | ~98°C |

85°C 초과 시 60초 쿨다운. `powermetrics`는 `sudo -n`으로 호출.  
전력 측정도 `powermetrics`를 사용하므로 실험 전 `sudo powermetrics --samplers cpu_power -n 1` 한 번 실행해 sudo 캐시 활성화 권장.

---

## Comparison Modes

| Mode | 설명 | 조건 |
|------|------|------|
| A | Practical stack | 백엔드별 최적 설정. 실사용 성능 비교. |
| B | Normalized engine | llamacpp + 동일 GGUF만 허용. 하드웨어 간 순수 비교. `comparison_mode=B` 설정 시 다른 백엔드 자동 제외. |
| C | Concurrency | concurrent_runner.py 전용. 동시 요청 처리 능력 측정. |

---

## Concurrency Benchmark

### Phase 1 — Closed-loop
N개 요청을 동시에 발사, 전부 완료 후 다음 run.
"서버가 N개를 동시에 처리할 수 있는가"를 측정.

CSV의 `concurrency_mode`:
- `parallel`: HTTP 백엔드 (Ollama, llama.cpp, vLLM). 서버가 실제로 N개를 병렬 처리.
- `queued_gpu`: MLX. in-process thread pool이지만 Metal GPU는 직렬 실행. 수치는 큐 대기 시간 + GPU 직렬화 관찰값. HTTP 백엔드와 동등하게 비교하면 안 된다.

### Phase 2 — Open-loop (선택적)
`config.yaml`의 `open_loop_rates: [0.5, 1.0, ...]`로 활성화.
고정 간격(`1/rate` 초)으로 요청을 주입한다.

### SLA 기준
`sla_pass=true`인 최대 concurrency = SLA를 만족하면서 처리 가능한 최대 동시 요청 수.

---

## 설계 제약 및 해석 주의사항

### 1. 입력 토큰 근사치 (`input_tokens`)

config의 `input_tokens: 64`는 **명목 값**이다. 실제 프롬프트 토큰 수는 다음 이유로 다를 수 있다:

- `CHARS_PER_TOKEN = 3.8`은 영문 tiktoken 기준 근사값. 모델/언어마다 다름.
- 꼬리 지시문(~45 tokens)이 고정 추가되므로, 텍스트 바디는 `64 - 45 = ~19 tokens`.
  실제 입력은 약 64 tokens이지만 텍스트 본문 비율이 작다.
- 백엔드마다 토크나이저가 달라 동일 문자열의 토큰 수가 수 % 차이날 수 있다.

**결론**: `input_tokens`는 실험 간 동일 프롬프트 문자열을 사용한다는 뜻으로 이해할 것.
절대 토큰 수로 해석할 때는 ±10% 오차를 감안해야 한다.
Prefill TPS는 `input_tokens(명목) / TTFT`로 계산되므로, 백엔드 간 prefill TPS 비교 시
native 값(`prefill_tps_source=native`)만 사용해야 한다.

---

### 2. 컨텍스트 윈도우 정책: 공정성 vs. 현실성

현재 설정: **전 백엔드 262,144 토큰 full context** (`context_window: 262144`).

**장점:**
- KV cache 크기, 메모리 레이아웃이 동등하여 백엔드 간 공정 비교 가능.
- llama.cpp가 최소 ctx 사용으로 유리한 상황을 방지.

**단점 / 현실성 한계:**
- 대부분의 실제 서비스는 4K–32K context를 사용한다.
  262K ctx 상태에서의 gen_tps가 실사용과 다를 수 있다 (메모리 압력, KV 관리 오버헤드 차이).
- vLLM / Ollama에서는 KV 블록을 미리 할당하므로 262K는 메모리를 많이 점유한다.
  이것이 gen_tps에 영향을 줄 수 있다.
- Ryzen AI 395(16GB iGPU)는 122B 모델의 262K ctx를 GPU에 올리지 못할 수 있다.

**Mode B(normalized engine) 시나리오:**
동일 하드웨어 → 하드웨어 간 비교라면 full ctx 통일이 옳다.
하드웨어별 최적 ctx를 쓰고 싶다면 Mode A + `hardware.id`로 구분하여 별도 실험으로 비교.

---

### 3. Open-loop 트래픽 모델: 고정 간격 vs. Poisson

현재 Phase 2는 **고정 간격(deterministic)** 도착 모델:
- rate = 1 req/s → 매 1.0초마다 1개 발사

실제 서비스 트래픽은 **Poisson 과정(확률적)**:
- rate = 1 req/s → 도착 간격이 지수분포(평균 1.0초), burst 가능

**측정값 해석 범위:**
고정 간격 open-loop로 측정 가능한 것:
- "이 rate를 유지할 때 p95 latency / success rate"
- "어느 rate부터 큐가 쌓이기 시작하는가 (Knee point)"
- "지속 처리량(sustained throughput)" 상한 추정

측정할 수 없는 것:
- Burst 내성 — Poisson 클러스터에서의 실제 p99 latency
- 순간 spike 처리 능력
- 완전한 M/M/1 큐 모델 기반 용량 설계

**결론**: 현재 open-loop 결과는 "이 rate를 초과하면 SLA가 깨진다"는
**용량 상한 추정**으로 사용할 것.
Poisson 도착 하에서는 같은 평균 rate에서도 latency가 더 높게 나올 수 있다.

---

### 4. Hit Rate는 생성 완주율이지 품질 지표가 아님

`hit_rate = output_tokens / max_tokens` (generation 트랙 전용, prefill은 -1).

**Hit rate가 높다는 것의 의미:**
- 모델이 max_tokens까지 생성을 이어갔다.
- EOS 없이 numbered FAQ 형식을 따라 계속 답변했다.

**Hit rate가 높아도 품질 문제가 있을 수 있는 경우:**
- 반복/루프 — 같은 질문과 답변을 계속 반복
- 저품질 장문 — 관련 없는 내용으로 채움
- Format 붕괴 — 번호 매기기가 흐트러지거나 hallucination

**Hit rate가 낮아도 정상인 경우:**
- gen-512에서 모델이 30개 질문 대신 간결한 답변 5개로 종료
- 내용이 완결됐기 때문에 EOS 발생 (의미 있는 종료)

**권장 해석:**
- hit_rate < 0.5: 모델이 지시를 따르지 않거나 EOS 경향 강함. 결과 주의.
- hit_rate > 0.9: numbered format이 EOS 억제에 효과적으로 작동. 결과 신뢰.
- 품질 검증이 필요하다면 결과 텍스트 샘플링 + 수동 확인 필요.

---

## Reproducibility

동일 실험 재현 시:
1. `hardware.id`를 `config.yaml`에서 현재 하드웨어로 설정
2. 동일 모델 파일 (GGUF hash, MLX revision 동일)
3. 동일 백엔드 버전
4. 백그라운드 프로세스 최소화
5. `./scripts/check_models.sh`로 모델 파일 존재 확인
6. macOS: `sudo powermetrics --samplers cpu_power,thermal -n 1` 실행으로 sudo 캐시 활성화
