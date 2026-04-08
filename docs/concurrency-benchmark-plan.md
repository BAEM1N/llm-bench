# Parallel / Concurrency Benchmark Plan

## 목적

현재 벤치마크는 **single-request / single-stream** 기준이다.
이 문서는 백엔드가 **동시에 여러 요청을 처리할 때** 성능이 어떻게 변하는지 측정하기 위한 병렬 실험 계획이다.

핵심 질문은 아래 3가지다.

1. **동시에 몇 개 요청까지 안정적으로 처리 가능한가?**
2. **지속 가능한 총 처리량(aggregate throughput)은 얼마인가?**
3. **실사용자 기준으로 몇 명까지 커버 가능한가?**

---

## 측정 대상

### 백엔드

1차 병렬 실험 대상:
- **llama.cpp**
- **Ollama**
- **LM Studio** (선택, GUI/수동 로드 제약이 있으면 후순위)

2차 대상:
- **MLX**
- **vLLM**

> 이유: 1차는 HTTP 서버형 백엔드라 "서빙 동시성" 비교가 자연스럽다.
> MLX는 현재 in-process Python API라서 1차에서는 별도 축으로 분리하는 것이 공정하다.

### 모델

전체 모델 사용:
- **Qwen3.5-9B**
- **Qwen3.5-27B**
- **Qwen3.5-35B-A3B**
- **Qwen3.5-122B-A10B**

### 양자화

우선순위:
1. **Q4_K_M** — 전체 비교 기준
2. **Q8_0** — 가능한 모델에 한해 추가

> 전체 sweep 기본축은 Q4_K_M으로 잡고, 시간이 허용되면 Q8_0 확장.

---

## 실험 질문 정의

### 1) 최대 동시 생성 수

"동시에 답변을 생성 중인 사용자"를 몇 명까지 버틸 수 있는지 측정한다.

정의:
- 특정 SLA를 만족하는 가장 높은 concurrency level = **최대 동시 생성 수**

### 2) 지속 가능한 요청률

초당 몇 개 요청까지 안정적으로 받는지 측정한다.

정의:
- 특정 SLA를 만족하는 가장 높은 arrival rate = **지속 가능한 요청률**

### 3) 환산 가능한 온라인 사용자 수

실사용자는 계속 생성 요청을 보내지 않는다. 따라서 아래 가정으로 환산한다.

- 평균 질문 간격 = `T_user`
- 평균 생성 시간 = `T_gen`
- 활성 비율 = `T_gen / T_user`
- 추정 온라인 사용자 수 = `최대 동시 생성 수 / 활성 비율`

예시:
- 최대 동시 생성 수 = 8
- 평균 생성 시간 = 6초
- 평균 질문 간격 = 30초
- 활성 비율 = 0.2
- 추정 온라인 사용자 수 = 8 / 0.2 = 40명

---

## 실험 단계

## Phase 1 — Closed-loop Concurrency

목적:
- 동시에 N개 요청을 넣었을 때 latency/throughput/failure가 어떻게 변하는지 측정

방식:
- concurrency = `1, 2, 4, 8, 12, 16, 24, 32`
- 각 level에서 동시에 N개 요청 발사
- 모든 요청이 끝나면 run 종료
- warmup 후 5회 반복 측정

장점:
- 구현 단순
- 최대 동시 생성 수 추정에 적합
- saturation point를 직관적으로 파악 가능

## Phase 2 — Open-loop Arrival Rate

목적:
- 실제 서비스처럼 초당 일정 요청이 들어올 때 queueing과 latency 붕괴 지점 측정

방식:
- request rate = `0.05, 0.1, 0.2, 0.5, 1.0 ... req/s`
- 일정 시간 동안 지속 주입
- p95 latency, fail rate, queue buildup 확인

장점:
- 지속 가능한 사용자 수 / 서비스 capacity 추정에 적합

> 구현 우선순위는 **Phase 1 먼저**, 이후 필요 시 Phase 2 추가.

---

## 트랙 설계

### Generation 중심 1차 실험

우선 아래 generation track부터 측정:
- `gen-512`
- `gen-2048`
- `gen-8192`

이유:
- 짧은 응답 / 중간 응답 / 긴 응답 모두 반영
- 동시성 증가 시 TTFT, 총 latency, aggregate TPS 변화가 잘 드러남

### Prefill 병렬 실험은 2차

후속 후보:
- `prefill-16k`
- `prefill-64k`
- `prefill-128k`

주의:
- prefill 병렬 실험은 KV cache와 메모리 압박이 매우 커서 OOM/timeout 비율이 높을 수 있음
- 1차 보고서는 generation concurrency만으로도 충분한 가치가 있음

---

## 공정성 규칙

### 1. 모델은 concurrency level sweep 동안 1회만 로드

- backend/model/quant 조합당 모델 로드 1회
- load 이후 concurrency level을 순서대로 실행
- load/unload 시간을 request latency에 포함하지 않음

### 2. 요청 프롬프트는 동일 의미 + nonce 추가

동일 프롬프트를 완전히 똑같이 보내면 prompt cache/dedup 영향이 생길 수 있다.
따라서 각 요청은 의미는 같되 마지막에 nonce를 붙인다.

예시:
- `Request nonce: C8-R3-I5`

### 3. timeout도 결과에 포함

- per-request timeout 발생 시 fail로 기록
- 병렬 실험에서는 timeout rate가 중요한 capacity 지표

### 4. concurrency level마다 warmup 수행

- concurrency 1 warmup
- concurrency 2 warmup
- ...

이렇게 해야 level 전환 직후의 캐시/스케줄링 초기화 영향을 완화할 수 있다.

### 5. Ollama / llama.cpp ctx 정책은 별도 명시

병렬 실험에서는 아래 두 모드를 구분한다.

- **Practical mode**: 현재 운영 기본값 유지
- **Fair mode**: 실제 필요 ctx만 맞춘 설정

특히 Ollama는 `num_ctx=262144` 고정 시 구조적으로 불리하므로, 최소한 문서에서 명시적으로 분리한다.

---

## SLA 정의

최대 동시 생성 수와 지속 가능한 요청률은 SLA 기준이 있어야 의미가 있다.

권장 기본 SLA:
- **success rate ≥ 99%**
- **p95 TTFT ≤ 3s** (`gen-512`, `gen-2048` 기준)
- **p95 total latency ≤ 30s** (`gen-512`, `gen-2048` 기준)
- **요청당 gen TPS p50 ≥ 10 tok/s**

긴 응답(`gen-8192`)은 별도 SLA 사용 권장:
- **success rate ≥ 99%**
- **p95 TTFT ≤ 5s**
- **p95 total latency ≤ 180s**

> 최종 보고서에는 backend/model별로 어떤 SLA를 통과했는지 표로 명시한다.

---

## 측정 지표

### 요청별(raw request metrics)

각 요청마다 저장:
- `timestamp`
- `hardware_id`
- `backend`
- `backend_version`
- `model`
- `quantization`
- `track_id`
- `concurrency`
- `run_number`
- `request_index`
- `status` (`ok`, `timeout`, `error`)
- `error_type`
- `ttft_ms`
- `gen_tps`
- `total_latency_s`
- `output_tokens`
- `start_ts`
- `first_token_ts`
- `end_ts`

### run 단위(summary metrics)

각 concurrency run마다 저장:
- `timestamp`
- `hardware_id`
- `backend`
- `model`
- `quantization`
- `track_id`
- `concurrency`
- `run_number`
- `completed_requests`
- `failed_requests`
- `success_rate`
- `makespan_s`
- `aggregate_output_tokens`
- `aggregate_gen_tps`
- `ttft_p50_ms`
- `ttft_p95_ms`
- `latency_p50_s`
- `latency_p95_s`
- `req_gen_tps_p50`
- `req_gen_tps_p95`
- `peak_memory_gb` (가능 시)

---

## 핵심 계산식

### 요청별 지표

- `TTFT = first_token_ts - start_ts`
- `Total latency = end_ts - start_ts`
- `Per-request gen TPS = output_tokens / (end_ts - first_token_ts)`

### run 단위 지표

- `Makespan = max(end_ts) - min(start_ts)`
- `Aggregate TPS = sum(output_tokens of successful requests) / makespan`
- `Success rate = completed / concurrency`

### 환산 사용자 수

- `Active ratio = avg_generation_time / avg_question_interval`
- `Estimated online users = max_concurrency / active_ratio`

---

## 실행 매트릭스

## Pass 1 — Coarse Scan

목적:
- 전체 모델 family의 대략적 saturation zone 탐색

대상:
- 모델: 전부
- 양자화: Q4_K_M
- 트랙: `gen-512`, `gen-2048`
- concurrency: `1, 2, 4, 8`
- backend: `ollama`, `llamacpp`

## Pass 2 — Boundary Refinement

목적:
- 각 조합의 최대 동시 생성 수를 더 촘촘히 찾기

예시:
- 9B / 35B-A3B: `8, 12, 16, 24, 32`
- 27B: `4, 6, 8, 10, 12`
- 122B: `1, 2, 3, 4, 6, 8`

## Pass 3 — Long Response Stress

대상:
- 상위 후보 조합만
- 트랙: `gen-8192`
- concurrency: Pass 2에서 찾은 경계 근처

## Pass 4 — Optional Extensions

- Q8_0 확장
- Prefill concurrency
- LM Studio 추가
- MLX 별도 축 비교
- Open-loop arrival rate

---

## 결과 파일 스키마

권장 출력 파일:
- `results/concurrent_raw_<timestamp>.csv`
- `results/concurrent_summary_<timestamp>.csv`

예시:

```csv
# concurrent_raw
 timestamp,hardware_id,backend,backend_version,model,quantization,track_id,concurrency,run_number,request_index,status,error_type,ttft_ms,gen_tps,total_latency_s,output_tokens,start_ts,first_token_ts,end_ts
```

```csv
# concurrent_summary
 timestamp,hardware_id,backend,model,quantization,track_id,concurrency,run_number,completed_requests,failed_requests,success_rate,makespan_s,aggregate_output_tokens,aggregate_gen_tps,ttft_p50_ms,ttft_p95_ms,latency_p50_s,latency_p95_s,req_gen_tps_p50,req_gen_tps_p95,peak_memory_gb
```

---

## 시각화 계획

필수 차트:

1. **Aggregate TPS vs Concurrency**
   - x: concurrency
   - y: aggregate tok/s
   - line: backend

2. **p95 Total Latency vs Concurrency**
   - saturation 지점 확인용

3. **p95 TTFT vs Concurrency**
   - interactive responsiveness 확인용

4. **Success Rate vs Concurrency**
   - timeout/error 급증 구간 확인용

5. **Scaling Efficiency**
   - `aggregate_tps_at_N / (single_request_tps × N)`
   - 1에 가까울수록 scaling 효율 우수

6. **Estimated Online Users Table**
   - 질문 간격 15초 / 30초 / 60초 가정별 환산 사용자 수

---

## 구현 제안

### 새 실행기 추가

기존 single benchmark와 분리:
- `src/concurrent_runner.py`

이유:
- 현재 `src/runner.py`는 단일 요청 벤치로 유지
- 스키마/로직/집계가 완전히 다름
- regression 위험 감소

### 백엔드 구현 방향

권장:
- HTTP backend에 대해 `async` 요청 처리 추가
- `httpx.AsyncClient` 기반 스트리밍 구현

대안:
- 기존 sync generate를 thread pool로 감싸기

권장 이유:
- 스트리밍 TTFT 측정이 async 쪽이 더 자연스럽고 정확함

### 설정 추가

`config.yaml` 또는 별도 파일에 아래 섹션 추가 권장:

```yaml
concurrency_benchmark:
  enabled: true
  levels: [1, 2, 4, 8, 12, 16, 24, 32]
  warmup_runs: 1
  measure_runs: 5
  request_timeout_s: 600
  stagger_ms: 0
  include_generation_tracks: ["gen-512", "gen-2048", "gen-8192"]
  include_prefill_tracks: []
  prompt_nonce: true
  sla:
    success_rate_min: 0.99
    ttft_p95_ms: 3000
    latency_p95_s: 30
    req_gen_tps_p50_min: 10
```

---

## 해석 가이드

### 좋은 결과
- concurrency 증가에 따라 aggregate TPS가 꾸준히 증가
- p95 TTFT / latency가 완만히 증가
- success rate 유지

### 나쁜 결과
- concurrency 증가 초반부터 TTFT 급증
- aggregate TPS 증가가 정체
- timeout / error 비율 급증

### 주의할 점
- aggregate TPS가 높아도 per-request latency가 너무 크면 interactive 용도로는 부적합
- 반대로 latency는 좋지만 aggregate TPS가 낮으면 소규모 사용자만 적합
- 따라서 **throughput / latency / failure**를 같이 봐야 한다

---

## 최종 산출물

최종 보고서에는 backend × model × quantization 조합마다 아래를 반드시 포함한다.

1. **최대 동시 생성 수**
2. **SLA 통과 최대 concurrency**
3. **최대 aggregate TPS**
4. **p95 TTFT / p95 latency at saturation**
5. **질문 간격 15초 / 30초 / 60초 기준 환산 온라인 사용자 수**

---

## 권장 시작점

가장 현실적인 첫 실행 순서:

1. `llamacpp`, `ollama`
2. 전 모델
3. Q4_K_M
4. `gen-512`, `gen-2048`
5. concurrency `1, 2, 4, 8`

이 coarse scan 결과를 본 뒤,
- 9B / 35B-A3B는 더 높은 concurrency로 확장
- 122B는 더 촘촘한 저구간 탐색
- 필요 시 `gen-8192`와 Q8_0 확장

---

## 한 줄 결론

병렬 실험은 단순히 "몇 명까지 되나"가 아니라,
**최대 동시 생성 수 + 지속 가능한 요청률 + 환산 사용자 수**를 같이 측정해야 의미가 있다.

이 계획은 현재 single-request 벤치와 분리된 **server concurrency benchmark**를 구축하는 기준 문서로 사용한다.
