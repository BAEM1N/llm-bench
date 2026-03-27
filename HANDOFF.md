# 핸드오프 문서

> 작성일: 2026-03-28
> 다음 세션 목표: 2026-03-29 (일) 실험 실행 및 REPORT.md 수치 기입

---

## 현재 상태 요약

### 코드 / 설정 — 완료
- `src/runner.py`, `src/backends/*.py`, `src/metrics.py`, `src/memory.py`, `src/thermal.py` 구현 완료
- `config.yaml` 전체 설정 완료 (모델 경로, 백엔드 옵션, 트랙 정의)
- `visualize.py` 차트 생성 스크립트 완료
- `scripts/download.sh`, `scripts/check_models.sh` 완료
- GitHub: https://github.com/BAEM1N/llm-bench

### 다운로드 — 진행 중 (2026-03-28 01:30 기준)

| 항목 | 상태 | 비고 |
|------|------|------|
| GGUF 9B Q4/Q8 | ✅ | |
| GGUF 27B Q4/Q8 | ✅ | |
| GGUF 35B Q4 | ⏳ curl 진행 중 | `~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/` |
| GGUF 35B Q8 | ⏳ curl 진행 중 | 동일 경로 |
| GGUF 122B Q4 | ✅ | 3-shard (10M + 47G + 25G) |
| MLX 9B 4bit/8bit | ✅ | |
| MLX 27B 4bit/8bit | ✅ | |
| MLX 35B 4bit | ✅ | |
| MLX 35B 8bit | ⏳ HF 진행 중 (1/8 shard) | `~/.cache/mlx/Qwen3.5-35B-A3B-8bit/` |
| MLX 122B 4bit | ✅ | |
| Ollama 9B Q4/Q8 | ✅ | |
| Ollama 27B Q4/Q8 | ✅ | |
| Ollama 35B Q4/Q8 | ✅ | |
| Ollama 122B Q4 | ⏳ 진행 중 (45%, ~81G) | PID 19523 |

---

## 실험 전 반드시 확인할 것

### 1. 다운로드 완료 확인
```bash
./scripts/check_models.sh
```

미완료 항목이 있으면 해당 다운로드 완료 후 진행.

### 2. 진행 중 프로세스 확인
```bash
ps aux | grep -E "(curl|huggingface|ollama pull)" | grep -v grep
```

### 3. GGUF 35B 경로 확인
35B Q4/Q8 모두 `~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/` 에 있어야 함 (unsloth, lmstudio-community 아님).

### 4. LM Studio 사전 설정 (수동)
각 모델 로드 시:
- Context Length: **262144**
- Flash Attention: **ON**
- GPU Offload: **100%**

LM Studio는 API로 이 값들을 제어할 수 없어서 GUI에서 직접 설정해야 함.

### 5. sudo 캐시 활성화 (thermal 측정용)
```bash
sudo powermetrics --samplers thermal -n 1
```

---

## 실험 실행

```bash
cd ~/Workspace/llm-bench

# 전체 실행
uv run python -m src.runner --config config.yaml

# 백엔드 선택 실행 (테스트용)
uv run python -m src.runner --backends ollama --models qwen3.5-9b --tracks gen-512

# 결과 경로 지정
uv run python -m src.runner --output results/bench_20260329.csv
```

실험 완료 후:
```bash
uv run python visualize.py results/bench_20260329.csv
```

---

## 알려진 이슈

| 이슈 | 내용 | 대응 |
|------|------|------|
| LM Studio prompt_tps | streaming API에서 미제공, 0으로 기록됨 | 결과 해석 시 제외 |
| 122B 컨텍스트 | 256K KV cache + 모델 메모리 합산 시 128GB 초과 가능 | OOM 시 config.yaml `context_window` 를 131072로 줄이고 재실행, 결과에 명시 |
| Thermal throttling | 장시간 실행 시 85°C 초과 가능 | config.yaml `thermal.enabled: true` 로 자동 쿨다운 적용 중 |
| HF CLI stall | 다운로드 중 stall 발생 시 | `curl -L -C - <url> -o <dest>` 로 resume |

---

## 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `config.yaml` | 전체 실험 설정 (모델 경로, 백엔드 옵션, 트랙 정의) |
| `REPORT.md` | 결과 보고서 템플릿 — 실험 후 `—` 수치 채우기 |
| `results/` | 벤치마크 CSV 저장 위치 |
| `charts/` | 시각화 차트 저장 위치 |
| `MODEL_STATUS.md` | 모델 다운로드 현황 |
| `CLAUDE.md` | 프로젝트 전체 컨텍스트 (새 세션 시작 시 자동 로드) |
