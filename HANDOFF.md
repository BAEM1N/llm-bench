# 핸드오프 문서

> 작성일: 2026-04-02  
> 상태: **Mac Apple Silicon 실험 완료**

---

## 완료된 것 (Mac, 2026-04-02 04:11 기준)

| 항목 | 상태 |
|------|------|
| 코드 / 설정 전체 | ✅ |
| MLX 전 모델/quant — gen + prefill | ✅ |
| llama.cpp 전 모델/quant — gen + prefill | ✅ |
| Ollama 전 모델/quant — gen + prefill | ✅ (27B prefill-128k 제외) |
| REPORT.md 업데이트 | ✅ |
| README.md 업데이트 | ✅ |
| docs/mac-apple-silicon.md 업데이트 | ✅ |

**총 187/189 측정 완료** — 미완료 2건은 Ollama 27B prefill-128k (메모리 부족, 재실행해도 동일)

---

## 결과 파일

| 파일 | 내용 |
|------|------|
| `results/full_gen_all.csv` | MLX + Ollama + llama.cpp 9B/27B gen 트랙 (오전 실험) |
| `results/llamacpp_remaining.csv` | llama.cpp 27B Q8/35B/122B gen 트랙 |
| `results/prefill_all.csv` | 전 백엔드/모델 prefill 트랙 (1k~128k) |

---

## 핵심 발견

1. **Gen TPS**: MLX > llama.cpp > Ollama. MLX 35B MoE가 142 tok/s로 전체 1위.
2. **Prefill**: llama.cpp Flash Attention 압도 — 128k에서 574K tok/s (TTFT 229ms).
3. **MoE prefill 동치**: llama.cpp 35B MoE(571K) ≈ 9B Dense(574K) — active params 3B 효과.
4. **Ollama KV 캐시**: num_ctx=262144 전체 pre-allocate → 2~3배 메모리, 27B 128k OOM.
5. **MLX 장문 prefill 한계**: 128k TTFT 9B=46초, 27B=162초 — Flash Attention 미적용 추정.

---

## 다음 단계

- [ ] DGX Spark (GB10) 실험 — `config.yaml`의 `hardware.id`를 `dgx-spark`로 변경
- [ ] Ryzen AI MAX+ 395 실험 — `hardware.id`를 `ryzen-ai-395`로 변경
- [ ] 시각화 (`uv run python visualize.py results/`)
- [ ] 크로스 플랫폼 비교 보고서 작성

---

## DGX Spark 실험 전 체크리스트

```bash
# 1. config.yaml hardware.id 변경
#    hardware:
#      id: dgx-spark

# 2. 모델 다운로드 확인
./scripts/download.sh dgx-spark --backends gguf vllm

# 3. 전체 실행
uv run python -m src.runner --config config.yaml \
  --output results/dgx_spark_gen_all.csv
```
