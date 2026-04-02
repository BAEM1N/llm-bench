# NVIDIA DGX Spark (GB10, 128GB Unified Memory)

**상태**: 🔜 실험 예정

---

## 하드웨어

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA GB10 (Blackwell) |
| 메모리 | 128GB Unified Memory |
| 대역폭 | ~1 TB/s |
| CUDA | 12.x |

---

## 예정 백엔드

| 백엔드 | 비고 |
|--------|------|
| **vLLM** | AWQ / GPTQ 양자화, CUDA |
| **llama.cpp** | CUDA 빌드, Flash Attention |
| **Ollama** | CUDA 지원 |
| ~~MLX~~ | Apple Silicon 전용 — 미지원 |

---

## 예정 모델 / 양자화

| 모델 | vLLM | llama.cpp | Ollama |
|------|------|-----------|--------|
| Qwen3.5-9B | AWQ-4bit | Q4_K_M / Q8_0 | q4_K_M / q8_0 |
| Qwen3.5-27B | AWQ-4bit | Q4_K_M / Q8_0 | q4_K_M / q8_0 |
| Qwen3.5-35B-A3B | AWQ-4bit | Q4_K_M / Q8_0 | q4_K_M / q8_0 |
| Qwen3.5-122B-A10B | AWQ-4bit | Q4_K_M | q4_K_M |

---

## 실행 계획

```bash
# config.yaml hardware.id 변경
hardware:
  id: dgx-spark
  description: "NVIDIA DGX Spark GB10 128GB"

# 실험 실행
uv run python -m src.runner \
  --config config.yaml \
  --backends vllm llamacpp ollama \
  --tracks gen-512 gen-2048 gen-4096 gen-8192
```

---

## 비교 포인트

- **vLLM vs llama.cpp CUDA**: 동일 GGUF vs AWQ, throughput 차이
- **GPU Unified Memory vs Apple Silicon**: 아키텍처 차이에 따른 메모리 대역폭 활용
- **MoE 모델**: CUDA vs Metal에서 35B/122B sparse activation 효율 비교
