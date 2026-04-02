# AMD Ryzen AI MAX+ 395 (128GB DDR5)

**상태**: 🔜 실험 예정

---

## 하드웨어

| 항목 | 값 |
|------|-----|
| CPU | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| iGPU | Radeon 890M (RDNA 3.5, 40 CU) |
| iGPU VRAM | ~16GB (shared from system memory) |
| 전체 메모리 | 최대 128GB DDR5 |
| ROCm 지원 | ✅ (ROCm 6.x) |

---

## 예정 백엔드

| 백엔드 | 비고 |
|--------|------|
| **llama.cpp (ROCm)** | `-DGGML_HIPBLAS=ON` 빌드, GPU offload |
| **Ollama** | ROCm 빌드 사용 |
| **vLLM (ROCm)** | ROCm 지원 시 추가, 미지원 시 제외 |
| ~~MLX~~ | Apple Silicon 전용 — 미지원 |

> vLLM ROCm 지원 여부는 실험 환경 세팅 시 확인. 미지원 또는 불안정 시 제외.

---

## GPU 오프로드 예상

| 모델 | Quant | iGPU (16GB VRAM) |
|------|-------|-----------------|
| 9B | Q4_K_M (~5GB) | Full GPU offload ✅ |
| 9B | Q8_0 (~9GB) | Full GPU offload ✅ |
| 27B | Q4_K_M (~15GB) | Full GPU offload ✅ |
| 27B | Q8_0 (~28GB) | CPU fallback ⚠️ |
| 35B-A3B | Q4_K_M (~18GB) | CPU fallback ⚠️ |
| 122B-A10B | Q4_K_M (~64GB) | CPU only ❌ |

> 16GB 초과 모델은 CPU/시스템 메모리로 fallback → 속도 저하 예상.

---

## 실행 계획

```bash
# llama.cpp ROCm 빌드
cmake -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# config.yaml hardware.id 변경
hardware:
  id: ryzen-ai-395
  description: "AMD Ryzen AI MAX+ 395 128GB DDR5"

# 실험 실행
uv run python -m src.runner \
  --config config.yaml \
  --backends llamacpp ollama \
  --tracks gen-512 gen-2048 gen-4096 gen-8192
```

---

## 비교 포인트

- **Apple Silicon vs Ryzen AI**: 동일 128GB 메모리 풀, 아키텍처 차이
- **CPU-only vs GPU offload**: 16GB iGPU 한계로 큰 모델은 CPU로 실행
- **MoE 모델**: sparse activation이 x86 CPU에서 어떻게 동작하는지
