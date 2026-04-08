# NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB Unified)

**상태**: 🔧 셋업 예정 (2026-04)
**최종 업데이트**: 2026-04-03

---

## 하드웨어

| 항목 | 값 |
|------|-----|
| SoC | NVIDIA GB10 (Grace ARM CPU + Blackwell GPU) |
| GPU | 6,144 CUDA cores, 5th-gen Tensor Cores |
| AI Compute | 1 PFLOP (sparse FP4) |
| CPU | 20-core ARM (10x Cortex-X925 + 10x Cortex-A725) |
| 메모리 | 128GB LPDDR5x unified, 256-bit, **273 GB/s** |
| 스토리지 | 1TB / 4TB NVMe M.2 (SED) |
| 네트워크 | 10 GbE RJ-45, ConnectX-7 (200 Gbps), Wi-Fi 7, BT 5.4 |
| 전원 | 240W USB-C (SoC TDP: 140W) — **반드시 기본 제공 어댑터 사용** |
| 크기 | 150 x 150 x 50.5 mm, 1.2 kg |
| 동작 온도 | 5-30°C (주변 온도) |
| 가격 | ~$4,699 |

---

## 소프트웨어 스택 (DGX OS 7.4.0, 2026-03)

| 구성요소 | 버전 |
|---------|------|
| DGX OS | 7.4.0 (Ubuntu 24.04 기반) |
| 커널 | 6.17 (Ubuntu HWE) |
| GPU 드라이버 | 580.142 |
| CUDA | **13.0.2** |
| Python | 3.12 |

**프리인스톨**: CUDA toolkit, cuDNN, TensorRT, Docker, NVIDIA Container Runtime, Ollama, JupyterLab.
별도 드라이버/CUDA 설치 불필요.

---

## 초기 셋업

```bash
# SSH 접속 후 기본 확인
nvidia-smi                # GPU 상태 (메모리는 "Not Supported" 정상)
cat /etc/os-release       # DGX OS 버전
nvcc --version            # CUDA 버전
free -h                   # 총 메모리 (128GB unified)

# 필수 패키지
sudo apt update && sudo apt install -y git cmake build-essential nvtop htop
```

> **주의**: `nvidia-smi`에서 총 메모리가 "Not Supported"로 나옴. iGPU unified 메모리 특성.
> 프로세스별 메모리 사용량은 정상 표시됨. 시스템 레벨은 `/proc/meminfo` 사용.

---

## llama.cpp 빌드 (CUDA, sm_121)

```bash
cd ~ && git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir -p build-gpu && cd build-gpu

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++

make -j"$(nproc)"
```

### 실행 팁
- `--no-mmap`: 모델 로드 **90초 → ~20초**로 단축. 필수 사용.
- `-ngl 99`: 전체 GPU 오프로드 (128GB unified이므로 전 모델 가능)
- `--flash-attn`: prefill 속도 향상
- 빌드 시간: ~2-4분

### 성능 참고치

| 모델 | Prefill (tps) | Gen (tps) |
|------|--------------|-----------|
| GPT-OSS-120B (MXFP4) | ~1,950 | ~60 |
| GPT-OSS-20B (MXFP4) | ~3,800 | ~86 |
| Qwen3 Coder 30B (Q8_0) | ~1,650 | ~44 |

> Apple M4 Max 대비 ~70-75% 속도. 메모리 대역폭 차이 (273 vs 546 GB/s).

---

## vLLM 셋업

Blackwell sm_121 아키텍처로 인해 **소스 빌드 필요**. 커뮤니티 원커맨드 인스톨러 권장:

```bash
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
```

또는 **NVIDIA Docker 컨테이너** (프리빌트) 사용.

### 환경변수

```bash
export TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

### 주의사항
- Eager mode 우선 사용 권장 (`--enforce-eager`)
- Triton 3.5.0 공식 릴리스에 sm_121a 버그 → dev 브랜치 빌드 필요
- 빌드 디스크 ~50GB, 시간 20-30분
- 테스트 완료 버전: vLLM 0.11.1rc4, PyTorch 2.9.0+cu130

---

## SGLang

- EAGLE3 speculative decoding으로 **2x throughput 부스트** 가능
- 배치 워크로드에 적합
- Llama 3.1 8B (batch 32): 7,949 tps prefill / 368 tps decode

---

## Ollama

프리인스톨됨. 별도 설정 불필요.

```bash
ollama pull qwen3.5:9b-q4_K_M
ollama run qwen3.5:9b-q4_K_M
```

> llama.cpp보다 3-4 tps 느림. 간편 사용 시 적합.

---

## NIM (NVIDIA Inference Microservice)

```bash
# NGC 인증 → Docker 컨테이너 실행
# 개발/테스트: 무료 (NVIDIA Developer Program)
# 프로덕션: AI Enterprise 라이선스 필요 (90일 트라이얼)
```

---

## 추론 엔진 비교 (커뮤니티 합의)

| 엔진 | 용도 | 비고 |
|------|------|------|
| **llama.cpp** | 단일유저, 최고 throughput | 기본 권장 |
| **Ollama** | 간편 사용 | 프리인스톨, 약간 느림 |
| **vLLM** | 멀티유저 서빙 | 단일유저에는 과잉, 셋업 복잡 |
| **SGLang** | 배치 워크로드 | speculative decoding 2x 부스트 |
| **NIM** | 프로덕션 배포 | Docker, 프리빌트, 라이선스 필요 |
| **TensorRT-LLM** | 이론적 최대 성능 | 컴파일 수시간, 비실용적 |

---

## 알려진 이슈

| 이슈 | 해결 |
|------|------|
| `nvidia-smi` 메모리 "Not Supported" | iGPU 정상 동작. `/proc/meminfo` 사용 |
| `cudaMemGetInfo` 과소 보고 | swap-reclaimable 메모리 미반영 |
| 전원 어댑터 | **반드시 기본 제공 어댑터** 사용. 서드파티 시 부팅 실패/성능 저하 |
| HDMI 딥슬립 | 모니터 물리 버튼으로 깨우기 |
| Triton sm_121a 버그 | dev 브랜치 빌드 필수 |
| 메모리 압박 시 | `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| 온도 모니터링 | GPU: nvidia-smi. CPU(Grace ARM): `/sys/class/thermal/thermal_zone*/temp` |

---

## CES 2026 업데이트

TensorRT-LLM 최적화 + speculative decoding으로 **출시 대비 2.5x 성능 향상** (SW only).

---

## 벤치마크 config

```yaml
hardware:
  id: "dgx-spark"
  description: "NVIDIA DGX Spark GB10 128GB unified"

backends:
  llamacpp:
    enabled: true
    binary: "/home/<user>/llama.cpp/build-gpu/bin/llama-server"
    n_gpu_layers: 99
    flash_attn: true
    extra_args: ["--no-mmap"]
  ollama:
    enabled: true
  vllm:
    enabled: true  # 소스 빌드 후
    tensor_parallel_size: 1  # 단일 GPU
    max_model_len: 65536     # KV cache 128GB unified에서 여유
  sglang:
    enabled: true
```

---

## 참고 링크

- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/index.html)
- [DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)
- [llama.cpp on DGX Spark (ARM Learning Path)](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/)
- [vLLM One-Command Installer](https://github.com/eelbaz/dgx-spark-vllm-setup)
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [LMSYS DGX Spark Review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [Choosing Inference Engine on DGX Spark](https://medium.com/sparktastic/choosing-an-inference-engine-on-dgx-spark-8a312dfcaac6)
