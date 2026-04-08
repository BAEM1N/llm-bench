# HP Z2 Mini G1a — AMD Ryzen AI MAX+ PRO 395 (Strix Halo, 128GB Unified)

**상태**: 🔧 셋업 예정 (2026-04)
**최종 업데이트**: 2026-04-03

---

## 하드웨어

| 항목 | 값 |
|------|-----|
| CPU | AMD Ryzen AI MAX+ PRO 395 — 16C/32T, Zen 5, 4nm, boost 5.1 GHz |
| iGPU | Radeon 8060S — 40 CU, RDNA 3.5, up to 2.9 GHz |
| NPU | XDNA 2 — 50 TOPS |
| 메모리 | **128GB LPDDR5x-8533 ECC**, unified (CPU/GPU/NPU 공유) |
| 메모리 대역폭 | **256 GB/s** |
| 스토리지 | Dual PCIe NVMe, up to 8TB (RAID 0/1) |
| 연결 | Dual Thunderbolt 4, 10GbE, 4x 4K 디스플레이 |
| GPU Arch ID | **gfx1151** |

---

## ⚠️ BIOS VRAM 설정 (가장 중요)

> **BIOS에서 dedicated VRAM을 96GB로 설정하면 안 됨** — OS 메모리 부족으로 모델 로드 멈춤.

### 올바른 설정:
1. BIOS VRAM → **Auto 또는 512MB**
2. Linux TTM 커널 파라미터로 GPU가 ~115-120GB 동적 접근 가능

```bash
# /etc/default/grub의 GRUB_CMDLINE_LINUX에 추가:
ttm.pages_limit=29360128 ttm.page_pool_size=29360128 \
amdttm.pages_limit=29360128 amdttm.page_pool_size=29360128
```

```bash
sudo update-grub && sudo reboot
```

**결과**: 122B Q4_K_M (~65GB) 포함 전 모델 `n_gpu_layers=99` 풀 GPU 오프로드 가능.

---

## OS / 커널 / ROCm

### 권장 환경
| 구성요소 | 권장 |
|---------|------|
| OS | **Ubuntu 24.04 LTS** (HP 공식 인증) |
| 커널 | **6.18.4+** 필수 (이전 버전 안정성 버그) |
| ROCm | **7.2** (gfx11-generic ISA 폴백 지원) |

### ROCm 설치

```bash
# ROCm 7.2 저장소 추가
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
# (repo URL은 AMD 공식 문서 참조)
sudo apt update
sudo apt install rocm-dev rocm-libs

# 확인
rocminfo  # gfx1151 감지 확인
```

> **참고**: gfx1151은 AMD 공식 호환 매트릭스에 없지만,
> [Strix Halo 최적화 가이드](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)를 공식 발행함 — 사실상 지원.

---

## llama.cpp 빌드

### ROCm/HIP 빌드 (권장 — 성능 최고)

```bash
cd ~ && git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build

cmake .. \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1151" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

```bash
# 런타임 환경변수
export ROCBLAS_USE_HIPBLASLT=1
```

### Vulkan 빌드 (대안 — 셋업 간단, 롱 컨텍스트 유리)

```bash
cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 프리빌트 바이너리 (가장 간단)

Lemonade SDK에서 ROCm 7 내장 llamacpp-rocm 릴리스:
https://github.com/lemonade-sdk/llamacpp-rocm/releases/latest/

### ROCm vs Vulkan 선택 기준

| 상황 | 권장 |
|------|------|
| 일반 gen/prefill | ROCm (더 빠름) |
| 롱 컨텍스트 (64K+) | Vulkan (ROCm KV cache 배치 버그) |
| 64GB+ 모델 로딩 | Vulkan (ROCm 로딩 느림) |
| 간단한 셋업 | Vulkan (ROCm 불필요) |

### 성능 기대치

| 모델 | Gen (tps) | 비고 |
|------|-----------|------|
| 30B Q8 | ~40 | ROCm + hipBLASLt |
| 120B MoE Q4 | ~45-50 | Vulkan, ROCm과 비슷 |

> Apple M4 Max급. M5 Max보다는 약간 느리지만 동일 메모리 용량 이점.

---

## Ollama (ROCm)

```bash
# gfx1151 미인식 → override 필수
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### systemd 서비스 설정

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d/
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.5.1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### 성능
- 30B Q8: ~40 tok/s
- 안전 시작점: 65K-96K context (30B → 36-45GB)
- 202K context 가능하지만 OOM 위험

---

## Lemonade Server (AMD 공식)

내부적으로 llama.cpp 사용. ROCm/Vulkan 지원.

### 설치 및 실행

```bash
# 프리빌트 (ROCm 7 내장 — 별도 ROCm 설치 불필요)
# github.com/lemonade-sdk/llamacpp-rocm/releases/latest/

# 또는 deb/rpm/snap 패키지
sudo apt install ./lemonade-server_*.deb

# 실행
lemonade-server serve --llamacpp rocm --port 8000

# 모델 다운로드
lemonade-server pull Qwen3.5-9B-GGUF
```

### API
- OpenAI 호환: `http://localhost:8000/api/v1/completions`
- Ollama 호환: `http://localhost:8000/api/chat`
- 성능 통계: `GET /api/v1/stats`
- 모델 관리: `POST /api/v1/load`, `POST /api/v1/unload`

### 내장 Qwen3.5 모델 (UD-Q4_K_XL)
- `Qwen3.5-9B-GGUF` (5.97 GB)
- `Qwen3.5-27B-GGUF` (16.7 GB)
- `Qwen3.5-35B-A3B-GGUF` (19.7 GB)
- `Qwen3.5-122B-A10B-GGUF` (68.4 GB)

> GPT-OSS-120B: Ubuntu 25.04에서 **40+ tok/s** out of the box.

---

## vLLM (ROCm) — 실험적

```bash
# 커뮤니티 빌드 사용
# kyuz0.github.io/amd-strix-halo-vllm-toolboxes/

# vLLM PR #25908에서 gfx1150/1151 지원 추가
# ROCm 6.4.4 권장 (6.4.1은 "invalid device function" 에러)
```

> **권장**: 일단 `enabled: false`. 안정화되면 시도.

---

## 알려진 이슈

| 이슈 | 영향 | 해결 |
|------|------|------|
| BIOS VRAM 96GB | OS 스타브, 모델 로드 멈춤 | Auto/512MB + TTM 파라미터 |
| 커널 < 6.18.4 | 불안정, 크래시 | 6.18.4+ 사용 |
| ROCm KV cache → shared mem | 롱 컨텍스트 성능 저하 | Vulkan 사용 |
| ROCm 64GB+ 모델 로딩 느림 | 긴 시작 시간 | Vulkan 또는 Lemonade 프리빌트 |
| Ollama gfx1151 미인식 | GPU 가속 없음 | `HSA_OVERRIDE_GFX_VERSION=11.5.1` |
| `--mlock` 무효 | UMA에서 의미 없음 | 무시 (메모리 공유 구조) |
| ROCm 7.0.2 즉시 크래시 | GPU 행 | ROCm 7.2 + 최신 amdgpu DKMS |
| hipMemcpyWithStream 병목 | 15GB+ 모델 decode 느림 | `ROCBLAS_USE_HIPBLASLT=1` |

---

## GPU 오프로드 테이블 (TTM 설정 후, ~120GB GPU 접근)

| 모델 | Quant | 크기 | GPU 오프로드 |
|------|-------|------|-------------|
| 9B | Q4_K_M | ~5GB | ✅ 전체 GPU |
| 9B | Q8_0 | ~9GB | ✅ 전체 GPU |
| 27B | Q4_K_M | ~15GB | ✅ 전체 GPU |
| 27B | Q8_0 | ~28GB | ✅ 전체 GPU |
| 35B-A3B | Q4_K_M | ~20GB | ✅ 전체 GPU |
| 35B-A3B | Q8_0 | ~37GB | ✅ 전체 GPU |
| 122B-A10B | Q4_K_M | ~65GB | ✅ 전체 GPU |

> TTM 설정 없이 (기본 16GB VRAM) → 27B Q8_0부터 CPU fallback.

---

## 벤치마크 config

```yaml
hardware:
  id: "ryzen-ai-max-395"
  description: "HP Z2 Mini G1a, Ryzen AI MAX+ PRO 395, 128GB unified"

backends:
  llamacpp:
    enabled: true
    binary: "/home/<user>/llama.cpp/build/bin/llama-server"
    n_gpu_layers: 99      # TTM 설정 후 전 모델 GPU
    flash_attn: true
    extra_args: []
  ollama:
    enabled: true         # HSA_OVERRIDE_GFX_VERSION=11.5.1 필요
  lemonade:
    enabled: true
    base_url: "http://localhost:8000"
  vllm:
    enabled: false        # 실험적, 안정화 후 활성화
```

---

## 참고 링크

- [HP Z2 Mini G1a 제품 페이지](https://www.hp.com/us-en/workstations/z2-mini-a.html)
- [ROCm Strix Halo 최적화 가이드](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [Lemonade SDK llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)
- [Lemonade Server](https://lemonade-server.ai/)
- [Framework Community — Strix Halo GPU LLM 테스트](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521)
- [Framework Community — ROCm 안정 설정 (2026-01)](https://community.frame.work/t/linux-rocm-january-2026-stable-configurations-update/79876)
- [llama.cpp/vLLM Strix Halo 툴박스](https://community.frame.work/t/llama-cpp-vllm-toolboxes-for-llm-inference-on-strix-halo/74916)
- [llama.cpp — Known-Good Strix Halo 스택](https://github.com/ggml-org/llama.cpp/discussions/20856)
- [Ollama Strix Halo ROCm 가이드](https://github.com/ollama/ollama/issues/14855)
- [vLLM PR #25908 — gfx1151 지원](https://github.com/vllm-project/vllm/pull/25908)
- [Jeff Geerling — AMD AI APU VRAM 할당](https://www.jeffgeerling.com/blog/2025/increasing-vram-allocation-on-amd-ai-apus-under-linux/)
- [StorageReview — HP Z2 Mini G1a GPT-OSS-120B 리뷰](https://www.storagereview.com/review/hp-z2-mini-g1a-review-running-gpt-oss-120b-without-a-discrete-gpu)
- [Phoronix — ROCm 7.0 Strix Halo 성능](https://www.phoronix.com/review/amd-rocm-7-strix-halo)
- [Strix Halo Wiki — llama.cpp 성능](https://strixhalo.wiki/AI/llamacpp-performance)
