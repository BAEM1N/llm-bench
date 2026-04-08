# vLLM / SGLang 환경 진단 및 개선 방안

> 조사일: 2026-04-08

## 목차

1. [RTX 3090×2](#1-rtx-3090×2)
2. [DGX Spark GB10](#2-dgx-spark-gb10)
3. [Ryzen AI MAX 395+](#3-ryzen-ai-max-395)
4. [기존 실험 데이터 검증](#4-기존-실험-데이터-검증)
5. [HuggingFace 모델 가용성](#5-huggingface-모델-가용성)
6. [SGLang 추가 가능성](#6-sglang-추가-가능성)
7. [액션 플랜](#7-액션-플랜)

---

## 1. RTX 3090×2

**호스트**: `baeumai@192.168.50.248` (linux-5950x-3090x2)

### 현재 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| vLLM | 0.19.0 설치됨 | **부분 동작** |
| torch | 2.10.0+cu128 | OK |
| CUDA Driver | 580.126.16 / CUDA 13.0 | OK |
| GPU | RTX 3090 ×2, sm_86 (Ampere) | 각 24GB |
| SGLang | 0.5.9 설치됨 | **완전 고장** (sm_100 빌드, sm_86 장비) |
| GPU P2P | **없음** | NVLink 없이 PCIe 경유 → TP 성능 저하 |
| FlashInfer | 0.6.6 | JIT 크래시 (ninja 미발견) |
| Triton | 3.6.0 | SparseMatrix import 경고 |

### 발견된 문제 (4건)

#### 문제 1: FlashInfer JIT 크래시 (Critical)

vLLM 0.19.0의 multiproc executor가 worker subprocess를 spawn할 때, `.venv/bin/ninja`가 worker PATH에 전파되지 않음.
FlashInfer가 prefill 커널을 JIT 컴파일 시도 → `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` → **서버 크래시**.

**영향**: prefill 트랙 전부 실패. gen 트랙은 warmup에서 커널이 컴파일되므로 일부 동작.

**해결**:
```bash
sudo apt install ninja-build   # 시스템 PATH에 ninja 설치
```

#### 문제 2: config.linux.yaml vllm 바이너리 경로

```yaml
# 현재 (잘못됨)
binary: "vllm"

# 수정
binary: "/home/baeumai/llm-bench/.venv/bin/vllm"
```

`vllm`이 시스템 PATH에 없으므로 실행 실패. `config.linux.run.yaml`에는 전체 경로가 있지만, 기본 `config.linux.yaml`은 수정 필요.

#### 문제 3: Triton 3.6.0 비호환

```
ERROR: Failed to import Triton kernels. cannot import name 'SparseMatrix' from 'triton_kernels.tensor'
```

비치명적 경고이나 MoE 모델(35B-A3B, 122B-A10B)의 Triton 커널 사용 불가 → fallback 경로. CSV `backend_version` 필드 오염.

#### 문제 4: 9B Q4/Q8 라벨 중복

vLLM은 GGUF를 지원하지 않으므로, 9B Q4_K_M과 Q8_0 모두 동일한 `Qwen/Qwen3.5-9B` (BF16, ~18GB) 모델을 사용.
CSV에 별도 행으로 기록되지만 **실질적으로 동일한 실험의 중복**.

### 모델 다운로드 상태

| 모델 | 크기 | 상태 | VRAM fit |
|------|------|------|----------|
| Qwen3.5-9B (BF16) | 19GB | 완료 | 단일 GPU OK |
| Qwen3.5-27B-GPTQ-Int4 | 29GB | 완료 | TP=2 필요 |
| Qwen3.5-35B-A3B-GPTQ-Int4 | 23GB | 완료 | 단일 GPU OK |
| Qwen3.5-122B-A10B-GPTQ-Int4 | 74GB | 완료 | 48GB 초과 → cpu_offload 필요 |
| Gemma 4 모델 | - | **미다운로드** | - |

### 수정 절차

```bash
ssh baeumai@192.168.50.248

# 1. ninja 시스템 설치
sudo apt install ninja-build

# 2. config 바이너리 경로 수정
cd ~/llm-bench
sed -i 's|binary: "vllm"|binary: "/home/baeumai/llm-bench/.venv/bin/vllm"|' config.linux.yaml

# 3. (선택) SGLang 별도 venv 설치
uv venv .sglang --python 3.12
source .sglang/bin/activate
uv pip install sglang sgl-kernel \
  --extra-index-url https://sgl-project.github.io/whl/cu129/ \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

---

## 2. DGX Spark GB10

**호스트**: `baeumai@192.168.50.251` (dgx-spark)

### 현재 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| vLLM (native) | 0.19.0 설치됨 | **완전 고장** (torch CPU-only) |
| vLLM (Docker) | **이미지 삭제됨** | `vllm-node` 없음 |
| torch | 2.10.0+**cpu** | CUDA 미지원 — 근본 원인 |
| 아키텍처 | aarch64 (ARM64) | PyPI wheel = x86_64 only |
| CUDA | 13.0 / Driver 580.82.09 | sm_121 (Blackwell) |
| 빌드 시스템 | `~/spark-vllm-docker/` | pre-built wheel 존재 |
| SGLang | 미설치 | NGC 컨테이너로 가능 |

### 근본 원인 분석

1. **torch CPU-only**: `uv`가 PyPI에서 aarch64 기본 wheel → `torch==2.10.0+cpu` (USE_CUDA=0)
2. **PyPI vLLM에 sm_121 미포함**: 표준 wheel은 sm_80/86/89/90만 컴파일
3. **CUDA 13 vs 12**: `_C.abi3.so`가 `libcudart.so.12` 요구, 시스템은 CUDA 13

**결과**: `from vllm import LLM` → `ImportError: libtorch_cuda.so: cannot open shared object file`

### 기존 Docker 벤치마크 데이터

이전에 `vllm-node` Docker 이미지로 측정한 163개 OK rows는 유효:

| 모델 | Gen TPS | Prefill TPS (16k) |
|------|--------:|------------------:|
| 9B BF16 | ~12.9 | ~6,774 |
| 27B GPTQ-Int4 | ~8.5 | ~1,601 |
| 35B-A3B GPTQ-Int4 | ~34.8 | ~4,328 |

커뮤니티 보고치와 비교:
- 27B Dense ~4 tok/s (메모리 대역폭 병목) — 우리 GPTQ 8.5 tok/s는 합리적
- 35B-A3B FP8 ~50 tok/s — 우리 GPTQ 34.8 tok/s도 합리적 (GPTQ < FP8)

### 수정 방안

#### 방법 A: `eugr/spark-vllm-docker` 재빌드 (추천)

```bash
ssh baeumai@192.168.50.251
cd ~/spark-vllm-docker
git pull
./build-and-copy.sh    # 2-3분, vllm-node 이미지 재생성

# Qwen3.5 전용 레시피 사용:
./run-recipe.sh qwen3.5-35b-a3b-fp8 --solo

# 수동 실행:
docker run -d --rm --gpus all --name vllm-bench \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm-node \
  vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --max-model-len 65536
```

- 901 stars, 활발히 유지보수 (2026-04-07 업데이트)
- Qwen3.5 레시피 포함: `qwen3.5-35b-a3b-fp8`, `qwen3.5-122b-fp8`
- pre-built wheels: vLLM 0.19.1rc1 + FlashInfer 0.6.7

#### 방법 B: NGC 컨테이너

```bash
docker pull nvcr.io/nvidia/vllm:26.03-py3
```

주의: Qwen3.5 transformers 버전 호환 문제 가능. eugr 이미지가 더 안정적.

#### 방법 C: 네이티브 원커맨드 설치 (Docker 없이)

```bash
curl -fsSL https://raw.githubusercontent.com/eelbaz/dgx-spark-vllm-setup/main/install.sh | bash
```

20-30분 소요. PyTorch 2.9.0+cu130, Triton 3.5.0, vLLM 0.11.1rc3+ 설치.

#### 방법 D: PyPI 패키지 (Docker 없이)

```bash
pip install dgx-spark-vllm
sudo dgx-spark-vllm-install
```

NGC 25.09 컨테이너 기반 추출. Qwen3.5 호환 미확인.

#### SGLang 추가

```bash
# NGC 컨테이너 (추천)
docker pull nvcr.io/nvidia/sglang:26.03-py3

# 주의사항:
# --attention-backend triton 필수 (sm_121a flashinfer 버그)
# --moe-runner-backend flashinfer_cutlass (MoE 모델)
# 동시성 16 이하 권장 (Triton JIT race condition)
```

커뮤니티 이미지: `scitrera/dgx-spark-sglang:0.5.10rc0`

---

## 3. Ryzen AI MAX 395+

**호스트**: `baeumai@192.168.50.245` (ryzen-ai-max-395, HP Z2 Mini G1a)

### 현재 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| vLLM | 0.18.0 **CUDA 빌드** | ROCm 장비에서 완전 무용 |
| ROCm | 6.4.1 | 정상 동작 |
| PyTorch | 2.5.1+rocm6.2 | GPU 인식 OK (103GB) |
| GPU | gfx1151 (RDNA 3.5), 40 CU | 96GB VRAM (unified) |
| Docker/Podman | **미설치** | 컨테이너 사용 불가 |
| SGLang | 미설치 | gfx1151 **미지원** |
| Kernel | 6.17.0-20-generic | OK (6.16.9+ 필요) |

### vLLM ROCm 가능성 — 공식 지원됨 (실험적)

vLLM PR [#25908](https://github.com/vllm-project/vllm/pull/25908) (2025-10 merge)로 gfx1151 공식 지원 아키텍처에 추가.

#### 알려진 제약사항

| 제약 | 내용 | 대응 |
|------|------|------|
| HIP Graph 크래시 | V1 엔진 HIP Graph capture 시 driver timeout | `--enforce-eager` 필수 |
| AITER 미지원 | gfx1151 KeyError | Triton/ROCm attention만 사용 |
| hipMemcpy 병목 | decode 시간의 82-95%가 메모리 복사 | 근본 해결 없음 (pytorch#171687) |
| 단일 요청 성능 | llama.cpp 대비 ~6.7x 느림 (120B 기준) | 동시 요청에서만 이점 |
| AWQ 대형 모델 segfault | libhsa-runtime64 segfault (vllm#37151) | GPTQ 사용 권장 |
| Qwen3.5 block_size | 하드코딩된 whitelist 거부 | kyuz0 패치 적용 필요 |

#### 커뮤니티 벤치마크 (kyuz0, gfx1151, Fedora 43)

| 모델 | 양자화 | Backend | tok/s |
|------|--------|---------|------:|
| Llama-3.1-8B | BF16 | ROCm | 434 |
| Gemma-3-12B | BF16 | ROCm | 268 |
| Qwen3-14B-AWQ | AWQ | ROCm | 180 |
| Qwen3-Coder-30B-A3B | GPTQ-4bit | Triton | 214 |
| GPT-OSS-120B | BF16 | ROCm | 75 |

### 수정 방안

#### 방법 A: AMD 공식 prebuilt wheel (추천, Docker 불필요)

```bash
ssh baeumai@192.168.50.245
cd ~/llm-bench
source .venv/bin/activate

# 1. 현재 CUDA vLLM 제거
pip uninstall vllm -y

# 2. PyTorch ROCm nightly (gfx1151 전용)
pip install --pre torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# 3. vLLM ROCm (gfx1151 전용 wheel)
pip install vllm --extra-index-url https://rocm.frameworks.amd.com/whl/gfx1151/

# 4. 환경변수 설정
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export HIP_FORCE_DEV_KERNARG=1
```

#### 방법 B: kyuz0 Docker (모든 패치 포함)

```bash
# 1. Podman 설치
sudo apt install podman

# 2. 이미지 pull
podman pull docker.io/kyuz0/vllm-therock-gfx1151:latest

# 3. 실행
podman run --ipc=host --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  kyuz0/vllm-therock-gfx1151:latest bash -lc \
  'source /torch-therock/.venv/bin/activate && \
   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
   vllm serve Qwen/Qwen3.5-9B \
   --enforce-eager \
   --host 0.0.0.0 --port 8000'
```

#### 벤치마크 시 필수 플래그

```bash
vllm serve <model> \
  --enforce-eager \              # HIP Graph 크래시 방지
  --no-enable-prefix-caching \   # 공정 비교
  --attention-backend triton \   # AITER 미지원이므로
  --host 0.0.0.0 --port 8000
```

#### 참고 자료

- [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) — Docker + 벤치마크
- [lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing) — 소스 빌드 스크립트
- [blog.epheo.eu/notes/strix-halo](https://blog.epheo.eu/notes/strix-halo/index.html) — 수동 빌드 가이드
- [AMD ROCm Strix Halo 최적화](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [Framework Community: vLLM 소스 빌드](https://community.frame.work/t/how-to-compiling-vllm-from-source-on-strix-halo/77241)

---

## 4. 기존 실험 데이터 검증

| 디바이스 | vLLM OK Rows | 문제점 | 데이터 유효성 |
|---------|:-----------:|--------|:----------:|
| 3090×2 | 140 | prefill 다수 실패 (FlashInfer), 9B Q4/Q8 중복 | gen: 유효, prefill: 부분 누락 |
| DGX Spark | 200 (Docker) | native 108 rows 전부 실패 | Docker 데이터 유효 |
| Ryzen AI | 0 | CUDA 빌드라 실행 불가 | 전면 재실험 필요 |

### 3090 데이터 상세

- **gen 트랙**: 9B(83 tok/s), 27B GPTQ(19 tok/s), 35B-A3B(156 tok/s) — 유효
- **prefill 트랙**: FlashInfer JIT 크래시로 대부분 실패. 일부 `--enforce-eager`로 성공한 데이터 존재
- **9B Q4/Q8 구분**: 의미 없음 (동일 BF16 모델). 분석 시 하나만 사용할 것
- **27B 19 tok/s**: GPTQ-Int4 + TP=2(PCIe only) 감안하면 합리적이나, BF16 대비 느림

### DGX Spark 데이터 상세

- **Docker 기반 163 rows**: 유효
- **native vLLM 시도**: 전부 `ImportError: libcudart.so.12` → 0 valid rows
- **9B 12.9 tok/s**: BF16 dense on GB10, 커뮤니티와 일치
- **27B 8.5 tok/s**: GPTQ-Int4, 커뮤니티 보고(~4 tok/s BF16)보다 높음 (GPTQ가 메모리 절약 → 더 빠를 수 있음)

---

## 5. HuggingFace 모델 가용성

### Qwen3.5 — config 정확함

| 모델 | vllm_model | 상태 | DL 수 |
|------|-----------|------|------:|
| 9B | `Qwen/Qwen3.5-9B` (BF16) | OK | - |
| 27B | `Qwen/Qwen3.5-27B-GPTQ-Int4` | OK (공식) | 315K |
| 35B-A3B | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | OK (공식) | 657K |
| 122B-A10B | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | OK (공식) | 277K |

### Gemma 4 — config 빈칸 발견, 모델 존재 확인

| 모델 | 현재 vllm_model | 추천 모델 | DL 수 |
|------|---------------|----------|------:|
| E2B | `google/gemma-4-E2B-it` (BF16, ~10GB) | OK | - |
| E4B | `google/gemma-4-E4B-it` (BF16, ~16GB) | OK | - |
| **26B-A4B** | **""** (빈칸) | `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | 97K |
| **31B** | **""** (빈칸) | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | 67K |

→ Gemma 4 26B-A4B, 31B의 vllm_model 필드를 채울 수 있음.

---

## 6. SGLang 추가 가능성

### 디바이스별 가능 여부

| 디바이스 | 가능 | 방법 | 비고 |
|---------|:----:|------|------|
| DGX Spark | O | NGC `nvcr.io/nvidia/sglang:26.03-py3` | `--attention-backend triton` 필수 |
| 3090×2 | O | pip (별도 venv) | sm_86 완전 지원, TP=2 가능 |
| Ryzen AI | X | - | CDNA만 지원 (MI300X/MI325X/MI355X) |
| Mac | X | - | CUDA/ROCm 전용 |

### SGLang vs vLLM 비교

| 항목 | SGLang | vLLM |
|------|--------|------|
| RadixAttention | O (핵심 기능) | O (APC) |
| 처리량 (H100 기준) | ~1.3x vLLM | baseline |
| GPTQ/AWQ | O (GPTQModel 경유) | O (네이티브) |
| ROCm | MI300X만 | MI300X + gfx1151 (실험적) |
| DGX Spark | NGC 컨테이너 | NGC/Docker |
| 모델 가중치 | vLLM과 동일 HF 포맷 | - |

### 기존 코드

`src/backends/sglang_backend.py` 이미 구현 완료:
- subprocess 서버 관리
- OpenAI `/v1/completions` 스트리밍
- `--disable-radix-cache` 지원
- `vllm_model` fallback 지원

config에서 `enabled: false` → `true`로 변경 + 서버 설치만 하면 바로 사용 가능.

### DGX Spark SGLang 벤치마크 시 주의

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --host 0.0.0.0 --port 30000 \
  --tp 1 \
  --attention-backend triton \       # sm_121a flashinfer 버그 회피
  --mem-fraction-static 0.75 \       # unified memory → 보수적
  --disable-radix-cache              # 벤치마크 공정성
```

---

## 7. 액션 플랜

### 즉시 실행 (환경 수정)

| # | 디바이스 | 작업 | 예상 시간 |
|---|---------|------|----------|
| 1 | 3090 | `sudo apt install ninja-build` | 1분 |
| 2 | 3090 | config.linux.yaml vllm binary 경로 수정 | 1분 |
| 3 | DGX | `cd ~/spark-vllm-docker && git pull && ./build-and-copy.sh` | 10분 |
| 4 | Ryzen | CUDA vllm 제거 → AMD ROCm wheel 설치 | 20분 |

### 추가 실험 (선택)

| # | 디바이스 | 작업 | 예상 시간 |
|---|---------|------|----------|
| 5 | DGX | SGLang NGC 컨테이너 pull + 벤치마크 | 2-3시간 |
| 6 | 3090 | SGLang 별도 venv 설치 + 벤치마크 | 2-3시간 |
| 7 | Ryzen | vLLM ROCm 벤치마크 (--enforce-eager) | 3-4시간 |
| 8 | 전체 | Gemma 4 26B-A4B/31B AWQ 모델 추가 | 1시간 (다운로드) |
| 9 | DGX/3090 | 122B-A10B GPTQ vLLM 실험 | 2시간 |

### 데이터 보정

- 3090 9B: Q4_K_M과 Q8_0 중 하나만 사용 (동일 BF16 모델)
- 3090 prefill: ninja 수정 후 재실험 필요
- Ryzen vLLM: 전면 신규 실험
- DGX 기존 Docker 데이터: 유효, 보존
