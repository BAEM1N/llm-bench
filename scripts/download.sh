#!/bin/bash
# ============================================================
# LLM Bench — 모델 다운로드 스크립트
# 디바이스별로 필요한 백엔드/모델만 선택적으로 다운로드
#
# 사용법:
#   ./scripts/download.sh [device] [--backends b1,b2] [--models m1,m2]
#
# device:
#   mac          Apple Silicon MacBook (기본값)
#   dgx-spark    NVIDIA DGX Spark (GB10 Grace Blackwell)
#   ryzen-ai     AMD Ryzen AI 395 (XDNA 3 NPU)
#
# 예시:
#   ./scripts/download.sh mac
#   ./scripts/download.sh mac --backends gguf,mlx
#   ./scripts/download.sh dgx-spark --backends gguf
#   ./scripts/download.sh mac --models 9b,27b
# ============================================================

set -euo pipefail

DEVICE="${1:-mac}"
BACKENDS="all"
MODELS="all"

# 인자 파싱
shift || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --backends) BACKENDS="$2"; shift 2 ;;
    --models)   MODELS="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── 색상 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── HuggingFace CLI ──
HF=""
for candidate in \
  "$(which hf 2>/dev/null)" \
  "$(which huggingface-cli 2>/dev/null)" \
  "$HOME/Workspace/llm-bench/.venv/bin/hf" \
  "$HOME/Workspace/llm-bench/.venv/bin/huggingface-cli" \
  "$HOME/Library/Python/3.9/bin/huggingface-cli" \
  "$HOME/.local/bin/huggingface-cli"; do
  [[ -x "$candidate" ]] && HF="$candidate" && break
done
[[ -z "$HF" ]] && error "huggingface-cli not found. Run: pip install huggingface_hub  또는  uv add huggingface_hub"

# HF CLI로 다운로드, stall 감지 시 curl fallback
hf_download() {
  local repo="$1" file="$2" dest="$3"
  mkdir -p "$dest"
  info "HF download: $repo / ${file:-<all>} → $dest"

  # hf CLI 시도 (신버전 우선)
  if "$HF" download "$repo" ${file:+--include "$file"} --local-dir "$dest"; then
    return 0
  fi

  # CLI 실패 시 단일 파일은 curl -L -C - 로 fallback
  if [[ -n "$file" ]] && [[ "$file" != *"*"* ]]; then
    warn "HF CLI 실패 → curl fallback: $file"
    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    local out="${dest}/${file##*/}"
    curl -L -C - --retry 5 --retry-delay 10 -o "$out" "$url"
  else
    error "HF download 실패: $repo / $file"
  fi
}

# ── 모델 필터 ──
want_model() {
  [[ "$MODELS" == "all" ]] && return 0
  [[ ",$MODELS," == *",$1,"* ]] && return 0
  return 1
}
want_backend() {
  [[ "$BACKENDS" == "all" ]] && return 0
  [[ ",$BACKENDS," == *",$1,"* ]] && return 0
  return 1
}

echo ""
echo "============================================"
echo " LLM Bench Download Script"
echo " Device  : $DEVICE"
echo " Backends: $BACKENDS"
echo " Models  : $MODELS"
echo "============================================"
echo ""

# ============================================================
# 디바이스별 분기
# ============================================================

case "$DEVICE" in

# ──────────────────────────────────────────
# mac — Apple Silicon MacBook
# 백엔드: GGUF (llama.cpp/LMStudio/Ollama), MLX
# ──────────────────────────────────────────
mac)
  GGUF_DIR="$HOME/.lmstudio/models"
  MLX_DIR="$HOME/.cache/mlx"

  # ── GGUF ──
  if want_backend "gguf" || want_backend "all"; then
    info "=== GGUF 다운로드 (mac) ==="

    if want_model "9b"; then
      hf_download "unsloth/Qwen3.5-9B-GGUF" "Qwen3.5-9B-Q4_K_M.gguf" "$GGUF_DIR/unsloth/Qwen3.5-9B-GGUF"
      hf_download "unsloth/Qwen3.5-9B-GGUF" "Qwen3.5-9B-Q8_0.gguf"   "$GGUF_DIR/unsloth/Qwen3.5-9B-GGUF"
    fi

    if want_model "27b"; then
      hf_download "unsloth/Qwen3.5-27B-GGUF" "Qwen3.5-27B-Q4_K_M.gguf" "$GGUF_DIR/unsloth/Qwen3.5-27B-GGUF"
      hf_download "unsloth/Qwen3.5-27B-GGUF" "Qwen3.5-27B-Q8_0.gguf"   "$GGUF_DIR/unsloth/Qwen3.5-27B-GGUF"
    fi

    if want_model "35b"; then
      hf_download "unsloth/Qwen3.5-35B-A3B-GGUF" "Qwen3.5-35B-A3B-Q4_K_M.gguf" "$GGUF_DIR/unsloth/Qwen3.5-35B-A3B-GGUF"
      hf_download "unsloth/Qwen3.5-35B-A3B-GGUF" "Qwen3.5-35B-A3B-Q8_0.gguf"   "$GGUF_DIR/unsloth/Qwen3.5-35B-A3B-GGUF"
    fi

    if want_model "122b"; then
      hf_download "unsloth/Qwen3.5-122B-A10B-GGUF" "Qwen3.5-122B-A10B-Q4_K_M-*.gguf" "$GGUF_DIR/unsloth/Qwen3.5-122B-A10B-GGUF"
    fi

    # ── Gemma 4 GGUF ──
    info "--- Gemma 4 GGUF ---"
    if want_model "gemma4-e2b"; then
      hf_download "unsloth/gemma-4-E2B-it-GGUF" "gemma-4-E2B-it-Q4_K_M.gguf" "$GGUF_DIR/unsloth/gemma-4-E2B-it-GGUF"
      hf_download "unsloth/gemma-4-E2B-it-GGUF" "gemma-4-E2B-it-Q8_0.gguf"   "$GGUF_DIR/unsloth/gemma-4-E2B-it-GGUF"
    fi
    if want_model "gemma4-e4b"; then
      hf_download "unsloth/gemma-4-E4B-it-GGUF" "gemma-4-E4B-it-Q4_K_M.gguf" "$GGUF_DIR/unsloth/gemma-4-E4B-it-GGUF"
      hf_download "unsloth/gemma-4-E4B-it-GGUF" "gemma-4-E4B-it-Q8_0.gguf"   "$GGUF_DIR/unsloth/gemma-4-E4B-it-GGUF"
    fi
    if want_model "gemma4-26b"; then
      hf_download "unsloth/gemma-4-26B-A4B-it-GGUF" "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" "$GGUF_DIR/unsloth/gemma-4-26B-A4B-it-GGUF"
      hf_download "unsloth/gemma-4-26B-A4B-it-GGUF" "gemma-4-26B-A4B-it-Q8_0.gguf"      "$GGUF_DIR/unsloth/gemma-4-26B-A4B-it-GGUF"
    fi
    if want_model "gemma4-31b"; then
      hf_download "unsloth/gemma-4-31B-it-GGUF" "gemma-4-31B-it-Q4_K_M.gguf" "$GGUF_DIR/unsloth/gemma-4-31B-it-GGUF"
      hf_download "unsloth/gemma-4-31B-it-GGUF" "gemma-4-31B-it-Q8_0.gguf"   "$GGUF_DIR/unsloth/gemma-4-31B-it-GGUF"
    fi
  fi

  # ── MLX ──
  if want_backend "mlx" || want_backend "all"; then
    info "=== MLX 다운로드 (mac) ==="

    if want_model "9b"; then
      hf_download "mlx-community/Qwen3.5-9B-4bit" "" "$MLX_DIR/Qwen3.5-9B-4bit"
      hf_download "mlx-community/Qwen3.5-9B-8bit" "" "$MLX_DIR/Qwen3.5-9B-8bit"
    fi

    if want_model "27b"; then
      hf_download "mlx-community/Qwen3.5-27B-4bit" "" "$MLX_DIR/Qwen3.5-27B-4bit"
      hf_download "mlx-community/Qwen3.5-27B-8bit" "" "$MLX_DIR/Qwen3.5-27B-8bit"
    fi

    if want_model "35b"; then
      hf_download "mlx-community/Qwen3.5-35B-A3B-4bit" "" "$MLX_DIR/Qwen3.5-35B-A3B-4bit"
    fi

    if want_model "122b"; then
      hf_download "mlx-community/Qwen3.5-122B-A10B-4bit" "" "$MLX_DIR/Qwen3.5-122B-A10B-4bit"
    fi

    # ── Gemma 4 MLX (unsloth 기반) ──
    info "--- Gemma 4 MLX ---"
    if want_model "gemma4-e2b"; then
      hf_download "unsloth/gemma-4-E2B-it-UD-MLX-4bit" "" "$MLX_DIR/gemma-4-E2B-it-UD-MLX-4bit"
      hf_download "unsloth/gemma-4-E2B-it-MLX-8bit"    "" "$MLX_DIR/gemma-4-E2B-it-MLX-8bit"
    fi
    if want_model "gemma4-e4b"; then
      hf_download "unsloth/gemma-4-E4B-it-UD-MLX-4bit" "" "$MLX_DIR/gemma-4-E4B-it-UD-MLX-4bit"
      hf_download "unsloth/gemma-4-E4B-it-MLX-8bit"    "" "$MLX_DIR/gemma-4-E4B-it-MLX-8bit"
    fi
    if want_model "gemma4-26b"; then
      # 26B-A4B: unsloth MLX 미제공 → mlx-community 사용
      hf_download "mlx-community/gemma-4-26b-a4b-it-4bit" "" "$MLX_DIR/gemma-4-26b-a4b-it-4bit"
      hf_download "mlx-community/gemma-4-26b-a4b-it-8bit" "" "$MLX_DIR/gemma-4-26b-a4b-it-8bit"
    fi
    # gemma4-31b MLX: unsloth/mlx-community 모두 미제공 → 추후 추가
  fi

  # ── Ollama ──
  if want_backend "ollama" || want_backend "all"; then
    command -v ollama &>/dev/null || error "ollama not found"
    info "=== Ollama pull (mac) ==="

    want_model "9b"   && ollama pull qwen3.5:9b-q4_K_M  && ollama pull qwen3.5:9b-q8_0
    want_model "27b"  && ollama pull qwen3.5:27b-q4_K_M && ollama pull qwen3.5:27b-q8_0
    want_model "35b"  && ollama pull qwen3.5:35b-a3b-q4_K_M && ollama pull qwen3.5:35b-a3b-q8_0
    want_model "122b" && ollama pull qwen3.5:122b-a10b-q4_K_M
    # Gemma 4 Ollama: 아직 공식 태그 미확인. 추가되면 여기에 pull 명령 추가.
  fi

  success "mac 다운로드 완료"
  ;;

# ──────────────────────────────────────────
# dgx-spark — NVIDIA DGX Spark (GB10, 128GB)
# 백엔드: GGUF (llama.cpp), vLLM (AWQ/GPTQ)
# ──────────────────────────────────────────
dgx-spark)
  GGUF_DIR="${GGUF_DIR:-/models/gguf}"
  VLLM_DIR="${VLLM_DIR:-/models/vllm}"

  if want_backend "gguf" || want_backend "all"; then
    info "=== GGUF 다운로드 (dgx-spark) ==="

    want_model "9b"   && hf_download "unsloth/Qwen3.5-9B-GGUF"        "Qwen3.5-9B-Q4_K_M.gguf"             "$GGUF_DIR/Qwen3.5-9B-GGUF"
    want_model "9b"   && hf_download "unsloth/Qwen3.5-9B-GGUF"        "Qwen3.5-9B-Q8_0.gguf"               "$GGUF_DIR/Qwen3.5-9B-GGUF"
    want_model "27b"  && hf_download "unsloth/Qwen3.5-27B-GGUF"       "Qwen3.5-27B-Q4_K_M.gguf"            "$GGUF_DIR/Qwen3.5-27B-GGUF"
    want_model "27b"  && hf_download "unsloth/Qwen3.5-27B-GGUF"       "Qwen3.5-27B-Q8_0.gguf"              "$GGUF_DIR/Qwen3.5-27B-GGUF"
    want_model "35b"  && hf_download "unsloth/Qwen3.5-35B-A3B-GGUF"   "Qwen3.5-35B-A3B-Q4_K_M.gguf"       "$GGUF_DIR/Qwen3.5-35B-A3B-GGUF"
    want_model "35b"  && hf_download "unsloth/Qwen3.5-35B-A3B-GGUF"   "Qwen3.5-35B-A3B-Q8_0.gguf"         "$GGUF_DIR/Qwen3.5-35B-A3B-GGUF"
    want_model "122b" && hf_download "unsloth/Qwen3.5-122B-A10B-GGUF" "Qwen3.5-122B-A10B-Q4_K_M-*.gguf"   "$GGUF_DIR/Qwen3.5-122B-A10B-GGUF"
  fi

  if want_backend "vllm" || want_backend "all"; then
    info "=== vLLM (AWQ) 다운로드 (dgx-spark) ==="
    # DGX Spark는 CUDA → AWQ FP8 권장
    want_model "9b"   && hf_download "Qwen/Qwen3.5-9B-Instruct-AWQ"        "" "$VLLM_DIR/Qwen3.5-9B-AWQ"
    want_model "27b"  && hf_download "Qwen/Qwen3.5-27B-Instruct-AWQ"       "" "$VLLM_DIR/Qwen3.5-27B-AWQ"
    want_model "35b"  && hf_download "Qwen/Qwen3.5-35B-A3B-Instruct-AWQ"   "" "$VLLM_DIR/Qwen3.5-35B-A3B-AWQ"
    want_model "122b" && hf_download "Qwen/Qwen3.5-122B-A10B-Instruct-AWQ" "" "$VLLM_DIR/Qwen3.5-122B-A10B-AWQ"
  fi

  success "dgx-spark 다운로드 완료"
  warn "config.yaml의 hardware.id를 'dgx-spark'으로 변경하세요"
  ;;

# ──────────────────────────────────────────
# ryzen-ai — AMD Ryzen AI 395 (XDNA 3 NPU + Radeon 890M)
# 백엔드: GGUF (llama.cpp ROCm), Ollama (ROCm)
# ──────────────────────────────────────────
ryzen-ai)
  GGUF_DIR="${GGUF_DIR:-$HOME/models/gguf}"

  if want_backend "gguf" || want_backend "all"; then
    info "=== GGUF 다운로드 (ryzen-ai) ==="
    # Ryzen AI 395 iGPU 16GB VRAM — 9B Q4/Q8, 27B Q4까지 GPU 추론 가능
    # 27B Q8, 35B, 122B는 CPU fallback 또는 부분 offload

    want_model "9b"  && hf_download "unsloth/Qwen3.5-9B-GGUF"      "Qwen3.5-9B-Q4_K_M.gguf"  "$GGUF_DIR/Qwen3.5-9B-GGUF"
    want_model "9b"  && hf_download "unsloth/Qwen3.5-9B-GGUF"      "Qwen3.5-9B-Q8_0.gguf"    "$GGUF_DIR/Qwen3.5-9B-GGUF"
    want_model "27b" && hf_download "unsloth/Qwen3.5-27B-GGUF"     "Qwen3.5-27B-Q4_K_M.gguf" "$GGUF_DIR/Qwen3.5-27B-GGUF"
    want_model "27b" && hf_download "unsloth/Qwen3.5-27B-GGUF"     "Qwen3.5-27B-Q8_0.gguf"   "$GGUF_DIR/Qwen3.5-27B-GGUF"
    want_model "35b" && hf_download "unsloth/Qwen3.5-35B-A3B-GGUF" "Qwen3.5-35B-A3B-Q4_K_M.gguf" "$GGUF_DIR/Qwen3.5-35B-A3B-GGUF"
    # 122B는 Ryzen AI 395 메모리(최대 128GB 시스템) 확인 후 선택
    want_model "122b" && warn "122B: Ryzen AI 395 시스템 메모리(≥128GB) 필요. 계속하려면 수동 실행하세요."
  fi

  if want_backend "ollama" || want_backend "all"; then
    command -v ollama &>/dev/null || error "ollama not found (ROCm 빌드 필요: https://ollama.com/download/linux)"
    info "=== Ollama pull (ryzen-ai) ==="

    want_model "9b"  && ollama pull qwen3.5:9b-q4_K_M  && ollama pull qwen3.5:9b-q8_0
    want_model "27b" && ollama pull qwen3.5:27b-q4_K_M && ollama pull qwen3.5:27b-q8_0
    want_model "35b" && ollama pull qwen3.5:35b-a3b-q4_K_M
  fi

  success "ryzen-ai 다운로드 완료"
  warn "config.yaml의 hardware.id를 'ryzen-ai-395'으로 변경하세요"
  warn "llama.cpp는 ROCm 빌드 사용: cmake -DGGML_HIPBLAS=ON ..."
  ;;

*)
  error "Unknown device: $DEVICE. 사용 가능: mac | dgx-spark | ryzen-ai"
  ;;
esac

echo ""
info "완료. 다운로드 확인:"
echo "  ./scripts/check_models.sh"
