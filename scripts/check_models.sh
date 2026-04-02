#!/bin/bash
# 모델 다운로드 완료 여부 확인

GGUF_DIR="$HOME/.lmstudio/models"
MLX_DIR="$HOME/.cache/mlx"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
prog() { echo -e "  ${YELLOW}…${NC} $*"; }

check_file() {
  local path="$1" label="$2"
  if [ -f "$path" ]; then
    size=$(du -sh "$path" 2>/dev/null | cut -f1)
    ok "$label ($size)"
  else
    # incomplete 확인
    dir=$(dirname "$path")
    if ls "$dir"/.cache/huggingface/download/*.incomplete 2>/dev/null | head -1 | grep -q .; then
      prog "$label (다운로드 중...)"
    else
      fail "$label"
    fi
  fi
}

check_dir() {
  local path="$1" label="$2"
  if [ -d "$path" ] && [ "$(ls "$path"/*.safetensors "$path"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    size=$(du -sh "$path" 2>/dev/null | cut -f1)
    ok "$label ($size)"
  elif [ -d "$path" ] && ls "$path"/.cache/huggingface/download/*.incomplete 2>/dev/null | head -1 | grep -q .; then
    prog "$label (다운로드 중...)"
  else
    fail "$label"
  fi
}

echo "=========================================="
echo " Model Download Status"
echo "=========================================="

echo ""
echo "── GGUF ──"
check_file "$GGUF_DIR/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"                "9B Q4_K_M"
check_file "$GGUF_DIR/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"                  "9B Q8_0"
check_file "$GGUF_DIR/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"              "27B Q4_K_M"
check_file "$GGUF_DIR/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf"                "27B Q8_0"
check_file "$GGUF_DIR/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"     "35B-A3B Q4_K_M"
check_file "$GGUF_DIR/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf" "35B-A3B Q8_0"
check_file "$GGUF_DIR/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_M-00003-of-00003.gguf" "122B-A10B Q4_K_M"

echo ""
echo "── MLX ──"
check_dir "$MLX_DIR/Qwen3.5-9B-4bit"        "9B 4bit"
check_dir "$MLX_DIR/Qwen3.5-9B-8bit"        "9B 8bit"
check_dir "$MLX_DIR/Qwen3.5-27B-4bit"       "27B 4bit"
check_dir "$MLX_DIR/Qwen3.5-27B-8bit"       "27B 8bit"
check_dir "$MLX_DIR/Qwen3.5-35B-A3B-4bit"   "35B-A3B 4bit"
check_dir "$MLX_DIR/Qwen3.5-35B-A3B-8bit"   "35B-A3B 8bit"
check_dir "$MLX_DIR/Qwen3.5-122B-A10B-4bit" "122B-A10B 4bit"

echo ""
echo "── Ollama ──"
if command -v ollama &>/dev/null; then
  for tag in "9b-q4_K_M" "9b-q8_0" "27b-q4_K_M" "27b-q8_0" "35b-a3b-q4_K_M" "35b-a3b-q8_0" "122b-a10b-q4_K_M"; do
    if ollama list 2>/dev/null | grep -q "qwen3.5:$tag"; then
      ok "qwen3.5:$tag"
    else
      fail "qwen3.5:$tag"
    fi
  done
else
  echo "  ollama not found"
fi

echo ""
echo "=========================================="
