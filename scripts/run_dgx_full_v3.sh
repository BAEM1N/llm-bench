#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# DGX Spark — Full Benchmark v3 (Track B + Track A, fresh clean run)
# GB10 Grace Blackwell SoC, 128GB unified memory
#
# Track B: llamacpp only (comparison_mode: B) — all 4 Qwen3.5 models
# Track A: llamacpp + ollama + vllm (comparison_mode: A)
#
# Usage:
#   nohup bash scripts/run_dgx_full_v3.sh > /tmp/dgx_bench_v3.log 2>&1 &
#   tail -f /tmp/dgx_bench_v3.log
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/config.dgx-spark.yaml"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/dgx_bench_v3.log"

echo "============================================================"
echo " DGX Spark — Full Benchmark v3 (Track B + Track A)"
echo " Started: $(date)"
echo " Config:  $CONFIG"
echo " Results: $RESULTS_DIR/"
echo "============================================================"

mkdir -p "$RESULTS_DIR"

# ─── Helper functions ─────────────────────────────────────────────────────────

log_section() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo " $(date)"
    echo "============================================================"
}

wait_cooldown() {
    local secs=${1:-60}
    echo "  Cooling down for ${secs}s..."
    sleep "$secs"
}

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 3
    fi
}

check_server() {
    local url=$1
    local timeout=${2:-30}
    local deadline=$((SECONDS + timeout))
    while [ $SECONDS -lt $deadline ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

set_comparison_mode() {
    local mode=$1
    sed -i "s/^comparison_mode:.*/comparison_mode: $mode/" "$CONFIG"
    echo "  comparison_mode set to: $mode"
}

# ─── System info ──────────────────────────────────────────────────────────────

log_section "System Info"
echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "CPU:      $(lscpu | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')"
echo "RAM:      $(free -h | awk '/Mem:/{print $2}' || echo 'unknown')"
echo "NVIDIA:   $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA:     $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Check GGUF models exist
echo ""
echo "Model files:"
for f in \
    ~/models/gguf/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    ~/models/gguf/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf \
    ~/models/gguf/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf \
    ~/models/gguf/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf \
    ~/models/gguf/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf \
    ~/models/gguf/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf \
    ~/models/gguf/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename "$f") ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
    fi
done

# Check llama-server binary
LLAMA_BIN="$HOME/llama.cpp/build-gpu/bin/llama-server"
echo ""
echo "llama-server: $LLAMA_BIN"
if [ -x "$LLAMA_BIN" ]; then
    echo "  OK"
else
    echo "  ERROR: binary not found!"
    exit 1
fi

# Check CUDA symlink
echo "CUDA symlink:"
ls -la /usr/local/cuda/lib64/libcudart.so.12 2>/dev/null || echo "  WARNING: CUDA symlink missing"

# Check ollama
echo ""
echo "Ollama:"
ollama --version 2>/dev/null || echo "  WARNING: ollama not found"

# Check vllm
echo ""
echo "vLLM:"
"$PROJECT_DIR/.venv/bin/vllm" --version 2>/dev/null || echo "  WARNING: vllm not found"


# =============================================================================
# PHASE 1: Track B — llamacpp only (comparison_mode: B)
# All Qwen3.5: 9B, 27B, 35B-A3B, 122B-A10B (all n_gpu_layers=99)
# =============================================================================

log_section "PHASE 1: Track B — llamacpp (normalized engine comparison)"

set_comparison_mode B

TRACKB_OUTPUT="$RESULTS_DIR/trackB_llamacpp_${TIMESTAMP}.csv"
echo "Output: $TRACKB_OUTPUT"

# Kill anything on llamacpp port
kill_port 8080

# Run each model individually for crash isolation
for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- Track B llamacpp: $model ---"
    kill_port 8080
    cd "$PROJECT_DIR"
    $PROJECT_DIR/.venv/bin/python -m src.runner \
        --config "$CONFIG" \
        --backends llamacpp \
        --models "$model" \
        --output "$TRACKB_OUTPUT" \
        2>&1 || echo "  WARNING: Track B llamacpp $model failed"
    kill_port 8080
    wait_cooldown 120
done

echo ""
echo "Track B complete: $TRACKB_OUTPUT"
if [ -f "$TRACKB_OUTPUT" ]; then
    lines=$(wc -l < "$TRACKB_OUTPUT")
    echo "  $lines lines"
fi

wait_cooldown 180  # Long cooldown before Track A


# =============================================================================
# PHASE 2: Track A — llamacpp (all models)
# comparison_mode: A, same llamacpp backend
# =============================================================================

log_section "PHASE 2: Track A — llamacpp (all Qwen3.5 models)"

set_comparison_mode A

TRACKA_OUTPUT="$RESULTS_DIR/trackA_${TIMESTAMP}.csv"
echo "Output: $TRACKA_OUTPUT"

kill_port 8080

for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- Track A llamacpp: $model ---"
    kill_port 8080
    cd "$PROJECT_DIR"
    $PROJECT_DIR/.venv/bin/python -m src.runner \
        --config "$CONFIG" \
        --backends llamacpp \
        --models "$model" \
        --output "$TRACKA_OUTPUT" \
        2>&1 || echo "  WARNING: Track A llamacpp $model failed"
    kill_port 8080
    wait_cooldown 120
done

echo "Track A llamacpp complete"
wait_cooldown 180


# =============================================================================
# PHASE 3: Track A — ollama (all Qwen3.5, skip Gemma4)
# =============================================================================

log_section "PHASE 3: Track A — ollama (all Qwen3.5 models)"

# Ensure ollama is running
if ! systemctl is-active --quiet ollama 2>/dev/null; then
    echo "  Starting ollama service..."
    sudo systemctl start ollama || true
    sleep 5
fi

# Verify ollama responds
if ! check_server "http://localhost:11434/api/version" 30; then
    echo "  ERROR: ollama not responding. Trying manual start..."
    ollama serve &
    sleep 10
fi

echo "  Ollama version: $(curl -sf http://localhost:11434/api/version | python3 -c 'import sys,json; print(json.load(sys.stdin).get("version","?"))' 2>/dev/null || echo 'unknown')"

# Check models are pulled
for model_tag in \
    "qwen3.5:9b-q4_K_M" "qwen3.5:9b-q8_0" \
    "qwen3.5:27b-q4_K_M" "qwen3.5:27b-q8_0" \
    "qwen3.5:35b-a3b-q4_K_M" "qwen3.5:35b-a3b-q8_0" \
    "qwen3.5:122b-a10b-q4_K_M"; do
    echo "  Checking: $model_tag"
    ollama list 2>/dev/null | grep -q "$(echo "$model_tag" | tr ':' ' ' | awk '{print $1}')" || \
        echo "  WARNING: $model_tag may not be pulled. Run: ollama pull $model_tag"
done

for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- Track A ollama: $model ---"
    cd "$PROJECT_DIR"
    $PROJECT_DIR/.venv/bin/python -m src.runner \
        --config "$CONFIG" \
        --backends ollama \
        --models "$model" \
        --output "$TRACKA_OUTPUT" \
        2>&1 || echo "  WARNING: Track A ollama $model failed"
    wait_cooldown 60
done

echo "Track A ollama complete"
wait_cooldown 180


# =============================================================================
# PHASE 4: Track A — vLLM (9B, 27B, 35B only — 122B too large for GPTQ on DGX)
# vLLM 0.19.0, tensor_parallel_size=1
# CUDA 12 symlink at /usr/local/cuda/lib64/libcudart.so.12
# =============================================================================

log_section "PHASE 4: Track A — vLLM (Qwen3.5 9B/27B/35B)"

kill_port 8000

VLLM_BIN="$PROJECT_DIR/.venv/bin/vllm"

if [ -x "$VLLM_BIN" ]; then
    echo "  vLLM binary: $VLLM_BIN"
    echo "  vLLM version: $($VLLM_BIN --version 2>/dev/null || echo 'unknown')"

    # vLLM models: 9B (BF16), 27B (GPTQ-Int4), 35B (GPTQ-Int4)
    # 122B GPTQ-Int4 is ~65GB — may OOM on vLLM KV cache allocation. Skip per user request.
    for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b; do
        echo ""
        echo "  --- Track A vLLM: $model ---"
        kill_port 8000
        sleep 3
        cd "$PROJECT_DIR"
        timeout 7200 "$PROJECT_DIR/.venv/bin/python" -m src.runner \
            --config "$CONFIG" \
            --backends vllm \
            --models "$model" \
            --output "$TRACKA_OUTPUT" \
            2>&1 || echo "  WARNING: Track A vLLM $model failed"
        kill_port 8000
        wait_cooldown 120
    done

    echo "Track A vLLM complete"
else
    echo "  SKIP: vLLM not found at $VLLM_BIN"
fi

kill_port 8000


# =============================================================================
# SUMMARY
# =============================================================================

log_section "Benchmark Complete"
echo "Finished: $(date)"
echo ""
echo "Result files:"
for f in "$TRACKB_OUTPUT" "$TRACKA_OUTPUT"; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f ($lines lines)"
    else
        echo "  $f (NOT CREATED)"
    fi
done

echo ""
echo "Config used: $CONFIG"
echo "Log: $LOG"
echo ""
echo "Notes:"
echo "  - Track B: llamacpp only, comparison_mode=B"
echo "  - Track A: llamacpp + ollama + vllm, comparison_mode=A"
echo "  - context_window: 65536 (prevents KV cache OOM)"
echo "  - vLLM: 9B/27B/35B only (122B skipped — GPTQ-Int4 too large)"
echo "  - All n_gpu_layers=99 (128GB unified fits everything)"
echo "  - DGX nvidia-smi 'Not Supported' for memory is NORMAL"
echo ""
echo "Done!"
