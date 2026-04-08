#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Ryzen AI MAX 395+ — Full Benchmark v2
# HP Z2 Mini G1a, 128GB unified memory, Radeon 8060S iGPU (RDNA 3.5, 40 CU)
# BIOS VRAM: 34GB, usable system RAM: ~94GB
#
# Backends: llamacpp (Vulkan), ollama (ROCm), vllm (ROCm experimental), lemonade
# Models:   Qwen3.5 9B/27B/35B-A3B/122B-A10B (Q4_K_M, Q8_0 where available)
#
# Usage:
#   nohup bash scripts/run_ryzen_full_v2.sh > /tmp/ryzen_bench_v2.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
# NOTE: NOT using set -e. Failures are handled per-phase to continue the full suite.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/config.ryzen-ai.yaml"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/ryzen_bench_v2.log"

# ROCm environment for gfx1151 (Strix Halo)
export HSA_OVERRIDE_GFX_VERSION=11.5.1

echo "============================================================"
echo " Ryzen AI MAX 395+ — Full Benchmark v2"
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

# ─── System info ──────────────────────────────────────────────────────────────

log_section "System Info"
echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "CPU:      $(lscpu | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')"
echo "RAM:      $(free -h | awk '/Mem:/{print $2}' || echo 'unknown')"
echo "ROCm:     $(rocminfo 2>/dev/null | grep -m1 'ROCk module' || echo 'unknown')"
echo "GPU:      $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: *//' || echo 'unknown')"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

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
    ~/models/gguf/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename "$f") ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
    fi
done

# ─── Phase 1: llamacpp (Vulkan) ──────────────────────────────────────────────
# Binary: ~/llama.cpp/build/bin/llama-server (Vulkan build)
# All models: 9B, 27B, 35B-A3B, 122B-A10B (Q4_K_M + Q8_0)
# 122B may timeout at 1800s — the runner handles this gracefully

log_section "Phase 1: llamacpp (Vulkan) — All Qwen3.5 models"

# Stop lemonade-server if running (it consumes GPU resources)
sudo snap stop lemonade-server 2>/dev/null || true
kill_port 8001

# Kill anything on port 8080
kill_port 8080

LLAMACPP_OUTPUT="$RESULTS_DIR/ryzen_llamacpp_v2_${TIMESTAMP}.csv"
echo "Output: $LLAMACPP_OUTPUT"

# Run each model individually to isolate crashes
# 96GB VRAM: all models fit. 256K context. 122B may still be slow.
for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- llamacpp: $model ---"
    kill_port 8080
    cd "$PROJECT_DIR"
    uv run python -m src.runner \
        --config "$CONFIG" \
        --backends llamacpp \
        --models "$model" \
        --output "$LLAMACPP_OUTPUT" \
        2>&1 | tee -a "$LOG.llamacpp" || echo "  WARNING: llamacpp $model failed"
    kill_port 8080
    wait_cooldown 120
done

echo "llamacpp phase complete: $LLAMACPP_OUTPUT"

# Ensure server is dead after run
kill_port 8080
wait_cooldown 120

# ─── Phase 2: ollama (ROCm) ──────────────────────────────────────────────────
# HSA_OVERRIDE_GFX_VERSION=11.5.1 (systemd override already configured)
# All Qwen3.5 models: 9b, 27b, 35b-a3b, 122b-a10b (Q4_K_M + Q8_0)

log_section "Phase 2: ollama (ROCm) — All Qwen3.5 models"

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

# Pull models if needed (ollama needs them pre-pulled)
for model in \
    "qwen3.5:9b-q4_K_M" "qwen3.5:9b-q8_0" \
    "qwen3.5:27b-q4_K_M" "qwen3.5:27b-q8_0" \
    "qwen3.5:35b-a3b-q4_K_M" "qwen3.5:35b-a3b-q8_0" \
    "qwen3.5:122b-a10b-q4_K_M"; do
    echo "  Checking: $model"
    if ! ollama list 2>/dev/null | grep -q "$(echo "$model" | cut -d: -f1).*$(echo "$model" | cut -d: -f2 | cut -d- -f1)"; then
        echo "  Pulling $model (this may take a while)..."
        ollama pull "$model" || echo "  WARNING: Failed to pull $model"
    fi
done

OLLAMA_OUTPUT="$RESULTS_DIR/ryzen_ollama_v2_${TIMESTAMP}.csv"
echo "Output: $OLLAMA_OUTPUT"

# Run each model individually for isolation
for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- ollama: $model ---"
    cd "$PROJECT_DIR"
    uv run python -m src.runner \
        --config "$CONFIG" \
        --backends ollama \
        --models "$model" \
        --output "$OLLAMA_OUTPUT" \
        2>&1 | tee -a "$LOG.ollama" || echo "  WARNING: ollama $model failed"
    wait_cooldown 60
done

echo "ollama phase complete: $OLLAMA_OUTPUT"
wait_cooldown 120

# ─── Phase 3: vLLM (ROCm — experimental on gfx1151) ─────────────────────────
# vLLM 0.19.0 installed in venv
# HSA_OVERRIDE_GFX_VERSION=11.5.1
# tensor_parallel_size=1 (single iGPU)
# Try 9b first as test — if fails, note error and skip

log_section "Phase 3: vLLM (ROCm experimental) — Qwen3.5-9B test"

VLLM_OUTPUT="$RESULTS_DIR/ryzen_vllm_v2_${TIMESTAMP}.csv"
VLLM_SUCCESS=false

# Kill anything on port 8000 (vllm and lemonade share this port)
kill_port 8000

# Check if vllm is available
if command -v vllm &>/dev/null || [ -f ~/venv/bin/vllm ]; then
    # Activate venv if vllm is in a venv
    if [ -f ~/venv/bin/activate ]; then
        source ~/venv/bin/activate
    fi

    echo "  vLLM binary: $(which vllm 2>/dev/null || echo '~/venv/bin/vllm')"
    echo "  Attempting vLLM with Qwen3.5-9B only (experimental)..."

    # Try just the 9B model first
    cd "$PROJECT_DIR"
    timeout 3600 uv run python -m src.runner \
        --config "$CONFIG" \
        --backends vllm \
        --models qwen3.5-9b \
        --output "$VLLM_OUTPUT" \
        2>&1 | tee -a "$LOG.vllm" && VLLM_SUCCESS=true || true

    if [ "$VLLM_SUCCESS" = true ]; then
        echo "  vLLM 9B succeeded! Trying remaining models..."

        # If 9B worked, try the rest
        for model in qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
            echo "  Trying vLLM: $model"
            kill_port 8000
            wait_cooldown 60
            timeout 7200 uv run python -m src.runner \
                --config "$CONFIG" \
                --backends vllm \
                --models "$model" \
                --output "$VLLM_OUTPUT" \
                2>&1 | tee -a "$LOG.vllm" || echo "  WARNING: vLLM $model failed"
        done
    else
        echo "  WARNING: vLLM failed on 9B. Skipping remaining models."
        echo "  Check $LOG.vllm for details."
    fi

    # Deactivate venv if we activated it
    if [ -f ~/venv/bin/activate ]; then
        deactivate 2>/dev/null || true
    fi
else
    echo "  SKIP: vllm not found. Install with: pip install vllm"
fi

kill_port 8000
wait_cooldown 120

# ─── Phase 4: lemonade ───────────────────────────────────────────────────────
# /snap/bin/lemonade-server serve --llamacpp rocm --port 8000
# Models use lemonade_model field: Qwen3.5-9B-GGUF, etc.
# Only Q4_K_M has lemonade_model set (Q8_0 lemonade_model is empty = skip)

log_section "Phase 4: lemonade — Qwen3.5 Q4_K_M models"

LEMONADE_OUTPUT="$RESULTS_DIR/ryzen_lemonade_v2_${TIMESTAMP}.csv"
LEMONADE_BIN="/snap/bin/lemonade-server"

kill_port 8000

if [ -x "$LEMONADE_BIN" ]; then
    echo "  Starting lemonade-server..."
    $LEMONADE_BIN serve --llamacpp rocm --port 8000 &
    LEMONADE_PID=$!
    echo "  lemonade PID: $LEMONADE_PID"

    # Wait for lemonade to be ready
    echo "  Waiting for lemonade to start..."
    if check_server "http://localhost:8000/api/v1/health" 60; then
        echo "  lemonade is ready"

        cd "$PROJECT_DIR"
        uv run python -m src.runner \
            --config "$CONFIG" \
            --backends lemonade \
            --output "$LEMONADE_OUTPUT" \
            2>&1 | tee -a "$LOG.lemonade" || true

        echo "lemonade phase complete: $LEMONADE_OUTPUT"
    else
        echo "  ERROR: lemonade failed to start within 60s"
        # Try alternate health endpoint
        if check_server "http://localhost:8000/api/v1/models" 10; then
            echo "  lemonade responding on /api/v1/models, proceeding..."
            cd "$PROJECT_DIR"
            uv run python -m src.runner \
                --config "$CONFIG" \
                --backends lemonade \
                --output "$LEMONADE_OUTPUT" \
                2>&1 | tee -a "$LOG.lemonade" || true
        else
            echo "  lemonade not responding on any endpoint. Skipping."
        fi
    fi

    # Kill lemonade
    echo "  Stopping lemonade (PID: $LEMONADE_PID)..."
    kill "$LEMONADE_PID" 2>/dev/null || true
    wait "$LEMONADE_PID" 2>/dev/null || true
    kill_port 8000
else
    echo "  SKIP: lemonade-server not found at $LEMONADE_BIN"
    echo "  Install: sudo snap install lemonade-server"
fi

# ─── Summary ──────────────────────────────────────────────────────────────────

log_section "Benchmark Complete"
echo "Finished: $(date)"
echo ""
echo "Result files:"
for f in "$RESULTS_DIR"/ryzen_*_v2_${TIMESTAMP}.csv; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f ($lines lines)"
    fi
done

echo ""
echo "Log files:"
for f in "$LOG" "$LOG.llamacpp" "$LOG.ollama" "$LOG.vllm" "$LOG.lemonade"; do
    if [ -f "$f" ]; then
        echo "  $f ($(du -h "$f" | cut -f1))"
    fi
done

echo ""
echo "Notes:"
if [ "$VLLM_SUCCESS" = false ] 2>/dev/null; then
    echo "  - vLLM: FAILED or SKIPPED on gfx1151 (experimental ROCm support)"
fi
echo "  - 122B models may have timed out (1800s limit per request)"
echo "  - Check individual log files for detailed errors"
echo ""
echo "Done!"
