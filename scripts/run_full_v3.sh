#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 3090 2-WAY Server — Full Benchmark v3 (Track B + Track A)
# Ryzen 9 5950X + RTX 3090 x2, 128GB DDR4 + 48GB VRAM
#
# Track B: Hardware comparison — llamacpp only (all Qwen3.5 models)
# Track A: Engine comparison — llamacpp + ollama + vllm
#
# Usage:
#   nohup bash scripts/run_full_v3.sh > /tmp/bench_v3.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/config.linux.yaml"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/bench_v3.log"

# Use project venv python
PYTHON="$PROJECT_DIR/.venv/bin/python"
export PATH="/usr/local/cuda/bin:$HOME/llama.cpp/build/bin:$HOME/.local/bin:$PROJECT_DIR/.venv/bin:$PATH"

echo "============================================================"
echo " 3090 2-WAY — Full Benchmark v3 (Track B + Track A)"
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

run_bench() {
    # Usage: run_bench <output_csv> <extra_args...>
    local output="$1"
    shift
    cd "$PROJECT_DIR"
    $PYTHON -m src.runner \
        --config "$CONFIG" \
        --output "$output" \
        "$@" \
        2>&1 || echo "  WARNING: benchmark run failed (args: $*)"
}

# ─── System info ──────────────────────────────────────────────────────────────

log_section "System Info"
echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "CPU:      $(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')"
echo "RAM:      $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo "CUDA:     $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "llama-server: $(which llama-server 2>/dev/null || echo 'not found')"
echo "vllm:     $($PYTHON -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'not installed')"
echo "Python:   $PYTHON"

# Check GGUF models exist
echo ""
echo "Model files:"
for f in \
    ~/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q8_0.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf \
    ~/.lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf; do
    if [ -f "$f" ]; then
        echo "  OK: $(basename "$f") ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
    fi
done

# ═════════════════════════════════════════════════════════════════════════════
# TRACK B: Hardware comparison — llamacpp only
# ═════════════════════════════════════════════════════════════════════════════

log_section "TRACK B: Hardware Comparison (llamacpp only)"

TRACKB_OUTPUT="$RESULTS_DIR/trackB_llamacpp_${TIMESTAMP}.csv"
echo "Output: $TRACKB_OUTPUT"

# Switch config to mode B
sed -i 's/^comparison_mode: A/comparison_mode: B/' "$CONFIG"
echo "  Config comparison_mode set to B"

# ── Track B: 9B, 27B, 35B-A3B (n_gpu_layers=99) ────────────────────────────

for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b; do
    echo ""
    echo "  --- Track B llamacpp: $model (ngl=99) ---"
    kill_port 8082
    run_bench "$TRACKB_OUTPUT" --backends llamacpp --models "$model"
    kill_port 8082
    wait_cooldown 120
done

# ── Track B: 122B (n_gpu_layers=20 for partial offload) ─────────────────────

log_section "Track B: 122B llamacpp (n_gpu_layers=20)"

# Temporarily set n_gpu_layers to 20 for 122B
sed -i 's/n_gpu_layers: 99/n_gpu_layers: 20/' "$CONFIG"
echo "  Config n_gpu_layers set to 20 for 122B"

kill_port 8082
run_bench "$TRACKB_OUTPUT" --backends llamacpp --models qwen3.5-122b-a10b
kill_port 8082

# Restore n_gpu_layers to 99
sed -i 's/n_gpu_layers: 20/n_gpu_layers: 99/' "$CONFIG"
echo "  Config n_gpu_layers restored to 99"

wait_cooldown 180

# Restore config to mode A
sed -i 's/^comparison_mode: B/comparison_mode: A/' "$CONFIG"
echo "  Config comparison_mode restored to A"

echo ""
echo "Track B complete: $TRACKB_OUTPUT"
if [ -f "$TRACKB_OUTPUT" ]; then
    echo "  Lines: $(wc -l < "$TRACKB_OUTPUT")"
fi

# ═════════════════════════════════════════════════════════════════════════════
# TRACK A: Engine comparison — llamacpp + ollama + vllm
# ═════════════════════════════════════════════════════════════════════════════

log_section "TRACK A: Engine Comparison"

TRACKA_OUTPUT="$RESULTS_DIR/trackA_${TIMESTAMP}.csv"
echo "Output: $TRACKA_OUTPUT"

# ── Phase A1: llamacpp — all models ─────────────────────────────────────────

log_section "Phase A1: llamacpp (all models)"

for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b; do
    echo ""
    echo "  --- Track A llamacpp: $model (ngl=99) ---"
    kill_port 8082
    run_bench "$TRACKA_OUTPUT" --backends llamacpp --models "$model"
    kill_port 8082
    wait_cooldown 120
done

# 122B with ngl=20
sed -i 's/n_gpu_layers: 99/n_gpu_layers: 20/' "$CONFIG"
echo "  Config n_gpu_layers set to 20 for 122B"

kill_port 8082
run_bench "$TRACKA_OUTPUT" --backends llamacpp --models qwen3.5-122b-a10b
kill_port 8082

sed -i 's/n_gpu_layers: 20/n_gpu_layers: 99/' "$CONFIG"
echo "  Config n_gpu_layers restored to 99"

wait_cooldown 180

# ── Phase A2: ollama — all models ───────────────────────────────────────────

log_section "Phase A2: ollama (all models)"

# Ensure ollama is running
if ! systemctl is-active --quiet ollama 2>/dev/null; then
    echo "  Starting ollama service..."
    sudo systemctl start ollama || true
    sleep 5
fi

# Verify ollama responds
for i in 1 2 3; do
    if curl -sf http://localhost:11434/api/version > /dev/null 2>&1; then
        break
    fi
    echo "  Waiting for ollama (attempt $i)..."
    sleep 5
done

echo "  Ollama version: $(curl -sf http://localhost:11434/api/version | python3 -c 'import sys,json; print(json.load(sys.stdin).get("version","?"))' 2>/dev/null || echo 'unknown')"

# Pull models if not present
for tag in \
    "qwen3.5:9b-q4_K_M" "qwen3.5:9b-q8_0" \
    "qwen3.5:27b-q4_K_M" "qwen3.5:27b-q8_0" \
    "qwen3.5:35b-a3b-q4_K_M" "qwen3.5:35b-a3b-q8_0" \
    "qwen3.5:122b-a10b-q4_K_M"; do
    if ! ollama list 2>/dev/null | grep -q "$(echo "$tag" | tr ':' ' ' | awk '{print $1}')"; then
        echo "  Pulling $tag..."
        ollama pull "$tag" || echo "  WARNING: Failed to pull $tag"
    fi
done

for model in qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b; do
    echo ""
    echo "  --- Track A ollama: $model ---"
    run_bench "$TRACKA_OUTPUT" --backends ollama --models "$model"
    wait_cooldown 60
done

wait_cooldown 120

# ── Phase A3: vLLM — selected models ───────────────────────────────────────
# 9B BF16 (~18GB), 27B GPTQ-Int4, 35B-A3B GPTQ-Int4, 122B GPTQ-Int4+cpu_offload
# vllm_quantization per quant is set in config.linux.yaml
# --dtype half is auto-added by vllm_backend.py for gptq/gptq_marlin

log_section "Phase A3: vLLM (9B BF16, 27B GPTQ, 35B GPTQ, 122B GPTQ)"

kill_port 8000

# 9B BF16
echo ""
echo "  --- Track A vLLM: qwen3.5-9b (BF16) ---"
run_bench "$TRACKA_OUTPUT" --backends vllm --models qwen3.5-9b
kill_port 8000
wait_cooldown 60

# 27B GPTQ-Int4
echo ""
echo "  --- Track A vLLM: qwen3.5-27b (GPTQ-Int4) ---"
run_bench "$TRACKA_OUTPUT" --backends vllm --models qwen3.5-27b
kill_port 8000
wait_cooldown 60

# 35B-A3B GPTQ-Int4
echo ""
echo "  --- Track A vLLM: qwen3.5-35b-a3b (GPTQ-Int4) ---"
run_bench "$TRACKA_OUTPUT" --backends vllm --models qwen3.5-35b-a3b
kill_port 8000
wait_cooldown 60

# 122B GPTQ-Int4 — cpu_offload_gb=20
echo ""
echo "  --- Track A vLLM: qwen3.5-122b-a10b (GPTQ-Int4, cpu_offload=20) ---"
sed -i 's/cpu_offload_gb: 0/cpu_offload_gb: 20/' "$CONFIG"
echo "  Config cpu_offload_gb set to 20 for 122B"

run_bench "$TRACKA_OUTPUT" --backends vllm --models qwen3.5-122b-a10b
kill_port 8000

# Restore cpu_offload_gb to 0
sed -i 's/cpu_offload_gb: 20/cpu_offload_gb: 0/' "$CONFIG"
echo "  Config cpu_offload_gb restored to 0"

wait_cooldown 120

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

log_section "Benchmark Complete"
echo "Finished: $(date)"
echo ""
echo "Result files:"
for f in "$TRACKB_OUTPUT" "$TRACKA_OUTPUT"; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f ($lines lines)"
    else
        echo "  $f (NOT FOUND — all runs may have failed)"
    fi
done

echo ""
echo "Config verification (should be back to defaults):"
grep -E '(comparison_mode|n_gpu_layers|cpu_offload_gb)' "$CONFIG" | head -5

echo ""
echo "Notes:"
echo "  - Track B: comparison_mode=B, llamacpp only, all 4 models"
echo "  - Track A: comparison_mode=A, llamacpp+ollama+vllm"
echo "  - 122B llamacpp: n_gpu_layers=20 (partial offload to CPU RAM)"
echo "  - 122B vLLM: cpu_offload_gb=20"
echo "  - All timeouts: 1800s per request"
echo "  - Prefill tracks: server restart between each (cold prefill)"
echo ""
echo "Done!"
