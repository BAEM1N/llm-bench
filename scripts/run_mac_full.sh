#!/bin/bash
# Full Track B + Track A benchmark on Mac (M5 Max, 128GB)
# Qwen3.5 models only (9B, 27B, 35B-A3B, 122B-A10B)
set -uo pipefail

cd /Users/baem1n/Workspace/llm-bench
TS=$(date +%Y%m%d_%H%M%S)
MODELS="qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b"

echo "[$(date)] === Track B: llamacpp only (normalized engine) ==="
uv run python -m src.runner \
  --config config.trackB.yaml \
  --models $MODELS \
  --output "results/trackB_llamacpp_${TS}.csv" 2>&1 | tee "results/trackB_llamacpp_${TS}.log" || true
echo "[$(date)] Track B done"

# Cool down between tracks
echo "[$(date)] Cooling down 120s..."
sleep 120

echo "[$(date)] === Track A: all backends (practical stack) ==="
uv run python -m src.runner \
  --config config.trackA.yaml \
  --backends llamacpp ollama mlx \
  --models $MODELS \
  --output "results/trackA_${TS}.csv" 2>&1 | tee "results/trackA_${TS}.log" || true
echo "[$(date)] Track A done"

# Cleanup
pkill -f "llama-server" 2>/dev/null

echo "[$(date)] === ALL DONE ==="
