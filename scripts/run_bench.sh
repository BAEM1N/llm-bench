#!/bin/bash
# 단일 벤치마크 스크립트 — 에이전트 없이 nohup으로 실행
set -uo pipefail

BENCH="$1"        # /home/baeumai/llm-bench or /Users/baem1n/Workspace/llm-bench
CONFIG_B="$2"     # Track B config path
CONFIG_A="$3"     # Track A config path
PYTHON="$4"       # python path

export PYTHONPATH="$BENCH"
cd "$BENCH"

TS=$(date +%Y%m%d_%H%M%S)

echo "[$(date)] === Track B: 하드웨어 비교 (llamacpp only) ==="
$PYTHON -m src.runner --config "$CONFIG_B" --output "results/trackB_${TS}.csv" 2>&1 | tee "results/trackB_${TS}.log" || true
echo "[$(date)] Track B 완료"

sleep 60

echo "[$(date)] === Track A: 엔진 비교 ==="
$PYTHON -m src.runner --config "$CONFIG_A" --output "results/trackA_${TS}.csv" 2>&1 | tee "results/trackA_${TS}.log" || true
echo "[$(date)] Track A 완료"

# 서버 프로세스 정리
pkill -f "llama-server" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
pkill -f "lemonade-server" 2>/dev/null

echo "[$(date)] === 전체 완료 ==="
