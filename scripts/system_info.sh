#!/bin/bash
# 실험 환경 정보 수집

echo "=== 시스템 정보 ==="
system_profiler SPHardwareDataType | grep -E "Model|Chip|Memory|Cores"

echo ""
echo "=== 메모리 ==="
sysctl hw.memsize | awk '{printf "Total RAM: %.0f GB\n", $2/1024/1024/1024}'

echo ""
echo "=== macOS 버전 ==="
sw_vers

echo ""
echo "=== 백엔드 버전 ==="
echo -n "Ollama: "; ollama --version 2>/dev/null || echo "not found"
echo -n "llama-server: "; llama-server --version 2>/dev/null | head -1 || echo "not found"
echo -n "mlx-lm: "; python3 -c "import mlx_lm; print(mlx_lm.__version__)" 2>/dev/null || echo "not found"
echo "LM Studio: 앱에서 직접 확인"

echo ""
echo "=== 전원 설정 ==="
pmset -g | grep -E "hibernatemode|sleep|powernap"
