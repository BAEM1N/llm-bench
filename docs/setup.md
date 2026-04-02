# Setup & Monitoring — MacBook Pro 14 (M5 Max)

## 1. 환경 설치

```bash
# uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 모델 다운로드 상태 확인
./scripts/check_models.sh

# 시스템 정보 수집
./scripts/system_info.sh
```

---

## 2. 온도 모니터링

### powermetrics (권장, sudo 필요)

```bash
# 온도 1회 확인
sudo powermetrics --samplers thermal -n 1 -i 1000

# SMC 기반 상세 온도
sudo powermetrics --samplers smc -n 1 -i 1000 | grep -i temp

# Thermal pressure level (nominal / moderate / heavy / critical)
sudo powermetrics --samplers thermal -n 1 -i 1000 | grep -i pressure
```

실험 전 sudo 캐시 활성화 권장:
```bash
sudo powermetrics --samplers thermal -n 1 -i 1000
# 한 번 실행하면 이후 ~5분간 sudo 없이도 runner가 호출 가능
```

NOPASSWD 설정 (영구적으로 sudo 없이 사용):
```bash
sudo visudo
# 아래 줄 추가:
# <username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

### iStats (대안, gem)

```bash
gem install iStats
istats            # 전체 센서
istats cpu temp   # CPU 온도만
```

---

## 3. 메모리 모니터링

### vm_stat — 유니파이드 메모리 실시간

```bash
# 1초마다 갱신
vm_stat 1

# 주요 필드 해석
# Pages active:    현재 사용 중
# Pages wired:     커널/고정 메모리
# Pages free:      즉시 사용 가능
# Pages compressed: 압축된 페이지 (스왑과 유사)
```

페이지 → GB 변환: `pages × 4096 / 1024^3`

### 프로세스별 메모리 (RSS)

```bash
# llama-server 메모리
ps -o pid,rss,command -p $(pgrep llama-server) | awk '{printf "%s  %.1f GB  %s\n", $1, $2/1024/1024, $3}'

# Ollama 서버 메모리
ps -o pid,rss,command -p $(pgrep ollama) | awk '{printf "%s  %.1f GB  %s\n", $1, $2/1024/1024, $3}'
```

### 메모리 압박 요약 (Memory Pressure)

```bash
memory_pressure
# Shows: System-wide memory free percentage
```

---

## 4. GPU (Metal) 모니터링

```bash
# Metal GPU 사용률 및 VRAM (Apple GPU)
sudo powermetrics --samplers gpu_power -n 1 -i 1000 | grep -E "GPU|Renderer|Tiler"

# MLX에서 직접 확인 (Python)
python3 -c "import mlx.core as mx; print(f'Peak memory: {mx.metal.get_peak_memory()/1e9:.2f} GB')"
```

---

## 5. 실험 중 통합 모니터링

```bash
# 별도 터미널에서 실행 — 5초마다 온도 + 메모리 출력
watch -n 5 'echo "=== $(date +%H:%M:%S) ===" && \
  sudo powermetrics --samplers thermal -n 1 -i 500 2>/dev/null | grep -i pressure && \
  vm_stat | grep -E "Pages (free|active|wired|compressed)" | \
  awk "{printf \"%s  %.1f GB\n\", \$0, (\$NF+0)*4096/1024/1024/1024}"'
```

---

## 6. Ollama 모니터링

```bash
# 로드된 모델 및 VRAM 사용 확인
ollama ps

# Ollama 서버 로그 (실시간)
tail -f ~/.ollama/logs/server.log

# Ollama 버전
ollama --version
```

---

## 7. llama.cpp 서버 모니터링

```bash
# 서버 상태 확인
curl -s http://localhost:8080/health | python3 -m json.tool

# 서버 프로세스 확인
pgrep -la llama-server
```

---

## 8. 실험 전 체크리스트

```bash
# 1. 모델 파일 확인
./scripts/check_models.sh

# 2. 백그라운드 프로세스 정리 (Chrome, Xcode 등 종료 권장)
# 3. 충전기 연결 확인 (배터리 모드에서 성능 throttle 발생)
# 4. sudo 캐시 활성화
sudo powermetrics --samplers thermal -n 1 -i 1000

# 5. 실험 실행
uv run python -m src.runner --config config.yaml --output results/my_run.csv
```
