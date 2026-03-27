"""macOS 온도 모니터링 — thermal throttling 감지 및 대기."""
import subprocess
import time
import platform
from typing import Optional


def get_cpu_temp_celsius() -> Optional[float]:
    """macOS에서 CPU die 온도 반환. sudo 없이 가능한 방법 우선."""
    if platform.system() != "Darwin":
        return None

    # 방법 1: powermetrics (sudo 필요 — 실험 전 sudo 캐시 활성화 권장)
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "--samplers", "thermal", "-n", "1", "-i", "100"],
            text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.splitlines():
            if "CPU die temperature" in line or "cpu_thermal" in line.lower():
                parts = line.split()
                for p in parts:
                    try:
                        return float(p.replace("C", "").replace("°", ""))
                    except ValueError:
                        continue
    except Exception:
        pass

    # 방법 2: istats (brew install istatmenus 또는 gem install iStats)
    try:
        out = subprocess.check_output(["istats", "cpu", "--value-only"], text=True, timeout=3)
        return float(out.strip().split()[0])
    except Exception:
        pass

    return None


def wait_for_cooldown(max_temp: float, check_interval: int, cooldown_wait: int, console=None) -> None:
    """온도가 max_temp 이하로 내려올 때까지 대기."""
    while True:
        temp = get_cpu_temp_celsius()
        if temp is None:
            return  # 온도 측정 불가 시 통과
        if temp <= max_temp:
            return
        msg = f"[yellow]온도 {temp:.1f}°C > {max_temp}°C — {cooldown_wait}s 쿨다운 대기...[/yellow]"
        if console:
            console.print(msg)
        else:
            print(msg)
        time.sleep(cooldown_wait)


def log_temp() -> dict:
    """현재 온도 + 타임스탬프 반환."""
    temp = get_cpu_temp_celsius()
    return {
        "cpu_temp_celsius": temp,
        "temp_available": temp is not None,
    }
