"""macOS 온도 모니터링 — thermal throttling 감지 및 대기."""
import subprocess
import time
import platform
from typing import Optional

# Apple Silicon thermal pressure level → 근사 온도 (°C)
_PRESSURE_TEMP = {
    "nominal": 60.0,
    "moderate": 75.0,
    "heavy": 88.0,
    "critical": 98.0,
}


def _get_thermal_pressure() -> Optional[str]:
    """powermetrics로 thermal pressure level 반환 (Apple Silicon)."""
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "--samplers", "thermal", "-n", "1", "-i", "100"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            if "Current pressure level" in line:
                return line.split(":")[-1].strip().lower()
    except Exception:
        pass
    return None


def get_cpu_temp_celsius() -> Optional[float]:
    """CPU 온도 반환. Apple Silicon은 pressure level → 근사값."""
    if platform.system() != "Darwin":
        return None

    pressure = _get_thermal_pressure()
    if pressure is not None:
        return _PRESSURE_TEMP.get(pressure, 60.0)

    return None


def get_thermal_pressure_level() -> Optional[str]:
    """현재 thermal pressure level 문자열 반환 (nominal/moderate/heavy/critical)."""
    return _get_thermal_pressure()


def wait_for_cooldown(max_temp: float, check_interval: int, cooldown_wait: int, console=None) -> None:
    """Heavy 이상 pressure 시 cooldown_wait초 대기."""
    while True:
        pressure = _get_thermal_pressure()
        if pressure is None:
            return  # 측정 불가 시 통과
        temp = _PRESSURE_TEMP.get(pressure, 0.0)
        if temp <= max_temp:
            return
        msg = f"[yellow]Thermal pressure: {pressure} (~{temp:.0f}°C) — {cooldown_wait}s 쿨다운 대기...[/yellow]"
        if console:
            console.print(msg)
        else:
            print(msg)
        time.sleep(cooldown_wait)


def log_temp() -> dict:
    """현재 thermal pressure → 근사 온도 반환."""
    pressure = _get_thermal_pressure()
    temp = _PRESSURE_TEMP.get(pressure, None) if pressure else None
    return {
        "cpu_temp_celsius": temp,
        "pressure_level": pressure,
        "temp_available": temp is not None,
    }
