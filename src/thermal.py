"""CPU 온도 모니터링 — thermal throttling 감지 및 대기.

플랫폼별 지원:
  macOS  : powermetrics (thermal pressure level → 근사 온도)
  Linux  : lm-sensors `sensors -j` → /sys/class/thermal fallback
"""
import glob
import json
import platform
import subprocess
import time
from typing import Optional


# ─── macOS (Apple Silicon) ──────────────────────────────────────────────────

_PRESSURE_TEMP = {
    "nominal": 60.0,
    "moderate": 75.0,
    "heavy": 88.0,
    "critical": 98.0,
}


def _macos_pressure() -> Optional[str]:
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


def _macos_temp() -> Optional[float]:
    pressure = _macos_pressure()
    return _PRESSURE_TEMP.get(pressure, None) if pressure else None


# ─── Linux (Ubuntu) ─────────────────────────────────────────────────────────

def _linux_sensors_temp() -> Optional[float]:
    """lm-sensors `sensors -j` 로 CPU 패키지 온도 읽기.

    패키지 설치: sudo apt install lm-sensors && sudo sensors-detect
    """
    try:
        out = subprocess.check_output(
            ["sensors", "-j"], text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        data = json.loads(out)
        # 우선순위: Package / Tctl(AMD) / CPU 포함 센서
        for readings in data.values():
            for sensor_name, values in readings.items():
                if any(k in sensor_name for k in ("Package", "Tctl", "CPU")):
                    for k, v in values.items():
                        if "input" in k and isinstance(v, (int, float)) and v > 0:
                            return float(v)
        # fallback: 첫 번째 양수 temp_input
        for readings in data.values():
            for values in readings.values():
                for k, v in values.items():
                    if "input" in k and isinstance(v, (int, float)) and v > 0:
                        return float(v)
    except Exception:
        pass
    return None


def _linux_sysfs_temp() -> Optional[float]:
    """/sys/class/thermal/thermal_zone* 에서 CPU 온도 읽기 (millidegree → °C).

    sensors 미설치 환경 fallback.
    """
    try:
        # type 파일로 CPU 관련 zone 탐색
        for type_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/type")):
            zone_dir = type_path.replace("/type", "")
            try:
                with open(type_path) as f:
                    zone_type = f.read().strip().lower()
                if any(k in zone_type for k in ("cpu", "pkg", "x86_pkg", "acpi")):
                    with open(f"{zone_dir}/temp") as f:
                        return int(f.read().strip()) / 1000.0
            except Exception:
                continue
        # fallback: thermal_zone0
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        pass
    return None


def _linux_temp() -> Optional[float]:
    temp = _linux_sensors_temp()
    return temp if temp is not None else _linux_sysfs_temp()


# ─── Public API ─────────────────────────────────────────────────────────────

def get_cpu_temp_celsius() -> Optional[float]:
    """현재 CPU 온도 반환 (°C). 측정 불가 시 None."""
    sys = platform.system()
    if sys == "Darwin":
        return _macos_temp()
    if sys == "Linux":
        return _linux_temp()
    return None


def get_thermal_pressure_level() -> Optional[str]:
    """macOS 전용 pressure level 반환. 비macOS는 None."""
    if platform.system() == "Darwin":
        return _macos_pressure()
    return None


def wait_for_cooldown(
    max_temp: float,
    check_interval: int,
    cooldown_wait: int,
    console=None,
) -> None:
    """온도가 max_temp 이하가 될 때까지 대기."""
    while True:
        temp = get_cpu_temp_celsius()
        if temp is None:
            return  # 측정 불가 시 통과
        if temp <= max_temp:
            return
        msg = f"[yellow]CPU {temp:.0f}°C > {max_temp:.0f}°C — {cooldown_wait}s 쿨다운 대기...[/yellow]"
        if console:
            console.print(msg)
        else:
            print(msg)
        time.sleep(cooldown_wait)


def log_temp() -> dict:
    """현재 온도 및 가용성 반환."""
    temp = get_cpu_temp_celsius()
    pressure = get_thermal_pressure_level()  # macOS only, else None
    return {
        "cpu_temp_celsius": temp,
        "pressure_level": pressure,
        "temp_available": temp is not None,
    }
