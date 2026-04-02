"""CPU 온도 + 전력 모니터링 — thermal throttling 감지, 전력/효율 측정.

플랫폼별 지원:
  macOS  : powermetrics (thermal pressure → 근사 온도, cpu_power_w / gpu_power_w)
  Linux  : lm-sensors `sensors -j` → /sys/class/thermal fallback (온도)
           Intel RAPL 200ms 샘플 (CPU 전력), nvidia-smi (GPU 전력)
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


def _linux_hwmon_temp() -> Optional[float]:
    """hwmon 드라이버에서 CPU 온도 직접 읽기.

    우선순위: k10temp (AMD Ryzen) → coretemp (Intel) → 첫 번째 hwmon temp
    lm-sensors 미설치 환경에서도 동작.
    """
    preferred = ("k10temp", "coretemp")
    fallback_hwmon = None

    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name_path = f"{hwmon_dir}/name"
            with open(name_path) as f:
                name = f.read().strip()
        except Exception:
            continue

        is_preferred = name in preferred
        if not is_preferred and fallback_hwmon is None:
            fallback_hwmon = hwmon_dir

        if is_preferred:
            # Tctl (AMD) 또는 Package id 0 (Intel) 우선
            for label_file in sorted(glob.glob(f"{hwmon_dir}/temp*_label")):
                try:
                    with open(label_file) as f:
                        label = f.read().strip()
                    if any(k in label for k in ("Tctl", "Package", "Tccd1")):
                        input_file = label_file.replace("_label", "_input")
                        with open(input_file) as f:
                            return int(f.read().strip()) / 1000.0
                except Exception:
                    continue
            # label 없으면 temp1_input (보통 Tctl)
            try:
                with open(f"{hwmon_dir}/temp1_input") as f:
                    return int(f.read().strip()) / 1000.0
            except Exception:
                continue

    # 최후 fallback: 아무 hwmon의 temp1
    if fallback_hwmon:
        try:
            with open(f"{fallback_hwmon}/temp1_input") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            pass
    return None


def _linux_temp() -> Optional[float]:
    # 우선순위: sensors -j → hwmon 직접 읽기 → sysfs thermal_zone
    temp = _linux_sensors_temp()
    if temp is not None:
        return temp
    temp = _linux_hwmon_temp()
    if temp is not None:
        return temp
    return _linux_sysfs_temp()


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


# ─── Power measurement ───────────────────────────────────────────────────────

def _parse_power_line(line: str) -> Optional[float]:
    """'CPU Power: 1808 mW' 또는 'CPU Power: 1.808 W' 형식을 W 단위로 파싱."""
    try:
        parts = line.strip().split()
        # 최소 형식: ["CPU", "Power:", "1808", "mW"]
        if len(parts) < 3:
            return None
        val = float(parts[2])
        unit = parts[3].lower() if len(parts) > 3 else "w"
        if unit == "mw":
            val /= 1000.0
        return round(val, 3)
    except (ValueError, IndexError):
        return None


def _macos_power() -> dict:
    """powermetrics cpu_power 샘플러로 CPU/GPU 전력 측정."""
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "200"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        cpu_w = gpu_w = -1.0
        for line in out.splitlines():
            line_l = line.strip()
            if line_l.startswith("CPU Power:"):
                v = _parse_power_line(line_l)
                if v is not None:
                    cpu_w = v
            elif line_l.startswith("GPU Power:"):
                v = _parse_power_line(line_l)
                if v is not None:
                    gpu_w = v
        return {"cpu_w": cpu_w, "gpu_w": gpu_w}
    except Exception:
        return {"cpu_w": -1.0, "gpu_w": -1.0}


def _linux_rapl_cpu_power() -> Optional[float]:
    """Intel RAPL 200ms 샘플로 CPU 패키지 평균 전력 측정."""
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    try:
        with open(rapl_path) as f:
            e1 = int(f.read())
        t1 = time.perf_counter()
        time.sleep(0.2)
        with open(rapl_path) as f:
            e2 = int(f.read())
        t2 = time.perf_counter()
        return round((e2 - e1) / 1e6 / (t2 - t1), 2)  # μJ/s → W
    except Exception:
        pass
    # AMD: /sys/class/powercap/amd_energy/intel-rapl:0/energy_uj (같은 인터페이스)
    for pattern in glob.glob("/sys/class/powercap/*/energy_uj"):
        try:
            with open(pattern) as f:
                e1 = int(f.read())
            t1 = time.perf_counter()
            time.sleep(0.2)
            with open(pattern) as f:
                e2 = int(f.read())
            t2 = time.perf_counter()
            return round((e2 - e1) / 1e6 / (t2 - t1), 2)
        except Exception:
            continue
    return None


def _linux_nvidia_gpu_power() -> Optional[float]:
    """nvidia-smi로 GPU 전력 합산 (W)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        total = sum(float(line.strip()) for line in out.strip().splitlines() if line.strip())
        return round(total, 2)
    except Exception:
        return None


def get_power_watts() -> dict:
    """CPU/GPU 전력 스냅샷 반환 (W). 측정 불가 필드는 -1."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        return _macos_power()
    if sys_name == "Linux":
        cpu_w = _linux_rapl_cpu_power()
        gpu_w = _linux_nvidia_gpu_power()
        return {
            "cpu_w": cpu_w if cpu_w is not None else -1.0,
            "gpu_w": gpu_w if gpu_w is not None else -1.0,
        }
    return {"cpu_w": -1.0, "gpu_w": -1.0}


def log_power() -> dict:
    """현재 전력 소비 스냅샷 반환."""
    d = get_power_watts()
    total = -1.0
    if d["cpu_w"] >= 0 and d["gpu_w"] >= 0:
        total = round(d["cpu_w"] + d["gpu_w"], 2)
    elif d["cpu_w"] >= 0:
        total = d["cpu_w"]
    elif d["gpu_w"] >= 0:
        total = d["gpu_w"]
    return {
        "cpu_power_w": d["cpu_w"],
        "gpu_power_w": d["gpu_w"],
        "total_power_w": total,
        "power_available": total >= 0,
    }
