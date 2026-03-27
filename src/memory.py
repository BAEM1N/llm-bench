import subprocess
import threading
import time


class MemoryMonitor:
    """macOS unified memory 사용량 모니터링."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._peak_gb = 0.0
        self._running = False
        self._thread = None

    def start(self) -> None:
        self._peak_gb = self._sample_gb()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak_gb

    def _loop(self) -> None:
        while self._running:
            sample = self._sample_gb()
            if sample > self._peak_gb:
                self._peak_gb = sample
            time.sleep(self.interval)

    def _sample_gb(self) -> float:
        try:
            out = subprocess.check_output(
                ["vm_stat"], text=True, stderr=subprocess.DEVNULL
            )
            stats = {}
            for line in out.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    stats[key.strip()] = int(val.strip().rstrip("."))

            page = 16384  # macOS page size bytes
            used_pages = (
                stats.get("Pages active", 0)
                + stats.get("Pages wired down", 0)
                + stats.get("Pages occupied by compressor", 0)
            )
            return round(used_pages * page / 1024**3, 2)
        except Exception:
            return 0.0
