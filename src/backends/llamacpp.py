import json
import time
import subprocess
import httpx
from pathlib import Path

from .base import BaseBackend, GenerateResult

STARTUP_TIMEOUT = 120  # seconds


class LlamaCppBackend(BaseBackend):
    name = "llamacpp"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        binary: str = "llama-server",
        n_gpu_layers: int = 99,
    ):
        self.base_url = base_url
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self._model_id: str = ""
        self._context_window: int = 262144
        self._proc: subprocess.Popen | None = None
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            out = subprocess.check_output(
                [self.binary, "--version"], text=True, stderr=subprocess.STDOUT
            )
            for line in out.splitlines():
                if "version" in line.lower() or "build" in line.lower():
                    return line.strip()
            return out.strip().splitlines()[0]
        except Exception:
            return "unknown"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        self._model_id = model_id
        self._context_window = context_window
        gguf_path = str(Path(gguf_path).expanduser())

        cmd = [
            self.binary,
            "--model", gguf_path,
            "--ctx-size", str(context_window),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--port", "8080",
            "--host", "127.0.0.1",
            "--no-mmap",
            "--log-disable",
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 서버 준비 대기
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError(f"llama-server did not start within {STARTUP_TIMEOUT}s")

    def unload_model(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> GenerateResult:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": True,
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        timings = {}

        with httpx.Client(timeout=300) as client:
            with client.stream("POST", f"{self.base_url}/completion", json=payload) as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("content") and t_first is None:
                        t_first = time.perf_counter()

                    if chunk.get("stop"):
                        timings = chunk.get("timings", {})
                        output_tokens = chunk.get("tokens_predicted", 0)
                        break

        t_end = time.perf_counter()
        t_first = t_first or t_end

        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start

        # llama.cpp timings 객체에서 정확한 값 사용
        gen_tps = timings.get("predicted_per_second", 0.0)
        prompt_tps = timings.get("prompt_per_second", 0.0)

        if gen_tps == 0.0:
            gen_duration = t_end - t_first
            gen_tps = output_tokens / gen_duration if gen_duration > 0 else 0.0

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
