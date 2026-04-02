import json
import time
import subprocess
import httpx
from pathlib import Path
from typing import Optional

from .base import BaseBackend, GenerateResult

STARTUP_TIMEOUT = 300  # seconds


class LlamaCppBackend(BaseBackend):
    name = "llamacpp"

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        binary: str = "llama-server",
        n_gpu_layers: int = 99,
        flash_attn: bool = True,
        batch_size: Optional[int] = 512,
        ubatch_size: Optional[int] = 512,
        threads: Optional[int] = None,
        threads_batch: Optional[int] = None,
        mlock: bool = False,
        extra_args: Optional[list] = None,
    ):
        self.base_url = base_url
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.flash_attn = flash_attn
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.threads_batch = threads_batch
        self.mlock = mlock
        self.extra_args = extra_args or []
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
        if self.flash_attn:
            cmd.extend(["--flash-attn", "on"])
        if self.batch_size is not None:
            cmd += ["--batch-size", str(self.batch_size)]
        if self.ubatch_size is not None:
            cmd += ["--ubatch-size", str(self.ubatch_size)]
        if self.threads is not None:
            cmd += ["--threads", str(self.threads)]
        if self.threads_batch is not None:
            cmd += ["--threads-batch", str(self.threads_batch)]
        if self.mlock:
            cmd.append("--mlock")
        if self.extra_args:
            cmd.extend(self.extra_args)

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
                    self._model_memory_gb = self._read_proc_rss_gb()
                    return
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError(f"llama-server did not start within {STARTUP_TIMEOUT}s")

    def _read_proc_rss_gb(self) -> float:
        """llama-server 프로세스 RSS (GB)."""
        try:
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(self._proc.pid)],
                text=True, stderr=subprocess.DEVNULL,
            )
            rss_kb = int(out.strip())
            return round(rss_kb / 1024 / 1024, 2)
        except Exception:
            return 0.0

    def get_model_memory_gb(self) -> float:
        return getattr(self, "_model_memory_gb", 0.0)

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
        token_count = 0
        output_tokens = 0
        timings = {}

        with httpx.Client(timeout=300) as client:
            with client.stream("POST", f"{self.base_url}/completion", json=payload) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    raw = line[6:] if line.startswith("data: ") else line
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    content = chunk.get("content", "")
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        token_count += 1  # 청크당 1토큰 (llama.cpp 기본)

                    if chunk.get("stop") is True:
                        timings = chunk.get("timings", {})
                        # tokens_predicted가 있으면 사용, 없으면 카운트 사용
                        output_tokens = chunk.get("tokens_predicted") or token_count
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

        source = "native" if prompt_tps > 0 else "ttft_estimate"
        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            prompt_tps_source=source,
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
