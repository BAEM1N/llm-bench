"""SGLang backend — NVIDIA CUDA (3090 2-WAY / DGX Spark) 용.

vLLM 대비 throughput 우위 케이스 많음 (FlashInfer 커널).
API: OpenAI-compatible /v1/completions (vLLM과 동일 포맷).

실행: python -m sglang.launch_server --model-path <hf_id> --tp 2 ...
설치: pip install sglang[all]

GPTQ / AWQ / FP8 / BF16 지원.
"""
import json
import sys
import time
import subprocess
from urllib.parse import urlparse
import httpx
from typing import Optional

from .base import BaseBackend, GenerateResult

STARTUP_TIMEOUT = 1800  # seconds — 대형 모델 로드 + 컴파일에 최대 30분


class SGLangBackend(BaseBackend):
    name = "sglang"

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,  # "gptq", "awq", "fp8", None
        cpu_offload_gb: float = 0.0,
        extra_args: Optional[list] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.cpu_offload_gb = cpu_offload_gb
        self.extra_args = extra_args or []
        self._model_id: str = ""
        self._context_window: int = 32768
        self._proc: Optional[subprocess.Popen] = None
        self._model_memory_gb: float = 0.0
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "sglang.version"],
                text=True, stderr=subprocess.DEVNULL,
            )
            return out.strip()
        except Exception:
            try:
                out = subprocess.check_output(
                    [sys.executable, "-c", "import sglang; print(sglang.__version__)"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                return out.strip()
            except Exception:
                return "unknown"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        self._model_id = model_id
        self._context_window = context_window

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 30000)

        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_id,
            "--host", host,
            "--port", port,
            "--tp", str(self.tensor_parallel_size),
            "--context-length", str(self.max_model_len or context_window),
            "--mem-fraction-static", str(self.gpu_memory_utilization),
            "--disable-radix-cache",              # prefix cache 비활성화 (cold prefill)
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.cpu_offload_gb > 0:
            cmd += ["--cpu-offload-gb", str(self.cpu_offload_gb)]
        if self.extra_args:
            cmd.extend(self.extra_args)

        self._server_log = open("/tmp/sglang_server.log", "w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._server_log,
            stderr=self._server_log,
        )

        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            # 프로세스가 이미 죽었으면 즉시 실패
            if self._proc.poll() is not None:
                self._server_log.close()
                with open("/tmp/sglang_server.log") as f:
                    tail = f.read()[-500:]
                raise RuntimeError(f"SGLang server exited (code={self._proc.returncode}): {tail}")
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    self._model_memory_gb = self._read_gpu_memory_gb()
                    return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"SGLang server did not start within {STARTUP_TIMEOUT}s")

    def _read_gpu_memory_gb(self) -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            total_mb = sum(int(l.strip()) for l in out.strip().splitlines() if l.strip())
            return round(total_mb / 1024, 2)
        except Exception:
            return 0.0

    def get_effective_context(self) -> int:
        return self.max_model_len or self._context_window

    def get_model_memory_gb(self) -> float:
        return self._model_memory_gb

    @property
    def memory_method(self) -> str:
        return "gpu_smi"

    def unload_model(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if hasattr(self, "_server_log") and self._server_log:
            self._server_log.close()
            self._server_log = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> GenerateResult:
        """OpenAI-compatible /v1/completions 스트리밍 (vLLM과 동일 포맷)."""
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        token_count = 0

        with httpx.Client(timeout=600) as client:
            with client.stream("POST", f"{self.base_url}/v1/completions", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    raw = line[6:] if line.startswith("data: ") else line
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        token_count += 1
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)

        if output_tokens == 0:
            output_tokens = token_count

        t_end = time.perf_counter()
        t_first = t_first or t_end
        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start
        gen_duration = t_end - t_first
        gen_tps = output_tokens / gen_duration if gen_duration > 0 and output_tokens > 0 else 0.0

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=0.0,
            prompt_tps_source="ttft_estimate",
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
