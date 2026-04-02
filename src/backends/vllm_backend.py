"""vLLM backend — DGX Spark (CUDA) / Ryzen AI 395 (ROCm) 용.

vLLM 서버를 서브프로세스로 실행하거나 외부에서 미리 실행된 서버에 연결.
AWQ / GPTQ 양자화 지원.

TODO: 실험 환경 세팅 후 구현 완료
"""
import json
import time
import subprocess
import httpx
from typing import Optional

from .base import BaseBackend, GenerateResult

STARTUP_TIMEOUT = 300  # seconds


class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        binary: str = "vllm",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,  # "awq", "gptq", None
        extra_args: Optional[list] = None,
    ):
        self.base_url = base_url
        self.binary = binary
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.extra_args = extra_args or []
        self._model_id: str = ""
        self._context_window: int = 262144
        self._proc: Optional[subprocess.Popen] = None
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            out = subprocess.check_output(
                [self.binary, "--version"], text=True, stderr=subprocess.STDOUT
            )
            return out.strip().splitlines()[0]
        except Exception:
            return "unknown"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        """vLLM 서버를 서브프로세스로 실행하거나 외부 서버에 연결."""
        self._model_id = model_id
        self._context_window = context_window

        cmd = [
            self.binary, "serve", model_id,
            "--port", "8000",
            "--host", "127.0.0.1",
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len or context_window),
            "--disable-log-requests",
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
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
                    self._model_memory_gb = self._read_gpu_memory_gb()
                    return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"vLLM server did not start within {STARTUP_TIMEOUT}s")

    def _read_gpu_memory_gb(self) -> float:
        """nvidia-smi 또는 rocm-smi로 GPU 메모리 사용량 측정."""
        # NVIDIA
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            total_mb = sum(int(line.strip()) for line in out.strip().splitlines() if line.strip())
            return round(total_mb / 1024, 2)
        except Exception:
            pass
        # AMD ROCm
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                text=True, stderr=subprocess.DEVNULL,
            )
            data = json.loads(out)
            total_bytes = sum(
                int(v.get("VRAM Total Used Memory (B)", 0))
                for v in data.values()
                if isinstance(v, dict)
            )
            return round(total_bytes / 1024**3, 2)
        except Exception:
            return 0.0

    def get_model_memory_gb(self) -> float:
        return getattr(self, "_model_memory_gb", 0.0)

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

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> GenerateResult:
        """OpenAI-compatible /v1/completions endpoint 스트리밍."""
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "stream": True,
            # 마지막 chunk에 usage 포함 요청 (vLLM 0.4.2+)
            "stream_options": {"include_usage": True},
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        token_count = 0  # stream_options 미지원 시 count 기반 폴백

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
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            if t_first is None:
                                t_first = time.perf_counter()
                            token_count += 1
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)

        # stream_options 미지원 구버전 폴백
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
            prompt_tps_source="ttft_estimate",  # vLLM은 prefill TPS 미제공
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
