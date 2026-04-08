"""vLLM backend — NVIDIA CUDA (3090 2-WAY / DGX Spark) / AMD ROCm (Ryzen AI 395) 용.

vLLM 서버를 서브프로세스로 실행하거나 외부에서 미리 실행된 서버에 연결.
AWQ / GPTQ / GGUF 양자화 지원.

GGUF 사용 시: vllm_model = GGUF 파일 경로, vllm_quantization = "gguf"
  → --load-format gguf 자동 추가. 별도 HF 다운로드 불필요.
  단, GGUF + tensor-parallel 조합은 vLLM에서 미지원 → tensor_parallel_size=1 필요.

AWQ 사용 시: vllm_model = HF 모델 ID (e.g. Qwen/Qwen3.5-9B-Instruct-AWQ)
  → HF에서 다운로드 필요. tensor-parallel 지원.
"""
import json
import os
import time
import subprocess
from urllib.parse import urlparse
import httpx
from typing import Optional

from .base import BaseBackend, GenerateResult

STARTUP_TIMEOUT = 1800  # seconds — vLLM 0.19+ torch.compile + CUDA graph 캡처에 최대 30분


class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        binary: str = "vllm",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,  # "awq", "gptq", "gguf", None
        cpu_offload_gb: float = 0.0,         # VRAM 부족 시 CPU RAM 오프로드 (GB)
        extra_args: Optional[list] = None,
    ):
        self.base_url = base_url
        self.binary = binary
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.cpu_offload_gb = cpu_offload_gb
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
        """vLLM 서버를 서브프로세스로 실행하거나 외부에서 이미 실행된 서버에 연결."""
        self._model_id = model_id
        self._context_window = context_window

        # 외부 서버가 이미 떠있으면 그냥 연결 (Docker 등)
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=5)
            if r.status_code == 200:
                self._proc = None
                self._model_memory_gb = self._read_gpu_memory_gb()
                return
        except Exception:
            pass

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = str(parsed.port or 8000)

        cmd = [
            self.binary, "serve", model_id,
            "--port", port,
            "--host", host,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len or context_window),
            "--no-enable-log-requests",
            "--no-enable-prefix-caching",     # prefix cache 비활성화 (cold prefill 측정)
        ]
        if self.quantization == "gguf":
            # GGUF 파일 직접 로드. tensor-parallel은 미지원 → tensor_parallel_size=1 권장.
            cmd += ["--load-format", "gguf"]
        elif self.quantization:
            cmd += ["--quantization", self.quantization]
            # GPTQ/AWQ/GPTQ-Marlin은 float16 필요 (bfloat16 미지원)
            if self.quantization in ("gptq", "gptq_marlin", "awq"):
                cmd += ["--dtype", "half"]
        if self.cpu_offload_gb > 0:
            # VRAM 초과 모델을 CPU RAM으로 오프로드.
            # 3090 2-WAY (48GB) + 122B Q4 (~65GB): --cpu-offload-gb 20 권장.
            cmd += ["--cpu-offload-gb", str(self.cpu_offload_gb)]
        if self.extra_args:
            cmd.extend(self.extra_args)

        self._server_log = open("/tmp/vllm_server.log", "w")
        # CUDA 라이브러리 경로 보장 (DGX Spark CUDA 13 + vLLM CUDA 12 호환)
        env = dict(os.environ)
        cuda_paths = "/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu"
        env["LD_LIBRARY_PATH"] = cuda_paths + ":" + env.get("LD_LIBRARY_PATH", "")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._server_log,
            stderr=self._server_log,
            env=env,
        )

        # 서버 준비 대기
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            # 프로세스가 이미 죽었으면 즉시 실패
            if self._proc.poll() is not None:
                self._server_log.close()
                with open("/tmp/vllm_server.log") as f:
                    tail = f.read()[-500:]
                raise RuntimeError(f"vLLM server exited (code={self._proc.returncode}): {tail}")
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

    def get_effective_context(self) -> int:
        """vLLM은 max_model_len이 실제 제약."""
        return self.max_model_len or self._context_window

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

        with httpx.Client(timeout=1800) as client:  # gen-8192 + 큰 모델: 최대 30분
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
