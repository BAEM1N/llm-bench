import json
import time
import httpx
import subprocess

from .base import BaseBackend, GenerateResult


class OllamaBackend(BaseBackend):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._model_id: str = ""
        self._context_window: int = 262144
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/api/version", timeout=5)
            return r.json().get("version", "unknown")
        except Exception:
            try:
                out = subprocess.check_output(["ollama", "--version"], text=True)
                return out.strip().split()[-1]
            except Exception:
                return "unknown"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        """Ollama는 pull로 미리 받아두고, API 호출 시 모델명으로 지정."""
        self._model_id = model_id
        self._context_window = context_window
        # 모델 로드 워밍업 (첫 로드 시 메모리 올라옴)
        self._ping_model()

    def unload_model(self) -> None:
        # Ollama는 자동 메모리 관리, 명시적 언로드는 불필요
        self._model_id = ""

    def _ping_model(self) -> None:
        """모델을 메모리에 올리기 위한 더미 요청."""
        try:
            httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self._model_id, "prompt": "", "keep_alive": "10m"},
                timeout=120,
            )
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> GenerateResult:
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_predict": max_tokens,
                "num_ctx": self._context_window,
            },
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        prompt_tokens = 0
        prompt_eval_duration_ns = 0
        eval_duration_ns = 0

        with httpx.Client(timeout=300) as client:
            with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if t_first is None and chunk.get("response"):
                        t_first = time.perf_counter()

                    if chunk.get("done"):
                        output_tokens = chunk.get("eval_count", 0)
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                        break

        t_end = time.perf_counter()
        t_first = t_first or t_end

        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start

        # Ollama가 직접 제공하는 정확한 타이밍 값 사용
        if eval_duration_ns > 0 and output_tokens > 0:
            gen_tps = output_tokens / (eval_duration_ns / 1e9)
        else:
            gen_duration = t_end - t_first
            gen_tps = output_tokens / gen_duration if gen_duration > 0 else 0.0

        if prompt_eval_duration_ns > 0 and prompt_tokens > 0:
            prompt_tps = prompt_tokens / (prompt_eval_duration_ns / 1e9)
        else:
            prompt_tps = 0.0

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
