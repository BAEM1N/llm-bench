import json
import time
import httpx

from .base import BaseBackend, GenerateResult


class LMStudioBackend(BaseBackend):
    """LM Studio — OpenAI-compatible API. 모델 로드는 앱에서 수동으로."""
    name = "lmstudio"

    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self._model_id: str = ""
        self._context_window: int = 262144
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=5)
            # LM Studio는 별도 버전 엔드포인트 없음
            return "lmstudio"
        except Exception:
            return "lmstudio"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        """LM Studio는 GUI에서 모델 로드. model_id만 기록."""
        self._model_id = model_id
        self._context_window = context_window

    def unload_model(self) -> None:
        self._model_id = ""

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
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": 0.0,
            "stream": True,
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        buffer = ""

        with httpx.Client(timeout=1800) as client:
            with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        buffer += delta
                        output_tokens += 1  # 청크당 1토큰 근사

                    # finish_reason 확인
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", output_tokens)

        t_end = time.perf_counter()
        t_first = t_first or t_end

        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start
        gen_duration = t_end - t_first
        gen_tps = output_tokens / gen_duration if gen_duration > 0 else 0.0

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=0.0,
            prompt_tps_source="ttft_estimate",  # LM Studio는 prefill TPS 미제공
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
