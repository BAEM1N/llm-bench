"""Lemonade Server backend — AMD Ryzen AI 395+ (Strix Halo) 전용.

AMD 공식 로컬 추론 서버. 내부적으로 llama.cpp (Vulkan/ROCm) 사용.
OpenAI-compatible API: /api/v1/completions, /api/v1/chat/completions.
모델 관리 API: /api/v1/load, /api/v1/unload.

설치: deb/rpm/snap 또는 소스 빌드 (C++ 네이티브).
실행: lemonade-server serve --port 8000 --llamacpp rocm
"""
import json
import time
import httpx

from .base import BaseBackend, GenerateResult

# Lemonade는 외부에서 미리 실행. 서버 자체를 서브프로세스로 띄우지 않음.
# → `lemonade-server serve --llamacpp rocm --port 8000` 으로 미리 시작 필요.


class LemonadeBackend(BaseBackend):
    name = "lemonade"

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._model_id: str = ""
        self._context_window: int = 262144
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            r = httpx.get(f"{self.base_url}/api/v1/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return data.get("version", "lemonade")
            return "lemonade"
        except Exception:
            return "lemonade"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        """Lemonade /api/v1/load 로 모델 로드."""
        self._model_id = model_id
        self._context_window = context_window

        # Lemonade load API 시도
        try:
            payload = {"model": model_id}
            r = httpx.post(
                f"{self.base_url}/api/v1/load",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
        except Exception:
            # load API 실패 시 이미 로드된 것으로 간주 (수동 로드)
            pass

        # 서버 응답 확인
        try:
            r = httpx.get(f"{self.base_url}/api/v1/models", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Lemonade server not responding: {e}")

    def unload_model(self) -> None:
        try:
            httpx.post(
                f"{self.base_url}/api/v1/unload",
                json={"model": self._model_id},
                timeout=30,
            )
        except Exception:
            pass
        self._model_id = ""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
    ) -> GenerateResult:
        """OpenAI-compatible /api/v1/completions 스트리밍."""
        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "stream": True,
        }

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        token_count = 0

        with httpx.Client(timeout=600) as client:
            with client.stream(
                "POST", f"{self.base_url}/api/v1/completions", json=payload
            ) as resp:
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
        gen_tps = (
            output_tokens / gen_duration
            if gen_duration > 0 and output_tokens > 0
            else 0.0
        )

        # Lemonade /api/v1/stats 에서 성능 통계 가져오기 (있으면)
        prompt_tps = 0.0
        prompt_tps_source = "ttft_estimate"
        try:
            sr = httpx.get(f"{self.base_url}/api/v1/stats", timeout=5)
            if sr.status_code == 200:
                stats = sr.json()
                pp_tps = stats.get("prompt_tps", 0.0)
                if pp_tps > 0:
                    prompt_tps = pp_tps
                    prompt_tps_source = "native"
        except Exception:
            pass

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            prompt_tps_source=prompt_tps_source,
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
