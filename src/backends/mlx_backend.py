import time

from .base import BaseBackend, GenerateResult


class MLXBackend(BaseBackend):
    name = "mlx"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_path: str = ""
        self._context_window: int = 262144
        self.version = self.get_version()

    def get_version(self) -> str:
        try:
            import mlx_lm
            return getattr(mlx_lm, "__version__", "unknown")
        except ImportError:
            return "not installed"

    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        from mlx_lm import load
        self._model_path = model_id  # MLX는 HuggingFace model ID 사용
        self._context_window = context_window
        self._model, self._tokenizer = load(model_id)

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        # MLX 메모리 해제
        try:
            import mlx.core as mx
            mx.clear_memory_pool()
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
        from mlx_lm import stream_generate

        t_start = time.perf_counter()
        t_first = None
        output_tokens = 0
        prompt_tokens = 0

        # 프롬프트 토큰 수 사전 계산
        if self._tokenizer:
            encoded = self._tokenizer.encode(prompt)
            prompt_tokens = len(encoded) if isinstance(encoded, list) else encoded.shape[0]

        for token in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            output_tokens += 1

        t_end = time.perf_counter()
        t_first = t_first or t_end

        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start
        gen_duration = t_end - t_first

        gen_tps = output_tokens / gen_duration if gen_duration > 0 else 0.0
        prompt_duration = (t_first - t_start)
        prompt_tps = prompt_tokens / prompt_duration if prompt_duration > 0 else 0.0

        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
