import time
from pathlib import Path

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
        import mlx.core as mx
        from mlx_lm import load
        expanded = str(Path(model_id).expanduser())
        model_ref = expanded if Path(expanded).exists() else model_id
        self._model_path = model_ref
        self._context_window = context_window
        mx.metal.reset_peak_memory()
        self._model, self._tokenizer = load(model_ref)
        # load() 완료 후 Metal peak = 모델 가중치 메모리
        self._model_memory_gb = round(mx.metal.get_peak_memory() / 1024**3, 2)

    def get_model_memory_gb(self) -> float:
        return getattr(self, "_model_memory_gb", 0.0)

    @property
    def memory_method(self) -> str:
        return "metal_peak"

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
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1.0 else 0.0)

        t_start = time.perf_counter()
        t_first = None
        last_resp = None

        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            if t_first is None:
                t_first = time.perf_counter()
            last_resp = resp

        t_end = time.perf_counter()
        t_first = t_first or t_end

        ttft_ms = (t_first - t_start) * 1000
        total_latency_s = t_end - t_start

        gen_tps = last_resp.generation_tps if last_resp else 0.0
        prompt_tps = last_resp.prompt_tps if last_resp else 0.0
        output_tokens = last_resp.generation_tokens if last_resp else 0

        source = "native" if prompt_tps > 0 else "ttft_estimate"
        return GenerateResult(
            ttft_ms=round(ttft_ms, 2),
            gen_tps=round(gen_tps, 2),
            prompt_tps=round(prompt_tps, 2),
            prompt_tps_source=source,
            total_latency_s=round(total_latency_s, 3),
            output_tokens=output_tokens,
        )
