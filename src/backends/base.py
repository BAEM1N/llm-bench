from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GenerateResult:
    ttft_ms: float          # Time to first token (ms)
    gen_tps: float          # Generation tokens/sec
    prompt_tps: float       # Prompt processing tokens/sec
    total_latency_s: float  # End-to-end latency (s)
    output_tokens: int


class BaseBackend(ABC):
    name: str = ""
    version: str = ""

    @abstractmethod
    def load_model(self, model_id: str, gguf_path: str, context_window: int) -> None:
        """모델 로드. 서버 시작 포함."""

    @abstractmethod
    def unload_model(self) -> None:
        """모델 언로드. 서버 종료."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
    ) -> GenerateResult:
        """추론 실행 및 타이밍 측정."""

    @abstractmethod
    def get_version(self) -> str:
        """백엔드 버전 반환."""
