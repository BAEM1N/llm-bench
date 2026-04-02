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
    prompt_tps_source: str = "native"  # "native" | "ttft_estimate"


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

    def get_model_memory_gb(self) -> float:
        """모델 로드 후 점유 메모리 반환 (GB). 측정 불가 시 0.0."""
        return 0.0

    @property
    def memory_method(self) -> str:
        """메모리 측정 방식 식별자. 서브클래스에서 오버라이드."""
        return "unknown"

    @abstractmethod
    def get_version(self) -> str:
        """백엔드 버전 반환."""
