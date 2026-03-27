from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import csv
import statistics


@dataclass
class BenchmarkResult:
    # 실험 메타
    timestamp: str
    hardware_id: str
    # 백엔드
    backend: str
    backend_version: str
    # 모델
    model: str
    architecture: str
    total_params: str
    active_params: str
    quantization: str
    # 트랙
    track_id: str       # gen-512, gen-2048, prefill-4k 등
    track_type: str     # generation | prefill
    input_tokens: int
    max_tokens: int
    # 측정값
    run_number: int
    ttft_ms: float          # Time to First Token
    prefill_tps: float      # prompt tokens / TTFT
    gen_tps: float          # generated tokens / gen time
    total_latency_s: float
    output_tokens: int
    peak_memory_gb: float
    cpu_temp_celsius: float # 측정 불가 시 -1
    context_window: int


CSV_FIELDS = [f.name for f in BenchmarkResult.__dataclass_fields__.values()]


def append_result(result: BenchmarkResult, path: Path) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(result))


def median_result(results: list[BenchmarkResult]) -> dict:
    if not results:
        return {}
    return {
        "ttft_ms":        statistics.median(r.ttft_ms for r in results),
        "prefill_tps":    statistics.median(r.prefill_tps for r in results),
        "gen_tps":        statistics.median(r.gen_tps for r in results),
        "total_latency_s":statistics.median(r.total_latency_s for r in results),
        "output_tokens":  statistics.median(r.output_tokens for r in results),
        "peak_memory_gb": statistics.median(r.peak_memory_gb for r in results),
    }
