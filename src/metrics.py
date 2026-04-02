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
    comparison_mode: str    # A=practical stack, B=normalized engine, C=concurrency
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
    track_id: str           # gen-512, gen-2048, prefill-4k 등
    track_type: str         # generation | prefill
    input_tokens: int
    max_tokens: int
    # 측정값
    run_number: int
    ttft_ms: float          # Time to First Token
    prefill_tps: float      # prompt tokens / TTFT
    gen_tps: float          # generated tokens / gen time
    total_latency_s: float
    output_tokens: int
    hit_rate: float         # output_tokens / max_tokens (gen 트랙), prefill은 -1
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


def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    rank = max(1, int(len(s) * p / 100 + 0.5))
    return s[min(rank - 1, len(s) - 1)]


def aggregate_results(results: list) -> dict:
    """median + p95 집계. runner._print_summary 및 run_track에서 사용."""
    if not results:
        return {}

    def med(key):
        return statistics.median(getattr(r, key) for r in results)

    def p95(key):
        return _percentile([getattr(r, key) for r in results], 95)

    return {
        "ttft_ms":          med("ttft_ms"),
        "ttft_p95_ms":      p95("ttft_ms"),
        "prefill_tps":      med("prefill_tps"),
        "gen_tps":          med("gen_tps"),
        "gen_tps_p95":      p95("gen_tps"),
        "total_latency_s":  med("total_latency_s"),
        "latency_p95_s":    p95("total_latency_s"),
        "output_tokens":    med("output_tokens"),
        "hit_rate":         med("hit_rate"),
        "peak_memory_gb":   med("peak_memory_gb"),
        "cpu_temp_celsius": med("cpu_temp_celsius"),
    }


# 하위 호환 alias
def median_result(results: list) -> dict:
    return aggregate_results(results)
