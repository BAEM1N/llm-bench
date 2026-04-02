"""메인 벤치마크 실행기 — Generation + Prefill 2-Track."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .metrics import BenchmarkResult, append_result, aggregate_results, median_result
from .thermal import wait_for_cooldown, log_temp, log_power
from .prompt_gen import build_prefill_prompt, build_generation_prompt
from .backends.ollama import OllamaBackend
from .backends.lmstudio import LMStudioBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.mlx_backend import MLXBackend
from .backends.vllm_backend import VLLMBackend

console = Console()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_backend(name: str, cfg: dict):
    if name == "ollama":
        return OllamaBackend(base_url=cfg["base_url"])
    if name == "lmstudio":
        return LMStudioBackend(base_url=cfg["base_url"])
    if name == "llamacpp":
        return LlamaCppBackend(
            base_url=cfg["base_url"],
            binary=cfg.get("binary", "llama-server"),
            n_gpu_layers=cfg.get("n_gpu_layers", 99),
            flash_attn=cfg.get("flash_attn", True),
            batch_size=cfg.get("batch_size", 512),
            ubatch_size=cfg.get("ubatch_size", 512),
            threads=cfg.get("threads"),
            threads_batch=cfg.get("threads_batch"),
            mlock=cfg.get("mlock", False),
            extra_args=cfg.get("extra_args", []),
        )
    if name == "mlx":
        return MLXBackend()
    if name == "vllm":
        return VLLMBackend(
            base_url=cfg.get("base_url", "http://localhost:8000"),
            binary=cfg.get("binary", "vllm"),
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len"),
            quantization=cfg.get("quantization"),
            extra_args=cfg.get("extra_args", []),
        )
    raise ValueError(f"Unknown backend: {name}")


def run_one(
    backend, prompt: str, max_tokens: int, gen_cfg: dict
):
    return backend.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        repeat_penalty=gen_cfg["repeat_penalty"],
    )


def run_track(
    backend,
    backend_name: str,
    model_cfg: dict,
    quant_cfg: dict,
    track: dict,
    track_type: str,
    bench_cfg: dict,
    thermal_cfg: dict,
    gen_cfg: dict,
    hardware_id: str,
    output_path: Path,
    model_memory_gb: float = 0.0,
    comparison_mode: str = "A",
) -> list[BenchmarkResult]:

    track_id = track["id"]
    max_tokens = track["max_tokens"]
    input_tokens = track["input_tokens"]

    # 프롬프트 생성 — 트랙 타입 무관하게 input_tokens로 길이 제어
    if track_type == "prefill":
        prompt = build_prefill_prompt(input_tokens)
    else:
        prompt = build_generation_prompt(input_tokens, max_tokens)

    # Ollama: full context 유지 (전 백엔드 동일 ctx)
    if backend_name == "ollama":
        backend._context_window = gen_cfg["context_window"]

    console.print(f"\n  [dim]{track_id}[/dim]  in≈{input_tokens} out={max_tokens}", end=" ")

    # 워밍업 (bench_cfg["warmup_runs"]회)
    for _ in range(bench_cfg.get("warmup_runs", 1)):
        try:
            run_one(backend, prompt, max_tokens, gen_cfg)
            console.print("🔥", end=" ")
        except Exception as e:
            console.print(f"[red]워밍업 실패: {e}[/red]")
            return []

    run_results = []

    for run_num in range(1, bench_cfg["measure_runs"] + 1):
        # 온도 체크
        if thermal_cfg.get("enabled"):
            wait_for_cooldown(
                thermal_cfg["max_temp_celsius"],
                thermal_cfg["check_interval"],
                thermal_cfg["cooldown_wait"],
                console,
            )

        temp_info = log_temp()
        power_info = log_power()

        try:
            result = run_one(backend, prompt, max_tokens, gen_cfg)
        except Exception as e:
            console.print(f"[red]Run {run_num} 실패: {e}[/red]")
            continue

        # backend native prefill_tps 우선, 없으면 input_tokens / TTFT 폴백
        if result.prompt_tps > 0:
            prefill_tps = result.prompt_tps
            prefill_tps_source = result.prompt_tps_source
        else:
            prefill_tps = (input_tokens / (result.ttft_ms / 1000)) if result.ttft_ms > 0 else 0.0
            prefill_tps_source = "ttft_estimate"

        # hit_rate = 생성 완주율 (output_tokens / max_tokens). 품질 지표가 아님.
        # hit_rate > 0.9: numbered format이 EOS 억제에 효과적으로 작동.
        # hit_rate < 0.5: 모델이 일찍 종료 → 결과 신뢰도 주의.
        # 반복/루프/저품질 장문도 hit_rate 높게 나올 수 있음.
        hit_rate = round(result.output_tokens / max_tokens, 4) if track_type == "generation" else -1.0

        total_power_w = power_info["total_power_w"]
        efficiency = round(result.gen_tps / total_power_w, 4) if total_power_w > 0 else -1.0

        bench_result = BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            hardware_id=hardware_id,
            comparison_mode=comparison_mode,
            backend=backend_name,
            backend_version=backend.version,
            model=model_cfg["id"],
            architecture=model_cfg["architecture"],
            total_params=model_cfg["total_params"],
            active_params=model_cfg["active_params"],
            quantization=quant_cfg["name"],
            track_id=track_id,
            track_type=track_type,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            run_number=run_num,
            ttft_ms=result.ttft_ms,
            prefill_tps=round(prefill_tps, 2),
            gen_tps=result.gen_tps,
            total_latency_s=result.total_latency_s,
            output_tokens=result.output_tokens,
            hit_rate=hit_rate,
            peak_memory_gb=model_memory_gb,
            cpu_temp_celsius=temp_info.get("cpu_temp_celsius") or -1,
            context_window=gen_cfg["context_window"],
            prefill_tps_source=prefill_tps_source,
            actual_runs=0,  # 루프 완료 후 일괄 업데이트
            cpu_power_w=power_info["cpu_power_w"],
            gpu_power_w=power_info["gpu_power_w"],
            efficiency_tps_per_w=efficiency,
            peak_memory_method=backend.memory_method,
            schema_version="2",
        )

        append_result(bench_result, output_path)
        run_results.append(bench_result)

        src_marker = "" if prefill_tps_source == "native" else "[yellow]~[/yellow]"
        console.print(
            f"[{run_num}] TTFT={result.ttft_ms:.0f}ms "
            f"Prefill={prefill_tps:.0f}{src_marker}tok/s "
            f"Gen={result.gen_tps:.1f}tok/s "
            f"Temp={temp_info.get('cpu_temp_celsius') or '?'}°C",
            end="  ",
        )

        if run_num < bench_cfg["measure_runs"]:
            time.sleep(bench_cfg["inter_run_sleep"])

    actual = len(run_results)
    target = bench_cfg["measure_runs"]

    if actual < target:
        console.print(
            f"\n  [yellow]⚠ {actual}/{target} runs 성공 — 통계 신뢰도 낮음[/yellow]"
        )

    if run_results:
        # actual_runs 소급 기록 (CSV는 이미 append됐으므로 in-memory만 갱신)
        for r in run_results:
            object.__setattr__(r, "actual_runs", actual) if hasattr(r, "__dataclass_fields__") else None

        agg = aggregate_results(run_results)
        hit = f"  hit={agg['hit_rate']:.0%}" if track_type == "generation" else ""
        n_note = f" (n={actual})" if actual < target else ""
        console.print(
            f"\n  [green]중앙값 → Prefill={agg['prefill_tps']:.0f}tok/s  "
            f"Gen={agg['gen_tps']:.1f}tok/s (p95={agg['gen_tps_p95']:.1f})  "
            f"TTFT={agg['ttft_ms']:.0f}ms (p95={agg['ttft_p95_ms']:.0f}ms)"
            f"{hit}{n_note}[/green]"
        )

    return run_results


def run_benchmark(config: dict, backends: list[str], models: list[str], output_path: Path) -> None:
    gen_cfg = config["generation"]
    bench_cfg = config["benchmark"]
    thermal_cfg = config.get("thermal", {"enabled": False})
    hardware_id = config.get("hardware", {}).get("id", "unknown")
    comparison_mode = config.get("comparison_mode", "A")

    gen_tracks = config.get("generation_tracks", [])
    prefill_tracks = config.get("prefill_tracks", [])
    all_results = []

    # Mode B: llamacpp 단일 엔진으로 강제 필터링
    if comparison_mode == "B":
        non_llama = [b for b in backends if b != "llamacpp"]
        if non_llama:
            console.print(
                f"[yellow]⚠ Mode B: {non_llama} 자동 제외 — "
                "normalized engine 비교는 llamacpp + 동일 GGUF만 허용[/yellow]"
            )
            backends = [b for b in backends if b == "llamacpp"]
        if not backends:
            console.print("[red]Mode B: llamacpp 백엔드가 없음 — 실험 중단[/red]")
            return

    for backend_name in backends:
        bcfg = config["backends"].get(backend_name, {})
        if not bcfg.get("enabled", True):
            console.print(f"[yellow]Skip {backend_name}[/yellow]")
            continue

        backend = make_backend(backend_name, bcfg)
        console.rule(f"[bold cyan]{backend_name.upper()} (v{backend.version})")

        for model_cfg in config["models"]:
            if models and model_cfg["id"] not in models:
                continue

            for quant_cfg in model_cfg["quantizations"]:
                # 백엔드별 모델 식별자
                if backend_name == "mlx":
                    model_ref = quant_cfg["mlx_model"]
                    if not model_ref:
                        console.print(f"[yellow]Skip MLX: {model_cfg['id']} {quant_cfg['name']} (mlx_model 미설정)[/yellow]")
                        continue
                elif backend_name == "ollama":
                    model_ref = quant_cfg["ollama_model"]
                elif backend_name == "lmstudio":
                    model_ref = quant_cfg["lmstudio_model"]
                elif backend_name == "vllm":
                    model_ref = quant_cfg.get("vllm_model") or quant_cfg.get("gguf_path", "")
                    if not model_ref:
                        console.print(f"[yellow]Skip vLLM: {model_cfg['id']} {quant_cfg['name']} (vllm_model 미설정)[/yellow]")
                        continue
                else:
                    model_ref = quant_cfg["gguf_path"]

                console.print(f"\n[bold]{model_cfg['id']}[/bold] [{quant_cfg['name']}]")

                # 전 백엔드 동일 full context (공정 비교)
                load_ctx = gen_cfg["context_window"]

                # 모델 로드
                try:
                    with Progress(SpinnerColumn(), TextColumn("모델 로드 중..."), TimeElapsedColumn(), transient=True) as p:
                        p.add_task("")
                        backend.load_model(model_ref, quant_cfg["gguf_path"], load_ctx)
                except Exception as e:
                    console.print(f"[red]로드 실패: {e}[/red]")
                    try:
                        backend.unload_model()
                    except Exception:
                        pass
                    continue

                model_memory_gb = backend.get_model_memory_gb()
                if model_memory_gb > 0:
                    console.print(f"  [dim]모델 메모리: {model_memory_gb:.2f} GB[/dim]")

                # Generation tracks
                console.print("  [bold yellow]── Generation ──[/bold yellow]")
                for track in gen_tracks:
                    results = run_track(
                        backend, backend_name, model_cfg, quant_cfg,
                        track, "generation",
                        bench_cfg, thermal_cfg, gen_cfg,
                        hardware_id, output_path, model_memory_gb,
                        comparison_mode,
                    )
                    all_results.extend(results)
                    time.sleep(bench_cfg["inter_track_sleep"])

                # Prefill tracks
                console.print("  [bold blue]── Prefill ──[/bold blue]")
                for track in prefill_tracks:
                    results = run_track(
                        backend, backend_name, model_cfg, quant_cfg,
                        track, "prefill",
                        bench_cfg, thermal_cfg, gen_cfg,
                        hardware_id, output_path, model_memory_gb,
                        comparison_mode,
                    )
                    all_results.extend(results)
                    time.sleep(bench_cfg["inter_track_sleep"])

                backend.unload_model()
                time.sleep(bench_cfg["inter_model_sleep"])

        time.sleep(bench_cfg["inter_backend_sleep"])

    _print_summary(all_results)
    console.print(f"\n[bold green]결과 저장: {output_path}[/bold green]")


def _print_summary(results: list[BenchmarkResult]) -> None:
    if not results:
        return
    console.rule("[bold]최종 요약")

    # 중앙값 집계
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r.backend, r.model, r.quantization, r.track_id)].append(r)

    table = Table(show_header=True, header_style="bold magenta")
    left = ["Backend", "Model", "Quant", "Track", "Mode"]
    for col in left + ["TTFT p50/p95(ms)", "Prefill TPS", "Gen p50/p95(tok/s)", "Lat p95(s)", "Hit%", "Mem(GB)"]:
        table.add_column(col, justify="left" if col in left else "right")

    for key, rs in sorted(groups.items()):
        agg = aggregate_results(rs)
        r = rs[0]
        hit = f"{agg['hit_rate']:.0%}" if r.track_type == "generation" else "—"
        table.add_row(
            r.backend, r.model, r.quantization, r.track_id, r.comparison_mode,
            f"{agg['ttft_ms']:.0f} / {agg['ttft_p95_ms']:.0f}",
            f"{agg['prefill_tps']:.0f}",
            f"{agg['gen_tps']:.1f} / {agg['gen_tps_p95']:.1f}",
            f"{agg['latency_p95_s']:.1f}",
            hit,
            f"{agg['peak_memory_gb']:.1f}",
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark — Generation + Prefill")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--backends", nargs="+",
                        default=["ollama", "llamacpp", "mlx"])
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--tracks", nargs="+", default=[],
                        help="특정 track만 실행 (e.g. gen-512 prefill-4k)")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # track 필터
    if args.tracks:
        config["generation_tracks"] = [t for t in config["generation_tracks"] if t["id"] in args.tracks]
        config["prefill_tracks"] = [t for t in config["prefill_tracks"] if t["id"] in args.tracks]

    output_path = Path(args.output) if args.output else (
        Path("results") / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    output_path.parent.mkdir(exist_ok=True)

    hw = config.get("hardware", {})
    console.print(f"[bold]LLM Benchmark[/bold] — {hw.get('description', '')}")
    console.print(f"출력 → {output_path}")
    console.print(f"Generation tracks: {[t['id'] for t in config['generation_tracks']]}")
    console.print(f"Prefill tracks:    {[t['id'] for t in config['prefill_tracks']]}")
    console.print(f"Thermal guard: {config.get('thermal', {}).get('enabled', False)}\n")

    run_benchmark(config, args.backends, args.models, output_path)


if __name__ == "__main__":
    main()
