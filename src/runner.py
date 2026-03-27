"""메인 벤치마크 실행기 — Generation + Prefill 2-Track."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .metrics import BenchmarkResult, append_result, median_result
from .memory import MemoryMonitor
from .thermal import wait_for_cooldown, log_temp
from .prompt_gen import build_prefill_prompt, build_generation_prompt
from .backends.ollama import OllamaBackend
from .backends.lmstudio import LMStudioBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.mlx_backend import MLXBackend

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
        )
    if name == "mlx":
        return MLXBackend()
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
) -> list[BenchmarkResult]:

    track_id = track["id"]
    max_tokens = track["max_tokens"]
    input_tokens = track["input_tokens"]

    # 프롬프트 생성
    if track_type == "prefill":
        prompt = build_prefill_prompt(input_tokens)
    else:
        prompt = build_generation_prompt(track_id)

    console.print(f"\n  [dim]{track_id}[/dim]  in≈{input_tokens} out={max_tokens}", end=" ")

    # 워밍업
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
        monitor = MemoryMonitor()
        monitor.start()

        try:
            result = run_one(backend, prompt, max_tokens, gen_cfg)
        except Exception as e:
            console.print(f"[red]Run {run_num} 실패: {e}[/red]")
            monitor.stop()
            continue
        finally:
            peak_mem = monitor.stop()

        # prefill_tps = prompt_tokens / TTFT
        prefill_tps = (input_tokens / (result.ttft_ms / 1000)) if result.ttft_ms > 0 else 0.0

        bench_result = BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            hardware_id=hardware_id,
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
            peak_memory_gb=peak_mem,
            cpu_temp_celsius=temp_info.get("cpu_temp_celsius") or -1,
            context_window=gen_cfg["context_window"],
        )

        append_result(bench_result, output_path)
        run_results.append(bench_result)

        console.print(
            f"[{run_num}] TTFT={result.ttft_ms:.0f}ms "
            f"Prefill={prefill_tps:.0f}tok/s "
            f"Gen={result.gen_tps:.1f}tok/s "
            f"Temp={temp_info.get('cpu_temp_celsius') or '?'}°C",
            end="  ",
        )

        if run_num < bench_cfg["measure_runs"]:
            time.sleep(bench_cfg["inter_run_sleep"])

    if run_results:
        med = median_result(run_results)
        console.print(
            f"\n  [green]중앙값 → Prefill={med['prefill_tps']:.0f}tok/s  "
            f"Gen={med['gen_tps']:.1f}tok/s  "
            f"TTFT={med['ttft_ms']:.0f}ms[/green]"
        )

    return run_results


def run_benchmark(config: dict, backends: list[str], models: list[str], output_path: Path) -> None:
    gen_cfg = config["generation"]
    bench_cfg = config["benchmark"]
    thermal_cfg = config.get("thermal", {"enabled": False})
    hardware_id = config.get("hardware", {}).get("id", "unknown")

    gen_tracks = config.get("generation_tracks", [])
    prefill_tracks = config.get("prefill_tracks", [])
    all_results = []

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
                else:
                    model_ref = quant_cfg["gguf_path"]

                console.print(f"\n[bold]{model_cfg['id']}[/bold] [{quant_cfg['name']}]")

                # 모델 로드
                try:
                    with Progress(SpinnerColumn(), TextColumn("모델 로드 중..."), TimeElapsedColumn(), transient=True) as p:
                        p.add_task("")
                        backend.load_model(model_ref, quant_cfg["gguf_path"], gen_cfg["context_window"])
                except Exception as e:
                    console.print(f"[red]로드 실패: {e}[/red]")
                    continue

                # Generation tracks
                console.print("  [bold yellow]── Generation ──[/bold yellow]")
                for track in gen_tracks:
                    results = run_track(
                        backend, backend_name, model_cfg, quant_cfg,
                        track, "generation",
                        bench_cfg, thermal_cfg, gen_cfg,
                        hardware_id, output_path,
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
                        hardware_id, output_path,
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
    for col in ["Backend", "Model", "Quant", "Track", "TTFT(ms)", "Prefill TPS", "Gen TPS", "Mem(GB)", "Temp(°C)"]:
        table.add_column(col, justify="right" if col not in ["Backend","Model","Quant","Track"] else "left")

    for key, rs in sorted(groups.items()):
        med = median_result(rs)
        r = rs[0]
        temp = f"{med.get('cpu_temp_celsius', -1):.0f}" if med.get("cpu_temp_celsius", -1) > 0 else "N/A"
        table.add_row(
            r.backend, r.model, r.quantization, r.track_id,
            f"{med['ttft_ms']:.0f}",
            f"{med['prefill_tps']:.0f}",
            f"{med['gen_tps']:.1f}",
            f"{med['peak_memory_gb']:.1f}",
            temp,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark — Generation + Prefill")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--backends", nargs="+",
                        default=["ollama", "lmstudio", "llamacpp", "mlx"])
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
