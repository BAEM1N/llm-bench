"""메인 벤치마크 실행기 — Generation + Prefill 2-Track.

v3: 실험 설계 개선
- OOM/skip → CSV 실패 row 기록
- Track A/B config 레벨 분리
- native context 강제 (줄이지 않음, OOM=실패 기록)
- run마다 prompt 재생성 (warmup과 measure 분리)
- model_id를 prompt generator에 전달 (tokenizer 정확화)
- 실행 순서 랜덤화
- cold prefill: prefill track마다 서버 재시작
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .metrics import BenchmarkResult, append_result, aggregate_results, median_result
from .thermal import wait_for_cooldown, log_temp, log_power
from .prompt_gen import build_prefill_prompt, build_generation_prompt, init_tokenizer
from .backends.ollama import OllamaBackend
from .backends.lmstudio import LMStudioBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.mlx_backend import MLXBackend
from .backends.vllm_backend import VLLMBackend
from .backends.sglang_backend import SGLangBackend
from .backends.lemonade_backend import LemonadeBackend

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
            cpu_offload_gb=cfg.get("cpu_offload_gb", 0.0),
            extra_args=cfg.get("extra_args", []),
        )
    if name == "sglang":
        return SGLangBackend(
            base_url=cfg.get("base_url", "http://localhost:30000"),
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len"),
            quantization=cfg.get("quantization"),
            cpu_offload_gb=cfg.get("cpu_offload_gb", 0.0),
            extra_args=cfg.get("extra_args", []),
        )
    if name == "lemonade":
        return LemonadeBackend(base_url=cfg.get("base_url", "http://localhost:8000"))
    raise ValueError(f"Unknown backend: {name}")


def run_one(backend, prompt: str, max_tokens: int, gen_cfg: dict):
    return backend.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        repeat_penalty=gen_cfg["repeat_penalty"],
    )


def _resolve_quant_label(backend_name: str, quant_cfg: dict) -> str:
    """vLLM/SGLang은 GPTQ/AWQ 모델을 사용하므로 실제 양자화 방식을 CSV에 기록."""
    if backend_name in ("vllm", "sglang"):
        vllm_q = quant_cfg.get("vllm_quantization")
        if vllm_q:
            return f"{quant_cfg['name']}({vllm_q})"
        vllm_model = quant_cfg.get("vllm_model", "")
        if "GPTQ" in vllm_model:
            return f"{quant_cfg['name']}(gptq)"
        if "AWQ" in vllm_model:
            return f"{quant_cfg['name']}(awq)"
        if vllm_model:
            return f"{quant_cfg['name']}(bf16)"
    return quant_cfg["name"]


def _make_skip_row(
    backend_name, backend_version, model_cfg, quant_cfg, track, track_type,
    hardware_id, comparison_mode, model_ctx, reason, memory_method="unknown",
) -> BenchmarkResult:
    """OOM/skip/로드실패 → CSV에 실패 row 기록."""
    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        hardware_id=hardware_id,
        comparison_mode=comparison_mode,
        backend=backend_name,
        backend_version=backend_version,
        model=model_cfg["id"],
        architecture=model_cfg["architecture"],
        total_params=model_cfg["total_params"],
        active_params=model_cfg["active_params"],
        quantization=quant_cfg["name"],
        track_id=track["id"],
        track_type=track_type,
        input_tokens=track["input_tokens"],
        max_tokens=track["max_tokens"],
        run_number=0,
        run_status=f"skip:{reason}",
        ttft_ms=-1.0,
        prefill_tps=0.0,
        gen_tps=-1.0,
        total_latency_s=-1.0,
        output_tokens=0,
        hit_rate=-1.0,
        peak_memory_gb=0.0,
        cpu_temp_celsius=-1,
        context_window=model_ctx,
        prefill_tps_source="none",
        actual_runs=0,
        cpu_power_w=-1.0,
        gpu_power_w=-1.0,
        efficiency_tps_per_w=-1.0,
        peak_memory_method=memory_method,
        schema_version="3",
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
    model_id: str = "",
) -> list[BenchmarkResult]:

    track_id = track["id"]
    max_tokens = track["max_tokens"]
    input_tokens = track["input_tokens"]

    # 모델 네이티브 컨텍스트 (줄이지 않음)
    model_ctx = model_cfg.get("context_window", gen_cfg["context_window"])
    if backend_name == "ollama":
        # Ollama: num_ctx를 실제 필요 크기로 설정 (모델 전체 ctx → KV cache 메모리 초과 방지)
        # 필요 크기 = input + output + 여유분(256 tokens)
        needed_ctx = input_tokens + max_tokens + 256
        backend._context_window = min(model_ctx, needed_ctx)

    console.print(f"\n  [dim]{track_id}[/dim]  in≈{input_tokens} out={max_tokens}", end=" ")

    # #4: warmup은 별도 prompt (nonce로 매번 다름)
    for _ in range(bench_cfg.get("warmup_runs", 1)):
        try:
            warmup_prompt = (build_prefill_prompt(input_tokens, model_id)
                            if track_type == "prefill"
                            else build_generation_prompt(input_tokens, max_tokens, model_id))
            run_one(backend, warmup_prompt, max_tokens, gen_cfg)
            console.print("🔥", end=" ")
        except Exception as e:
            console.print(f"[red]워밍업 실패: {e}[/red]")
            return []

    pending: list[BenchmarkResult] = []
    run_results: list[BenchmarkResult] = []

    def _make_row(run_num: int, status: str, result=None,
                  prefill_tps=0.0, prefill_tps_source="ttft_estimate",
                  hit_rate=-1.0, temp_info=None, power_info=None) -> BenchmarkResult:
        ti = temp_info or {}
        pi = power_info or {"cpu_power_w": -1.0, "gpu_power_w": -1.0, "total_power_w": -1.0}
        total_power_w = pi["total_power_w"]
        efficiency = round((result.gen_tps / total_power_w), 4) if (
            result and total_power_w > 0) else -1.0
        return BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            hardware_id=hardware_id,
            comparison_mode=comparison_mode,
            backend=backend_name,
            backend_version=backend.version,
            model=model_cfg["id"],
            architecture=model_cfg["architecture"],
            total_params=model_cfg["total_params"],
            active_params=model_cfg["active_params"],
            quantization=_resolve_quant_label(backend_name, quant_cfg),
            track_id=track_id,
            track_type=track_type,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            run_number=run_num,
            run_status=status,
            ttft_ms=result.ttft_ms if result else -1.0,
            prefill_tps=round(prefill_tps, 2),
            gen_tps=result.gen_tps if result else -1.0,
            total_latency_s=result.total_latency_s if result else -1.0,
            output_tokens=result.output_tokens if result else 0,
            hit_rate=hit_rate,
            peak_memory_gb=model_memory_gb,
            cpu_temp_celsius=ti.get("cpu_temp_celsius") or -1,
            context_window=model_ctx,
            prefill_tps_source=prefill_tps_source,
            actual_runs=-1,
            cpu_power_w=pi["cpu_power_w"],
            gpu_power_w=pi["gpu_power_w"],
            efficiency_tps_per_w=efficiency,
            peak_memory_method=backend.memory_method,
            schema_version="3",
        )

    for run_num in range(1, bench_cfg["measure_runs"] + 1):
        if thermal_cfg.get("enabled"):
            wait_for_cooldown(
                thermal_cfg["max_temp_celsius"],
                thermal_cfg["check_interval"],
                thermal_cfg["cooldown_wait"],
                console,
            )

        temp_info = log_temp()
        power_info = log_power()

        # #4: 매 run마다 새 prompt 생성 (nonce prefix로 cache 완전 차단)
        if track_type == "prefill":
            prompt = build_prefill_prompt(input_tokens, model_id)
        else:
            prompt = build_generation_prompt(input_tokens, max_tokens, model_id)

        try:
            result = run_one(backend, prompt, max_tokens, gen_cfg)
        except Exception as e:
            console.print(f"[red]Run {run_num} 실패: {e}[/red]")
            pending.append(_make_row(run_num, "failed",
                                     temp_info=temp_info, power_info=power_info))
            continue

        if result.prompt_tps > 0:
            prefill_tps = result.prompt_tps
            prefill_tps_source = result.prompt_tps_source
        else:
            prefill_tps = (input_tokens / (result.ttft_ms / 1000)) if result.ttft_ms > 0 else 0.0
            prefill_tps_source = "ttft_estimate"

        hit_rate = round(result.output_tokens / max_tokens, 4) if track_type == "generation" else -1.0

        bench_result = _make_row(run_num, "ok", result,
                                 prefill_tps, prefill_tps_source,
                                 hit_rate, temp_info, power_info)
        pending.append(bench_result)
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
    for row in pending:
        object.__setattr__(row, "actual_runs", actual)
        append_result(row, output_path)

    if actual < target:
        console.print(
            f"\n  [yellow]⚠ {actual}/{target} runs 성공 — 통계 신뢰도 낮음[/yellow]"
        )

    if run_results:
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


def run_benchmark(config: dict, backends: list[str], models: list[str], output_path: Path, quants: list[str] | None = None) -> None:
    gen_cfg = config["generation"]
    bench_cfg = config["benchmark"]
    thermal_cfg = config.get("thermal", {"enabled": False})
    hardware_id = config.get("hardware", {}).get("id", "unknown")
    comparison_mode = config.get("comparison_mode", "A")

    gen_tracks = list(config.get("generation_tracks", []))
    prefill_tracks = list(config.get("prefill_tracks", []))
    all_results = []

    # 실행 순서 랜덤화
    random.shuffle(gen_tracks)
    random.shuffle(prefill_tracks)
    console.print(f"[dim]Randomized gen order: {[t['id'] for t in gen_tracks]}[/dim]")
    console.print(f"[dim]Randomized prefill order: {[t['id'] for t in prefill_tracks]}[/dim]")

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

    # backend / model 순서 랜덤화
    random.shuffle(backends)
    model_list = list(config["models"])
    random.shuffle(model_list)
    config["models"] = model_list
    console.print(f"[dim]Randomized backends: {backends}[/dim]")
    console.print(f"[dim]Randomized models: {[m['id'] for m in model_list]}[/dim]")

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

            # #5: tokenizer 초기화 (model_id 전달)
            model_id = model_cfg.get("id", "")

            for quant_cfg in model_cfg["quantizations"]:
                if quants and quant_cfg["name"] not in quants:
                    continue
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
                elif backend_name == "sglang":
                    model_ref = quant_cfg.get("sglang_model") or quant_cfg.get("vllm_model", "")
                    if not model_ref:
                        console.print(f"[yellow]Skip SGLang: {model_cfg['id']} {quant_cfg['name']} (sglang_model 미설정)[/yellow]")
                        continue
                elif backend_name == "lemonade":
                    model_ref = quant_cfg.get("lemonade_model") or quant_cfg.get("gguf_path", "")
                    if not model_ref:
                        console.print(f"[yellow]Skip Lemonade: {model_cfg['id']} {quant_cfg['name']} (lemonade_model 미설정)[/yellow]")
                        continue
                else:
                    model_ref = quant_cfg["gguf_path"]

                console.print(f"\n[bold]{model_cfg['id']}[/bold] [{quant_cfg['name']}]")

                # per-model context_window, config 상한 적용
                model_ctx = model_cfg.get("context_window", gen_cfg["context_window"])
                load_ctx = min(gen_cfg["context_window"], model_ctx)

                # vLLM / SGLang: quantization type을 quant_cfg에서 per-model 재설정
                if backend_name == "vllm" and "vllm_quantization" in quant_cfg:
                    backend.quantization = quant_cfg.get("vllm_quantization")
                if backend_name == "sglang":
                    q = quant_cfg.get("sglang_quantization") or quant_cfg.get("vllm_quantization")
                    if q is not None:
                        backend.quantization = q

                # 모델 로드 — 실패 시 전 track에 실패 row 기록
                try:
                    with Progress(SpinnerColumn(), TextColumn("모델 로드 중..."), TimeElapsedColumn(), transient=True) as p:
                        p.add_task("")
                        backend.load_model(model_ref, quant_cfg["gguf_path"], load_ctx)
                except Exception as e:
                    console.print(f"[red]로드 실패: {e}[/red]")
                    # #1: OOM/로드실패 → 전 track에 실패 row 기록
                    for track in gen_tracks + prefill_tracks:
                        tt = "generation" if track["id"].startswith("gen") else "prefill"
                        row = _make_skip_row(
                            backend_name, backend.version, model_cfg, quant_cfg,
                            track, tt, hardware_id, comparison_mode, model_ctx,
                            f"load_fail:{str(e)[:80]}", backend.memory_method,
                        )
                        append_result(row, output_path)
                    try:
                        backend.unload_model()
                    except Exception:
                        pass
                    continue

                model_memory_gb = backend.get_model_memory_gb()
                if model_memory_gb > 0:
                    console.print(f"  [dim]모델 메모리: {model_memory_gb:.2f} GB[/dim]")

                # Generation tracks — ctx 초과 시 실패 row 기록 (skip 아닌 기록)
                _server_backends = {"llamacpp", "vllm", "sglang"}
                _gen_server_dead = False  # 서버 crash 감지 플래그
                console.print("  [bold yellow]── Generation ──[/bold yellow]")
                for track in gen_tracks:
                    needed = track["input_tokens"] + track["max_tokens"]
                    if needed > model_ctx:
                        console.print(f"  [yellow]OOM record: {track['id']} ({needed} > ctx {model_ctx})[/yellow]")
                        row = _make_skip_row(
                            backend_name, backend.version, model_cfg, quant_cfg,
                            track, "generation", hardware_id, comparison_mode,
                            model_ctx, "ctx_exceeded", backend.memory_method,
                        )
                        append_result(row, output_path)
                        continue

                    # 서버 crash 후 자동 재시작 (gen track 간)
                    if _gen_server_dead and backend_name in _server_backends:
                        console.print("  [yellow]서버 crash 감지 → 재시작 시도...[/yellow]")
                        try:
                            backend.unload_model()
                        except Exception:
                            pass
                        time.sleep(5)
                        try:
                            backend.load_model(model_ref, quant_cfg["gguf_path"], load_ctx)
                            model_memory_gb = backend.get_model_memory_gb()
                            _gen_server_dead = False
                            console.print("  [green]서버 재시작 성공[/green]")
                        except Exception as e:
                            console.print(f"  [red]서버 재시작 실패: {e} → 남은 gen track 전부 skip[/red]")
                            for remaining_track in gen_tracks[gen_tracks.index(track):]:
                                rn = remaining_track["input_tokens"] + remaining_track["max_tokens"]
                                if rn > model_ctx:
                                    continue
                                row = _make_skip_row(
                                    backend_name, backend.version, model_cfg, quant_cfg,
                                    remaining_track, "generation", hardware_id, comparison_mode,
                                    model_ctx, f"server_crash:{str(e)[:60]}", backend.memory_method,
                                )
                                append_result(row, output_path)
                            break

                    results = run_track(
                        backend, backend_name, model_cfg, quant_cfg,
                        track, "generation",
                        bench_cfg, thermal_cfg, gen_cfg,
                        hardware_id, output_path, model_memory_gb,
                        comparison_mode, model_id,
                    )
                    all_results.extend(results)

                    # run_track이 빈 결과 반환 (warmup 실패) → 서버 상태 확인
                    if not results and backend_name in _server_backends:
                        try:
                            import httpx
                            r = httpx.get(f"{bcfg.get('base_url', 'http://localhost:8080')}/health", timeout=3)
                            if r.status_code != 200:
                                _gen_server_dead = True
                        except Exception:
                            _gen_server_dead = True

                    time.sleep(bench_cfg["inter_track_sleep"])

                # Prefill tracks — cold prefill: 서버 백엔드는 track마다 재시작
                console.print("  [bold blue]── Prefill ──[/bold blue]")
                for track in prefill_tracks:
                    needed = track["input_tokens"] + track["max_tokens"]
                    if needed > model_ctx:
                        console.print(f"  [yellow]OOM record: {track['id']} ({needed} > ctx {model_ctx})[/yellow]")
                        row = _make_skip_row(
                            backend_name, backend.version, model_cfg, quant_cfg,
                            track, "prefill", hardware_id, comparison_mode,
                            model_ctx, "ctx_exceeded", backend.memory_method,
                        )
                        append_result(row, output_path)
                        continue
                    # cold prefill: 서버 백엔드는 track마다 재시작
                    if backend_name in _server_backends:
                        backend.unload_model()
                        time.sleep(3)
                        try:
                            backend.load_model(model_ref, quant_cfg["gguf_path"], load_ctx)
                            model_memory_gb = backend.get_model_memory_gb()
                        except Exception as e:
                            console.print(f"[red]Prefill 재시작 실패: {e}[/red]")
                            row = _make_skip_row(
                                backend_name, backend.version, model_cfg, quant_cfg,
                                track, "prefill", hardware_id, comparison_mode,
                                model_ctx, f"reload_fail:{str(e)[:80]}", backend.memory_method,
                            )
                            append_result(row, output_path)
                            continue
                    results = run_track(
                        backend, backend_name, model_cfg, quant_cfg,
                        track, "prefill",
                        bench_cfg, thermal_cfg, gen_cfg,
                        hardware_id, output_path, model_memory_gb,
                        comparison_mode, model_id,
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
    parser.add_argument("--backends", nargs="+", default=None,
                        help="실행할 백엔드 목록. 미지정 시 config의 enabled 백엔드 전체.")
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--tracks", nargs="+", default=[],
                        help="특정 track만 실행 (e.g. gen-512 prefill-4k)")
    parser.add_argument("--quant", nargs="+", default=[],
                        help="특정 양자화만 실행 (e.g. Q4_K_M Q8_0)")
    args = parser.parse_args()

    config = load_config(Path(args.config))

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

    backends = args.backends or [
        name for name, cfg in config.get("backends", {}).items()
        if cfg.get("enabled", True)
    ]
    run_benchmark(config, backends, args.models, output_path, args.quant)


if __name__ == "__main__":
    main()
