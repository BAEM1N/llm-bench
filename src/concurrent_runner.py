"""동시성(Concurrency) 벤치마크 실행기 — Closed-loop Phase 1.

단일 요청 벤치(runner.py)와 완전 분리.
HTTP 서버형 백엔드(llamacpp, ollama)에 N개 요청을 동시에 발사하고
전부 완료되면 run을 종료한다.

출력:
  results/concurrent_raw_<ts>.csv      — 요청별 raw 지표
  results/concurrent_summary_<ts>.csv  — run 단위 집계
"""
import argparse
import asyncio
import csv
import json
import random
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .prompt_gen import build_generation_prompt, build_prefill_prompt
from .runner import make_backend, load_config

console = Console()


# ─── Nonce ──────────────────────────────────────────────────────────────────

def _nonce() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _add_nonce(prompt: str) -> str:
    """동일 프롬프트 캐시/dedup 방지용 nonce 추가."""
    return prompt + f"\n\n[Request nonce: {_nonce()}]"


# ─── Async streaming requests ───────────────────────────────────────────────

async def _stream_ollama(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    context_window: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    timeout: float,
) -> dict:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": max_tokens,
            "num_ctx": context_window,
        },
    }
    start_ts = time.perf_counter()
    first_token_ts: Optional[float] = None
    output_tokens = 0
    eval_duration_ns = 0

    try:
        async with client.stream(
            "POST", f"{base_url}/api/generate",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=10.0),
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if first_token_ts is None and chunk.get("response"):
                    first_token_ts = time.perf_counter()
                if chunk.get("done"):
                    output_tokens = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    break

        end_ts = time.perf_counter()
        first_token_ts = first_token_ts or end_ts
        ttft_ms = (first_token_ts - start_ts) * 1000
        total_latency_s = end_ts - start_ts

        if eval_duration_ns > 0 and output_tokens > 0:
            gen_tps = output_tokens / (eval_duration_ns / 1e9)
        else:
            gen_dur = end_ts - first_token_ts
            gen_tps = output_tokens / gen_dur if gen_dur > 0 else 0.0

        return dict(
            status="ok", error_type="",
            ttft_ms=round(ttft_ms, 2), gen_tps=round(gen_tps, 2),
            total_latency_s=round(total_latency_s, 3), output_tokens=output_tokens,
            start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts,
        )

    except httpx.TimeoutException:
        end_ts = time.perf_counter()
        return dict(
            status="timeout", error_type="timeout",
            ttft_ms=0.0, gen_tps=0.0,
            total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
            start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts,
        )
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(
            status="error", error_type=str(e)[:120],
            ttft_ms=0.0, gen_tps=0.0,
            total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
            start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts,
        )


async def _stream_llamacpp(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    timeout: float,
) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stream": True,
    }
    start_ts = time.perf_counter()
    first_token_ts: Optional[float] = None
    token_count = 0
    output_tokens = 0
    gen_tps = 0.0

    try:
        async with client.stream(
            "POST", f"{base_url}/completion",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=10.0),
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                raw = line[6:] if line.startswith("data: ") else line
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                content = chunk.get("content", "")
                if content:
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
                    token_count += 1
                if chunk.get("stop") is True:
                    timings = chunk.get("timings", {})
                    output_tokens = chunk.get("tokens_predicted") or token_count
                    gen_tps = timings.get("predicted_per_second", 0.0)
                    break

        end_ts = time.perf_counter()
        first_token_ts = first_token_ts or end_ts
        ttft_ms = (first_token_ts - start_ts) * 1000
        total_latency_s = end_ts - start_ts

        if gen_tps == 0.0:
            gen_dur = end_ts - first_token_ts
            gen_tps = output_tokens / gen_dur if gen_dur > 0 else 0.0

        return dict(
            status="ok", error_type="",
            ttft_ms=round(ttft_ms, 2), gen_tps=round(gen_tps, 2),
            total_latency_s=round(total_latency_s, 3), output_tokens=output_tokens,
            start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts,
        )

    except httpx.TimeoutException:
        end_ts = time.perf_counter()
        return dict(
            status="timeout", error_type="timeout",
            ttft_ms=0.0, gen_tps=0.0,
            total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
            start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts,
        )
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(
            status="error", error_type=str(e)[:120],
            ttft_ms=0.0, gen_tps=0.0,
            total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
            start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts,
        )


async def _closed_loop_run(
    backend_name: str,
    base_url: str,
    model_id: str,
    context_window: int,
    prompts: list,
    max_tokens: int,
    gen_cfg: dict,
    timeout: float,
) -> list:
    """N개 요청을 동시 발사하고 전부 완료되면 반환."""
    async with httpx.AsyncClient() as client:
        if backend_name == "ollama":
            coros = [
                _stream_ollama(
                    client, base_url, model_id, p, max_tokens,
                    context_window, gen_cfg["temperature"],
                    gen_cfg["top_p"], gen_cfg["repeat_penalty"], timeout,
                )
                for p in prompts
            ]
        else:  # llamacpp
            coros = [
                _stream_llamacpp(
                    client, base_url, p, max_tokens,
                    gen_cfg["temperature"], gen_cfg["top_p"],
                    gen_cfg["repeat_penalty"], timeout,
                )
                for p in prompts
            ]
        return list(await asyncio.gather(*coros))


# ─── Statistics helpers ──────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    # nearest-rank method
    rank = max(1, int(len(s) * p / 100 + 0.5))
    return s[min(rank - 1, len(s) - 1)]


def _compute_summary(raw_results: list) -> dict:
    ok = [r for r in raw_results if r["status"] == "ok"]
    failed = len(raw_results) - len(ok)
    success_rate = len(ok) / len(raw_results) if raw_results else 0.0

    if not ok:
        return dict(
            completed_requests=0, failed_requests=failed, success_rate=0.0,
            makespan_s=0.0, aggregate_output_tokens=0, aggregate_gen_tps=0.0,
            ttft_p50_ms=0.0, ttft_p95_ms=0.0,
            latency_p50_s=0.0, latency_p95_s=0.0,
            req_gen_tps_p50=0.0, req_gen_tps_p95=0.0,
        )

    makespan_s = (
        max(r["end_ts"] for r in raw_results) - min(r["start_ts"] for r in raw_results)
    )
    agg_tokens = sum(r["output_tokens"] for r in ok)
    agg_gen_tps = agg_tokens / makespan_s if makespan_s > 0 else 0.0

    ttft_vals = [r["ttft_ms"] for r in ok]
    lat_vals = [r["total_latency_s"] for r in ok]
    tps_vals = [r["gen_tps"] for r in ok]

    return dict(
        completed_requests=len(ok),
        failed_requests=failed,
        success_rate=round(success_rate, 4),
        makespan_s=round(makespan_s, 3),
        aggregate_output_tokens=agg_tokens,
        aggregate_gen_tps=round(agg_gen_tps, 2),
        ttft_p50_ms=round(_percentile(ttft_vals, 50), 2),
        ttft_p95_ms=round(_percentile(ttft_vals, 95), 2),
        latency_p50_s=round(_percentile(lat_vals, 50), 3),
        latency_p95_s=round(_percentile(lat_vals, 95), 3),
        req_gen_tps_p50=round(_percentile(tps_vals, 50), 2),
        req_gen_tps_p95=round(_percentile(tps_vals, 95), 2),
    )


# ─── SLA ────────────────────────────────────────────────────────────────────

def _check_sla(summary: dict, sla_cfg: dict, track_id: str) -> tuple:
    violations = []

    if summary["success_rate"] < sla_cfg.get("success_rate_min", 0.99):
        violations.append(
            f"success_rate {summary['success_rate']:.1%} < {sla_cfg['success_rate_min']:.1%}"
        )

    # gen-8192는 완화된 SLA
    if "8192" in track_id:
        ttft_limit = sla_cfg.get("ttft_p95_ms_long", 5000)
        lat_limit = sla_cfg.get("latency_p95_s_long", 180)
    else:
        ttft_limit = sla_cfg.get("ttft_p95_ms", 3000)
        lat_limit = sla_cfg.get("latency_p95_s", 30)

    if summary["ttft_p95_ms"] > ttft_limit:
        violations.append(f"p95 TTFT {summary['ttft_p95_ms']:.0f}ms > {ttft_limit}ms")
    if summary["latency_p95_s"] > lat_limit:
        violations.append(f"p95 latency {summary['latency_p95_s']:.1f}s > {lat_limit}s")
    if summary["req_gen_tps_p50"] < sla_cfg.get("req_gen_tps_p50_min", 10):
        violations.append(
            f"p50 gen TPS {summary['req_gen_tps_p50']:.1f} < {sla_cfg['req_gen_tps_p50_min']}"
        )

    return len(violations) == 0, violations


# ─── CSV output ─────────────────────────────────────────────────────────────

RAW_FIELDS = [
    "timestamp", "hardware_id", "backend", "backend_version",
    "model", "architecture", "total_params", "active_params", "quantization",
    "track_id", "concurrency", "run_number", "request_index",
    "status", "error_type",
    "ttft_ms", "gen_tps", "total_latency_s", "output_tokens",
    "start_ts", "first_token_ts", "end_ts",
]

SUMMARY_FIELDS = [
    "timestamp", "hardware_id", "backend", "backend_version",
    "model", "architecture", "total_params", "active_params", "quantization",
    "track_id", "concurrency", "run_number",
    "completed_requests", "failed_requests", "success_rate",
    "makespan_s", "aggregate_output_tokens", "aggregate_gen_tps",
    "ttft_p50_ms", "ttft_p95_ms",
    "latency_p50_s", "latency_p95_s",
    "req_gen_tps_p50", "req_gen_tps_p95",
    "peak_memory_gb", "sla_pass",
]


def _append_csv(row: dict, path: Path, fields: list) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


# ─── Main benchmark orchestration ───────────────────────────────────────────

def run_concurrent_benchmark(
    config: dict,
    backends: list,
    models: list,
    raw_path: Path,
    summary_path: Path,
) -> None:
    gen_cfg = config["generation"]
    hw_id = config.get("hardware", {}).get("id", "unknown")
    cc_cfg = config.get("concurrency_benchmark", {})

    levels = cc_cfg.get("levels", [1, 2, 4, 8])
    warmup_runs = cc_cfg.get("warmup_runs", 1)
    measure_runs = cc_cfg.get("measure_runs", 5)
    timeout = cc_cfg.get("request_timeout_s", 600)
    sla_cfg = cc_cfg.get("sla", {})
    use_nonce = cc_cfg.get("prompt_nonce", True)

    include_gen = set(cc_cfg.get("include_generation_tracks", ["gen-512", "gen-2048", "gen-8192"]))
    include_prefill = set(cc_cfg.get("include_prefill_tracks", []))

    gen_tracks = [t for t in config.get("generation_tracks", []) if t["id"] in include_gen]
    prefill_tracks = [t for t in config.get("prefill_tracks", []) if t["id"] in include_prefill]
    active_tracks = gen_tracks + prefill_tracks

    for backend_name in backends:
        if backend_name not in ("ollama", "llamacpp"):
            console.print(
                f"[yellow]Skip {backend_name}: Phase 1은 ollama/llamacpp만 지원[/yellow]"
            )
            continue

        bcfg = config["backends"].get(backend_name, {})
        if not bcfg.get("enabled", True):
            console.print(f"[yellow]Skip {backend_name} (disabled)[/yellow]")
            continue

        base_url = bcfg.get(
            "base_url",
            "http://localhost:8080" if backend_name == "llamacpp" else "http://localhost:11434",
        )

        for model_cfg in config["models"]:
            if models and model_cfg["id"] not in models:
                continue

            for quant_cfg in model_cfg["quantizations"]:
                if backend_name == "ollama":
                    model_ref = quant_cfg.get("ollama_model", "")
                else:
                    model_ref = quant_cfg.get("gguf_path", "")

                if not model_ref:
                    console.print(
                        f"[yellow]Skip {backend_name}: {model_cfg['id']} {quant_cfg['name']} (model_ref 미설정)[/yellow]"
                    )
                    continue

                # llamacpp: max concurrency에 맞춰 --parallel 설정 (1회 로드로 전 레벨 커버)
                if backend_name == "llamacpp":
                    max_level = max(levels)
                    bcfg_patched = dict(bcfg)
                    existing_extra = list(bcfg.get("extra_args", []))
                    bcfg_patched["extra_args"] = existing_extra + ["--parallel", str(max_level)]
                    backend = make_backend(backend_name, bcfg_patched)
                else:
                    backend = make_backend(backend_name, bcfg)

                console.rule(
                    f"[bold cyan]{backend_name.upper()} (v{backend.version}) · "
                    f"{model_cfg['id']} [{quant_cfg['name']}]"
                )

                try:
                    with Progress(
                        SpinnerColumn(), TextColumn("모델 로드 중..."),
                        TimeElapsedColumn(), transient=True,
                    ) as p:
                        p.add_task("")
                        backend.load_model(
                            model_ref,
                            quant_cfg.get("gguf_path", ""),
                            gen_cfg["context_window"],
                        )
                except Exception as e:
                    console.print(f"[red]로드 실패: {e}[/red]")
                    continue

                mem_gb = backend.get_model_memory_gb()
                if mem_gb > 0:
                    console.print(f"  [dim]모델 메모리: {mem_gb:.2f} GB[/dim]")

                for track in active_tracks:
                    track_id = track["id"]
                    max_tokens = track["max_tokens"]
                    input_tokens = track["input_tokens"]

                    if "prefill" in track_id:
                        base_prompt = build_prefill_prompt(input_tokens)
                    else:
                        base_prompt = build_generation_prompt(input_tokens)

                    console.print(
                        f"\n[bold yellow]── Track: {track_id}[/bold yellow]"
                        f"  in≈{input_tokens} out={max_tokens}"
                    )

                    for level in levels:
                        # Ollama: per-track ctx (llama.cpp는 서버 시작 시 설정됨)
                        if backend_name == "ollama":
                            track_ctx = min(
                                input_tokens + max_tokens + 256,
                                gen_cfg["context_window"],
                            )
                            backend._context_window = track_ctx
                        else:
                            track_ctx = gen_cfg["context_window"]

                        # Warmup (결과 버림)
                        for _ in range(warmup_runs):
                            prompts = [
                                _add_nonce(base_prompt) if use_nonce else base_prompt
                                for _ in range(level)
                            ]
                            asyncio.run(
                                _closed_loop_run(
                                    backend_name, base_url, model_ref, track_ctx,
                                    prompts, max_tokens, gen_cfg, timeout,
                                )
                            )
                        console.print(f"  [dim]warmup done (c={level})[/dim]", end="  ")

                        # Measure
                        for run_num in range(1, measure_runs + 1):
                            prompts = [
                                _add_nonce(base_prompt) if use_nonce else base_prompt
                                for _ in range(level)
                            ]
                            raw_results = asyncio.run(
                                _closed_loop_run(
                                    backend_name, base_url, model_ref, track_ctx,
                                    prompts, max_tokens, gen_cfg, timeout,
                                )
                            )

                            ts = datetime.now().isoformat()
                            meta = dict(
                                timestamp=ts,
                                hardware_id=hw_id,
                                backend=backend_name,
                                backend_version=backend.version,
                                model=model_cfg["id"],
                                architecture=model_cfg["architecture"],
                                total_params=model_cfg["total_params"],
                                active_params=model_cfg["active_params"],
                                quantization=quant_cfg["name"],
                                track_id=track_id,
                                concurrency=level,
                                run_number=run_num,
                            )

                            # Raw CSV
                            for idx, r in enumerate(raw_results):
                                _append_csv(
                                    {**meta, "request_index": idx, **r},
                                    raw_path, RAW_FIELDS,
                                )

                            # Summary CSV
                            summary = _compute_summary(raw_results)
                            sla_pass, violations = _check_sla(summary, sla_cfg, track_id)
                            _append_csv(
                                {**meta, "peak_memory_gb": mem_gb, "sla_pass": sla_pass, **summary},
                                summary_path, SUMMARY_FIELDS,
                            )

                            sla_icon = "[green]✓[/green]" if sla_pass else "[red]✗[/red]"
                            console.print(
                                f"c={level} run={run_num} {sla_icon} "
                                f"agg={summary['aggregate_gen_tps']:.0f}tok/s "
                                f"p95TTFT={summary['ttft_p95_ms']:.0f}ms "
                                f"p95lat={summary['latency_p95_s']:.1f}s "
                                f"ok={summary['completed_requests']}/{level}",
                                end="  ",
                            )
                            if violations:
                                console.print(f"\n    [red]SLA 위반: {', '.join(violations)}[/red]", end="")
                        console.print()

                backend.unload_model()

    console.print(f"\n[bold green]Raw     → {raw_path}[/bold green]")
    console.print(f"[bold green]Summary → {summary_path}[/bold green]")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Concurrency Benchmark — Closed-loop Phase 1"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backends", nargs="+", default=["llamacpp", "ollama"])
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    cc_cfg = config.get("concurrency_benchmark", {})
    hw = config.get("hardware", {})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    raw_path = output_dir / f"concurrent_raw_{ts}.csv"
    summary_path = output_dir / f"concurrent_summary_{ts}.csv"

    console.print(f"[bold]LLM Concurrency Benchmark[/bold] — {hw.get('description', '')}")
    console.print(f"Backends : {args.backends}")
    console.print(f"Levels   : {cc_cfg.get('levels', [1, 2, 4, 8])}")
    console.print(f"Tracks   : {cc_cfg.get('include_generation_tracks', [])}")
    console.print(f"Runs     : warmup={cc_cfg.get('warmup_runs', 1)}, measure={cc_cfg.get('measure_runs', 5)}\n")

    run_concurrent_benchmark(config, args.backends, args.models, raw_path, summary_path)


if __name__ == "__main__":
    main()
