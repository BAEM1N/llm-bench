"""동시성(Concurrency) 벤치마크 실행기 — Phase 1 (Closed-loop) + Phase 2 (Open-loop).

단일 요청 벤치(runner.py)와 완전 분리.

Phase 1 — Closed-loop:
  N개 요청을 동시에 발사하고 전부 완료되면 run 종료.
  최대 동시 생성 수 / SLA 통과 concurrency 측정에 적합.

Phase 2 — Open-loop:
  일정 arrival rate(req/s)로 duration 동안 요청을 주입.
  지속 가능한 처리량 / 큐 쌓임 지점 측정에 적합.

지원 백엔드:
  Phase 1: llamacpp, ollama, vllm (HTTP 서버형), mlx (in-process thread pool)
  Phase 2: llamacpp, ollama, vllm

출력:
  results/concurrent_raw_<ts>.csv      — 요청별 raw 지표
  results/concurrent_summary_<ts>.csv  — run 단위 집계
"""
import argparse
import asyncio
import concurrent.futures
import csv
import json
import random
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .prompt_gen import build_generation_prompt, build_prefill_prompt
from .runner import make_backend, load_config
from .metrics import percentile as _percentile

console = Console()


# ─── Nonce ──────────────────────────────────────────────────────────────────

def _nonce() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _add_nonce(prompt: str) -> str:
    """동일 프롬프트 캐시/dedup 방지용 nonce 추가."""
    return prompt + f"\n\n[Request nonce: {_nonce()}]"


# ─── Async streaming — Ollama ────────────────────────────────────────────────

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
            resp.raise_for_status()
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
        gen_dur = end_ts - first_token_ts
        gen_tps = (output_tokens / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 and output_tokens > 0 \
                  else (output_tokens / gen_dur if gen_dur > 0 else 0.0)

        return dict(status="ok", error_type="",
                    ttft_ms=round((first_token_ts - start_ts) * 1000, 2),
                    gen_tps=round(gen_tps, 2),
                    total_latency_s=round(end_ts - start_ts, 3),
                    output_tokens=output_tokens,
                    start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts)

    except httpx.TimeoutException:
        end_ts = time.perf_counter()
        return dict(status="timeout", error_type="timeout", ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(status="error", error_type=str(e)[:120], ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)


# ─── Async streaming — llama.cpp ─────────────────────────────────────────────

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
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                raw = line[6:] if line.startswith("data: ") else line
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if chunk.get("content"):
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
        if gen_tps == 0.0:
            gen_dur = end_ts - first_token_ts
            gen_tps = output_tokens / gen_dur if gen_dur > 0 else 0.0

        return dict(status="ok", error_type="",
                    ttft_ms=round((first_token_ts - start_ts) * 1000, 2),
                    gen_tps=round(gen_tps, 2),
                    total_latency_s=round(end_ts - start_ts, 3),
                    output_tokens=output_tokens,
                    start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts)

    except httpx.TimeoutException:
        end_ts = time.perf_counter()
        return dict(status="timeout", error_type="timeout", ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(status="error", error_type=str(e)[:120], ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)


# ─── Async streaming — vLLM (/v1/completions SSE) ───────────────────────────

async def _stream_vllm(
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    timeout: float,
) -> dict:
    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repeat_penalty,
        "stream": True,
    }
    start_ts = time.perf_counter()
    first_token_ts: Optional[float] = None
    output_tokens = 0

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions",
            json=payload,
            timeout=httpx.Timeout(timeout, connect=10.0),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                raw = line[6:] if line.startswith("data: ") else line
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    if text and first_token_ts is None:
                        first_token_ts = time.perf_counter()
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", output_tokens)

        end_ts = time.perf_counter()
        first_token_ts = first_token_ts or end_ts
        gen_dur = end_ts - first_token_ts
        gen_tps = output_tokens / gen_dur if gen_dur > 0 and output_tokens > 0 else 0.0

        return dict(status="ok", error_type="",
                    ttft_ms=round((first_token_ts - start_ts) * 1000, 2),
                    gen_tps=round(gen_tps, 2),
                    total_latency_s=round(end_ts - start_ts, 3),
                    output_tokens=output_tokens,
                    start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts)

    except httpx.TimeoutException:
        end_ts = time.perf_counter()
        return dict(status="timeout", error_type="timeout", ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(status="error", error_type=str(e)[:120], ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)


# ─── MLX — thread pool (in-process, GPU 직렬 실행) ─────────────────────────

def _sync_mlx_request(model, tokenizer, prompt: str, max_tokens: int,
                       temperature: float, top_p: float) -> dict:
    """MLX stream_generate를 동기 실행. ThreadPoolExecutor에서 호출됨.

    Metal GPU는 단일 스레드로 직렬 실행됨 — throughput은 단일 요청과 동일.
    concurrency 수치는 '동시 요청 수' 아닌 '큐 대기 시간' 관점에서 해석해야 함.
    """
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1.0 else 0.0)
    start_ts = time.perf_counter()
    first_token_ts: Optional[float] = None
    last_resp = None

    try:
        for resp in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens, sampler=sampler):
            if first_token_ts is None:
                first_token_ts = time.perf_counter()
            last_resp = resp

        end_ts = time.perf_counter()
        first_token_ts = first_token_ts or end_ts
        output_tokens = last_resp.generation_tokens if last_resp else 0
        gen_tps = last_resp.generation_tps if last_resp else 0.0

        return dict(status="ok", error_type="",
                    ttft_ms=round((first_token_ts - start_ts) * 1000, 2),
                    gen_tps=round(gen_tps, 2),
                    total_latency_s=round(end_ts - start_ts, 3),
                    output_tokens=output_tokens,
                    start_ts=start_ts, first_token_ts=first_token_ts, end_ts=end_ts)
    except Exception as e:
        end_ts = time.perf_counter()
        return dict(status="error", error_type=str(e)[:120], ttft_ms=0.0, gen_tps=0.0,
                    total_latency_s=round(end_ts - start_ts, 3), output_tokens=0,
                    start_ts=start_ts, first_token_ts=end_ts, end_ts=end_ts)


async def _closed_loop_mlx(model, tokenizer, prompts: list, max_tokens: int,
                             gen_cfg: dict, timeout: float) -> list:
    """MLX 동시 요청 — ThreadPoolExecutor로 병렬 제출, GPU는 직렬 처리."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            loop.run_in_executor(
                executor, _sync_mlx_request,
                model, tokenizer, p, max_tokens,
                gen_cfg["temperature"], gen_cfg["top_p"],
            )
            for p in prompts
        ]
        try:
            return list(await asyncio.wait_for(
                asyncio.gather(*futures), timeout=timeout * len(prompts)
            ))
        except asyncio.TimeoutError:
            return [dict(status="timeout", error_type="timeout", ttft_ms=0.0, gen_tps=0.0,
                         total_latency_s=timeout, output_tokens=0,
                         start_ts=0.0, first_token_ts=0.0, end_ts=0.0)] * len(prompts)


# ─── Closed-loop dispatcher ──────────────────────────────────────────────────

async def _closed_loop_run(
    backend_name: str,
    base_url: str,
    model_id: str,
    context_window: int,
    prompts: list,
    max_tokens: int,
    gen_cfg: dict,
    timeout: float,
    mlx_model=None,
    mlx_tokenizer=None,
) -> list:
    """N개 요청을 동시 발사하고 전부 완료되면 반환."""
    if backend_name == "mlx":
        return await _closed_loop_mlx(
            mlx_model, mlx_tokenizer, prompts, max_tokens, gen_cfg, timeout
        )

    async with httpx.AsyncClient() as client:
        if backend_name == "ollama":
            coros = [
                _stream_ollama(client, base_url, model_id, p, max_tokens,
                               context_window, gen_cfg["temperature"],
                               gen_cfg["top_p"], gen_cfg["repeat_penalty"], timeout)
                for p in prompts
            ]
        elif backend_name == "vllm":
            coros = [
                _stream_vllm(client, base_url, model_id, p, max_tokens,
                             gen_cfg["temperature"], gen_cfg["top_p"],
                             gen_cfg["repeat_penalty"], timeout)
                for p in prompts
            ]
        else:  # llamacpp
            coros = [
                _stream_llamacpp(client, base_url, p, max_tokens,
                                 gen_cfg["temperature"], gen_cfg["top_p"],
                                 gen_cfg["repeat_penalty"], timeout)
                for p in prompts
            ]
        return list(await asyncio.gather(*coros))


# ─── Open-loop (Phase 2) ─────────────────────────────────────────────────────

async def _open_loop_run(
    backend_name: str,
    base_url: str,
    model_id: str,
    context_window: int,
    base_prompt: str,
    max_tokens: int,
    gen_cfg: dict,
    rate: float,
    duration_s: float,
    use_nonce: bool,
    timeout: float,
) -> list:
    """rate req/s로 duration_s 동안 요청을 주입, 전부 완료 후 반환.

    실제 도착 간격 = 1/rate 초 (고정, Poisson 미사용).
    큐 쌓임 / p95 latency 붕괴 지점 탐색에 사용.
    """
    interval = 1.0 / rate
    tasks = []
    wall_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        while time.perf_counter() - wall_start < duration_s:
            prompt = _add_nonce(base_prompt) if use_nonce else base_prompt

            if backend_name == "ollama":
                coro = _stream_ollama(client, base_url, model_id, prompt, max_tokens,
                                      context_window, gen_cfg["temperature"],
                                      gen_cfg["top_p"], gen_cfg["repeat_penalty"], timeout)
            elif backend_name == "vllm":
                coro = _stream_vllm(client, base_url, model_id, prompt, max_tokens,
                                    gen_cfg["temperature"], gen_cfg["top_p"],
                                    gen_cfg["repeat_penalty"], timeout)
            else:  # llamacpp
                coro = _stream_llamacpp(client, base_url, prompt, max_tokens,
                                        gen_cfg["temperature"], gen_cfg["top_p"],
                                        gen_cfg["repeat_penalty"], timeout)

            tasks.append(asyncio.create_task(coro))
            await asyncio.sleep(interval)

        # 남은 요청 완료 대기
        results = []
        for task in tasks:
            try:
                results.append(await task)
            except Exception as e:
                results.append(dict(status="error", error_type=str(e)[:80],
                                    ttft_ms=0.0, gen_tps=0.0, total_latency_s=0.0,
                                    output_tokens=0, start_ts=0.0,
                                    first_token_ts=0.0, end_ts=0.0))
        return results


# ─── Statistics ──────────────────────────────────────────────────────────────

def _compute_summary(raw_results: list) -> dict:
    ok = [r for r in raw_results if r["status"] == "ok"]
    failed = len(raw_results) - len(ok)
    success_rate = len(ok) / len(raw_results) if raw_results else 0.0

    if not ok:
        return dict(completed_requests=0, failed_requests=failed, success_rate=0.0,
                    makespan_s=0.0, aggregate_output_tokens=0, aggregate_gen_tps=0.0,
                    ttft_p50_ms=0.0, ttft_p95_ms=0.0,
                    latency_p50_s=0.0, latency_p95_s=0.0,
                    req_gen_tps_p50=0.0, req_gen_tps_p95=0.0)

    makespan_s = max(r["end_ts"] for r in raw_results) - min(r["start_ts"] for r in raw_results)
    agg_tokens = sum(r["output_tokens"] for r in ok)

    ttft_vals = [r["ttft_ms"] for r in ok]
    lat_vals  = [r["total_latency_s"] for r in ok]
    tps_vals  = [r["gen_tps"] for r in ok]

    return dict(
        completed_requests=len(ok),
        failed_requests=failed,
        success_rate=round(success_rate, 4),
        makespan_s=round(makespan_s, 3),
        aggregate_output_tokens=agg_tokens,
        aggregate_gen_tps=round(agg_tokens / makespan_s if makespan_s > 0 else 0.0, 2),
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
    ttft_limit = sla_cfg.get("ttft_p95_ms_long" if "8192" in track_id else "ttft_p95_ms", 3000)
    lat_limit  = sla_cfg.get("latency_p95_s_long" if "8192" in track_id else "latency_p95_s", 30)

    if summary["ttft_p95_ms"] > ttft_limit:
        violations.append(f"p95 TTFT {summary['ttft_p95_ms']:.0f}ms > {ttft_limit}ms")
    if summary["latency_p95_s"] > lat_limit:
        violations.append(f"p95 latency {summary['latency_p95_s']:.1f}s > {lat_limit}s")
    if summary["req_gen_tps_p50"] < sla_cfg.get("req_gen_tps_p50_min", 10):
        violations.append(
            f"p50 gen TPS {summary['req_gen_tps_p50']:.1f} < {sla_cfg['req_gen_tps_p50_min']}"
        )
    return len(violations) == 0, violations


# ─── Online user estimation ──────────────────────────────────────────────────

def _print_online_user_table(
    sla_max_concurrency: dict,
    avg_gen_times: dict,
    user_intervals: list,
) -> None:
    """SLA 통과 최대 concurrency → 환산 온라인 사용자 수 테이블 출력.

    active_ratio = T_gen / T_user
    estimated_online_users = max_concurrency / active_ratio
    """
    if not sla_max_concurrency:
        return

    console.rule("[bold]환산 온라인 사용자 수 추정")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Track")
    table.add_column("SLA max c")
    table.add_column("avg T_gen (s)")
    for interval in user_intervals:
        table.add_column(f"T_user={interval}s")

    for track_id, max_c in sorted(sla_max_concurrency.items()):
        if max_c == 0:
            continue
        avg_gen = avg_gen_times.get(track_id, 0.0)
        row = [track_id, str(max_c), f"{avg_gen:.1f}"]
        for interval in user_intervals:
            if avg_gen > 0:
                active_ratio = avg_gen / interval
                est = int(max_c / active_ratio) if active_ratio > 0 else 0
                row.append(str(est))
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


# ─── CSV output ──────────────────────────────────────────────────────────────

RAW_FIELDS = [
    "timestamp", "hardware_id", "comparison_mode", "backend", "backend_version",
    "model", "architecture", "total_params", "active_params", "quantization",
    "track_id", "concurrency", "run_number", "request_index",
    "status", "error_type",
    "ttft_ms", "gen_tps", "total_latency_s", "output_tokens",
    "start_ts", "first_token_ts", "end_ts",
]

SUMMARY_FIELDS = [
    "timestamp", "hardware_id", "comparison_mode", "backend", "backend_version",
    "model", "architecture", "total_params", "active_params", "quantization",
    "track_id", "concurrency", "run_number",
    "concurrency_mode",  # "parallel" (HTTP backends) | "queued_gpu" (MLX)
    "completed_requests", "failed_requests", "success_rate",
    "makespan_s", "aggregate_output_tokens", "aggregate_gen_tps",
    "ttft_p50_ms", "ttft_p95_ms", "latency_p50_s", "latency_p95_s",
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


# ─── HTTP backend filter ─────────────────────────────────────────────────────

_HTTP_BACKENDS = {"ollama", "llamacpp", "vllm"}
_ALL_BACKENDS  = _HTTP_BACKENDS | {"mlx"}


def _get_model_ref(backend_name: str, quant_cfg: dict) -> str:
    if backend_name == "ollama":
        return quant_cfg.get("ollama_model", "")
    if backend_name == "vllm":
        return quant_cfg.get("vllm_model") or quant_cfg.get("gguf_path", "")
    if backend_name == "mlx":
        return quant_cfg.get("mlx_model", "")
    return quant_cfg.get("gguf_path", "")  # llamacpp


def _default_base_url(backend_name: str) -> str:
    return {
        "llamacpp": "http://localhost:8080",
        "ollama":   "http://localhost:11434",
        "vllm":     "http://localhost:8000",
        "mlx":      "",
    }.get(backend_name, "")


# ─── Main benchmark orchestration ────────────────────────────────────────────

def run_concurrent_benchmark(
    config: dict,
    backends: list,
    models: list,
    raw_path: Path,
    summary_path: Path,
) -> None:
    gen_cfg  = config["generation"]
    hw_id    = config.get("hardware", {}).get("id", "unknown")
    cc_cfg   = config.get("concurrency_benchmark", {})

    levels        = cc_cfg.get("levels", [1, 2, 4, 8])
    warmup_runs   = cc_cfg.get("warmup_runs", 1)
    measure_runs  = cc_cfg.get("measure_runs", 5)
    timeout       = cc_cfg.get("request_timeout_s", 600)
    sla_cfg       = cc_cfg.get("sla", {})
    use_nonce     = cc_cfg.get("prompt_nonce", True)
    user_intervals = cc_cfg.get("user_interval_assumptions_s", [15, 30, 60])

    # Phase 2 open-loop 설정
    ol_rates    = cc_cfg.get("open_loop_rates", [])          # [] → Phase 2 skip
    ol_duration = cc_cfg.get("open_loop_duration_s", 60)

    include_gen    = set(cc_cfg.get("include_generation_tracks", ["gen-512", "gen-2048", "gen-8192"]))
    include_prefill = set(cc_cfg.get("include_prefill_tracks", []))

    gen_tracks    = [t for t in config.get("generation_tracks", []) if t["id"] in include_gen]
    prefill_tracks = [t for t in config.get("prefill_tracks", []) if t["id"] in include_prefill]
    active_tracks = gen_tracks + prefill_tracks

    for backend_name in backends:
        if backend_name not in _ALL_BACKENDS:
            console.print(f"[yellow]Skip {backend_name}: 지원되지 않는 백엔드[/yellow]")
            continue
        if backend_name == "mlx" and ol_rates:
            console.print("[yellow]MLX: open-loop (Phase 2)는 HTTP 백엔드 전용 — Phase 1만 실행[/yellow]")

        bcfg = config["backends"].get(backend_name, {})
        if not bcfg.get("enabled", True):
            console.print(f"[yellow]Skip {backend_name} (disabled)[/yellow]")
            continue

        base_url = bcfg.get("base_url", _default_base_url(backend_name))

        for model_cfg in config["models"]:
            if models and model_cfg["id"] not in models:
                continue

            for quant_cfg in model_cfg["quantizations"]:
                model_ref = _get_model_ref(backend_name, quant_cfg)
                if not model_ref:
                    console.print(
                        f"[yellow]Skip {backend_name}: {model_cfg['id']} {quant_cfg['name']} (model_ref 미설정)[/yellow]"
                    )
                    continue

                # llamacpp: max concurrency에 맞춰 --parallel (1회 로드로 전 레벨 커버)
                if backend_name == "llamacpp":
                    bcfg_patched = dict(bcfg)
                    bcfg_patched["extra_args"] = list(bcfg.get("extra_args", [])) + [
                        "--parallel", str(max(levels))
                    ]
                    backend = make_backend(backend_name, bcfg_patched)
                else:
                    backend = make_backend(backend_name, bcfg)

                console.rule(
                    f"[bold cyan]{backend_name.upper()} (v{backend.version}) · "
                    f"{model_cfg['id']} [{quant_cfg['name']}]"
                )
                if backend_name == "mlx":
                    console.print(
                        "  [dim]MLX: in-process thread pool — Metal GPU 직렬 실행, "
                        "concurrency 수치는 큐 대기 시간 관점에서 해석 권장[/dim]"
                    )

                try:
                    with Progress(SpinnerColumn(), TextColumn("모델 로드 중..."),
                                  TimeElapsedColumn(), transient=True) as p:
                        p.add_task("")
                        backend.load_model(model_ref, quant_cfg.get("gguf_path", ""),
                                           gen_cfg["context_window"])
                except Exception as e:
                    console.print(f"[red]로드 실패: {e}[/red]")
                    continue

                mem_gb = backend.get_model_memory_gb()
                if mem_gb > 0:
                    console.print(f"  [dim]모델 메모리: {mem_gb:.2f} GB[/dim]")

                # MLX 모델 객체 추출 (thread pool용)
                mlx_model = getattr(backend, "_model", None)
                mlx_tokenizer = getattr(backend, "_tokenizer", None)

                # 온라인 사용자 추정을 위한 per-track 수집
                sla_max_concurrency: dict[str, int] = {}
                avg_gen_times: dict[str, float] = {}

                # ── Phase 1: Closed-loop ─────────────────────────────────────

                for track in active_tracks:
                    track_id    = track["id"]
                    max_tokens  = track["max_tokens"]
                    input_tokens = track["input_tokens"]

                    base_prompt = (build_prefill_prompt(input_tokens) if "prefill" in track_id
                                   else build_generation_prompt(input_tokens, max_tokens))

                    console.print(
                        f"\n[bold yellow]── Track: {track_id}[/bold yellow]"
                        f"  in≈{input_tokens} out={max_tokens}"
                    )

                    # Ollama: full context
                    if backend_name == "ollama":
                        backend._context_window = gen_cfg["context_window"]
                    track_ctx = gen_cfg["context_window"]

                    track_sla_max = 0
                    track_gen_times = []

                    for level in levels:
                        # Warmup
                        for _ in range(warmup_runs):
                            prompts = [_add_nonce(base_prompt) if use_nonce else base_prompt
                                       for _ in range(level)]
                            asyncio.run(_closed_loop_run(
                                backend_name, base_url, model_ref, track_ctx,
                                prompts, max_tokens, gen_cfg, timeout,
                                mlx_model, mlx_tokenizer,
                            ))
                        console.print(f"  [dim]warmup done (c={level})[/dim]", end="  ")

                        level_sla_passed = True

                        for run_num in range(1, measure_runs + 1):
                            prompts = [_add_nonce(base_prompt) if use_nonce else base_prompt
                                       for _ in range(level)]
                            raw_results = asyncio.run(_closed_loop_run(
                                backend_name, base_url, model_ref, track_ctx,
                                prompts, max_tokens, gen_cfg, timeout,
                                mlx_model, mlx_tokenizer,
                            ))

                            ts = datetime.now().isoformat()
                            cc_mode = "queued_gpu" if backend_name == "mlx" else "parallel"
                            meta = dict(
                                timestamp=ts, hardware_id=hw_id, comparison_mode="C",
                                backend=backend_name, backend_version=backend.version,
                                model=model_cfg["id"], architecture=model_cfg["architecture"],
                                total_params=model_cfg["total_params"],
                                active_params=model_cfg["active_params"],
                                quantization=quant_cfg["name"],
                                track_id=track_id, concurrency=level, run_number=run_num,
                                concurrency_mode=cc_mode,
                            )

                            for idx, r in enumerate(raw_results):
                                _append_csv({**meta, "request_index": idx, **r},
                                            raw_path, RAW_FIELDS)

                            summary = _compute_summary(raw_results)
                            sla_pass, violations = _check_sla(summary, sla_cfg, track_id)
                            _append_csv({**meta, "peak_memory_gb": mem_gb,
                                         "sla_pass": sla_pass, **summary},
                                        summary_path, SUMMARY_FIELDS)

                            if not sla_pass:
                                level_sla_passed = False

                            # 평균 gen time 수집 (온라인 사용자 추정용)
                            ok_results = [r for r in raw_results if r["status"] == "ok"]
                            if ok_results:
                                track_gen_times.extend(r["total_latency_s"] for r in ok_results)

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
                                console.print(
                                    f"\n    [red]SLA 위반: {', '.join(violations)}[/red]", end=""
                                )
                        console.print()

                        if level_sla_passed:
                            track_sla_max = level

                    sla_max_concurrency[track_id] = track_sla_max
                    if track_gen_times:
                        avg_gen_times[track_id] = sum(track_gen_times) / len(track_gen_times)

                # ── Phase 2: Open-loop (HTTP 백엔드만) ───────────────────────

                if ol_rates and backend_name in _HTTP_BACKENDS:
                    console.print("\n[bold magenta]── Phase 2: Open-loop Arrival Rate ──[/bold magenta]")
                    for track in active_tracks:
                        track_id     = track["id"]
                        max_tokens   = track["max_tokens"]
                        input_tokens = track["input_tokens"]
                        base_prompt  = (build_prefill_prompt(input_tokens) if "prefill" in track_id
                                        else build_generation_prompt(input_tokens, max_tokens))

                        for rate in ol_rates:
                            console.print(
                                f"  [dim]{track_id} @ {rate} req/s, {ol_duration}s[/dim]", end=" "
                            )
                            raw_results = asyncio.run(_open_loop_run(
                                backend_name, base_url, model_ref, gen_cfg["context_window"],
                                base_prompt, max_tokens, gen_cfg,
                                rate, ol_duration, use_nonce, timeout,
                            ))

                            ts = datetime.now().isoformat()
                            summary = _compute_summary(raw_results)
                            sla_pass, violations = _check_sla(summary, sla_cfg, track_id)
                            meta = dict(
                                timestamp=ts, hardware_id=hw_id, comparison_mode="C",
                                backend=backend_name, backend_version=backend.version,
                                model=model_cfg["id"], architecture=model_cfg["architecture"],
                                total_params=model_cfg["total_params"],
                                active_params=model_cfg["active_params"],
                                quantization=quant_cfg["name"],
                                # concurrency 필드에 rate 기록 (open-loop 식별용)
                                track_id=f"{track_id}@{rate}rps", concurrency=-1,
                                run_number=1,
                            )
                            for idx, r in enumerate(raw_results):
                                _append_csv({**meta, "request_index": idx, **r},
                                            raw_path, RAW_FIELDS)
                            _append_csv({**meta, "peak_memory_gb": mem_gb,
                                         "sla_pass": sla_pass, **summary},
                                        summary_path, SUMMARY_FIELDS)

                            sla_icon = "[green]✓[/green]" if sla_pass else "[red]✗[/red]"
                            console.print(
                                f"{sla_icon} agg={summary['aggregate_gen_tps']:.0f}tok/s "
                                f"p95lat={summary['latency_p95_s']:.1f}s "
                                f"ok={summary['completed_requests']}/{len(raw_results)}"
                            )

                # ── 온라인 사용자 추정 ────────────────────────────────────────
                _print_online_user_table(sla_max_concurrency, avg_gen_times, user_intervals)

                backend.unload_model()

    console.print(f"\n[bold green]Raw     → {raw_path}[/bold green]")
    console.print(f"[bold green]Summary → {summary_path}[/bold green]")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Concurrency Benchmark — Closed-loop Phase 1 + Open-loop Phase 2"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backends", nargs="+", default=["llamacpp", "ollama"])
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    cc_cfg = config.get("concurrency_benchmark", {})
    hw     = config.get("hardware", {})
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    raw_path     = output_dir / f"concurrent_raw_{ts}.csv"
    summary_path = output_dir / f"concurrent_summary_{ts}.csv"

    console.print(f"[bold]LLM Concurrency Benchmark[/bold] — {hw.get('description', '')}")
    console.print(f"Backends : {args.backends}")
    console.print(f"Levels   : {cc_cfg.get('levels', [1, 2, 4, 8])}")
    console.print(f"Tracks   : {cc_cfg.get('include_generation_tracks', [])}")
    console.print(f"Runs     : warmup={cc_cfg.get('warmup_runs', 1)}, "
                  f"measure={cc_cfg.get('measure_runs', 5)}")
    ol_rates = cc_cfg.get("open_loop_rates", [])
    if ol_rates:
        console.print(f"Phase 2  : open-loop rates={ol_rates} "
                      f"duration={cc_cfg.get('open_loop_duration_s', 60)}s")
    console.print()

    run_concurrent_benchmark(config, args.backends, args.models, raw_path, summary_path)


if __name__ == "__main__":
    main()
