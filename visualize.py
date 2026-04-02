"""벤치마크 결과 시각화."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
BACKENDS = ["mlx", "llamacpp", "ollama"]
PALETTE = dict(zip(BACKENDS, sns.color_palette("muted", len(BACKENDS))))
BACKEND_LABELS = {"mlx": "MLX", "llamacpp": "llama.cpp", "ollama": "Ollama"}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # run_number=1은 warmup — 제외
    df = df[df["run_number"] > 1].copy()
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """중앙값 집계."""
    return (
        df.groupby(["backend", "model", "architecture", "total_params",
                    "active_params", "quantization", "track_id", "track_type"])
        .agg(
            ttft_ms=("ttft_ms", "median"),
            gen_tps=("gen_tps", "median"),
            prefill_tps=("prefill_tps", "median"),
            peak_memory_gb=("peak_memory_gb", "median"),
            output_tokens=("output_tokens", "median"),
            total_latency_s=("total_latency_s", "median"),
        )
        .reset_index()
    )


def chart_gen_tps(df: pd.DataFrame, out: Path) -> None:
    """백엔드별 Generation TPS (모델별, gen 트랙별)."""
    gen = df[df["track_type"] == "generation"]
    tracks = sorted(gen["track_id"].unique())
    n = len(tracks)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, track_id in zip(axes, tracks):
        sub = gen[gen["track_id"] == track_id]
        pivot = sub.pivot_table(index="model", columns="backend", values="gen_tps")
        pivot.rename(columns=BACKEND_LABELS, inplace=True)
        pivot.plot(kind="bar", ax=ax, color=[PALETTE.get(b, "gray") for b in [
            k for k, v in BACKEND_LABELS.items() if v in pivot.columns
        ]])
        ax.set_title(f"Gen TPS — {track_id}")
        ax.set_xlabel("")
        ax.set_ylabel("tokens/sec")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="backend")
    fig.tight_layout()
    fig.savefig(out / "01_gen_tps.png", dpi=150)
    plt.close(fig)


def chart_ttft(df: pd.DataFrame, out: Path) -> None:
    """백엔드별 TTFT (모델별, gen-512 기준)."""
    sub = df[(df["track_id"] == "gen-512") & (df["track_type"] == "generation")]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, quant in zip(axes, ["Q4_K_M", "Q8_0"]):
        q = sub[sub["quantization"] == quant]
        if q.empty:
            ax.set_visible(False)
            continue
        pivot = q.pivot_table(index="model", columns="backend", values="ttft_ms")
        pivot.rename(columns=BACKEND_LABELS, inplace=True)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"TTFT (ms) — gen-512, {quant}")
        ax.set_xlabel("")
        ax.set_ylabel("ms")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "02_ttft.png", dpi=150)
    plt.close(fig)


def chart_prefill_scaling(df: pd.DataFrame, out: Path) -> None:
    """Prefill TPS — 입력 길이별 스케일링."""
    pf = df[df["track_type"] == "prefill"].copy()
    if pf.empty:
        return
    # track_id에서 입력 토큰 수 추출
    track_order = ["prefill-1k", "prefill-4k", "prefill-16k", "prefill-64k", "prefill-128k"]
    pf = pf[pf["track_id"].isin(track_order)].copy()
    pf["track_id"] = pd.Categorical(pf["track_id"], categories=track_order, ordered=True)

    models = sorted(pf["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = pf[(pf["model"] == model) & (pf["quantization"] == "Q4_K_M")]
        for backend in BACKENDS:
            bsub = sub[sub["backend"] == backend].sort_values("track_id")
            if bsub.empty:
                continue
            ax.plot(bsub["track_id"], bsub["prefill_tps"] / 1000,
                    marker="o", label=BACKEND_LABELS[backend], color=PALETTE[backend])
        ax.set_title(f"Prefill TPS — {model} (Q4)")
        ax.set_xlabel("Context length")
        ax.set_ylabel("Prefill TPS (K tok/s)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="backend")
    fig.tight_layout()
    fig.savefig(out / "03_prefill_scaling.png", dpi=150)
    plt.close(fig)


def chart_quantization(df: pd.DataFrame, out: Path) -> None:
    """양자화별 TPS 비교 (Q4 vs Q8, gen-512)."""
    sub = df[(df["track_id"] == "gen-512") & (df["track_type"] == "generation")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=sub, x="model", y="gen_tps", hue="quantization", ax=ax)
    ax.set_title("Quantization Impact — Gen TPS (gen-512)")
    ax.set_xlabel("")
    ax.set_ylabel("tokens/sec")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "04_quantization.png", dpi=150)
    plt.close(fig)


def chart_memory(df: pd.DataFrame, out: Path) -> None:
    """모델 로드 메모리 비교."""
    sub = df[(df["track_id"] == "gen-512") & (df["quantization"] == "Q4_K_M")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = sub.pivot_table(index="model", columns="backend", values="peak_memory_gb")
    pivot.rename(columns=BACKEND_LABELS, inplace=True)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Model Load Memory — Q4_K_M (GB)")
    ax.set_xlabel("")
    ax.set_ylabel("GB")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "05_memory.png", dpi=150)
    plt.close(fig)


def chart_e2e_latency(df: pd.DataFrame, out: Path) -> None:
    """E2E 레이턴시 — gen-8192, Q4_K_M."""
    sub = df[(df["track_id"] == "gen-8192") & (df["quantization"] == "Q4_K_M")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = sub.pivot_table(index="model", columns="backend", values="total_latency_s")
    pivot.rename(columns=BACKEND_LABELS, inplace=True)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("E2E Latency — gen-8192, Q4_K_M (seconds)")
    ax.set_xlabel("")
    ax.set_ylabel("seconds")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "06_e2e_latency.png", dpi=150)
    plt.close(fig)


def chart_heatmap(df: pd.DataFrame, out: Path) -> None:
    """종합 히트맵: 백엔드 × 모델 Gen TPS (gen-512, Q4_K_M)."""
    sub = df[(df["track_id"] == "gen-512") & (df["quantization"] == "Q4_K_M")]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="model", columns="backend", values="gen_tps")
    pivot.rename(columns=BACKEND_LABELS, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_title("Gen TPS Heatmap — gen-512, Q4_K_M")
    fig.tight_layout()
    fig.savefig(out / "07_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("csv", nargs="+", help="결과 CSV 파일 경로 (여러 개 가능)")
    parser.add_argument("--out", default="charts", help="차트 출력 디렉토리")
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.csv]
    raw = pd.concat(dfs, ignore_index=True)
    raw = raw[raw["run_number"] > 1]  # warmup 제외

    df = aggregate(raw)
    print(f"데이터 로드: {len(raw)}행 → 집계 후 {len(df)}행")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    chart_gen_tps(df, out_dir)
    chart_ttft(df, out_dir)
    chart_prefill_scaling(df, out_dir)
    chart_quantization(df, out_dir)
    chart_memory(df, out_dir)
    chart_e2e_latency(df, out_dir)
    chart_heatmap(df, out_dir)

    print(f"차트 저장 완료: {out_dir}/")


if __name__ == "__main__":
    main()
