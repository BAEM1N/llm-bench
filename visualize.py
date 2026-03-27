"""벤치마크 결과 시각화."""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
BACKENDS = ["ollama", "lmstudio", "llamacpp", "mlx"]
PALETTE = dict(zip(BACKENDS, sns.color_palette("muted", len(BACKENDS))))


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 중앙값 집계
    return (
        df.groupby(["backend", "model", "architecture", "total_params",
                    "active_params", "quantization", "prompt_id"])
        .agg(
            ttft_ms=("ttft_ms", "median"),
            gen_tps=("gen_tps", "median"),
            prompt_tps=("prompt_tps", "median"),
            peak_memory_gb=("peak_memory_gb", "median"),
            output_tokens=("output_tokens", "median"),
        )
        .reset_index()
    )


def chart_gen_tps(df: pd.DataFrame, out: Path) -> None:
    """백엔드별 Generation TPS (모델별)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, prompt_id in zip(axes, ["short", "medium", "long"]):
        sub = df[df["prompt_id"] == prompt_id]
        pivot = sub.pivot_table(index="model", columns="backend", values="gen_tps")
        pivot.plot(kind="bar", ax=ax, color=[PALETTE.get(b, "gray") for b in pivot.columns])
        ax.set_title(f"Generation TPS — {prompt_id}")
        ax.set_xlabel("")
        ax.set_ylabel("tokens/sec")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="backend")
    fig.tight_layout()
    fig.savefig(out / "01_gen_tps.png", dpi=150)
    plt.close(fig)


def chart_ttft(df: pd.DataFrame, out: Path) -> None:
    """백엔드별 TTFT (모델별)."""
    sub = df[df["prompt_id"] == "medium"]
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = sub.pivot_table(index="model", columns="backend", values="ttft_ms")
    pivot.plot(kind="bar", ax=ax, color=[PALETTE.get(b, "gray") for b in pivot.columns])
    ax.set_title("Time to First Token (TTFT) — medium prompt")
    ax.set_xlabel("")
    ax.set_ylabel("ms")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "02_ttft.png", dpi=150)
    plt.close(fig)


def chart_quantization(df: pd.DataFrame, out: Path) -> None:
    """양자화별 TPS 비교 (Q4 vs Q8)."""
    sub = df[df["prompt_id"] == "medium"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=sub, x="model", y="gen_tps", hue="quantization", ax=ax)
    ax.set_title("Quantization Impact on Generation TPS")
    ax.set_xlabel("")
    ax.set_ylabel("tokens/sec")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "03_quantization.png", dpi=150)
    plt.close(fig)


def chart_scaling(df: pd.DataFrame, out: Path) -> None:
    """모델 크기 스케일링 곡선 (Dense 모델)."""
    dense = df[(df["architecture"] == "dense") & (df["prompt_id"] == "medium") & (df["quantization"] == "Q4_K_M")]
    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in dense["backend"].unique():
        sub = dense[dense["backend"] == backend].sort_values("total_params")
        ax.plot(sub["model"], sub["gen_tps"], marker="o", label=backend, color=PALETTE.get(backend))
    ax.set_title("Dense Model Scaling (Q4_K_M, medium prompt)")
    ax.set_xlabel("Model")
    ax.set_ylabel("tokens/sec")
    ax.legend(title="backend")
    fig.tight_layout()
    fig.savefig(out / "04_dense_scaling.png", dpi=150)
    plt.close(fig)


def chart_dense_vs_moe(df: pd.DataFrame, out: Path) -> None:
    """Dense vs MoE 비교."""
    sub = df[(df["prompt_id"] == "medium") & (df["quantization"] == "Q4_K_M")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label in zip(axes, ["gen_tps", "peak_memory_gb"], ["Gen TPS (tok/s)", "Peak Memory (GB)"]):
        sns.barplot(data=sub, x="backend", y=metric, hue="architecture", ax=ax)
        ax.set_title(f"Dense vs MoE — {label}")
        ax.set_xlabel("")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out / "05_dense_vs_moe.png", dpi=150)
    plt.close(fig)


def chart_memory_efficiency(df: pd.DataFrame, out: Path) -> None:
    """메모리 효율: TPS per GB."""
    sub = df[(df["prompt_id"] == "medium") & (df["peak_memory_gb"] > 0)].copy()
    sub["tps_per_gb"] = sub["gen_tps"] / sub["peak_memory_gb"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sub, x="peak_memory_gb", y="gen_tps",
        hue="backend", style="model", s=120, ax=ax,
        palette=PALETTE,
    )
    ax.set_title("Memory Efficiency (Gen TPS vs Peak Memory)")
    ax.set_xlabel("Peak Memory (GB)")
    ax.set_ylabel("Gen TPS (tok/s)")
    fig.tight_layout()
    fig.savefig(out / "06_memory_efficiency.png", dpi=150)
    plt.close(fig)


def chart_heatmap(df: pd.DataFrame, out: Path) -> None:
    """종합 히트맵: 백엔드 × 모델 Gen TPS."""
    sub = df[(df["prompt_id"] == "medium") & (df["quantization"] == "Q4_K_M")]
    pivot = sub.pivot_table(index="model", columns="backend", values="gen_tps")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_title("Gen TPS Heatmap — Q4_K_M, medium prompt")
    ax.set_xlabel("Backend")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(out / "07_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("csv", help="결과 CSV 파일 경로")
    parser.add_argument("--out", default="charts", help="차트 출력 디렉토리")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    df = load(csv_path)
    print(f"데이터 로드: {len(df)}행")

    chart_gen_tps(df, out_dir)
    chart_ttft(df, out_dir)
    chart_quantization(df, out_dir)
    chart_scaling(df, out_dir)
    chart_dense_vs_moe(df, out_dir)
    chart_memory_efficiency(df, out_dir)
    chart_heatmap(df, out_dir)

    print(f"차트 저장 완료: {out_dir}/")


if __name__ == "__main__":
    main()
