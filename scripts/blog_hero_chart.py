"""Blog hero chart: memory bandwidth vs Gen TPS (BAEUM palette).

Run: uv run python scripts/blog_hero_chart.py
Output: ~/Workspace/baem1n.github.io/src/assets/images/llm-bench/bandwidth-vs-gen-tps.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

BAEUM_TEAL = "#83ced2"
BAEUM_TEAL_DARK = "#1a5c5f"
BAEUM_CORAL = "#ed7969"
BAEUM_CORAL_DARK = "#8a3028"
BAEUM_TEXT = "#1e293b"
BAEUM_SUB = "#64748b"
BAEUM_LINE = "#e2e8f0"

HARDWARE = [
    ("Ryzen AI MAX 395+", 256, 58.0, BAEUM_CORAL_DARK),
    ("DGX Spark GB10",    273, 59.6, BAEUM_TEAL_DARK),
    ("M5 Max",            546, 94.1, BAEUM_TEAL),
    ("RTX 3090 ×2",       936, 138.9, BAEUM_CORAL),
]

OUT = Path.home() / "Workspace/baem1n.github.io/src/assets/images/llm-bench/bandwidth-vs-gen-tps.png"


def pick_font() -> str:
    for name in ("Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", "Helvetica"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "sans-serif"


def main() -> None:
    font = pick_font()
    plt.rcParams.update({
        "font.family": font,
        "axes.edgecolor": BAEUM_SUB,
        "axes.labelcolor": BAEUM_TEXT,
        "xtick.color": BAEUM_SUB,
        "ytick.color": BAEUM_SUB,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    fig.patch.set_facecolor("#f8fafa")
    ax.set_facecolor("#f8fafa")

    xs = np.array([b for _, b, _, _ in HARDWARE])
    ys = np.array([t for _, _, t, _ in HARDWARE])

    slope, intercept = np.polyfit(xs, ys, 1)
    line_x = np.linspace(200, 1000, 100)
    line_y = slope * line_x + intercept
    ss_res = np.sum((ys - (slope * xs + intercept)) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.plot(line_x, line_y, linestyle="--", linewidth=1.2, color=BAEUM_LINE,
            label=f"linear fit  (R² = {r2:.3f})", zorder=1)

    label_offset = {
        "Ryzen AI MAX 395+": (0, -24, "center"),
        "DGX Spark GB10":    (18, 14, "left"),
        "M5 Max":            (18, 0, "left"),
        "RTX 3090 ×2":       (-18, 0, "right"),
    }
    for name, bw, tps, color in HARDWARE:
        ax.scatter(bw, tps, s=220, color=color, edgecolor="white",
                   linewidth=2, zorder=3)
        dx, dy, ha = label_offset[name]
        ax.annotate(name, (bw, tps), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11, color=BAEUM_TEXT, fontweight="600",
                    ha=ha, va="center")

    ax.set_xlim(180, 1020)
    ax.set_ylim(0, 160)
    ax.set_xlabel("Memory bandwidth (GB/s)", fontsize=12, color=BAEUM_TEXT,
                  labelpad=10)
    ax.set_ylabel("Generation throughput (tok/s)", fontsize=12, color=BAEUM_TEXT,
                  labelpad=10)
    ax.set_title("Memory bandwidth predicts Qwen3.5 generation speed",
                 fontsize=15, color=BAEUM_TEXT, fontweight="700",
                 loc="left", pad=30)
    ax.text(0, 1.02,
            "llama.cpp · Qwen3.5-35B-A3B MoE · Q4_K_M · gen-512 (5-run median)",
            transform=ax.transAxes, fontsize=10, color=BAEUM_SUB)

    ax.grid(True, linestyle="-", linewidth=0.6, color=BAEUM_LINE, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=10,
              labelcolor=BAEUM_SUB)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, facecolor="#f8fafa", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
