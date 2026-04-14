"""Consolidate all benchmark CSV results into per-device and all-devices merged files.

Output:
  results/consolidated/
    mac.csv
    dgx-spark.csv
    ryzen-ai.csv
    linux-3090x2.csv
    all_devices.csv
"""
import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "consolidated"
OUT.mkdir(exist_ok=True)

# Mapping: device -> list of CSV files
DEVICE_FILES = {
    "mac": [
        # Track A
        RESULTS / "trackA_20260405_012423.csv",
        RESULTS / "trackA_20260405_013727.csv",
        RESULTS / "trackA_llamacpp_20260406_172334.csv",
        RESULTS / "trackA_mlx_35b_q4.csv",
        RESULTS / "trackA_mlx_35b.csv",
        RESULTS / "trackA_mlx_remaining.csv",
        RESULTS / "trackA_ollama_20260407_054117.csv",
        RESULTS / "trackA_ollama_20260407_172927.csv",
        RESULTS / "trackA_ollama_remaining_20260409_000320.csv",
        RESULTS / "trackA_ollama_resume_20260408_214448.csv",
        RESULTS / "ollama_122b_prefill128k.csv",
        RESULTS / "ollama_27b_q8_prefill.csv",
        RESULTS / "sup_ollama_27b_q8_prefill.csv",
        # Track B
        RESULTS / "trackB_llamacpp_20260405_012423.csv",
        RESULTS / "trackB_llamacpp_20260405_013727.csv",
    ],
    "dgx-spark": list((RESULTS / "remote" / "dgx-spark").glob("*.csv")),
    "ryzen-ai": list((RESULTS / "remote" / "ryzen-ai").glob("*.csv")),
    "linux-3090x2": list((RESULTS / "remote" / "3090").glob("*.csv")),
}


def read_csv_rows(path: Path) -> list[dict]:
    """Read CSV, skip header, return list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove exact duplicate rows (same timestamp+backend+model+quant+track+run_number)."""
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("timestamp", ""),
            r.get("backend", ""),
            r.get("model", ""),
            r.get("quantization", ""),
            r.get("track_id", ""),
            r.get("run_number", ""),
        )
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def filter_qwen_only(rows: list[dict]) -> list[dict]:
    """Keep only qwen3.5 models, drop gemma4."""
    return [r for r in rows if r.get("model", "").startswith("qwen3.5")]


def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"  SKIP {path.name} (0 rows)")
        return
    fieldnames = rows[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


def main():
    all_rows = []
    for device, files in DEVICE_FILES.items():
        print(f"\n[{device}]")
        device_rows = []
        for f in sorted(files):
            if not f.exists():
                print(f"  WARN: {f} not found, skipping")
                continue
            rows = read_csv_rows(f)
            print(f"  {f.name}: {len(rows)} rows")
            device_rows.extend(rows)

        # Filter Qwen only, deduplicate
        device_rows = filter_qwen_only(device_rows)
        device_rows = deduplicate(device_rows)
        # Sort by timestamp
        device_rows.sort(key=lambda r: r.get("timestamp", ""))

        write_csv(device_rows, OUT / f"{device}.csv")
        all_rows.extend(device_rows)

    # All devices combined
    all_rows = deduplicate(all_rows)
    all_rows.sort(key=lambda r: (r.get("hardware_id", ""), r.get("timestamp", "")))
    write_csv(all_rows, OUT / "all_devices.csv")

    # Summary
    print("\n=== Summary ===")
    for device in DEVICE_FILES:
        p = OUT / f"{device}.csv"
        if p.exists():
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {device}: {n} rows")
    p = OUT / "all_devices.csv"
    if p.exists():
        with open(p) as f:
            n = sum(1 for _ in f) - 1
        print(f"  all_devices: {n} rows")


if __name__ == "__main__":
    main()
