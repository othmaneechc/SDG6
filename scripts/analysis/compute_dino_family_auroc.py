#!/usr/bin/env python3
"""Compute DINO-family AUROC on PW-s and SW-s from SDG6 run artifacts.

This script reads evaluation outputs in:
  <runs_root>/dino-knn
  <runs_root>/dinov2-knn
  <runs_root>/dinov3-knn

Outputs are written to explicit output paths:
  - dino_family_auroc_pws_sws.csv
  - dino_family_auroc_pws_sws.md
  - dino_family_auroc_vs_k_pws_sws.png

Notes:
  - dino/dinov2 use signed max-vote confidence from prediction CSVs.
  - dinov3 uses the same score when prediction CSVs are available; otherwise
    it falls back to a hard-label AUROC proxy:
    0.5 * (TPR + TNR) from confusion matrices.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    runs_root_default = Path(os.environ.get("SDG6_RUNS_ROOT", repo_root / "runs")).expanduser()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=runs_root_default,
        help="Path to experiment runs directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "outputs" / "tables" / "dino_family_auroc_pws_sws.csv",
        help="Path where AUROC CSV output is written.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo_root / "outputs" / "reports" / "dino_family_auroc_pws_sws.md",
        help="Path where AUROC markdown summary is written.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=repo_root / "outputs" / "figures" / "dino_family_auroc_vs_k_pws_sws.png",
        help="Path where AUROC-vs-k figure is written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="PW-s,SW-s",
        help="Comma-separated datasets to include (default: PW-s,SW-s).",
    )
    return parser.parse_args()


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via Mann-Whitney rank statistic with average tie handling."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_total = y_true.size
    n_pos = int(y_true.sum())
    n_neg = int(n_total - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]

    ranks_sorted = np.empty(n_total, dtype=np.float64)
    i = 0
    while i < n_total:
        j = i
        while j + 1 < n_total and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j + 2) / 2.0  # average 1-based rank
        i = j + 1

    ranks = np.empty(n_total, dtype=np.float64)
    ranks[order] = ranks_sorted
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def load_dino_and_dinov2(
    runs_root: Path, keep_datasets: set[str]
) -> list[dict[str, str | int | float]]:
    records: list[dict[str, str | int | float]] = []
    patterns = [
        runs_root / "dino-knn" / "*" / "checkpoint.pth" / "confusion" / "test_predictions_k*.csv",
        runs_root
        / "dinov2-knn"
        / "*"
        / "teacher_checkpoint.pth"
        / "confusion"
        / "test_predictions_k*.csv",
    ]

    for pattern in patterns:
        for file_path_str in sorted(glob.glob(str(pattern))):
            file_path = Path(file_path_str)
            rel = file_path.relative_to(runs_root)
            parts = rel.parts
            run_dir, dataset, checkpoint = parts[0], parts[1], parts[2]
            if dataset not in keep_datasets:
                continue

            k_match = re.search(r"_k(\d+)\.csv$", file_path.name)
            if not k_match:
                continue
            k = int(k_match.group(1))

            model = "dino" if run_dir == "dino-knn" else "dinov2"
            y_true, pred_idx, conf = [], [], []
            with file_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    y_true.append(int(row["true_label_idx"]))
                    pred_idx.append(int(row["pred_label_idx"]))
                    conf.append(float(row["confidence"]))

            y_true_arr = np.asarray(y_true, dtype=np.int64)
            pred_idx_arr = np.asarray(pred_idx, dtype=np.int64)
            conf_arr = np.asarray(conf, dtype=np.float64)

            # confidence is max predicted-class vote; use signed value so
            # higher means more evidence for positive class (idx=1).
            signed_score = np.where(pred_idx_arr == 1, conf_arr, -conf_arr)
            auroc = auc_rank(y_true_arr, signed_score)

            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k": k,
                    "auroc": auroc,
                    "metric_type": "auroc_signed_conf",
                    "checkpoint": checkpoint,
                    "source_file": str(file_path),
                }
            )

    return records


def load_prediction_csv_auroc(
    file_path: Path,
    *,
    dataset: str,
    model: str,
    checkpoint: str,
) -> dict[str, str | int | float] | None:
    k_match = re.search(r"_k(\d+)\.csv$", file_path.name)
    if not k_match:
        return None
    k = int(k_match.group(1))

    y_true, pred_idx, conf = [], [], []
    with file_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["true_label_idx"]))
            pred_idx.append(int(row["pred_label_idx"]))
            conf.append(float(row["confidence"]))

    if not y_true:
        return None

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    pred_idx_arr = np.asarray(pred_idx, dtype=np.int64)
    conf_arr = np.asarray(conf, dtype=np.float64)
    signed_score = np.where(pred_idx_arr == 1, conf_arr, -conf_arr)
    auroc = auc_rank(y_true_arr, signed_score)
    return {
        "dataset": dataset,
        "model": model,
        "k": k,
        "auroc": auroc,
        "metric_type": "auroc_signed_conf",
        "checkpoint": checkpoint,
        "source_file": str(file_path),
    }


def parse_confusion_from_txt(path: Path) -> tuple[int, int, int, int]:
    lines = path.read_text().splitlines()
    marker = "Confusion matrix (rows=true, cols=pred):"
    idx = lines.index(marker)
    row_0 = lines[idx + 1].strip().split()
    row_1 = lines[idx + 2].strip().split()
    tn, fp = int(row_0[0]), int(row_0[1])
    fn, tp = int(row_1[0]), int(row_1[1])
    return tn, fp, fn, tp


def load_dinov3_from_predictions(
    runs_root: Path, keep_datasets: set[str]
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    pattern = runs_root / "dinov3-knn" / "*" / "*" / "confusion" / "test_predictions_k*.csv"
    for file_path_str in sorted(glob.glob(str(pattern))):
        file_path = Path(file_path_str)
        rel = file_path.relative_to(runs_root)
        parts = rel.parts
        dataset, checkpoint = parts[1], parts[2]
        if dataset not in keep_datasets:
            continue
        row = load_prediction_csv_auroc(
            file_path,
            dataset=dataset,
            model="dinov3",
            checkpoint=checkpoint,
        )
        if row is not None:
            rows.append(row)
    return rows


def load_dinov3_proxy(
    runs_root: Path, keep_datasets: set[str]
) -> list[dict[str, str | int | float]]:
    raw_rows: list[dict[str, str | int | float]] = []
    pattern = runs_root / "dinov3-knn" / "*" / "*" / "confusion" / "test_k*.txt"

    for file_path_str in sorted(glob.glob(str(pattern))):
        file_path = Path(file_path_str)
        rel = file_path.relative_to(runs_root)
        parts = rel.parts
        dataset, checkpoint = parts[1], parts[2]
        if dataset not in keep_datasets:
            continue

        k_match = re.search(r"test_k(\d+)\.txt$", file_path.name)
        if not k_match:
            continue
        k = int(k_match.group(1))

        try:
            tn, fp, fn, tp = parse_confusion_from_txt(file_path)
        except (ValueError, IndexError):
            continue

        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        auroc_proxy = 0.5 * (tpr + tnr)
        raw_rows.append(
            {
                "dataset": dataset,
                "model": "dinov3",
                "k": k,
                "auroc": auroc_proxy,
                "metric_type": "auroc_hardlabel_proxy",
                "checkpoint": checkpoint,
                "source_file": str(file_path),
            }
        )

    # Keep best checkpoint per dataset/k so dinov3 is a single line per dataset.
    best_per_dataset_k: dict[tuple[str, int], dict[str, str | int | float]] = {}
    for row in raw_rows:
        key = (str(row["dataset"]), int(row["k"]))
        if key not in best_per_dataset_k or float(row["auroc"]) > float(best_per_dataset_k[key]["auroc"]):
            best_per_dataset_k[key] = row

    return sorted(best_per_dataset_k.values(), key=lambda r: (str(r["dataset"]), int(r["k"])))


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fields = ["dataset", "model", "k", "auroc", "metric_type", "checkpoint", "source_file"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_summary(path: Path, rows: list[dict[str, str | int | float]], datasets: list[str]) -> None:
    model_order = ["dino", "dinov2", "dinov3"]
    with path.open("w") as f:
        f.write("# DINO-family AUROC (PW-s and SW-s)\n\n")
        f.write("Models: dino, dinov2, dinov3\n\n")
        has_d3_proxy = any(r["model"] == "dinov3" and r["metric_type"] == "auroc_hardlabel_proxy" for r in rows)
        if has_d3_proxy:
            f.write("Note: some/all dinov3 values are hard-label AUROC proxies from confusion matrices.\n\n")
        else:
            f.write("Note: dinov3 values are computed from per-sample prediction confidences.\n\n")
        for dataset in datasets:
            f.write(f"## {dataset}\n\n")
            f.write("| model | k | AUROC |\n")
            f.write("|---|---:|---:|\n")
            for model in model_order:
                model_rows = sorted(
                    [r for r in rows if r["dataset"] == dataset and r["model"] == model],
                    key=lambda r: int(r["k"]),
                )
                for row in model_rows:
                    f.write(f"| {model} | {int(row['k'])} | {float(row['auroc']):.6f} |\n")
            f.write("\n")
            f.write("| model | best k | best AUROC |\n")
            f.write("|---|---:|---:|\n")
            for model in model_order:
                model_rows = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
                if not model_rows:
                    continue
                best = max(model_rows, key=lambda r: float(r["auroc"]))
                f.write(f"| {model} | {int(best['k'])} | {float(best['auroc']):.6f} |\n")
            f.write("\n")


def plot_auroc(path: Path, rows: list[dict[str, str | int | float]], datasets: list[str]) -> None:
    model_order = ["dino", "dinov2", "dinov3"]
    colors = {"dino": "#1f77b4", "dinov2": "#ff7f0e", "dinov3": "#2ca02c"}
    labels = {"dino": "dino", "dinov2": "dinov2", "dinov3": "dinov3"}

    y_vals = [float(r["auroc"]) for r in rows]
    y_min = max(0.0, min(y_vals) - 0.02)
    y_max = min(1.0, max(y_vals) + 0.02)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.8), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]
        for model in model_order:
            model_rows = sorted(
                [r for r in rows if r["dataset"] == dataset and r["model"] == model],
                key=lambda r: int(r["k"]),
            )
            if not model_rows:
                continue
            metric_type = str(model_rows[0]["metric_type"])
            is_proxy = any(str(r["metric_type"]) == "auroc_hardlabel_proxy" for r in model_rows)
            linestyle = "--" if is_proxy else "-"
            label = labels[model] + (" (proxy)" if is_proxy else "")
            ax.plot(
                [int(r["k"]) for r in model_rows],
                [float(r["auroc"]) for r in model_rows],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=colors[model],
                linestyle=linestyle,
                label=label,
            )

        ax.set_title(f"{dataset} test AUROC vs k")
        ax.set_xlabel("k")
        ax.set_ylabel("AUROC")
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)


def print_best(rows: list[dict[str, str | int | float]], datasets: list[str]) -> None:
    model_order = ["dino", "dinov2", "dinov3"]
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        for model in model_order:
            model_rows = sorted(
                [r for r in rows if r["dataset"] == dataset and r["model"] == model],
                key=lambda r: int(r["k"]),
            )
            if not model_rows:
                continue
            series = ", ".join([f"k={int(r['k'])}: {float(r['auroc']):.6f}" for r in model_rows])
            print(f"{model}: {series}")
            best = max(model_rows, key=lambda r: float(r["auroc"]))
            print(f"best {model}: k={int(best['k'])}, auroc={float(best['auroc']):.6f}")


def main() -> None:
    args = parse_args()
    keep_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | int | float]] = []
    records.extend(load_dino_and_dinov2(args.runs_root, keep_datasets))
    dinov3_rows = load_dinov3_from_predictions(args.runs_root, keep_datasets)
    if dinov3_rows:
        # Keep best checkpoint per dataset/k to preserve one dinov3 line per dataset.
        best_per_dataset_k: dict[tuple[str, int], dict[str, str | int | float]] = {}
        for row in dinov3_rows:
            key = (str(row["dataset"]), int(row["k"]))
            if key not in best_per_dataset_k or float(row["auroc"]) > float(best_per_dataset_k[key]["auroc"]):
                best_per_dataset_k[key] = row
        records.extend(sorted(best_per_dataset_k.values(), key=lambda r: (str(r["dataset"]), int(r["k"]))))
    else:
        records.extend(load_dinov3_proxy(args.runs_root, keep_datasets))
    records = sorted(records, key=lambda r: (str(r["dataset"]), str(r["model"]), int(r["k"])))

    if not records:
        raise RuntimeError("No AUROC records found. Check runs path and dataset filters.")

    datasets = [d for d in ["PW-s", "SW-s"] if d in keep_datasets]
    if not datasets:
        datasets = sorted({str(r["dataset"]) for r in records})

    csv_path = args.output_csv
    md_path = args.output_md
    png_path = args.output_figure

    write_csv(csv_path, records)
    write_summary(md_path, records, datasets)
    plot_auroc(png_path, records, datasets)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {png_path}")
    print_best(records, datasets)


if __name__ == "__main__":
    main()
