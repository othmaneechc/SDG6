#!/usr/bin/env python3
"""Prepare SDG6 datasets for piped water and sewage labels."""
import argparse
import os
import random
import shutil
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

VALID_LABELS = {0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test splits and symlink images.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/work/lamlab/data"),
        help="Root directory containing R7, R8, R9 folders.",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        default=["R7", "R8", "R9"],
        help="Survey rounds to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Where to write the dataset folders (PW-s, SW-s, ...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for splits.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train fraction for each modality."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation fraction for each modality."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Parallel workers for linking/copying.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (symlink by default).",
    )
    return parser.parse_args()


def load_labels(csv_paths: Sequence[Path]) -> Dict[Tuple[float, float], Dict[str, float]]:
    frames = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Label CSV not found: {path}")
        frames.append(
            pd.read_csv(
                path,
                usecols=["EA_GPS_LA", "EA_GPS_LO", "EA_SVC_B", "EA_SVC_C"],
            )
        )
    df = pd.concat(frames, ignore_index=True)

    df["EA_GPS_LA"] = pd.to_numeric(df["EA_GPS_LA"], errors="coerce")
    df["EA_GPS_LO"] = pd.to_numeric(df["EA_GPS_LO"], errors="coerce")
    df["EA_SVC_B"] = pd.to_numeric(df["EA_SVC_B"], errors="coerce")
    df["EA_SVC_C"] = pd.to_numeric(df["EA_SVC_C"], errors="coerce")

    df = df[df["EA_SVC_B"].isin(VALID_LABELS) & df["EA_SVC_C"].isin(VALID_LABELS)]
    df = df.dropna(subset=["EA_GPS_LA", "EA_GPS_LO"])
    df["lat"] = df["EA_GPS_LA"].round(6)
    df["lon"] = df["EA_GPS_LO"].round(6)

    grouped = df.groupby(["lat", "lon"]).agg({"EA_SVC_B": "mean", "EA_SVC_C": "mean"})
    labels: Dict[Tuple[float, float], Dict[str, float]] = {}
    for (lat, lon), row in grouped.iterrows():
        labels[(lat, lon)] = {"piped": float(row["EA_SVC_B"]), "sewage": float(row["EA_SVC_C"])}
    return labels


def parse_image_filename(path: Path) -> Tuple[float, float, str]:
    """Extract (lat, lon, date_key) from filenames like sentinel_image_<lat>_<lon>_<date>.tif."""
    stem = path.stem
    if "_image_" not in stem:
        raise ValueError(f"Unexpected filename: {path.name}")
    _, tail = stem.split("_image_", 1)
    parts = tail.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename: {path.name}")
    lat = round(float(parts[0]), 6)
    lon = round(float(parts[1]), 6)
    date_key = "_".join(parts[2:])
    return lat, lon, date_key


def collect_images(modality: str, roots: Iterable[Path]) -> Dict[Tuple[float, float], List[Tuple[str, Path]]]:
    """Collect all images per (lat, lon), keeping every date."""
    collected: Dict[Tuple[float, float], List[Tuple[str, Path]]] = defaultdict(list)
    pattern = f"{modality}_image_*.tif"
    for root in roots:
        if not root.is_dir():
            continue
        iter_paths = root.rglob(pattern)
        for img_path in tqdm(iter_paths, desc=f"Indexing {modality} in {root}", unit="file"):
            try:
                lat, lon, date_key = parse_image_filename(img_path)
            except ValueError:
                continue
            key = (lat, lon)
            collected[key].append((date_key, img_path))
    # Sort lists deterministically by date then filename.
    for key, entries in collected.items():
        entries.sort(key=lambda x: (x[0], x[1].name))
    return collected


def split_keys(keys: List[Tuple[float, float]], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    # Deduplicate so each (lat, lon) is assigned to exactly one split.
    ordered = sorted(set(keys))
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train : n_train + n_val]),
        "test": set(ordered[n_train + n_val :]),
    }


def invert_splits(split_sets: Dict[str, set]) -> Dict[Tuple[float, float], str]:
    lookup: Dict[Tuple[float, float], str] = {}
    for split, keys in split_sets.items():
        for key in keys:
            lookup[key] = split
    return lookup


def label_folder(label_key: str, label_value: float) -> str:
    """Map label value to a class-specific folder name."""
    is_positive = label_value >= 0.5
    if label_key == "piped":
        return "pipedwater" if is_positive else "no_pipedwater"
    if label_key == "sewage":
        return "sewage" if is_positive else "no_sewage"
    raise ValueError(f"Unknown label key: {label_key}")


def materialize(job: Tuple[Path, Path], copy_flag: bool) -> None:
    src, dst = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_flag:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def run_materialization(jobs: List[Tuple[Path, Path]], copy_flag: bool, workers: int, desc: str) -> None:
    if not jobs:
        return
    worker = partial(materialize, copy_flag=copy_flag)
    with Pool(processes=workers) as pool:
        list(tqdm(pool.imap_unordered(worker, jobs), total=len(jobs), desc=desc, unit="file"))


def summarize(dataset_name: str, tasks: List[Tuple[Path, Path, float, str]]) -> None:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "pos": 0})
    for _, _, label, split in tasks:
        summary[split]["count"] += 1
        if label >= 0.5:
            summary[split]["pos"] += 1
    print(f"\n{dataset_name}:")
    for split in ("train", "val", "test"):
        count = summary[split]["count"]
        pos = summary[split]["pos"]
        neg = count - pos
        print(f"  {split:<5} {count:6d} images | pos={pos:6d} neg={neg:6d}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [args.base_dir / f"{name}.csv" for name in ("R7", "R8", "R9")]
    labels = load_labels(csv_paths)
    print(f"Loaded {len(labels)} unique (lat, lon) label pairs.")

    sentinel_dirs = [args.base_dir / rnd / "sentinel" for rnd in args.rounds]
    landsat_dirs = [args.base_dir / rnd / "landsat" for rnd in args.rounds]

    sentinel_images = collect_images("sentinel", sentinel_dirs)
    landsat_images = collect_images("landsat", landsat_dirs)

    merged_images: Dict[Tuple[float, float], List[Tuple[str, Path]]] = defaultdict(list)
    for key, entries in sentinel_images.items():
        merged_images[key].extend(entries)
    for key, entries in landsat_images.items():
        merged_images[key].extend(entries)
    for key in merged_images:
        merged_images[key].sort(key=lambda x: (x[0], x[1].name))

    modalities = {
        "sentinel": sentinel_images,
        "landsat": landsat_images,
        "merged": merged_images,
    }

    dataset_map = {
        ("sentinel", "piped"): "PW-s",
        ("sentinel", "sewage"): "SW-s",
        ("landsat", "piped"): "PW-l",
        ("landsat", "sewage"): "SW-l",
        ("merged", "piped"): "PW-m",
        ("merged", "sewage"): "SW-m",
    }

    for modality, image_map in modalities.items():
        records = []
        for key, img_path in image_map.items():
            if key not in labels:
                continue
            entry = labels[key]
            for date_key, path in img_path:
                records.append(
                    {
                        "key": key,
                        "path": path,
                        "date": date_key,
                        "piped": entry["piped"],
                        "sewage": entry["sewage"],
                    }
                )
        if not records:
            print(f"No labeled samples found for modality '{modality}'.")
            continue

        split_sets = split_keys([r["key"] for r in records], args.train_ratio, args.val_ratio, args.seed)
        split_lookup = invert_splits(split_sets)

        for label_key in ("piped", "sewage"):
            dataset_name = dataset_map[(modality, label_key)]
            dataset_root = args.output_dir / dataset_name
            tasks: List[Tuple[Path, Path, float, str]] = []
            jobs: List[Tuple[Path, Path]] = []
            for rec in records:
                split = split_lookup[rec["key"]]
                label = float(rec[label_key])
                class_dir = label_folder(label_key, label)
                dst = dataset_root / split / class_dir / rec["path"].name
                tasks.append((rec["path"], dst, label, split))
                jobs.append((rec["path"].resolve(), dst))

            run_materialization(jobs, args.copy, args.workers, f"Writing {dataset_name}")
            summarize(dataset_name, tasks)


if __name__ == "__main__":
    main()

"""
python3 data/prepare_datasets.py --output-dir /home/mila/e/echchabo/scratch
"""
