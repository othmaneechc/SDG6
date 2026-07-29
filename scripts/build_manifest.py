#!/usr/bin/env python3
"""Build a manifest of survey imagery instead of materializing ImageFolder trees.

Why a manifest
--------------
data/prepare_datasets.py symlinks every image into
``<dataset>/<split>/<class>/`` for each of six datasets. On a filesystem with a
per-user inode quota that is expensive: the sentinel datasets alone cost
2 x 188,876 = 377,752 extra inodes, and PW-s and SW-s point at *the same
images* with the same split -- only the label differs.

This script instead records one row per image with both labels and the split, so
the imagery is read in place. It reproduces prepare_datasets.py exactly:

* labels come from EA_SVC_B (piped) and EA_SVC_C (sewage), restricted to {0, 1},
  grouped by the EA GPS point rounded to 6 dp, and averaged within the EA;
* the binary label is ``fraction >= 0.5``;
* the split is drawn over deduplicated (lat, lon) keys with random.Random(seed)
  and the same ratio arithmetic, so a given seed reproduces the original assignment.

The within-EA fraction is kept alongside the thresholded label because it is
discarded by the >= 0.5 rule and is useful as a soft target.

Usage:
    python scripts/build_manifest.py --base-dir DATABASE/RAW --out data/manifest_sentinel.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

VALID_LABELS = {0, 1}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", type=Path, default=repo / "DATABASE" / "RAW")
    p.add_argument("--rounds", nargs="+", default=["R7", "R8", "R9"])
    p.add_argument("--modality", default="sentinel")
    p.add_argument("--out", type=Path, default=repo / "data" / "manifest_sentinel.csv")
    # Defaults match data/prepare_datasets.py, which is what the released
    # artifacts were produced with.
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expect-train", type=int, default=None,
                   help="Assert the train row count (e.g. 149632 from the "
                        "released k-NN classifier) to confirm the split matches.")
    return p.parse_args()


def _mode(series: pd.Series):
    """Most common non-null value, or None. Country and urban/rural are constant
    within an enumeration area in principle, but take the mode defensively."""
    s = series.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if len(m) else None


def load_labels(csv_paths: list[Path]) -> dict[tuple[float, float], dict[str, object]]:
    """Mirror of prepare_datasets.load_labels, plus COUNTRY and URBRUR.

    COUNTRY enables leave-one-country-out validation; URBRUR supports the
    urban/rural baseline and the settlement-stratified analysis.
    """
    frames = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Label CSV not found: {path}")
        cols = ["EA_GPS_LA", "EA_GPS_LO", "EA_SVC_B", "EA_SVC_C"]
        available = set(pd.read_csv(path, nrows=0).columns)
        extra = [c for c in ("COUNTRY", "URBRUR") if c in available]
        frames.append(pd.read_csv(path, usecols=cols + extra))
    df = pd.concat(frames, ignore_index=True)
    for col in ("EA_GPS_LA", "EA_GPS_LO", "EA_SVC_B", "EA_SVC_C"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["EA_SVC_B"].isin(VALID_LABELS) & df["EA_SVC_C"].isin(VALID_LABELS)]
    df = df.dropna(subset=["EA_GPS_LA", "EA_GPS_LO"])
    df["lat"] = df["EA_GPS_LA"].round(6)
    df["lon"] = df["EA_GPS_LO"].round(6)

    agg = {"EA_SVC_B": "mean", "EA_SVC_C": "mean"}
    for c in ("COUNTRY", "URBRUR"):
        if c in df.columns:
            agg[c] = _mode
    grouped = df.groupby(["lat", "lon"]).agg(agg)
    out: dict[tuple[float, float], dict[str, object]] = {}
    for (lat, lon), r in grouped.iterrows():
        out[(lat, lon)] = {
            "piped": float(r["EA_SVC_B"]),
            "sewage": float(r["EA_SVC_C"]),
            "country": r["COUNTRY"] if "COUNTRY" in grouped.columns else None,
            "urbrur": r["URBRUR"] if "URBRUR" in grouped.columns else None,
        }
    return out


def parse_image_filename(path: Path) -> tuple[float, float, str]:
    """Mirror of prepare_datasets.parse_image_filename."""
    stem = path.stem
    if "_image_" not in stem:
        raise ValueError(f"Unexpected filename: {path.name}")
    _, tail = stem.split("_image_", 1)
    parts = tail.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename: {path.name}")
    return round(float(parts[0]), 6), round(float(parts[1]), 6), "_".join(parts[2:])


def split_keys(keys: list[tuple[float, float]], train_ratio: float,
               val_ratio: float, seed: int) -> dict[tuple[float, float], str]:
    """Mirror of prepare_datasets.split_keys + invert_splits."""
    rng = random.Random(seed)
    ordered = sorted(set(keys))
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    lookup = {}
    for k in ordered[:n_train]:
        lookup[k] = "train"
    for k in ordered[n_train:n_train + n_val]:
        lookup[k] = "val"
    for k in ordered[n_train + n_val:]:
        lookup[k] = "test"
    return lookup


def main() -> int:
    args = parse_args()
    labels = load_labels([args.base_dir / f"{r}.csv" for r in args.rounds])
    print(f"Loaded {len(labels)} unique (lat, lon) label pairs.")

    rows = []
    unmatched = 0
    unparsed = 0
    for rnd in args.rounds:
        root = args.base_dir / rnd / args.modality
        if not root.is_dir():
            print(f"  [warn] missing {root}")
            continue
        found = 0
        for img in root.rglob(f"{args.modality}_image_*.tif"):
            try:
                lat, lon, date = parse_image_filename(img)
            except ValueError:
                unparsed += 1
                continue
            entry = labels.get((lat, lon))
            if entry is None:
                unmatched += 1
                continue
            rows.append({
                "path": str(img), "round": rnd, "date": date,
                "lat": lat, "lon": lon,
                "pw_frac": entry["piped"], "sw_frac": entry["sewage"],
                "country": entry["country"], "urbrur": entry["urbrur"],
            })
            found += 1
        print(f"  {rnd}/{args.modality}: {found} labelled images")

    if not rows:
        raise SystemExit("No labelled images found; check --base-dir/--modality.")

    df = pd.DataFrame(rows)
    lookup = split_keys(list(zip(df["lat"], df["lon"])),
                        args.train_ratio, args.val_ratio, args.seed)
    df["split"] = [lookup[(la, lo)] for la, lo in zip(df["lat"], df["lon"])]
    df["pw_label"] = (df["pw_frac"] >= 0.5).astype(int)
    df["sw_label"] = (df["sw_frac"] >= 0.5).astype(int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_loc = df[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"\nWrote {args.out}  ({len(df)} images, {n_loc} unique locations)")
    print(f"  unmatched (no label at that GPS point): {unmatched}; unparsed: {unparsed}")
    for split in ("train", "val", "test"):
        s = df[df["split"] == split]
        if s.empty:
            continue
        print(f"  {split:<5} images={len(s):>7}  locations={s[['lat','lon']].drop_duplicates().shape[0]:>6}"
              f"  pw_pos={int(s['pw_label'].sum()):>7}  sw_pos={int(s['sw_label'].sum()):>7}")

    if args.expect_train is not None:
        actual = int((df["split"] == "train").sum())
        if actual != args.expect_train:
            print(f"\nMISMATCH: train rows {actual} != expected {args.expect_train}")
            return 1
        print(f"\nMATCH: train rows == {actual}, split reproduces the released artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
