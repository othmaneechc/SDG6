#!/usr/bin/env python3
"""Quick integrity check for downloaded country tiles (TIF/ZIP)."""

from __future__ import annotations

import argparse
import zipfile
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm

DEFAULT_EXTS = (".tif", ".tiff", ".zip")


def iter_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def check_tif(path: Path) -> Tuple[bool, str]:
    if path.stat().st_size == 0:
        return False, "empty file"
    # Prefer tifffile, then rasterio, then PIL.
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(str(path)):
            return True, "ok"
    except ImportError:
        pass
    except Exception as exc:
        return False, f"tifffile failed: {exc}"

    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as src:
            _ = src.meta
        return True, "ok"
    except ImportError:
        pass
    except Exception as exc:
        return False, f"rasterio failed: {exc}"

    try:
        with Image.open(path) as img:
            img.verify()
        return True, "ok"
    except Exception as exc:
        return False, f"PIL verify failed: {exc}"


def check_zip(path: Path) -> Tuple[bool, str]:
    if path.stat().st_size == 0:
        return False, "empty file"
    try:
        with zipfile.ZipFile(path, "r") as zf:
            corrupt = zf.testzip()
            if corrupt:
                return False, f"zip corrupt entry: {corrupt}"
        return True, "ok"
    except Exception as exc:
        return False, f"zip open failed: {exc}"


def _check_file(fpath: Path) -> Tuple[bool, str, str]:
    if fpath.suffix.lower() in {".tif", ".tiff"}:
        ok, msg = check_tif(fpath)
    elif fpath.suffix.lower() == ".zip":
        ok, msg = check_zip(fpath)
    else:
        return True, "skipped", str(fpath)
    return ok, msg, str(fpath)


def validate_country(country_dir: Path, limit: int | None, exts: Iterable[str], sample: int) -> Tuple[int, int, List[str]]:
    files = []
    for fpath in iter_files(country_dir, exts):
        files.append(fpath)
        if limit and len(files) >= limit:
            break

    if not files:
        return 0, 0, []

    failures = 0
    samples: List[str] = []
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        for ok, msg, path_str in tqdm(
            pool.imap_unordered(_check_file, files),
            total=len(files),
            desc=f"Files {country_dir.name}",
            unit="file",
        ):
            if not ok:
                failures += 1
                if len(samples) < sample:
                    samples.append(f"{Path(path_str).name}: {msg}")
    return len(files), failures, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate downloaded tiles for corruption")
    parser.add_argument("--countries-dir", type=Path, required=True, help="Root with per-country folders")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated subset of countries to check")
    parser.add_argument("--limit", type=int, default=50, help="Max files per country to check (0 = all)")
    parser.add_argument("--sample", type=int, default=5, help="Sample N failure reasons per country")
    args = parser.parse_args()

    root = args.countries_dir
    if not root.exists():
        raise FileNotFoundError(f"Countries dir not found: {root}")

    subset = None
    if args.countries:
        subset = {c.strip().lower() for c in args.countries.split(",") if c.strip()}

    total_failures = 0
    total_checked = 0
    for cdir in tqdm(sorted(p for p in root.iterdir() if p.is_dir()), desc="Countries", unit="country"):
        if subset and cdir.name.lower() not in subset:
            continue
        limit = None if args.limit == 0 else args.limit
        checked, failures, samples = validate_country(cdir, limit=limit, exts=DEFAULT_EXTS, sample=args.sample)
        total_checked += checked
        total_failures += failures
        if failures:
            sample_msg = "; ".join(samples)
            print(f"{cdir.name}: {failures}/{checked} failed (e.g., {sample_msg})")
    print(f"TOTAL failed {total_failures} of {total_checked} checked")


if __name__ == "__main__":  # pragma: no cover
    main()
