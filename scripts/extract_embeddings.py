#!/usr/bin/env python3
"""Extract encoder embeddings for every image in the manifest, once.

Features do not depend on the task: PW-s and SW-s use the same imagery and the
same split, and differ only in their labels. So this runs the encoder once and
stores both label columns alongside the features, which halves the GPU work and
makes the embeddings reusable for re-splitting (spatial blocks,
leave-one-country-out) without re-encoding.

Output, one file per split under ``<out>/<model>/``::

    train.npz  features (N, D) float32
               paths    (N,)   str
               pw_label (N,)   int64
               sw_label (N,)   int64
               lat, lon (N,)   float64

Usage:
    python scripts/extract_embeddings.py --model dinov2 \
        --weights runs/pretrained/dinov2/teacher_checkpoint.pth \
        --dinov2-config scripts/configs/sat_vit.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from models import load_model  # noqa: E402
from sdg6.embedding import extract_embeddings  # noqa: E402
from sdg6.manifest import build_manifest_dataloaders, load_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2")
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifest_sentinel.csv")
    p.add_argument("--out", type=Path, default=REPO / "runs" / "embeddings")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--dinov2-config", type=Path, default=None,
                   help="Required for the dinov2 adapter.")
    p.add_argument("--checkpoint-key", default="teacher")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0,
                   help="Debug: cap rows per split (0 = all).")
    # Galileo input-scaling controls, for the preprocessing diagnostic. The tiles
    # are uint8 RGB (0-255) but Galileo normalizes against Sentinel-2 reflectance
    # stats (0-10000 scale); --value-scale multiplies the raw input before
    # Galileo's own normalize step so it can be brought into the expected range.
    p.add_argument("--value-scale", type=float, default=1.0)
    p.add_argument("--galileo-normalize", dest="galileo_normalize",
                   action="store_true", default=True)
    p.add_argument("--no-galileo-normalize", dest="galileo_normalize",
                   action="store_false")
    return p.parse_args()


def build_adapter(args: argparse.Namespace):
    # Galileo takes a weights *directory* and a band specification rather than a
    # checkpoint path; defaults here mirror scripts/configs/galileo.yaml.
    if args.model == "galileo":
        print(f"galileo input: value_scale={args.value_scale} "
              f"normalize={args.galileo_normalize}")
        return load_model(
            "galileo",
            weights_dir=args.weights,
            input_resolution_m=10,
            patch_size=0,
            band_indices=[0, 1, 2],
            band_names=["B2", "B3", "B4"],
            value_scale=args.value_scale,
            normalize=args.galileo_normalize,
            compute_ndvi=False,
            default_month_index=5,
            pad_square=True,
            pad_to_patch_flag=True,
        )

    kwargs = dict(weights=args.weights, resize_size=args.resize, crop_size=args.crop)
    if args.model == "dinov2":
        if args.dinov2_config is None:
            raise SystemExit("--dinov2-config is required for the dinov2 adapter")
        kwargs.update(config_file=args.dinov2_config, checkpoint_key=args.checkpoint_key)
    elif args.model in {"dino", "dinov3"}:
        kwargs.update(checkpoint_key=args.checkpoint_key)
    return load_model(args.model, **kwargs)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest.groupby("split", group_keys=False).head(args.limit)
    print(f"manifest: {len(manifest)} rows from {args.manifest}")

    adapter = build_adapter(args)
    print(f"adapter: {adapter.name}  device={adapter.device}  dim={adapter.output_dim}")

    # Labels are attached from the manifest afterwards, so the task passed here
    # only selects which label the loader carries; features are identical.
    loaders, _ = build_manifest_dataloaders(
        manifest, task="pw", transform=adapter.transform, reader=adapter.reader,
        batch_size=args.batch_size, num_workers=args.num_workers,
        splits=args.splits, collate_fn=adapter.collate_fn,
    )

    out_dir = args.out / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, loader in loaders.items():
        feats, _, paths = extract_embeddings(adapter, loader, desc=f"{split}")
        if len(paths) != len(feats):
            raise RuntimeError(
                f"{split}: {len(paths)} paths vs {len(feats)} features -- "
                "some samples were dropped, so labels cannot be aligned by position."
            )
        # Align labels to the returned paths rather than trusting row order.
        sub = manifest.set_index("path")
        idx = sub.loc[paths]
        np.savez_compressed(
            out_dir / f"{split}.npz",
            features=feats.astype(np.float32),
            paths=np.array(paths),
            pw_label=idx["pw_label"].to_numpy(np.int64),
            sw_label=idx["sw_label"].to_numpy(np.int64),
            lat=idx["lat"].to_numpy(np.float64),
            lon=idx["lon"].to_numpy(np.float64),
        )
        print(f"wrote {out_dir / f'{split}.npz'}  features={feats.shape}")

    print("EMBEDDING EXTRACTION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
