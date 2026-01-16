"""CLI entrypoint for model-agnostic embedding extraction and k-NN evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from models import available_models, load_model, resolve_device, resolve_dtype
from sdg6 import data, embedding, knn


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified embedding + k-NN evaluation.")
    parser.add_argument("--model", type=str, choices=available_models(), required=True, help="Which encoder to use.")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint path or Hugging Face id.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with train/val/test splits.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, cuda:1, ...).")

    parser.add_argument("--k-values", type=str, default="1,5,10,20,50,100", help="Comma-separated k list.")
    parser.add_argument("--knn-metric", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--knn-weights", type=str, choices=["uniform", "distance", "softmax"], default="uniform")
    parser.add_argument("--knn-softmax-temp", type=float, default=0.07, help="Temperature for softmax weights.")
    parser.add_argument("--pca-dim", type=int, default=0, help="Apply PCA to this dimension before k-NN (0 disables).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write embeddings + confusion matrices.")

    # DINO-only knobs
    dino = parser.add_argument_group("dino")
    dino.add_argument("--dino-arch", type=str, default="vit_base", help="DINO backbone (vit_small|vit_base|resnet50...).")
    dino.add_argument("--dino-patch-size", type=int, default=8, help="Patch size for ViT variants.")
    dino.add_argument("--dino-checkpoint-key", type=str, default="teacher", help="Checkpoint state_dict key.")
    dino.add_argument("--dino-resize", type=int, default=256, help="Resize shorter side before center crop.")
    dino.add_argument("--dino-crop", type=int, default=224, help="Center crop size.")

    # DINOv3-only knobs
    dv3 = parser.add_argument_group("dinov3")
    dv3.add_argument("--dinov3-weights-type", type=str, choices=["auto", "lvd", "sat"], default="auto")

    # Galileo-only knobs
    gal = parser.add_argument_group("galileo")
    gal.add_argument("--galileo-band-indices", type=str, default=None, help="Comma-separated band indices to select.")
    gal.add_argument("--galileo-band-names", type=str, default=None, help="Comma-separated band names.")
    gal.add_argument("--galileo-value-scale", type=float, default=1.0, help="Scale applied to raw band values.")
    gal.add_argument("--galileo-normalize", action="store_true", help="Apply Galileo pretraining normalization.")
    gal.add_argument("--galileo-compute-ndvi", action="store_true", help="Compute NDVI if B4/B8 present.")
    gal.add_argument("--galileo-input-resolution-m", type=int, default=10, help="Meters-per-pixel for positional encoding.")
    gal.add_argument("--galileo-patch-size", type=int, default=0, help="Patch size override (0 uses checkpoint config).")
    gal.add_argument("--galileo-default-month-index", type=int, default=5, help="Fallback month index (0-11).")
    gal.add_argument("--galileo-no-month-from-name", action="store_true", help="Disable month parsing from filename.")
    gal.add_argument("--galileo-no-pad-square", action="store_true", help="Disable square padding.")
    gal.add_argument("--galileo-no-pad-to-patch", action="store_true", help="Disable padding to patch size.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    data_root = args.data_dir.resolve()

    if args.model == "dino":
        adapter = load_model(
            "dino",
            weights=args.weights,
            device=device,
            dtype=dtype,
            arch=args.dino_arch,
            patch_size=args.dino_patch_size,
            checkpoint_key=args.dino_checkpoint_key,
            resize_size=args.dino_resize,
            crop_size=args.dino_crop,
        )
    elif args.model == "dinov3":
        adapter = load_model(
            "dinov3",
            weights=args.weights,
            device=device,
            dtype=dtype,
            weights_type=args.dinov3_weights_type,
        )
    else:
        adapter = load_model(
            "galileo",
            weights_dir=args.weights,
            device=device,
            dtype=dtype,
            input_resolution_m=args.galileo_input_resolution_m,
            patch_size=args.galileo_patch_size,
            band_indices=_parse_int_list(args.galileo_band_indices),
            band_names=_parse_str_list(args.galileo_band_names),
            value_scale=args.galileo_value_scale,
            normalize=args.galileo_normalize,
            compute_ndvi=args.galileo_compute_ndvi,
            default_month_index=args.galileo_default_month_index,
            parse_month_from_name=not args.galileo_no_month_from_name,
            pad_square=not args.galileo_no_pad_square,
            pad_to_patch_flag=not args.galileo_no_pad_to_patch,
        )

    loaders, class_names = data.build_dataloaders(
        data_root,
        transform=adapter.transform,
        reader=adapter.reader,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=("train", "val", "test"),
        shuffle_train=False,
        collate_fn=adapter.collate_fn,
    )
    if not class_names:
        raise RuntimeError("No class folders found; k-NN evaluation needs labeled data.")

    amp_dtype = dtype
    split_outputs = embedding.extract_all_splits(adapter, loaders, amp_dtype=amp_dtype, device=device)

    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        weight_tag = Path(args.weights).name.replace("/", "_")
        out_dir = Path("runs") / f"{args.model}-knn" / data_root.name / weight_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist embeddings for reuse.
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for split, (feats, labels, paths) in split_outputs.items():
        np.savez_compressed(emb_dir / f"{split}.npz", features=feats, labels=labels, paths=np.array(paths))

    train_feats, train_labels, _ = split_outputs.get("train", (np.zeros((0, 1)), np.zeros((0,)), []))
    eval_sets = {
        split: (feats, labels)
        for split, (feats, labels, _) in split_outputs.items()
        if split != "train"
    }
    if not eval_sets:
        raise RuntimeError("No evaluation splits found (need val/ or test/ folders).")

    results = knn.evaluate_knn(
        train_feats,
        train_labels,
        eval_sets,
        class_names=class_names,
        k_values=args.k_values,
        metric=args.knn_metric,
        weights=args.knn_weights,
        softmax_temp=args.knn_softmax_temp,
        pca_dim=args.pca_dim,
        output_dir=out_dir,
    )

    print("\n=== k-NN results ===")
    for res in results:
        k = res["k"]
        split_lines = []
        for split_name, metrics in res["splits"].items():
            acc = metrics.get("acc", float("nan"))
            split_lines.append(f"{split_name}: {acc*100:.2f}%")
        print(f"k={k:3d} -> " + " | ".join(split_lines))
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
