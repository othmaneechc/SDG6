"""CLI entrypoint for model-agnostic embedding extraction and k-NN evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from models import available_models, load_model, resolve_device, resolve_dtype
from sdg6 import data, embedding, knn


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if not value:
        return None
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if not value:
        return None
    return [v.strip() for v in str(value).split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified embedding + k-NN evaluation.")
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config file to set defaults.")
    parser.add_argument("--model", type=str, choices=available_models(), required=False, help="Which encoder to use.")
    parser.add_argument("--weights", type=str, required=False, help="Checkpoint path or Hugging Face id.", nargs="+")
    parser.add_argument("--data-dir", type=Path, required=False, help="Root directory with train/val/test splits.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="If set with --datasets, resolves each dataset under this root.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional space/comma separated dataset names under --data-root.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, cuda:1, ...).")

    parser.add_argument("--k-values", type=str, default="1,5,10,20,50,100", help="Comma-separated k list.")
    parser.add_argument("--knn-metric", type=str, choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--knn-weights", type=str, choices=["uniform", "distance", "softmax"], default="uniform")
    parser.add_argument("--knn-softmax-temp", type=float, default=0.07, help="Temperature for softmax weights.")
    parser.add_argument("--pca-dim", type=int, default=0, help="Apply PCA to this dimension before k-NN (0 disables).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final output directory (skip dataset/weights nesting if set).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for outputs when --output-dir is not provided.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Persist per-split embeddings to disk (npz).",
    )
    parser.add_argument(
        "--no-save-embeddings",
        dest="save_embeddings",
        action="store_false",
        help="Do not write embeddings to disk; still runs k-NN in memory.",
    )
    parser.set_defaults(save_embeddings=False)

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
    # Distributed setup (torchrun/SLURM).
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed = world_size > 1
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Two-stage parse to allow config defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre.parse_known_args()

    parser = build_parser()

    # Apply defaults from config if provided.
    if pre_args.config and pre_args.config.exists():
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required when using --config") from exc
        with pre_args.config.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict):
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    if not args.model:
        parser.error("Provide --model or set 'model' in the config file.")
    if not args.weights:
        parser.error("Provide --weights or set 'weights' in the config file.")

    # Normalize paths in case they came from config defaults.
    if args.data_root and not isinstance(args.data_root, Path):
        args.data_root = Path(args.data_root)
    if args.data_dir and not isinstance(args.data_dir, Path):
        args.data_dir = Path(args.data_dir)
    if args.output_dir and not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)
    if args.output_root and not isinstance(args.output_root, Path):
        args.output_root = Path(args.output_root)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    # Resolve datasets and weights
    dataset_names = _parse_str_list(args.datasets) or [None]
    weights_list = args.weights if isinstance(args.weights, list) else [args.weights]

    for dataset in dataset_names:
        if dataset and args.data_root:
            data_root = (args.data_root / dataset).resolve()
        elif dataset:
            data_root = Path(dataset).resolve()
        elif args.data_dir:
            data_root = args.data_dir.resolve()
        else:
            raise RuntimeError("Provide --data-dir or --data-root with --datasets")

        for weights in weights_list:
            if not distributed or rank == 0:
                print("------------------------------------------------------------")
                print(f"[rank {rank}] Model          : {args.model}")
                print(f"[rank {rank}] Weights        : {weights}")
                print(f"[rank {rank}] Datasets       : {dataset_names}")
                print(f"[rank {rank}] Data directory : {data_root}")
                print(f"[rank {rank}] Output root    : {args.output_root or 'runs/<model>-knn/...'}")
                print(f"[rank {rank}] Batch size     : {args.batch_size}")
                print(f"[rank {rank}] Workers        : {args.num_workers}")
                print(f"[rank {rank}] K-values       : {args.k_values}")
                print(f"[rank {rank}] KNN metric     : {args.knn_metric}")
                print(f"[rank {rank}] Save emb       : {args.save_embeddings}")
                if args.model == "dino":
                    print(f"[rank {rank}] DINO arch      : {args.dino_arch}  patch: {args.dino_patch_size}  key: {args.dino_checkpoint_key}")
                elif args.model == "dinov3":
                    print(f"[rank {rank}] DINOv3 weights : {weights}  type: {args.dinov3_weights_type}")
                else:
                    print(
                        f"[rank {rank}] Galileo bands  : "
                        f"indices={args.galileo_band_indices or 'all'} "
                        f"names={args.galileo_band_names or 'infer'} "
                        f"norm={args.galileo_normalize} ndvi={args.galileo_compute_ndvi}"
                    )
                print("------------------------------------------------------------")

            if args.model == "dino":
                adapter = load_model(
                    "dino",
                    weights=weights,
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
                    weights=weights,
                    device=device,
                    dtype=dtype,
                    weights_type=args.dinov3_weights_type,
                )
            else:
                adapter = load_model(
                    "galileo",
                    weights_dir=weights,
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
                distributed=distributed,
                world_size=world_size,
                rank=rank,
            )
            if not distributed or rank == 0:
                for split_name, loader in loaders.items():
                    try:
                        ds_len = len(loader.dataset)  # type: ignore[attr-defined]
                    except Exception:
                        ds_len = "unknown"
                    print(f"[dataset][rank {rank}] {split_name}: {ds_len} images (batch_size={args.batch_size})")
            if not class_names:
                raise RuntimeError("No class folders found; k-NN evaluation needs labeled data.")

            amp_dtype = dtype
            split_outputs = embedding.extract_all_splits(adapter, loaders, amp_dtype=amp_dtype, device=device)

            if distributed and world_size > 1:
                # Gather embeddings/labels/paths from all ranks.
                gathered: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
                for split, (feats, labels, paths) in split_outputs.items():
                    obj = (feats, labels, paths)
                    buf = [None for _ in range(world_size)]
                    dist.all_gather_object(buf, obj)
                    if rank == 0:
                        all_feats = [b[0] for b in buf if b[0].size]
                        all_labels = [b[1] for b in buf if b[1].size]
                        all_paths = sum([b[2] for b in buf], [])
                        if all_feats:
                            feats_cat = np.concatenate(all_feats, axis=0)
                            labels_cat = np.concatenate(all_labels, axis=0)
                        else:
                            feats_cat = np.zeros((0, adapter.output_dim), dtype=np.float32)
                            labels_cat = np.zeros((0,), dtype=np.int64)
                        gathered[split] = (feats_cat, labels_cat, all_paths)
                split_outputs = gathered

            # Only rank 0 continues to k-NN / saving.
            if distributed and rank != 0:
                continue

            if args.output_dir:
                out_dir = args.output_dir.resolve()
            else:
                weight_tag = Path(weights).name.replace("/", "_")
                base = args.output_root if args.output_root else Path("runs")
                out_dir = base / f"{args.model}-knn" / data_root.name / weight_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            # Log hyperparameters for reproducibility.
            hparams = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
            hparams["weights"] = weights
            hparams["data_dir"] = str(data_root)
            hparams["output_dir"] = str(out_dir)
            try:
                import yaml  # type: ignore
                (out_dir / "hparams.yaml").write_text(yaml.safe_dump(hparams, sort_keys=True))
            except Exception:
                (out_dir / "hparams.json").write_text(json.dumps(hparams, indent=2, sort_keys=True))

            # Persist embeddings for reuse.
            if args.save_embeddings:
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
