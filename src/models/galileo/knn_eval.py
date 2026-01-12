#!/usr/bin/env python3
"""Distributed k-NN classification with Galileo encoder embeddings."""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

from models.galileo.encoder import (
    DEFAULT_MONTH,
    GalileoInput,
    build_s2_input,
    encode_batch,
    infer_band_names,
    load_encoder,
    to_month_index,
)

# KNN logic adapted from nasaharvest/galileo (MIT):
# /home/mila/e/echchabo/projects/galileo/src/eval/knn.py
import torch.nn as nn

# -----------------------
# Limit CPU threading
# -----------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

# NCCL settings (for multi-GPU stability)
os.environ.pop("NCCL_BLOCKING_WAIT", None)
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_DEBUG", "WARN")
os.environ.setdefault("TORCH_NCCL_TIMEOUT", "1800")

RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = "29500"

DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{4})_(\d{2})_(\d{2})"),
    re.compile(r"(\d{8})"),
]


def log_msg(message: str, *, rank: int | None = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]"
    if rank is not None:
        prefix += f"[GPU {rank}]"
    print(f"{prefix} {message}", flush=True)


def init_distributed() -> tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed Galileo k-NN evaluation.")
    p.add_argument("--data-dir", type=Path, required=True, help="Root with train/val/test splits.")
    p.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Folder containing Galileo config.json + encoder.pt.",
    )
    p.add_argument("--k-values", type=str, default="1,5,10,20,50,100", help="Comma-separated list of k.")
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory to save/load embeddings (default: DATA_DIR/_galileo_embeddings).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Emit per-GPU progress every N samples; 0 disables progress logs.",
    )
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction (higher uses more GPU memory).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Computation dtype for feature extraction (default: auto prefers bf16, else fp32).",
    )
    p.add_argument(
        "--input-resolution-m",
        type=int,
        default=10,
        help="Ground sample distance (meters per pixel) for spatial encoding.",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=0,
        help="Patch size override (0 uses model config base patch size).",
    )
    p.add_argument(
        "--band-indices",
        type=str,
        default=None,
        help="Comma-separated 0-based indices to select bands from input TIFF.",
    )
    p.add_argument(
        "--band-names",
        type=str,
        default=None,
        help="Comma-separated band names matching selected channels (e.g. B2,B3,B4,B8).",
    )
    p.add_argument(
        "--value-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to raw band values (e.g. 0.0001 for reflectance).")
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Apply Galileo pretraining normalization to space-time bands.",
    )
    p.add_argument(
        "--compute-ndvi",
        action="store_true",
        help="Compute NDVI if B4 and B8 are present.",
    )
    p.add_argument(
        "--pad-square",
        action="store_true",
        default=True,
        help="Pad inputs to square if height != width (default: enabled).",
    )
    p.add_argument(
        "--no-pad-square",
        dest="pad_square",
        action="store_false",
        help="Disable square padding.",
    )
    p.add_argument(
        "--pad-to-patch",
        action="store_true",
        default=True,
        help="Pad inputs so H,W are divisible by patch size (default: enabled).",
    )
    p.add_argument(
        "--no-pad-to-patch",
        dest="pad_to_patch",
        action="store_false",
        help="Disable patch-size padding.",
    )
    p.add_argument(
        "--default-month-index",
        type=int,
        default=DEFAULT_MONTH,
        help="Fallback month index (0-11) when parsing fails.",
    )
    p.add_argument(
        "--no-month-from-name",
        action="store_true",
        help="Disable parsing month from filename; always use --default-month-index.",
    )
    p.add_argument(
        "--knn-softmax-temp",
        type=float,
        default=0.07,
        help="Softmax temperature for weighted cosine k-NN.",
    )
    p.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="Apply PCA to this dimensionality before k-NN; 0 disables PCA.",
    )
    return p.parse_args()


def resolve_amp_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    if args.dtype == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def discover_class_names(data_dir: Path) -> list[str]:
    classes = set()
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for child in split_dir.iterdir():
            if child.is_dir():
                classes.add(child.name)
    if not classes:
        raise FileNotFoundError(f"No class folders found under {data_dir}")
    return sorted(classes)


def parse_month_from_path(path: Path) -> int | None:
    stem = path.stem
    for pattern in DATE_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        if len(match.groups()) == 3:
            return int(match.group(2))
        if len(match.groups()) == 1:
            date_str = match.group(1)
            if len(date_str) == 8:
                return int(date_str[4:6])
    return None


def read_multiband(path: Path) -> np.ndarray:
    try:
        import rasterio
    except Exception:
        rasterio = None

    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        return arr

    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    h, w, c = arr.shape
    if h == w:
        return arr
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")


def pad_to_patch(arr: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_size <= 0:
        return arr
    h, w, c = arr.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts or None


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [int(v.strip()) for v in value.split(",") if v.strip()]
    return parts or None


def _run_knn_for_k(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    test_embeddings: torch.Tensor,
    num_classes: int,
    k: int,
    device: torch.device,
    temperature: float,
    *,
    skip_idx: bool = False,
) -> torch.Tensor:
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)

    cos = nn.CosineSimilarity(dim=-1)
    all_preds: list[torch.Tensor] = []
    for idx in range(test_embeddings.shape[0]):
        test_embedding = test_embeddings[idx].unsqueeze(dim=0).repeat(train_embeddings.shape[0], 1)
        sims = cos(test_embedding, train_embeddings)
        top_k = torch.topk(sims, k=k)

        if skip_idx:
            top_k_values = top_k.values[1:]
            top_k_indices = top_k.indices[1:]
        else:
            top_k_values = top_k.values
            top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = torch.nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        distances = top_k_values.clone().div_(temperature).exp_()
        weighted_sum_onehots = (distances.unsqueeze(dim=1) * fetched_onehots).sum(dim=0)
        prediction = torch.argmax(weighted_sum_onehots)
        all_preds.append(prediction)

    return torch.stack(all_preds, dim=0).cpu()


def build_embeddings(
    split_dir: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    class_to_idx: dict[str, int],
    log_interval: int,
    embed_batch_size: int,
    *,
    rank: int,
    world_size: int,
    amp_dtype: torch.dtype,
    patch_size: int,
    input_resolution_m: int,
    band_indices: list[int] | None,
    band_names: list[str] | None,
    value_scale: float,
    normalize: bool,
    compute_ndvi: bool,
    pad_square: bool,
    pad_to_patch_flag: bool,
    default_month_index: int,
    parse_months: bool,
) -> tuple[np.ndarray, np.ndarray]:
    files: list[tuple[Path, int]] = []
    for class_name, label_int in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for fname in os.listdir(class_dir):
            files.append((class_dir / fname, label_int))

    random.shuffle(files)
    files = files[rank::world_size]
    total = len(files)
    log_msg(f"{split_dir.name}: rank {rank} assigned {total} files", rank=rank)

    X_local: list[np.ndarray] = []
    y_local: list[int] = []
    start_time = time.time()
    batch_inputs: list[GalileoInput] = []
    batch_labels: list[int] = []

    for idx, (fpath, label_int) in enumerate(files):
        try:
            arr = read_multiband(fpath)
        except Exception:
            log_msg(f"Skipping unreadable file {fpath}", rank=rank)
            continue

        if arr.ndim != 3:
            log_msg(f"Skipping non-3D array {fpath} shape={arr.shape}", rank=rank)
            continue

        if band_indices is not None:
            arr = arr[:, :, band_indices]

        sample_band_names = band_names
        if sample_band_names is None:
            if arr.shape[2] > 10 and band_indices is None:
                arr = arr[:, :, :10]
            sample_band_names = infer_band_names(arr.shape[2])

        arr = arr.astype(np.float32) * value_scale
        if pad_square:
            arr = pad_to_square(arr)
        if pad_to_patch_flag:
            arr = pad_to_patch(arr, patch_size)

        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"Input {fpath} is not square after padding (H={arr.shape[0]}, W={arr.shape[1]})."
            )

        month_idx = default_month_index
        if parse_months:
            parsed_month = parse_month_from_path(fpath)
            if parsed_month is not None:
                month_idx = to_month_index(parsed_month)
        months_t = torch.tensor([month_idx], dtype=torch.long)

        s2 = torch.from_numpy(arr)
        sample = build_s2_input(
            s2,
            band_names=sample_band_names,
            months=months_t,
            normalize=normalize,
            compute_ndvi=compute_ndvi,
        )
        batch_inputs.append(sample)
        batch_labels.append(label_int)

        flush = (len(batch_inputs) == embed_batch_size) or (idx == total - 1)
        if flush:
            batch = GalileoInput.stack(batch_inputs).to(device)
            feats = encode_batch(
                encoder,
                batch,
                patch_size=patch_size,
                input_resolution_m=input_resolution_m,
                amp_dtype=amp_dtype,
            ).float().cpu()
            if feats.ndim == 1:
                feats = feats[None, :]
            X_local.extend([feat.numpy() for feat in feats])
            y_local.extend(batch_labels)
            batch_inputs.clear()
            batch_labels.clear()

        if log_interval and (idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = total - (idx + 1)
            eta = remaining / rate if rate > 0 else float("inf")
            log_msg(
                f"{split_dir.name}: processed {idx + 1}/{total} "
                f"({(idx + 1) / total * 100:.1f}%) | "
                f"elapsed {elapsed:.1f}s | "
                f"ETA {eta:.1f}s",
                rank=rank,
            )

    elapsed = time.time() - start_time
    log_msg(
        f"{split_dir.name}: finished {len(X_local)}/{total} samples in {elapsed:.1f}s",
        rank=rank,
    )

    if len(X_local) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(X_local), np.array(y_local, dtype=np.int64)


def run_single(args: argparse.Namespace, *, rank: int, world_size: int, device: torch.device) -> None:
    encoder, config = load_encoder(args.weights_dir, device)
    encoder_config = config.get("model", {}).get("encoder", {})
    patch_size = args.patch_size or int(encoder_config.get("max_patch_size", encoder.base_patch_size))
    amp_dtype = resolve_amp_dtype(args)

    log_msg(f"Loading Galileo encoder from {args.weights_dir}", rank=rank)
    log_msg(
        f"Patch size={patch_size}, input_resolution_m={args.input_resolution_m}, "
        f"normalize={args.normalize}, ndvi={args.compute_ndvi}",
        rank=rank,
    )
    log_msg(f"AMP dtype: {amp_dtype}", rank=rank)

    if device.type == "cuda" and amp_dtype == torch.float16:
        encoder = encoder.half()

    splits = ["train", "val", "test"]
    Xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}

    ckpt_dir = args.ckpt_dir
    if rank == 0 and not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    confusion_dir = ckpt_dir / "confusion"
    if rank == 0:
        confusion_dir.mkdir(parents=True, exist_ok=True)
        class_names_path = confusion_dir / "class_names.txt"
        if not class_names_path.exists():
            class_names_path.write_text("\n".join(args.class_names))
    if world_size > 1:
        dist.barrier()

    class_to_idx = {name: idx for idx, name in enumerate(args.class_names)}

    for split in splits:
        split_dir = args.data_dir / split

        log_msg(f"Computing embeddings for: {split}", rank=rank)

        X_local, y_local = build_embeddings(
            split_dir,
            encoder,
            device=device,
            class_to_idx=class_to_idx,
            log_interval=args.log_interval,
            embed_batch_size=args.embed_batch_size,
            rank=rank,
            world_size=world_size,
            amp_dtype=amp_dtype,
            patch_size=patch_size,
            input_resolution_m=args.input_resolution_m,
            band_indices=args.band_indices,
            band_names=args.band_names,
            value_scale=args.value_scale,
            normalize=args.normalize,
            compute_ndvi=args.compute_ndvi,
            pad_square=args.pad_square,
            pad_to_patch_flag=args.pad_to_patch,
            default_month_index=args.default_month_index,
            parse_months=not args.no_month_from_name,
        )

        if world_size > 1:
            gathered_X: list[np.ndarray | None] = [None for _ in range(world_size)]
            gathered_y: list[np.ndarray | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_X, X_local)
            dist.all_gather_object(gathered_y, y_local)
            if rank == 0:
                valid_x = [x for x in gathered_X if x is not None and x.size]
                valid_y = [y for y in gathered_y if y is not None and y.size]
                if valid_x and valid_y:
                    X_concat = np.concatenate(valid_x, axis=0)
                    y_concat = np.concatenate(valid_y, axis=0)
                else:
                    X_concat = np.zeros((0, 1), dtype=np.float32)
                    y_concat = np.zeros((0,), dtype=np.int64)
                Xs[split] = X_concat
                ys[split] = y_concat
        else:
            Xs[split] = X_local
            ys[split] = y_local

        if rank == 0:
            if np.isnan(Xs[split]).any():
                raise ValueError(f"NaNs detected in embeddings for split={split}; aborting.")
            log_msg(
                f"{split}: {Xs[split].shape[0]} samples, feat_dim={Xs[split].shape[1]}",
                rank=rank,
            )

        if world_size > 1:
            dist.barrier()

    def save_confusion(split_name: str, preds: np.ndarray, targets: np.ndarray, k_val: int) -> Path:
        cm = confusion_matrix(targets, preds, labels=list(range(len(args.class_names))))
        report = classification_report(
            targets,
            preds,
            labels=list(range(len(args.class_names))),
            target_names=args.class_names,
            zero_division=0,
        )
        out_path = confusion_dir / f"confusion_{split_name}_k{k_val}.txt"
        with open(out_path, "w") as f:
            f.write(f"Split: {split_name}, k={k_val}\n")
            f.write("Confusion matrix (rows=true, cols=pred):\n")
            for row in cm:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            f.write("\nClassification report:\n")
            f.write(report)
            f.write("\n")
        return out_path

    if world_size > 1:
        dist.barrier()
    if rank != 0:
        return

    if np.isnan(Xs["train"]).any() or np.isnan(Xs["val"]).any() or np.isnan(Xs["test"]).any():
        raise ValueError("NaNs detected in gathered embeddings before k-NN; aborting.")

    log_msg("==============================", rank=rank)
    log_msg("   Galileo k-NN Eval         ", rank=rank)
    log_msg("==============================", rank=rank)

    Xtr = Xs["train"]
    ytr = ys["train"]
    Xval = Xs["val"]
    yval = ys["val"]
    Xte = Xs["test"]
    yte = ys["test"]

    if args.pca_dim > 0 and Xtr.size:
        feat_dim = Xtr.shape[1]
        if args.pca_dim < feat_dim:
            log_msg(f"Applying PCA: {feat_dim} -> {args.pca_dim} dims for k-NN", rank=rank)
            pca = PCA(n_components=args.pca_dim, random_state=0)
            Xtr = pca.fit_transform(Xtr)
            Xval = pca.transform(Xval)
            Xte = pca.transform(Xte)
        else:
            log_msg(
                f"Skipping PCA because requested {args.pca_dim} >= feature dim {feat_dim}",
                rank=rank,
            )

    log_msg(f"Train size: {len(ytr)}, Val size: {len(yval)}, Test size: {len(yte)}", rank=rank)

    k_list = [int(k) for k in args.k_values.split(",") if k.strip()]
    results: list[tuple[int, float, float]] = []

    log_msg(
        f"k-NN settings: cosine + softmax(temp={args.knn_softmax_temp}), pca_dim={args.pca_dim}",
        rank=rank,
    )

    train_embeddings = torch.from_numpy(Xtr).float()
    train_labels = torch.from_numpy(ytr).long()
    val_embeddings = torch.from_numpy(Xval).float()
    test_embeddings = torch.from_numpy(Xte).float()

    for k in k_list:
        log_msg(f"Running k-NN with k = {k} ...", rank=rank)
        val_preds = _run_knn_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=val_embeddings,
            num_classes=len(args.class_names),
            k=k,
            device=device,
            temperature=args.knn_softmax_temp,
        ).numpy()
        test_preds = _run_knn_for_k(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            num_classes=len(args.class_names),
            k=k,
            device=device,
            temperature=args.knn_softmax_temp,
        ).numpy()

        val_acc = float((val_preds == yval).mean()) if len(yval) else 0.0
        test_acc = float((test_preds == yte).mean()) if len(yte) else 0.0
        results.append((k, val_acc, test_acc))

        save_confusion("val", val_preds, yval, k)
        save_confusion("test", test_preds, yte, k)

        log_msg(f"k={k}: val_acc={val_acc:.4f} | test_acc={test_acc:.4f}", rank=rank)

    log_msg("\nSummary:")
    for k, val_acc, test_acc in results:
        log_msg(f"k={k:>3} | val={val_acc:.4f} | test={test_acc:.4f}", rank=rank)


def main() -> None:
    args = parse_args()

    if args.ckpt_dir is None:
        args.ckpt_dir = args.data_dir / "_galileo_embeddings"

    args.class_names = discover_class_names(args.data_dir)

    args.band_indices = parse_int_list(args.band_indices)
    args.band_names = parse_csv_list(args.band_names)

    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    args.default_month_index = int(args.default_month_index)
    if args.default_month_index < 0 or args.default_month_index > 11:
        raise ValueError("default-month-index must be in [0, 11]")

    log_msg(
        f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}",
        rank=rank,
    )

    run_single(args, rank=rank, world_size=world_size, device=device)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
