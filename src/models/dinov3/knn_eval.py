#!/usr/bin/env python3
"""Distributed k-NN classification with DINOv3 backbones (Hugging Face weights)."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.transforms import v2 as T
from datetime import datetime

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

STATS = {
    "lvd": (
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    "sat": (
        (0.430, 0.411, 0.296),
        (0.213, 0.156, 0.143),
    ),
}


def log_msg(message: str, *, rank: int | None = None) -> None:
    """Timestamped logging helper with optional rank prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]"
    if rank is not None:
        prefix += f"[GPU {rank}]"
    print(f"{prefix} {message}", flush=True)


def init_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank); initialize if launched via torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed DINOv3 k-NN evaluation.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "hf"],
        default="hf",
        help="Model source: Hugging Face model id (local repo removed).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Hugging Face model id or local snapshot directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Unused for Hugging Face source (kept for backward compatibility).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,5,10,20,50,100",
        help="Comma-separated list of k values.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory to save/load embeddings (default: DATA_DIR/_embeddings).",
    )
    parser.add_argument(
        "--weights-type",
        type=str,
        choices=["auto", "lvd", "sat"],
        default="auto",
        help="Which normalization stats to use; 'auto' uses the HF processor values.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Emit per-GPU progress every N samples; 0 disables progress logs.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (higher uses more GPU memory).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Computation dtype for feature extraction (default: auto prefers bf16, else fp32).",
    )
    parser.add_argument(
        "--knn-metric",
        type=str,
        choices=["cosine", "l2"],
        default="cosine",
        help="Distance metric for k-NN evaluation.",
    )
    parser.add_argument(
        "--knn-weights",
        type=str,
        choices=["uniform", "distance", "softmax"],
        default="uniform",
        help="Neighbor weighting strategy (softmax applies exp(-d/temp)).",
    )
    parser.add_argument(
        "--knn-softmax-temp",
        type=float,
        default=0.07,
        help="Temperature used when --knn-weights=softmax (ignored otherwise).",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="Apply PCA to this dimensionality before k-NN; 0 disables PCA.",
    )
    return parser.parse_args()


def resolve_weights_type(weights: Path | str, override: str) -> str:
    if override != "auto":
        return override
    name = str(weights).lower()
    if "lvd" in name:
        return "lvd"
    if "sat" in name:
        return "sat"
    return "sat"


def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    crop_size: int = 224,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> T.Compose:
    transforms = [T.ToImage(), T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC)]
    if crop_size:
        transforms.append(T.CenterCrop(crop_size))
    transforms.extend(
        [
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(transforms)


def make_transform(
    weights_type: str,
    resize_size: int = 256,
    crop_size: int = 224,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> T.Compose:
    if mean is None or std is None:
        mean, std = STATS[weights_type]
    # Match upstream eval: resize shorter side then center crop
    return make_classification_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        mean=mean,
        std=std,
    )


def resolve_hf_sizes(processor) -> tuple[int, int]:
    resize_size = None
    crop_size = None

    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        resize_size = size.get("shortest_edge") or size.get("height") or size.get("width")

    crop = getattr(processor, "crop_size", None)
    if isinstance(crop, dict):
        crop_size = crop.get("height") or crop.get("width")

    return int(resize_size or 256), int(crop_size or 224)


def load_hf_model(model_id: str):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for --source hf") from exc

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, processor


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


def extract_cls(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    source: str,
    *,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad(), torch.amp.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        img_tensor = img_tensor.to(device, non_blocking=True)

        if source == "hf":
            outputs = model(pixel_values=img_tensor)
            if isinstance(outputs, (tuple, list)):
                last_hidden = outputs[0] if outputs else None
                pooler = None
            else:
                last_hidden = getattr(outputs, "last_hidden_state", None)
                pooler = getattr(outputs, "pooler_output", None)

            if pooler is not None:
                cls = pooler
            elif last_hidden is not None:
                cls = last_hidden[:, 0]
            else:
                raise KeyError("HF model outputs missing last_hidden_state")
        else:
            out = model.forward_features(img_tensor)
            if isinstance(out, list):
                out = out[0]

            if "x_norm_clstoken" in out:
                cls = out["x_norm_clstoken"]
            elif "x_cls" in out:
                cls = out["x_cls"]
            elif "x_prenorm" in out:
                cls = out["x_prenorm"][:, 0]
            else:
                raise KeyError(f"No CLS token in forward_features output: {out.keys()}")

        # L2 normalize like upstream k-NN eval
        cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.cpu()


def build_embeddings(
    split_dir: Path,
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device,
    class_to_idx: dict[str, int],
    log_interval: int,
    embed_batch_size: int,
    source: str,
    *,
    rank: int,
    world_size: int,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    files: list[tuple[Path, int]] = []
    for class_name, label_int in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for fname in os.listdir(class_dir):
            files.append((class_dir / fname, label_int))

    random.shuffle(files)
    # Shard file list across ranks for data parallel extraction
    files = files[rank::world_size]
    total = len(files)
    log_msg(f"{split_dir.name}: rank {rank} assigned {total} files", rank=rank)
    X_local: list[np.ndarray] = []
    y_local: list[int] = []
    start_time = time.time()
    batch_imgs: list[torch.Tensor] = []
    batch_labels: list[int] = []

    for idx, (fpath, label_int) in enumerate(files):
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            log_msg(f"Skipping corrupted file {fpath}", rank=rank)
            continue

        img_t = transform(img)
        batch_imgs.append(img_t)
        batch_labels.append(label_int)

        flush = (len(batch_imgs) == embed_batch_size) or (idx == total - 1)
        if flush:
            batch_tensor = torch.stack(batch_imgs, dim=0)
            feats = extract_cls(model, batch_tensor, device, source, amp_dtype=amp_dtype)
            # feats shape: (B, dim)
            if feats.ndim == 1:
                feats = feats[None, :]
            X_local.extend([feat.numpy() for feat in feats])
            y_local.extend(batch_labels)
            batch_imgs.clear()
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
        f"{split_dir.name}: finished {len(X_local)}/{total} samples "
        f"in {elapsed:.1f}s",
        rank=rank,
    )

    if len(X_local) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(X_local), np.array(y_local, dtype=np.int64)


def resolve_amp_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    if args.dtype == "fp32":
        return torch.float32
    # auto: prefer bf16 if available on CUDA, else fp32 for stability
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def make_softmax_weights_fn(temp: float):
    def _weights(distances):
        d = np.asarray(distances)
        return np.exp(-d / temp)

    return _weights


def run_single(args: argparse.Namespace, *, rank: int, world_size: int, device: torch.device) -> None:
    if args.source != "hf":
        raise RuntimeError("local source not supported: dinov3 repo removed")

    model, processor = load_hf_model(args.weights)
    resize_size, crop_size = resolve_hf_sizes(processor)
    if args.weights_type != "auto":
        mean, std = STATS[args.weights_type]
    else:
        mean = tuple(getattr(processor, "image_mean", STATS["lvd"][0]))
        std = tuple(getattr(processor, "image_std", STATS["lvd"][1]))
    transform = make_transform(
        weights_type="auto",
        resize_size=resize_size,
        crop_size=crop_size,
        mean=mean,
        std=std,
    )
    amp_dtype = resolve_amp_dtype(args)

    log_msg(f"Loading HF model {args.weights}", rank=rank)
    log_msg(
        f"Normalization (hf): mean={mean}, std={std}, resize={resize_size}, crop={crop_size}",
        rank=rank,
    )
    log_msg(f"AMP dtype: {amp_dtype}", rank=rank)
    model = model.to(device).eval()
    if device.type == "cuda" and amp_dtype == torch.float16:
        model = model.half()

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
            model,
            transform,
            device=device,
            class_to_idx=class_to_idx,
            log_interval=args.log_interval,
            embed_batch_size=args.embed_batch_size,
            source=args.source,
            rank=rank,
            world_size=world_size,
            amp_dtype=amp_dtype,
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
    log_msg("   DINOv3 k-NN Eval          ", rank=rank)
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
            log_msg(
                f"Applying PCA: {feat_dim} -> {args.pca_dim} dims for k-NN", rank=rank
            )
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

    # Configure metric and neighbor weights
    metric_arg = "cosine" if args.knn_metric == "cosine" else "euclidean"
    if args.knn_weights == "softmax":
        weights_arg = make_softmax_weights_fn(args.knn_softmax_temp)
        log_msg(
            f"k-NN settings: metric={metric_arg}, weights=softmax(temp={args.knn_softmax_temp}), "
            f"pca_dim={args.pca_dim}",
            rank=rank,
        )
    else:
        weights_arg = args.knn_weights
        log_msg(
            f"k-NN settings: metric={metric_arg}, weights={weights_arg}, pca_dim={args.pca_dim}",
            rank=rank,
        )

    for k in k_list:
        log_msg(f"Training k-NN with k = {k} ...", rank=rank)
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric_arg,
            weights=weights_arg,
            n_jobs=8,
        )
        knn.fit(Xtr, ytr)
        val_preds = knn.predict(Xval)
        test_preds = knn.predict(Xte)
        val_acc = float(np.mean(val_preds == yval))
        test_acc = float(np.mean(test_preds == yte))

        log_msg(f"  -> Val:  {val_acc*100:.2f}% | Test: {test_acc*100:.2f}%", rank=rank)
        val_cm_path = save_confusion("val", val_preds, yval, k)
        test_cm_path = save_confusion("test", test_preds, yte, k)
        log_msg(f"  -> Confusion saved: val={val_cm_path.name}, test={test_cm_path.name}", rank=rank)
        results.append((k, val_acc, test_acc))

    log_msg("==== Summary (k-NN) ====", rank=rank)
    log_msg("   k    |   Val Acc   |  Test Acc", rank=rank)
    log_msg("---------------------------------", rank=rank)
    for k, v, t in results:
        log_msg(f"{k:6d} | {v*100:9.2f}% | {t*100:9.2f}%", rank=rank)
    log_msg("================================", rank=rank)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for k-NN evaluation.")

    args.data_dir = args.data_dir.resolve()
    dataset_name = args.data_dir.name

    if args.source != "hf":
        raise RuntimeError("local source not supported: dinov3 repo removed")

    weight_tag = args.model_name or Path(args.weights).name or args.weights.replace("/", "_")

    default_ckpt = RUNS_ROOT / "dinov3-knn-cache" / dataset_name / weight_tag
    args.ckpt_dir = (args.ckpt_dir or default_ckpt).resolve()
    args.class_names = discover_class_names(args.data_dir)

    os.environ.setdefault("MASTER_ADDR", DEFAULT_MASTER_ADDR)
    os.environ.setdefault("MASTER_PORT", DEFAULT_MASTER_PORT)

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log_msg(f"World size: {world_size}, rank: {rank}, local_rank: {local_rank}, device: {device}", rank=rank)
    log_msg(f"Data directory: {args.data_dir}", rank=rank)
    log_msg(f"Output directory: {args.ckpt_dir}", rank=rank)

    run_single(args, rank=rank, world_size=world_size, device=device)

    if world_size > 1 and dist.is_initialized():
        # Do not barrier here: rank 0 may still be running k-NN while others have exited.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
