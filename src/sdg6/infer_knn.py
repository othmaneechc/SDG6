"""Run k-NN inference on arbitrary images using a saved classifier artifact."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from models import load_model, resolve_device, resolve_dtype
from sdg6 import embedding
from sdg6.data import DEFAULT_EXTS, Sample, collate_samples, read_rgb_image
from sdg6.knn import load_knn_classifier, predict_knn_with_probs


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], transform, reader=read_rgb_image):
        self.paths = paths
        self.transform = transform
        self.reader = reader

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = self.reader(path)
        if self.transform is not None:
            try:
                image = self.transform(image, path=path)
            except TypeError:
                image = self.transform(image)
        return Sample(image=image, label=-1, path=str(path))


def _gather_paths(input_dir: Path | None, input_list: Path | None) -> list[Path]:
    paths: list[Path] = []
    if input_list:
        for line in input_list.read_text().splitlines():
            line = line.strip()
            if line:
                paths.append(Path(line))
    if input_dir:
        for ext in DEFAULT_EXTS:
            paths.extend(sorted(input_dir.rglob(f"*{ext}")))
    return paths


def _parse_k_values(value: str | Iterable[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        return [int(p) for p in parts]
    return [int(v) for v in value]


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 k-NN inference")
    parser.add_argument("--config", type=Path, required=True, help="Path to inference YAML config.")
    args = parser.parse_args()

    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SystemExit("PyYAML is required to parse the inference config.") from exc

    cfg = yaml.safe_load(args.config.read_text()) or {}

    def get(key: str, default=None):
        return cfg.get(key, default)

    classifier_path = Path(get("knn_classifier_path")).expanduser().resolve()
    input_dir = get("input_dir")
    input_list = get("input_list")
    output_csv = Path(get("output_csv", "knn_predictions.csv")).expanduser().resolve()

    input_dir = Path(input_dir).expanduser().resolve() if input_dir else None
    input_list = Path(input_list).expanduser().resolve() if input_list else None

    paths = _gather_paths(input_dir, input_list)
    if not paths:
        raise RuntimeError("No input images found. Provide input_dir or input_list.")

    device = resolve_device(get("device"))
    dtype = resolve_dtype(get("dtype", "auto"), device)

    adapter = load_model(
        get("model", "dinov2"),
        weights=get("weights"),
        config_file=get("dinov2_config"),
        checkpoint_key=get("dinov2_checkpoint_key", "teacher"),
        resize_size=int(get("dinov2_resize", 256)),
        crop_size=int(get("dinov2_crop", 224)),
        device=device,
        dtype=dtype,
    )

    dataset = ImageListDataset(paths, transform=adapter.transform, reader=adapter.reader)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(get("batch_size", 64)),
        num_workers=int(get("num_workers", 4)),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_samples,
    )

    feats, _, out_paths = embedding.extract_embeddings(adapter, loader, device=device, amp_dtype=dtype)

    classifier = load_knn_classifier(classifier_path)
    k_values = _parse_k_values(get("k_values"))
    temperature = get("knn_softmax_temp")

    preds = predict_knn_with_probs(classifier, feats, k_values=k_values, temperature=temperature)
    class_names = classifier["class_names"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    k_list = list(preds.keys())
    headers = ["path"]
    for k in k_list:
        headers += [f"pred_k{k}", f"pred_class_k{k}", f"prob_k{k}"]

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx, path in enumerate(out_paths):
            row = [path]
            for k in k_list:
                pred_ids, probs = preds[k]
                pred = int(pred_ids[idx])
                prob = float(probs[idx])
                row += [pred, class_names[pred], f"{prob:.6f}"]
            writer.writerow(row)

    print(f"Wrote predictions to {output_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
