"""Model-agnostic k-NN evaluation helpers aligned to DINOv2 logic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def parse_k_values(value: str | Iterable[int]) -> List[int]:
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        return [int(p) for p in parts]
    return [int(v) for v in value]


def _as_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1)


def _knn_softmax_vote(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    *,
    num_classes: int,
    k_values: List[int],
    temperature: float,
    chunk_size: int = 1024,
) -> dict[int, np.ndarray]:
    max_k = max(k_values)
    preds_per_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}

    for start in range(0, eval_features.shape[0], chunk_size):
        end = min(start + chunk_size, eval_features.shape[0])
        feats = eval_features[start:end]

        sims = feats @ train_features.T
        topk_sims, indices = sims.topk(max_k, dim=1, largest=True, sorted=True)
        neighbor_labels = train_labels[indices]

        weights = torch.softmax(topk_sims / temperature, dim=1)
        weighted = torch.nn.functional.one_hot(neighbor_labels, num_classes=num_classes) * weights.unsqueeze(-1)

        for k in k_values:
            probs = weighted[:, :k, :].sum(dim=1)
            preds = probs.argmax(dim=1)
            preds_per_k[k].append(preds.cpu().numpy())

    return {k: np.concatenate(chunks, axis=0) for k, chunks in preds_per_k.items()}


def _knn_softmax_vote_with_probs(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    *,
    num_classes: int,
    k_values: List[int],
    temperature: float,
    chunk_size: int = 1024,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    max_k = max(k_values)
    preds_per_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    probs_per_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}

    for start in range(0, eval_features.shape[0], chunk_size):
        end = min(start + chunk_size, eval_features.shape[0])
        feats = eval_features[start:end]

        sims = feats @ train_features.T
        topk_sims, indices = sims.topk(max_k, dim=1, largest=True, sorted=True)
        neighbor_labels = train_labels[indices]

        weights = torch.softmax(topk_sims / temperature, dim=1)
        weighted = torch.nn.functional.one_hot(neighbor_labels, num_classes=num_classes) * weights.unsqueeze(-1)

        for k in k_values:
            probs = weighted[:, :k, :].sum(dim=1)
            preds = probs.argmax(dim=1)
            max_probs = probs.max(dim=1).values
            preds_per_k[k].append(preds.cpu().numpy())
            probs_per_k[k].append(max_probs.cpu().numpy())

    return {
        k: (np.concatenate(preds_per_k[k], axis=0), np.concatenate(probs_per_k[k], axis=0))
        for k in k_values
    }


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_sets: Dict[str, Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, list[str]]],
    *,
    class_names: List[str],
    k_values: str | Iterable[int],
    metric: str = "cosine",
    weights: str = "uniform",
    softmax_temp: float = 0.07,
    pca_dim: int = 0,
    output_dir: Path | None = None,
    classifier_path: Path | None = None,
) -> List[dict]:
    """Evaluate k-NN using DINOv2-style softmax voting over cosine similarities."""
    if train_features.size == 0:
        raise ValueError("Empty training embeddings; cannot run k-NN.")

    if metric != "cosine":
        raise ValueError("DINOv2-aligned k-NN only supports cosine similarity.")
    if weights != "softmax":
        raise ValueError("DINOv2-aligned k-NN only supports softmax voting.")
    if pca_dim:
        raise ValueError("DINOv2-aligned k-NN does not apply PCA.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = _normalize(_as_torch(train_features, device))
    ytr = torch.as_tensor(train_labels, device=device, dtype=torch.long)

    k_list = parse_k_values(k_values)
    num_classes = len(class_names)

    if classifier_path is not None:
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            classifier_path,
            train_features=train_features,
            train_labels=train_labels,
            class_names=np.array(class_names),
            k_values=np.array(k_list, dtype=np.int64),
            softmax_temp=np.array([softmax_temp], dtype=np.float32),
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        confusion_dir = output_dir / "confusion"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        (confusion_dir / "class_names.txt").write_text("\n".join(class_names))
    else:
        confusion_dir = None

    results: list[dict] = []
    split_metrics_per_k: dict[int, dict[str, dict]] = {k: {} for k in k_list}
    for split_name, split_data in eval_sets.items():
        if len(split_data) == 3:
            Xev, yev, eval_paths = split_data
        else:
            Xev, yev = split_data
            eval_paths = None

        if Xev.size == 0:
            for k in k_list:
                split_metrics_per_k[k][split_name] = {"acc": float("nan")}
            continue

        Xev_t = _normalize(_as_torch(Xev, device))
        preds_with_probs_per_k = _knn_softmax_vote_with_probs(
            Xtr,
            ytr,
            Xev_t,
            num_classes=num_classes,
            k_values=k_list,
            temperature=softmax_temp,
        )

        for k, (preds, max_probs) in preds_with_probs_per_k.items():
            acc = float(np.mean(preds == yev))
            split_metrics_per_k[k][split_name] = {"acc": acc}

            if confusion_dir is not None:
                cm = confusion_matrix(yev, preds, labels=list(range(len(class_names))))
                report = classification_report(
                    yev,
                    preds,
                    labels=list(range(len(class_names))),
                    target_names=class_names,
                    zero_division=0,
                )
                out_path = confusion_dir / f"{split_name}_k{k}.txt"
                with out_path.open("w") as f:
                    f.write(f"Split: {split_name}, k={k}\n")
                    f.write("Confusion matrix (rows=true, cols=pred):\n")
                    for row in cm:
                        f.write(" ".join(str(int(x)) for x in row) + "\n")
                    f.write("\nClassification report:\n")
                    f.write(report)
                    f.write("\n")

                if eval_paths is not None:
                    pred_csv_path = confusion_dir / f"{split_name}_predictions_k{k}.csv"
                    with pred_csv_path.open("w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "path",
                                "true_label_idx",
                                "true_label",
                                "pred_label_idx",
                                "pred_label",
                                "confidence",
                            ]
                        )
                        for path, true_idx, pred_idx, confidence in zip(eval_paths, yev, preds, max_probs):
                            writer.writerow(
                                [
                                    path,
                                    int(true_idx),
                                    class_names[int(true_idx)],
                                    int(pred_idx),
                                    class_names[int(pred_idx)],
                                    float(confidence),
                                ]
                            )

    for k in k_list:
        results.append({"k": k, "splits": split_metrics_per_k[k]})

    return results


def load_knn_classifier(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "train_features": data["train_features"],
        "train_labels": data["train_labels"],
        "class_names": [str(x) for x in data["class_names"].tolist()],
        "k_values": [int(x) for x in data["k_values"].tolist()],
        "softmax_temp": float(np.array(data["softmax_temp"]).reshape(-1)[0]),
    }


def predict_knn(
    classifier: dict,
    eval_features: np.ndarray,
    *,
    k_values: Iterable[int] | None = None,
    temperature: float | None = None,
) -> dict[int, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = _normalize(_as_torch(classifier["train_features"], device))
    ytr = torch.as_tensor(classifier["train_labels"], device=device, dtype=torch.long)
    Xev = _normalize(_as_torch(eval_features, device))

    k_list = list(k_values) if k_values is not None else list(classifier["k_values"])
    temp = float(temperature) if temperature is not None else float(classifier["softmax_temp"])
    num_classes = len(classifier["class_names"])
    return _knn_softmax_vote(
        Xtr,
        ytr,
        Xev,
        num_classes=num_classes,
        k_values=k_list,
        temperature=temp,
    )


def predict_knn_with_probs(
    classifier: dict,
    eval_features: np.ndarray,
    *,
    k_values: Iterable[int] | None = None,
    temperature: float | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = _normalize(_as_torch(classifier["train_features"], device))
    ytr = torch.as_tensor(classifier["train_labels"], device=device, dtype=torch.long)
    Xev = _normalize(_as_torch(eval_features, device))

    k_list = list(k_values) if k_values is not None else list(classifier["k_values"])
    temp = float(temperature) if temperature is not None else float(classifier["softmax_temp"])
    num_classes = len(classifier["class_names"])
    return _knn_softmax_vote_with_probs(
        Xtr,
        ytr,
        Xev,
        num_classes=num_classes,
        k_values=k_list,
        temperature=temp,
    )
