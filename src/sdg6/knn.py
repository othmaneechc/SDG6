"""Model-agnostic k-NN evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def parse_k_values(value: str | Iterable[int]) -> List[int]:
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
        return [int(p) for p in parts]
    return [int(v) for v in value]


def _softmax_weights(temp: float) -> Callable[[np.ndarray], np.ndarray]:
    def _fn(distances: np.ndarray) -> np.ndarray:
        d = np.asarray(distances)
        return np.exp(-d / temp)

    return _fn


def evaluate_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_sets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    class_names: List[str],
    k_values: str | Iterable[int],
    metric: str = "cosine",
    weights: str = "uniform",
    softmax_temp: float = 0.07,
    pca_dim: int = 0,
    output_dir: Path | None = None,
) -> List[dict]:
    """Fit a k-NN classifier on train embeddings and evaluate on provided splits."""
    if train_features.size == 0:
        raise ValueError("Empty training embeddings; cannot run k-NN.")

    metric_arg = "cosine" if metric == "cosine" else "euclidean"
    weights_arg: str | Callable[[np.ndarray], np.ndarray]
    if weights == "softmax":
        weights_arg = _softmax_weights(softmax_temp)
    else:
        weights_arg = weights

    Xtr = train_features
    ytr = train_labels
    if pca_dim and Xtr.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0)
        Xtr = pca.fit_transform(Xtr)
        eval_sets = {name: (pca.transform(x), y) for name, (x, y) in eval_sets.items()}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        confusion_dir = output_dir / "confusion"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        (confusion_dir / "class_names.txt").write_text("\n".join(class_names))
    else:
        confusion_dir = None

    results: list[dict] = []
    for k in parse_k_values(k_values):
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric_arg,
            weights=weights_arg,
            n_jobs=8,
        )
        knn.fit(Xtr, ytr)

        split_metrics: dict[str, dict] = {}
        for split_name, (Xev, yev) in eval_sets.items():
            if Xev.size == 0:
                split_metrics[split_name] = {"acc": float("nan")}
                continue
            preds = knn.predict(Xev)
            acc = float(np.mean(preds == yev))
            split_metrics[split_name] = {"acc": acc}

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

        results.append({"k": k, "splits": split_metrics})

    return results
