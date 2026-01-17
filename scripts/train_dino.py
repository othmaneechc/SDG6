"""Launcher for DINO pretraining/fine-tuning driven by a YAML config."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required to parse the training config.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="DINO training wrapper.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/configs/dino_pt.yaml"),
        help="Path to YAML with training hyperparameters.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text()) or {}

    def get(key: str, default):
        return cfg.get(key, default)

    # Threading / NCCL env defaults (overrideable in config).
    env_defaults = {
        "OMP_NUM_THREADS": get("omp_num_threads", 8),
        "MKL_NUM_THREADS": get("mkl_num_threads", 8),
        "NUMEXPR_NUM_THREADS": get("numexpr_num_threads", 8),
        "TORCH_NCCL_BLOCKING_WAIT": get("torch_nccl_blocking_wait", 1),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": get("torch_nccl_async_error_handling", 1),
        "TORCH_NCCL_DEBUG": get("torch_nccl_debug", "WARN"),
        "TORCH_NCCL_TIMEOUT": get("torch_nccl_timeout", 1800),
        "MASTER_ADDR": get("master_addr", "127.0.0.1"),
    }
    for k, v in env_defaults.items():
        os.environ.setdefault(k, str(v))

    gpus = int(get("gpus_per_node", 2))
    master_port = int(get("master_port", 29500))
    data_path = Path(get("data_path", "/path/to/train_data")).expanduser().resolve()
    output_root = Path(get("output_root", repo / "runs" / "dino-train")).expanduser()
    arch = str(get("arch", "vit_base"))
    patch_size = int(get("patch_size", 8))
    epochs = int(get("epochs", 200))
    batch_size_per_gpu = int(get("batch_size_per_gpu", 64))
    lr = float(get("lr", 0.001))
    use_fp16 = bool(get("use_fp16", True))

    data_name = data_path.name
    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{arch}_p{patch_size}_{data_name}_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training DINO: arch={arch}, patch={patch_size}, data={data_path}, out={output_dir}")

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(gpus),
        "--master_port",
        str(master_port),
        "-m",
        "dino.main_dino",
        "--arch",
        arch,
        "--data_path",
        str(data_path),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch_size_per_gpu",
        str(batch_size_per_gpu),
        "--patch_size",
        str(patch_size),
        "--lr",
        str(lr),
        "--use_fp16",
        str(use_fp16).lower(),
    ]

    subprocess.run(cmd, check=True, cwd=repo)


if __name__ == "__main__":  # pragma: no cover
    main()
