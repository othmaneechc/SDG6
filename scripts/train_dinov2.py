"""Launcher for DINOv2 pretraining driven by a YAML config."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required to parse the training config.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 training wrapper.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/configs/dinov2_pt.yaml"),
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

    if bool(get("unset_slurm_env", True)):
        os.environ.pop("SLURM_JOB_ID", None)
        os.environ.pop("SLURM_PROCID", None)
        os.environ.pop("SLURM_LOCALID", None)

    gpus = int(get("gpus_per_node", 2))
    master_port = int(get("master_port", 29500))
    dinov2_repo = Path(get("dinov2_repo", "/dkucc/home/oe23/dinov2_")).expanduser().resolve()
    config_file = Path(get("config_file", dinov2_repo / "dinov2/configs/train/sat_vit.yaml")).expanduser().resolve()
    output_dir = Path(get("output_dir", "/tmp/dinov2-outputs")).expanduser().resolve()
    extra_args = get("extra_args", []) or []

    if not dinov2_repo.exists():
        raise FileNotFoundError(f"DINOv2 repo not found: {dinov2_repo}")
    if not config_file.exists():
        raise FileNotFoundError(f"DINOv2 config not found: {config_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "--project",
        str(repo),
        "--",
        "torchrun",
        "--nproc_per_node",
        str(gpus),
        "--master_port",
        str(master_port),
        "dinov2/train/train.py",
        "--config-file",
        str(config_file),
        "--output-dir",
        str(output_dir),
    ]
    cmd += [str(arg) for arg in extra_args]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{dinov2_repo}:{env.get('PYTHONPATH', '')}"

    subprocess.run(cmd, check=True, cwd=dinov2_repo, env=env)


if __name__ == "__main__":  # pragma: no cover
    main()
