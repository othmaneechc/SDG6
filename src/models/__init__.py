"""Model registry exposing a uniform loading API."""

from __future__ import annotations

from models.base import DEFAULT_EXTS, ModelAdapter, resolve_device, resolve_dtype

__all__ = [
    "DEFAULT_EXTS",
    "ModelAdapter",
    "available_models",
    "load_model",
    "resolve_device",
    "resolve_dtype",
]


def _builders():
    from models.dino import load_model as load_dino
    from models.dinov3 import load_model as load_dinov3
    from models.galileo import load_model as load_galileo

    return {
        "dino": load_dino,
        "dinov3": load_dinov3,
        "galileo": load_galileo,
    }


def available_models() -> list[str]:
    return sorted(_builders().keys())


def load_model(name: str, **kwargs) -> ModelAdapter:
    builders = _builders()
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(builders))}")
    return builders[name](**kwargs)
