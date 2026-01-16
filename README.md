# SDG6 Tracker – unified embedding + k-NN benchmark

This repo now exposes a single, model-agnostic pipeline for extracting embeddings and running k-NN benchmarks across multiple remote-sensing foundation models.

## What changed
- **One dataloader for everything**: `sdg6.data.build_dataloaders` feeds the same batch schema (`{"image", "label", "path"}`) to embedding, k-NN evaluation, and DINO training loops (set `allow_unlabeled=True` for self-supervised data).
- **Single-file model adapters** under `src/models/`:
  - `dino.py` wraps the DINO ViT dependency (provide your checkpoint path).
  - `dinov3.py` wraps Hugging Face checkpoints.
  - `galileo.py` wraps the local Galileo encoder + weights folder.
- **Model-agnostic CLI**: switch models with one flag; everything else (data handling, embedding extraction, k-NN) stays identical.
- **Dependencies only**: DINO code comes from the installed `dino` package; Galileo is fully contained in `src/models/galileo.py`.

## Layout
```
src/
  models/               # Single-file adapters (dino.py, dinov3.py, galileo.py)
  sdg6/                 # Data loader, embedding, k-NN, CLI
```

## Quickstart
The code assumes ImageFolder splits: `data_dir/{train,val,test}/{class_name}/*.tif|*.png|...`.

```bash
export PYTHONPATH=src

# DINOv3 (Hugging Face)
python -m sdg6.cli \
  --model dinov3 \
  --weights facebook/dinov3-vitb16-pretrain-lvd1689m \
  --data-dir /path/to/PW-m \
  --output-dir runs/dinov3-PW-m

# Galileo (local weights folder containing config.json + encoder.pt)
python -m sdg6.cli \
  --model galileo \
  --weights /path/to/galileo_weights \
  --data-dir /path/to/PW-m \
  --galileo-band-names B2,B3,B4,B8 \
  --galileo-compute-ndvi

# DINO (dependency checkpoint)
python -m sdg6.cli \
  --model dino \
  --weights /path/to/dino/checkpoint.pth \
  --dino-arch vit_base --dino-patch-size 8 --dino-checkpoint-key teacher \
  --data-dir /path/to/PW-m
```

Outputs:
- Embeddings cached under `OUTPUT_DIR/embeddings/<split>.npz`
- Confusion matrices + reports under `OUTPUT_DIR/confusion/`

## Adding a new model
1. Drop a new file under `src/models/<name>.py` that returns a `ModelAdapter` with:
   - `transform(image, path=None)`, `reader(path)`, `collate_fn`, and `encode(batch)` returning L2-normalized features.
2. Register it in `src/models/__init__.py`.
3. Run `python -m sdg6.cli --model <name> ...` — no other code changes needed.

## Notes on training
- The unified dataloader works for DINO pretraining/fine-tuning; pass `allow_unlabeled=True` to use unlabeled data while keeping the same batch schema.
- Strong data augmentations can be plugged in via the `transform` argument without changing downstream code.

## Citations / acknowledgements
- DINO (Caron et al. 2021) via the `dino` dependency.
- DINOv3 (Oquab et al. 2023) via Hugging Face.
- Galileo encoder adapted from `nasaharvest/galileo` (MIT), consolidated into `src/models/galileo.py`.
