# SDG6 Tracker – Tracking Progress Towards SDG 6 Using Satellite Imagery

This repository contains the code for the paper:

Tracking Progress Towards Sustainable Development Goal 6 Using Satellite Imagery
https://arxiv.org/abs/2411.19093

It provides a unified pipeline for:
- pretraining DINO/DINOv2 backbones
- extracting embeddings
- k‑NN evaluation/classification
- large‑scale inference on new imagery

The core design is a model‑agnostic CLI and adapter layer, so you can switch encoders without touching the data/metric code.

## Repository layout
- src/models: single‑file adapters (dino, dinov2, dinov3, galileo)
- src/sdg6: data loader, embedding extraction, k‑NN logic, CLI, inference
- scripts: SLURM sbatch files and YAML configs

## Environment
Use uv to sync dependencies in this repo:

```bash
uv sync
```

## Pretraining (DINOv2)
Pretraining is driven by a YAML config and a wrapper that launches torchrun.

1) Edit the pretraining config:

scripts/configs/dinov2_pt.yaml

Key fields:
- dinov2_repo: local clone of the DINOv2 repo
- config_file: DINOv2 training config YAML (satellite config)
- output_dir: output directory for checkpoints and logs
- gpus_per_node, master_port

2) Launch pretraining:

```bash
bash scripts/dinov2_pt.sbatch
```

## k‑NN evaluation / classification
Evaluation uses the unified CLI and the DINOv2‑aligned k‑NN logic (cosine similarity + softmax voting).

1) Configure the eval:

scripts/configs/dinov2.yaml

Key fields:
- data_dir: dataset with train/val/test class folders
- weights: DINOv2 checkpoint (teacher_checkpoint.pth)
- dinov2_config: DINOv2 training config
- knn_classifier_path: where to save the k‑NN classifier artifact

2) Run k‑NN evaluation:

```bash
bash scripts/dinov2.sbatch
```

Outputs:
- Embeddings (optional): output_dir/embeddings
- Confusion reports: output_dir/confusion
- k‑NN classifier artifact: path set in knn_classifier_path

## Inference on arbitrary images
Inference uses the saved k‑NN classifier artifact and a DINOv2 encoder to produce predictions on new images.

1) Configure inference:

scripts/configs/dinov2_infer.yaml

Key fields:
- weights, dinov2_config: DINOv2 encoder settings
- knn_classifier_path: saved artifact from evaluation
- input_dir or input_list: images to score
- output_csv: where predictions are written

2) Run inference on many countries:

```bash
bash scripts/dinov2_infer.sbatch
```

This sbatch loops over data/countries and writes SW/PW predictions under data/inference/<country>.

## Adding a new model
1) Create a new adapter in src/models/<name>.py that returns a ModelAdapter:
   - transform(image, path=None)
   - reader(path)
   - collate_fn
   - encode(batch) returning L2‑normalized features
2) Register it in src/models/__init__.py
3) Run the unified CLI:

```bash
python -m sdg6.cli --model <name> ...
```

## Notes
- The unified dataloader expects ImageFolder structure for classification: data_dir/train|val|test/<class_name>/*.tif
- The k‑NN logic is aligned to DINOv2: cosine similarity + softmax voting (temperature configurable).

## Acknowledgements
- DINO (Caron et al. 2021) via the dino dependency
- DINOv3 (Oquab et al. 2023) via Hugging Face
- Galileo encoder adapted from nasaharvest/galileo (MIT)
