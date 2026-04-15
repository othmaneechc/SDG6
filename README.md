# SDG6 Tracker

This repository contains the code for the paper:

Monitoring access to piped water and sanitation infrastructure in Africa at disaggregated scales using satellite imagery and self-supervised learning https://arxiv.org/abs/2411.19093

It provides a unified pipeline for:
- pretraining DINO/DINOv2 backbones
- extracting embeddings
- k‑NN evaluation/classification
- large‑scale inference on new imagery

The core design is a model‑agnostic CLI and adapter layer, so you can switch encoders without touching the data/metric code.

## Repository layout
- src/models: single‑file adapters (dino, dinov2, dinov3, prithvi, galileo)
- src/sdg6: data loader, embedding extraction, k‑NN logic, CLI, inference
- scripts/configs: YAML configs for train/eval/inference
- scripts/slurm: SLURM launchers
- scripts/training: Python wrappers for pretraining
- scripts/analysis: Python analysis workflows (including converted notebooks)
- outputs: generated artifacts (`figures/`, `tables/`, `reports/`)

## External Sources
- DINO adapter: https://github.com/facebookresearch/dino (Caron et al., 2021, https://arxiv.org/abs/2104.14294)
- DINOv2 adapter: https://github.com/facebookresearch/dinov2 (Oquab et al., 2023, https://arxiv.org/abs/2304.07193)
- DINOv3 adapter: https://github.com/facebookresearch/dinov3 and HF model cards under https://huggingface.co/facebook
- Prithvi adapter: TerraTorch registry (https://github.com/IBM/terratorch) + IBM/NASA Geospatial checkpoints (https://huggingface.co/ibm-nasa-geospatial)
- Galileo adapter: adapted from https://github.com/nasaharvest/galileo

## Data and Weights
- Galileo model weights: hosted on Hugging Face (base model used here), e.g. https://huggingface.co/nasaharvest/galileo
- DINO and DINOv2 weights: Zenodo, https://zenodo.org/records/19156085
- Inference results: Zenodo, https://zenodo.org/records/19156085
- Population density patches: Zenodo, https://zenodo.org/records/19156085
- Afrobarometer imagery tiles: Zenodo, https://zenodo.org/records/14740420

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
sbatch scripts/slurm/dinov2_pt.sbatch
```

## k‑NN evaluation/classification
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
sbatch scripts/slurm/dinov2.sbatch
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
sbatch scripts/slurm/dinov2_infer.sbatch
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
