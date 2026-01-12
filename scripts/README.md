# DINOv3 k-NN batch job (HF weights)

- Script: `scripts/dinov3/knn_eval_dinov3.sbatch` (or `scripts/dinov3_knn.sbatch`) runs a single k-NN eval job.
- Driver: `src/models/dinov3/knn_eval.py` loads a DINOv3 backbone from Hugging Face and computes k-NN metrics on train/val/test splits.

## Required inputs
- `DATA_DIR` (env var): folder with `train/val/test` ImageFolder splits (default `/home/mila/e/echchabo/scratch/PW-m`).
- `WEIGHTS` (env var): Hugging Face model id (default `facebook/dinov3-vitb16-pretrain-lvd1689m`).
- Optional env vars: `WEIGHTS_TYPE` (`auto|lvd|sat`), `K_VALUES`, `EMBED_BATCH_SIZE`, `CKPT_DIR`, `VENV_ACTIVATE`.

## Submit
```bash
cd /home/mila/e/echchabo/projects/SDG6-Tracker
sbatch scripts/dinov3/knn_eval_dinov3.sbatch
```

# Galileo k-NN batch job (encoder embeddings)

- Script: `scripts/galileo/knn_eval_galileo.sbatch` runs a Galileo encoder k-NN eval job.
- Driver: `src/models/galileo/knn_eval.py` loads `single_file_galileo` weights and computes k-NN metrics on train/val/test splits.
  The k-NN voting logic is adapted from `nasaharvest/galileo` (`src/eval/knn.py`, MIT License).

## Required inputs
- `DATA_DIR` (env var): folder with `train/val/test` ImageFolder splits.
- `WEIGHTS_DIR` (env var): folder containing Galileo `config.json` + `encoder.pt`.
- Optional env vars: `BAND_INDICES`, `BAND_NAMES`, `VALUE_SCALE`, `NORMALIZE`, `COMPUTE_NDVI`.

## Submit
```bash
cd /home/mila/e/echchabo/projects/SDG6-Tracker
sbatch scripts/galileo/knn_eval_galileo.sbatch
```
