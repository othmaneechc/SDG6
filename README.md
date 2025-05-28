# SDG6

This repository implements a deep‐learning pipeline to **predict piped water and sewage access** from satellite imagery, using a self‐supervised DINO backbone. See our paper on arXiv for full details: https://arxiv.org/abs/2411.19093.

---

## Repository Structure

```text
SDG6/
├── model/                   # PyTorch inference code
│   ├── predict.py           # CLI for running inference on .tif tiles
│   └── prediction.sh        
│   └── utils.py             # helper functions (pre‐/post‐processing)
│   └── vision_tranformer.py
│
├── weights/imbalanced_pw/   # pretrained model weights  
│   └── checkpoint.pth
│   └── knn_classifier.pth
│
└── README.md                # This file
```

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/othmaneechc/SDG6.git
   cd SDG6
   ```

2. **Create & activate a Python environment** (tested with Python 3.9)  
   ```bash
   conda create -n sdg6 python=3.9 -y
   conda activate sdg6
   ```

3. **Install dependencies**  
    Check https://github.com/facebookresearch/dino for details.

---

## Preparing Your Data

- **Satellite tiles**:  
  Place your downloaded Sentinel/Landsat `.tif` tiles in a folder, e.g. `data/images/`.

- **Optional metadata**:  
  If you have survey shapefiles or population rasters, store them (and adjust paths) under `data/`.

---

## Running Predictions

All-in-one Bash wrapper: **prediction.sh**  
This script will:

1. Loop over each `.tif` tile in your input folder  
2. Invoke `python model/predict.py` with the right flags  
3. Save output masks (NumPy or GeoTIFF) into your designated output directory

### Usage

```bash
python /model/predict.py \
        --checkpoint_path '${CHECKPOINT_PATH}' \
        --knn_classifier_path '${KNN_CLASSIFIER_PATH}' \
        --csv_path '${CSV_PATH}' \
        --directory '${DIRECTORY}'"
```
---

## Citation

If you use this code in your research, please cite:

> Echchabi, O., Lahlou, A., Talty, N., Manto, J., & Lam, K. L. (2025). Tracking Progress Towards SDG 6 Using Satellite Imagery. *arXiv preprint arXiv:2411.19093*.

---

## Contact

For questions or support, please reach out at **othmane.echchabi@mail.mcgill.ca**.
