#!/usr/bin/env python
import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import utils
import vision_transformer as vits  # Ensure this module is available

# ----------------------------- #
#        ImageDataset Class     #
# ----------------------------- #

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images.
    """
    def __init__(self, image_paths, transform):
        """
        Args:
            image_paths (list): List of image file paths.
            transform (callable): Transformations to apply to images.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # In case of error, create a blank image.
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        transformed_image = self.transform(image)
        file_name = os.path.basename(image_path)
        return transformed_image, file_name

# ----------------------------- #
#        Model Loading          #
# ----------------------------- #

def load_model(checkpoint_path, model_arch='vit_base', patch_size=16, checkpoint_key='teacher', device=torch.device('cpu')):
    """
    Load the Vision Transformer model with pretrained weights.
    """
    if 'vit' in model_arch:
        model = vits.__dict__[model_arch](patch_size=patch_size, num_classes=0)
    else:
        raise ValueError(f"Model architecture {model_arch} is not supported.")

    utils.load_pretrained_weights(model, checkpoint_path, checkpoint_key, model_arch, patch_size)
    model.eval()
    model.to(device)
    return model

# ----------------------------- #
#  Feature Extraction & Prediction  #
# ----------------------------- #

def extract_features_and_predict(model, dataloader, device, knn):
    """
    Extract features from images and predict with k-NN.
    """
    features_list = []
    file_names = []
    with torch.no_grad():
        for images, fnames in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            for fname, feat in zip(fnames, outputs):
                file_names.append(fname)
                features_list.append(feat)
    feature_matrix = np.array(features_list)
    predictions = knn.predict(feature_matrix)
    return file_names, predictions

# ----------------------------- #
#         Worker Function       #
# ----------------------------- #

def run_worker(rank, world_size, args, image_paths, transform):
    """
    Each spawned process runs this function on its assigned GPU.
    Only the worker with rank 0 will print log messages.
    """
    # Define a logging helper: only print if rank is 0.
    log = print if rank == 0 else lambda *a, **k: None

    # Assign the current process to a GPU.
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    log(f"[Worker {rank}] Running on device: {device}")

    # Partition the image list among workers.
    total_images = len(image_paths)
    chunk_size = total_images // world_size
    start_idx = rank * chunk_size
    end_idx = total_images if rank == world_size - 1 else (rank + 1) * chunk_size
    local_image_paths = image_paths[start_idx:end_idx]
    log(f"[Worker {rank}] Processing {len(local_image_paths)} images.")

    # Prepare the dataset and dataloader for this worker.
    dataset = ImageDataset(local_image_paths, transform)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # Load the model on this GPU.
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_arch='vit_base',
        patch_size=8,  # Adjust as needed
        checkpoint_key='teacher',
        device=device
    )

    # Load and set up the k-NN classifier.
    if not os.path.exists(args.knn_classifier_path):
        raise FileNotFoundError(f"k-NN classifier not found at {args.knn_classifier_path}")
    knn_checkpoint = torch.load(args.knn_classifier_path, map_location='cpu')
    if "train_features" not in knn_checkpoint or "train_labels" not in knn_checkpoint:
        raise KeyError("k-NN checkpoint must contain 'train_features' and 'train_labels'.")
    train_features = knn_checkpoint["train_features"].cpu().numpy()
    train_labels = knn_checkpoint["train_labels"].cpu().numpy()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, train_labels)
    log(f"[Worker {rank}] k-NN classifier loaded and fitted.")

    # Extract features and perform predictions.
    file_names, predictions = extract_features_and_predict(model, dataloader, device, knn)

    # Save the worker's results to a temporary CSV file.
    worker_csv = f"{args.csv_path}_worker{rank}.csv"
    results = [{"file_name": fname, "prediction": pred} for fname, pred in zip(file_names, predictions)]
    pd.DataFrame(results).to_csv(worker_csv, index=False)
    log(f"[Worker {rank}] Saved predictions to {worker_csv}")

# ----------------------------- #
#           Main Script         #
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Prediction without nn.DataParallel")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--knn_classifier_path", type=str, required=True,
                        help="Path to the k-NN classifier checkpoint file")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Output CSV file path (final combined CSV)")
    parser.add_argument("--directory", type=str, required=True,
                        help="Directory containing images")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise FileNotFoundError(f"Image directory not found: {args.directory}")

    # List all .tif images in the directory.
    all_files = os.listdir(args.directory)
    image_paths = [os.path.join(args.directory, f)
                   for f in all_files if f.lower().endswith(".tif")]

    if len(image_paths) == 0:
        raise ValueError(f"No .tif images found in {args.directory}")

    print(f"Total images found: {len(image_paths)}")

    # Define image transformations.
    transform = pth_transforms.Compose([
        pth_transforms.Resize(64, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
    ])

    # Determine the number of GPUs available.
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Detected {world_size} GPU(s).")

    if world_size > 1:
        print("Spawning multiple processes for multi-GPU inference...")
        mp.spawn(run_worker,
                 args=(world_size, args, image_paths, transform),
                 nprocs=world_size,
                 join=True)
        # After all workers finish, combine their CSV results.
        all_dfs = []
        for rank in range(world_size):
            worker_csv = f"{args.csv_path}_worker{rank}.csv"
            if os.path.exists(worker_csv):
                df = pd.read_csv(worker_csv)
                all_dfs.append(df)
            else:
                print(f"Warning: {worker_csv} not found.")
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(args.csv_path, index=False)
            print(f"Combined predictions saved to {args.csv_path}")
        else:
            print("No results to combine.")
    else:
        # Single GPU (or CPU) fallback.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on a single device: {device}")

        model = load_model(
            checkpoint_path=args.checkpoint_path,
            model_arch='vit_base',
            patch_size=8,
            checkpoint_key='teacher',
            device=device
        )

        if not os.path.exists(args.knn_classifier_path):
            raise FileNotFoundError(f"k-NN classifier not found at {args.knn_classifier_path}")
        knn_checkpoint = torch.load(args.knn_classifier_path, map_location='cpu')
        train_features = knn_checkpoint["train_features"].cpu().numpy()
        train_labels = knn_checkpoint["train_labels"].cpu().numpy()
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_features, train_labels)
        print("k-NN classifier loaded and fitted.")

        dataset = ImageDataset(image_paths, transform)
        dataloader = DataLoader(dataset,
                                batch_size=64,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True)
        file_names, predictions = extract_features_and_predict(model, dataloader, device, knn)
        results = [{"file_name": fname, "prediction": pred} for fname, pred in zip(file_names, predictions)]
        pd.DataFrame(results).to_csv(args.csv_path, index=False)
        print(f"Predictions saved to {args.csv_path}")

if __name__ == "__main__":
    main()
