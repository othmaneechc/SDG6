#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import utils
import vision_transformer as vits  # Ensure this module is available

# ----------------------------- #
#        ImageDataset Class     #
# ----------------------------- #

class ImageDataset(Dataset):
    """Dataset that returns transformed image, label, and path."""

    def __init__(self, items, transform):
        """
        Args:
            items (list[tuple[str,int]]): (path,label_index)
            transform (callable): Transformations to apply to images.
        """
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        transformed_image = self.transform(image)
        return transformed_image, label, image_path

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
    """Extract features from images and predict with k-NN."""
    features_list = []
    labels_list = []
    paths_list = []
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting features", leave=False):
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            features_list.extend(outputs)
            labels_list.extend(labels)
            paths_list.extend(paths)
    feature_matrix = np.array(features_list)
    predictions = knn.predict(feature_matrix)
    return paths_list, np.array(labels_list), predictions

# ----------------------------- #
#         Worker Function       #
# ----------------------------- #

def collect_items_with_labels(root_dir, exts=(".tif", ".tiff", ".jpg", ".jpeg", ".png")):
    """Collect (path, label_idx) pairs assuming ImageFolder-style subdirectories."""
    classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
    if not classes:
        raise ValueError(f"No class subdirectories found in {root_dir}")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(exts):
                items.append((os.path.join(cls_dir, fname), class_to_idx[cls]))
    if not items:
        raise ValueError(f"No images with extensions {exts} found under {root_dir}")
    return items, classes

# ----------------------------- #
#           Main Script         #
# ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Single-GPU prediction with confusion matrix.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--knn_classifier_path", type=str, required=True,
                        help="Path to the k-NN classifier checkpoint file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store confusion matrix and logs")
    parser.add_argument("--directory", type=str, required=True,
                        help="ImageFolder-style root containing class subdirectories")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise FileNotFoundError(f"Image directory not found: {args.directory}")

    os.makedirs(args.output_dir, exist_ok=True)
    items, classes = collect_items_with_labels(args.directory)
    class_names_path = os.path.join(args.output_dir, "class_names.txt")
    if not os.path.exists(class_names_path):
        with open(class_names_path, "w") as f:
            f.write("\n".join(classes))
    print(f"Found {len(items)} images across {len(classes)} classes.")

    # Define image transformations.
    transform = pth_transforms.Compose([
        pth_transforms.Resize(64, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
    ])

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

    dataset = ImageDataset(items, transform)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    paths, labels, predictions = extract_features_and_predict(model, dataloader, device, knn)

    acc = float((predictions == labels).mean()) if len(labels) else 0.0
    print(f"Accuracy: {acc*100:.2f}% on {len(labels)} samples")

    cm = confusion_matrix(labels, predictions, labels=list(range(len(classes))))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )
    cm_path = os.path.join(args.output_dir, "confusion.txt")
    with open(cm_path, "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}% on {len(labels)} samples\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("\nClassification report:\n")
        f.write(report)
        f.write("\n")
    print(f"Confusion matrix and report saved to {cm_path}")

if __name__ == "__main__":
    main()
