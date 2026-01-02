import os

# -----------------------
# Limit CPU threading
# -----------------------
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")

# NCCL settings (for multi-GPU stability)
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_TIMEOUT"] = "1800"

import random
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from torchvision.transforms import v2
from tqdm import tqdm

# Path to your local DINOv3 repo (where hubconf.py lives)
REPO_DIR = "/home/mila/e/echchabo/projects/SDG6-Tracker/src/dinov3"


############################################################
# TRANSFORMS
############################################################
def make_transform(resize_size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143),
        ),
    ])


############################################################
# FEATURE EXTRACTION (CLS token)
############################################################
def extract_cls(model, img_tensor, device):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        img_tensor = img_tensor.to(device, non_blocking=True)
        out = model.forward_features(img_tensor)

        if isinstance(out, list):
            out = out[0]

        if "x_norm_clstoken" in out:
            cls = out["x_norm_clstoken"]
        elif "x_cls" in out:
            cls = out["x_cls"]
        elif "x_prenorm" in out:
            cls = out["x_prenorm"][:, 0]
        else:
            raise KeyError(f"No CLS token in forward_features output: {out.keys()}")

        cls = cls.squeeze(0)
        return cls.cpu().numpy()


############################################################
# PER-GPU EMBEDDING EXTRACTION
############################################################
def build_embeddings(split_dir, model, transform, gpu_id, world_size):
    labels = sorted(os.listdir(split_dir))
    files = []

    # read class folders
    for label_str in labels:
        class_dir = os.path.join(split_dir, label_str)
        if not os.path.isdir(class_dir):
            continue

        try:
            label_int = int(float(label_str))   # handles "0.0", "1.0"
        except ValueError:
            continue

        for fname in os.listdir(class_dir):
            files.append((os.path.join(class_dir, fname), label_int))

    # Shuffle then shard across GPUs to reduce imbalance
    random.shuffle(files)
    files = files[gpu_id::world_size]

    device = torch.device(f"cuda:{gpu_id}")
    X_local, y_local = [], []

    iterable = tqdm(
        files,
        desc=f"GPU {gpu_id} - {os.path.basename(split_dir)}",
        position=gpu_id,
        leave=False,
    )

    for fpath, label_int in iterable:
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception:
            print(f"[GPU {gpu_id}] Skipping corrupted file {fpath}")
            continue

        img_t = transform(img).unsqueeze(0)
        feat = extract_cls(model, img_t, device)
        X_local.append(feat)
        y_local.append(label_int)

    if len(X_local) == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(X_local), np.array(y_local)


############################################################
# MAIN WORKER (EACH GPU)
############################################################
def main_worker(gpu_id, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=gpu_id,
    )
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    transform = make_transform()

    # ----------------------------
    # Load model (no DDP)
    # ----------------------------
    if gpu_id == 0:
        print(f"\n[Rank 0] Loading model {args.model_name}")
        print(f"[Rank 0] Using weights: {args.weights}\n")

    model = torch.hub.load(
        REPO_DIR,
        args.model_name,
        source="local",
        weights=args.weights,
    )
    model = model.to(device).eval().half()

    splits = ["train", "val", "test"]
    Xs, ys = {}, {}

    # ----------------------------
    # CHECKPOINTING DIRECTORY
    # ----------------------------
    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    else:
        ckpt_dir = os.path.join(args.data_dir, "_embeddings")

    if gpu_id == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    # ----------------------------
    # PROCESS EACH SPLIT
    # ----------------------------
    for split in splits:
        split_dir = os.path.join(args.data_dir, split)

        X_path = os.path.join(ckpt_dir, f"X_{split}.npy")
        y_path = os.path.join(ckpt_dir, f"y_{split}.npy")

        # ---------------------------------------------------
        # RANK 0: CHECK IF CHECKPOINT EXISTS
        # ---------------------------------------------------
        use_ckpt = False
        if gpu_id == 0 and os.path.exists(X_path) and os.path.exists(y_path):
            print(f"[Rank 0] Found cached embeddings for {split} in {ckpt_dir}. Loading...")
            Xs[split] = np.load(X_path, mmap_mode="r")
            ys[split] = np.load(y_path, mmap_mode="r")
            use_ckpt = True

        # Broadcast decision to all ranks (if multi-GPU)
        if args.world_size > 1:
            out_list = [use_ckpt]
            dist.broadcast_object_list(out_list, src=0)
            use_ckpt = out_list[0]

        # ---------------------------------------------------
        # If cached → skip inference on all GPUs
        # ---------------------------------------------------
        if use_ckpt:
            dist.barrier()
            continue

        # ---------------------------------------------------
        # Otherwise compute embeddings distributed
        # ---------------------------------------------------
        if gpu_id == 0:
            print(f"[Rank 0] Computing embeddings for: {split}")

        X_local, y_local = build_embeddings(
            split_dir, model, transform, gpu_id, args.world_size
        )

        # gather
        X_list = [None] * args.world_size
        y_list = [None] * args.world_size
        dist.all_gather_object(X_list, X_local)
        dist.all_gather_object(y_list, y_local)

        # assemble on rank 0
        if gpu_id == 0:
            Xs[split] = np.concatenate([x for x in X_list if x is not None and len(x) > 0], axis=0)
            ys[split] = np.concatenate([y for y in y_list if y is not None and len(y) > 0], axis=0)

            print(f"[Rank 0] {split}: {Xs[split].shape[0]} samples, "
                  f"feat_dim={Xs[split].shape[1]}")

            # SAVE CHECKPOINT
            np.save(X_path, Xs[split])
            np.save(y_path, ys[split])

        dist.barrier()

    # -------------------------------------------------------
    # ONLY RANK 0 → RUN k-NN
    # -------------------------------------------------------
    if gpu_id == 0:
        print("\n==============================")
        print("   DINOv3 Satellite k-NN Eval")
        print("==============================")

        Xtr = Xs["train"];  ytr = ys["train"]
        Xval = Xs["val"];   yval = ys["val"]
        Xte = Xs["test"];   yte = ys["test"]

        print(f"Train size: {len(ytr)}, Val size: {len(yval)}, Test size: {len(yte)}")

        k_list = [int(k) for k in args.k_values.split(",")]
        results = []

        for k in k_list:
            print(f"Training k-NN with k = {k} ...")

            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric="cosine",
                n_jobs=8,
            )
            knn.fit(Xtr, ytr)
            val_acc = knn.score(Xval, yval)
            test_acc = knn.score(Xte, yte)

            print(f"  -> Val:  {val_acc*100:.2f}% | Test: {test_acc*100:.2f}%")
            results.append((k, val_acc, test_acc))

        print("\n==== Summary (cosine k-NN) ====")
        print("   k    |   Val Acc   |  Test Acc")
        print("---------------------------------")
        for k, v, t in results:
            print(f"{k:6d} | {v*100:9.2f}% | {t*100:9.2f}%")
        print("================================")

    dist.barrier()
    dist.destroy_process_group()


############################################################
# MAIN
############################################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--k_values", type=str, default="1,5,10,20,50,100")
    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count()
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to save/load embeddings. "
             "If not set, defaults to DATA_DIR/_embeddings"
    )

    args = parser.parse_args()
    print("Using GPUs:", args.world_size)

    mp.spawn(
        main_worker,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()


'''torchrun --nproc_per_node=1 knn_eval.py \
  --data_dir /home/mila/e/echchabo/scratch/PW-m \
  --model_name dinov3_vitl16 \
  --weights /home/mila/e/echchabo/projects/SDG6-Tracker/src/dinov3/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --k_values 1,5,10,20,50,100
'''
