"""
NeuroScan-AI: Dataset Ingestion, Preprocessing & Augmentation Module
Downloads genuine Brain Tumor MRI scans from the benchmark dataset,
applies medical image augmentations, and yields PyTorch DataLoaders.
"""

import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_DISPLAY_NAMES = {
    'glioma': 'Glioma Tumor',
    'meningioma': 'Meningioma Tumor',
    'no_tumor': 'Normal (No Tumor)',
    'pituitary': 'Pituitary Tumor'
}
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

HF_REPO_ID = "PranomVignesh/MRI-Images-of-Brain-Tumor"
HF_BASE_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/timri"
HF_TREE_API = f"https://huggingface.co/api/datasets/{HF_REPO_ID}/tree/main/timri"


def fetch_file_list(split: str, category: str):
    """Fetch list of image filenames for a split and category from Hugging Face."""
    cat_dir = "no-tumor" if category == "no_tumor" else category
    url = f"{HF_TREE_API}/{split}/{cat_dir}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return [item.get("path") for item in data if item.get("type") == "file" and item.get("path").endswith((".jpg", ".png", ".jpeg"))]
    except Exception as e:
        print(f"Warning: Failed to fetch online tree for {split}/{category}: {e}")
        return []


def download_single_task(args):
    """Worker task for thread pool."""
    rel_path, dest = args
    if os.path.exists(dest):
        return True
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {rel_path}: {e}")
        return False


def prepare_dataset(data_dir: str = "data", n_train: int = 100, n_val: int = 25, n_test: int = 25, max_workers: int = 20):
    """
    Downloads and prepares a balanced split of genuine MRI scans for all 4 classes using multi-threading.
    """
    print("=" * 60)
    print("NeuroScan-AI: Preparing Genuine Brain Tumor MRI Dataset (Fast Parallel Mode)")
    print("=" * 60)

    splits = {
        'train': n_train,
        'valid': n_val,
        'test': n_test
    }

    download_tasks = []

    for split_name, count_per_class in splits.items():
        for cat in CLASS_NAMES:
            cat_folder = os.path.join(data_dir, split_name, cat)
            os.makedirs(cat_folder, exist_ok=True)
            
            existing = [f for f in os.listdir(cat_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
            needed = count_per_class - len(existing)
            if needed <= 0:
                print(f"[{split_name.upper()}] {cat}: {len(existing)} images already present.")
                continue

            # Fetch file list from HF
            files = fetch_file_list(split_name, cat)
            candidates = files[:needed]
            for fpath in candidates:
                fname = os.path.basename(fpath)
                dest = os.path.join(cat_folder, fname)
                download_tasks.append((fpath, dest))

    if download_tasks:
        print(f"Downloading {len(download_tasks)} images concurrently using {max_workers} worker threads...")
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_single_task, task) for task in download_tasks]
            for fut in as_completed(futures):
                if fut.result():
                    completed += 1
                if completed % 50 == 0 or completed == len(download_tasks):
                    print(f"Progress: {completed}/{len(download_tasks)} images downloaded.")

    print("\nDataset preparation completed successfully.")
    print_dataset_summary(data_dir)


def print_dataset_summary(data_dir: str = "data"):
    """Prints a clear breakdown table of the dataset images per split and class."""
    print("\nDataset Summary Breakdown:")
    print("-" * 55)
    print(f"{'Split':<10} | {'Class':<15} | {'Count':<8} | {'Display Name':<20}")
    print("-" * 55)
    total = 0
    for split in ['train', 'valid', 'test']:
        for cat in CLASS_NAMES:
            p = os.path.join(data_dir, split, cat)
            c = len(os.listdir(p)) if os.path.exists(p) else 0
            total += c
            print(f"{split:<10} | {cat:<15} | {c:<8} | {CLASS_DISPLAY_NAMES[cat]:<20}")
    print("-" * 55)
    print(f"Total dataset images across all splits: {total}\n")


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for Brain Tumor MRI Images."""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for cat in CLASS_NAMES:
            cat_dir = os.path.join(root_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            for fname in os.listdir(cat_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(cat_dir, fname)
                    self.samples.append((fpath, CLASS_TO_IDX[cat]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def get_transforms():
    """Returns training and validation/test torchvision transform pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir: str = "data", batch_size: int = 32, num_workers: int = 0):
    """Builds PyTorch DataLoaders for train, valid, and test splits."""
    train_tf, eval_tf = get_transforms()

    train_ds = BrainTumorDataset(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds = BrainTumorDataset(os.path.join(data_dir, 'valid'), transform=eval_tf)
    test_ds = BrainTumorDataset(os.path.join(data_dir, 'test'), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    prepare_dataset(data_dir="data", n_train=100, n_val=25, n_test=25)
