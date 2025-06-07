import os
import argparse
import csv
from pathlib import Path
from typing import List, Tuple
from sklearn.metrics import f1_score
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import random

def set_global_seed(seed: int = 42):
    """
    Make training as deterministic as reasonably possible.
    Call before creating any datasets, models, or dataloaders.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def detect_crack_score(
    img: np.ndarray,
    canny_low: int = 40,
    canny_high: int = 120,
) -> float:
    """
    Score = length of the longest skeleton component × its elongation ratio.
    Tuned for 227×227 tiles.  Requires *no* opencv-contrib.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # morphological thinning (Zhang–Suen like)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(edges)
    eroded = edges.copy()
    while True:
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, elem)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        eroded = cv2.erode(eroded, elem)
        if cv2.countNonZero(eroded) == 0:
            break

    # largest connected skeleton component
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(
        (skel > 0).astype("uint8"), connectivity=8
    )
    best = 0.0
    for i in range(1, num):
        length = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ratio = max(w, h) / (min(w, h) + 1e-3)
        best = max(best, length * ratio)
    return best


def build_pseudo_labels(train_dir: Path, out_csv: Path, thresh: float = 300.0):
    """
    Assign label 1 (crack) if score ≥ thresh, else 0.  Writes CSV for PyTorch.
    """
    rows: List[Tuple[str, int]] = []
    for p in tqdm(sorted(train_dir.glob("*.jpg")), desc="pseudo-labeling"):
        score = detect_crack_score(cv2.imread(str(p)))
        rows.append((str(p), int(score >= thresh)))
    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows([("file", "label"), *rows])


class CrackDataset(Dataset):
    def __init__(self, csv_file: Path, tf):
        self.items = [(r["file"], int(r["label"])) for r in csv.DictReader(csv_file.open())]
        self.tf = tf

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx):
        f, y = self.items[idx]
        return self.tf(Image.open(f).convert("RGB")), y


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([transforms.ToTensor()])
    return train_tf, val_tf


def accuracy(model, loader, device):
    model.eval()
    good = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            good += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return good / total if total else 0.0


def train_model(csv_path, out_path, bs=32, epochs=10, lr=1e-3):
    g = torch.Generator()
    g.manual_seed(42)
    tr_tf, val_tf = build_transforms()
    full = CrackDataset(csv_path, tr_tf)
    val_len = int(0.1 * len(full))
    train_ds, val_ds = random_split(
        full,
        [len(full) - val_len, val_len],
        generator=g,
    )
    val_ds.dataset.tf = val_tf

    nw = 0 if os.name == "nt" else min(4, os.cpu_count())
    tr_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        generator=g
    )
    val_loader = DataLoader(val_ds, bs, False, num_workers=nw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        val_acc = accuracy(model, val_loader, device)
        print(f"Epoch {ep:02d}  val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), out_path)
            print("  ↳  saved")


class TestFolderDataset(Dataset):
    def __init__(self, root: Path, tf):
        self.fp, self.lb = [], []
        for name, lbl in (("normal", 0), ("anomaly", 1)):
            for p in (root / name).glob("*.jpg"):
                self.fp.append(str(p)); self.lb.append(lbl)
        self.tf = tf
    def __len__(self): return len(self.fp)
    def __getitem__(self, idx):
        return self.tf(Image.open(self.fp[idx]).convert("RGB")), self.lb[idx]

def build_test_loader(root, tf, bs=32):
    nw = 0 if os.name == "nt" else min(4, os.cpu_count())
    return DataLoader(TestFolderDataset(root, tf), batch_size=bs,
                      shuffle=False, num_workers=nw)

def evaluate(model_path: Path, test_root: Path, metric: str = "acc"):
    """
    metric = "acc" → report accuracy
    metric = "f1"  → report binary F1-score (positive class = crack)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = build_transforms()
    loader = build_test_loader(test_root, val_tf)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device); model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = logits.argmax(1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    if metric == "acc":
        score = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"{test_root.name} accuracy: {score:.4f}")
    elif metric == "f1":
        score = f1_score(y_true, y_pred, pos_label=1)
        print(f"{test_root.name} F1-score: {score:.4f}")
    else:
        raise ValueError("metric must be 'acc' or 'f1'")

def dump_predictions(model_path: Path, test_root: Path, out_csv: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = build_transforms()

    class PredDS(Dataset):
        def __init__(self, root, tf):
            self.paths, self.labels = [], []
            for n,l in (("normal",0),("anomaly",1)):
                for p in (root/n).glob("*.jpg"):
                    self.paths.append(str(p)); self.labels.append(l)
            self.tf=tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img), self.labels[i], self.paths[i]

    loader = DataLoader(PredDS(test_root, val_tf), batch_size=32,
                        shuffle=False, num_workers=0)

    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device)); model.to(device); model.eval()

    rows = []
    with torch.no_grad():
        for x, y_true, paths in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), 1)[:, 1]      # P(crack)
            preds = (probs > 0.5).long().cpu()
            for pth, tl, pl, pa in zip(paths, y_true, preds, probs.cpu()):
                rows.append((pth, int(tl), int(pl), float(pa)))

    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows([("file","true","pred","p_crack"), *rows])
    print(f"Saved predictions to {out_csv}")


def main():
    set_global_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=str)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.data_root)
    csv_path = root / "pseudo_labels.csv"
    if not csv_path.exists():
        build_pseudo_labels(root / "train", csv_path, thresh=300.0)

    model_pth = root / "baseline_model.pth"
    train_model(csv_path, model_pth, epochs=args.epochs)

    evaluate(model_pth, root / "test_balanced", metric="acc")
    evaluate(model_pth, root / "test_unbalanced", metric="f1")

    dump_predictions(model_pth, root / "test_balanced",  root / "test_balanced_preds.csv")
    dump_predictions(model_pth, root / "test_unbalanced", root / "test_unbalanced_preds.csv")

if __name__ == "__main__":
    main()
