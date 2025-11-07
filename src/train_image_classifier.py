#!/usr/bin/env python3
# train_image_classifier.py
# Train an image sentiment classifier (3 classes) on data/memes.csv.

import argparse, os, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from PIL import Image, ImageOps

def open_rgb_safe(path: str | Path) -> Image.Image:
    """Open image and convert to RGB, correctly handling P/RGBA + transparency."""
    img = Image.open(path)
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        # composite onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


NUM_CLASSES = 3
LABELS = {0:"negative", 1:"positive", 2:"neutral"}

class MemeImageDataset(Dataset):
    def __init__(self, df, img_col="image_path", label_col="label", size=224, train=True):
        self.paths = df[img_col].tolist()
        self.labels = df[label_col].astype(int).tolist()
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomApply([T.GaussianBlur(3)], p=0.2),
                T.ToTensor(),
                T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[idx]
        try:
            img = open_rgb_safe(p)
        except Exception:
            img = Image.new("RGB", (224,224), (255,255,255))
        x = self.tf(img)
        return x, torch.tensor(y, dtype=torch.long)

def build_model(arch="resnet18", num_classes=NUM_CLASSES):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_dim = m.fc.in_features
        m.fc = nn.Linear(in_dim, num_classes)
        return m
    elif arch == "vgg19":
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m
    else:
        raise ValueError("Unsupported arch. Use resnet18 or vgg19")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro}, y, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/memes.csv")
    ap.add_argument("--out_dir", default="outputs/image_cls")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18","vgg19"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    if not {"image_path","label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: image_path,label")
    df = df.dropna(subset=["image_path","label"])
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed, stratify=df["label"])

    train_set = MemeImageDataset(train_df, size=args.img_size, train=True)
    val_set   = MemeImageDataset(val_df,   size=args.img_size, train=False)

    from torch.utils.data import WeightedRandomSampler

     # --- Balanced Sampling ---
     # count samples per class
    class_count = train_df["label"].value_counts().sort_index().to_dict()  # {0:..., 1:..., 2:...}

# assign inverse frequency as weight
    sample_weights = train_df["label"].map(lambda y: 1.0 / max(1, class_count[int(y)])).values.astype(np.float32)

# sampler with replacement = True (so all classes get balanced batches)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights),
                                num_samples=len(sample_weights),
                                replacement=True)

# use sampler for training loader
    train_loader = DataLoader(
      train_set,
      batch_size=args.batch_size,
      sampler=sampler,       
      num_workers=args.num_workers,
      pin_memory=True
      )

# validation loader 
    val_loader = DataLoader(
     val_set,
     batch_size=args.batch_size,
     shuffle=False,
     num_workers=args.num_workers,
     pin_memory=True
      )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch).to(device)

    cls_counts = train_df["label"].value_counts().sort_index().values.astype(np.float32)
    inv = 1.0 / np.maximum(cls_counts, 1.0)
    cls_w = torch.tensor(inv / inv.sum() * NUM_CLASSES, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_f1 = -1.0
    best_path = Path(args.out_dir)/"best_model.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * xb.size(0)

        train_loss = running / max(1, len(train_set))
        metrics, y_true, y_pred = evaluate(model, val_loader, device)

        print(f"[{epoch:02d}/{args.epochs}] train_loss={train_loss:.4f} "
              f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(),
                        "arch": args.arch,
                        "img_size": args.img_size}, best_path)
            rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred)
            with open(Path(args.out_dir)/"val_report.txt","w") as f: f.write(rep)
            np.savetxt(Path(args.out_dir)/"val_confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")
            with open(Path(args.out_dir)/"val_metrics.json","w") as f: json.dump(metrics, f, indent=2)
    scheduler.step()

    print(f"[DONE] Best macro-F1={best_f1:.4f}, saved to {best_path}")

if __name__ == "__main__":
    main()
