#!/usr/bin/env python3
"""
export_image_features.py
Extract image embeddings using pretrained CNN (e.g., ResNet18) and save them to .npy + .csv.
"""
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, df, img_size=224):
        self.df = df
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])
        try:
            img = Image.open(path).convert("RGB")
            img = self.tf(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img, label, path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["image_path","label"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = getattr(models, args.arch)(weights=None)
    base.fc = nn.Identity()  # remove classifier
    model = base
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()

    loader = DataLoader(ImageDataset(df, args.img_size),
                        batch_size=args.batch_size, shuffle=False, num_workers=4)
    feats, labels, paths = [], [], []
    with torch.no_grad():
        for xb, yb, pb in tqdm(loader, desc="Extracting"):
            xb = xb.to(device)
            h = model(xb).cpu().numpy()
            feats.append(h); labels += yb.tolist(); paths += pb
    feats = np.concatenate(feats, axis=0)
    np.save(args.out_npy, feats)
    pd.DataFrame({"image_path": paths, "label": labels}).to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved {feats.shape} -> {args.out_npy}")

if __name__ == "__main__":
    main()
