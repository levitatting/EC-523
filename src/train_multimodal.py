#!/usr/bin/env python3
"""
train_multimodal.py
Fuse text (BERT) + image features for meme sentiment (3 classes).
- Loads your fine-tuned text branch checkpoint (encoder weights)
- Loads pre-exported image features (from export_image_features.py)
- Concatenates [text_cls(768)] and [img_proj] -> MLP head
- Class-weighted CE (optional) + label smoothing
- Cosine LR + AMP
"""

import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModel
import re

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

NUM_CLASSES = 3

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, img_feat_map, img_feat_dim, text_col="text", path_col="image_path", label_col="label", train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.img_feat_map = img_feat_map  # dict path->np.array
        self.img_feat_dim = img_feat_dim
        self.text_col = text_col
        self.path_col = path_col
        self.label_col = label_col
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        txt = clean_text(str(row.get(self.text_col, "") or ""))
        enc = self.tokenizer(
            txt, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        p = row[self.path_col]
        img_feat = self.img_feat_map.get(p)
        if img_feat is None:
            img_feat = np.zeros(self.img_feat_dim, dtype=np.float32)
        img_feat = torch.tensor(img_feat, dtype=torch.float32)

        y = int(row[self.label_col])
        return input_ids, attn_mask, img_feat, torch.tensor(y, dtype=torch.long)

class FusionModel(nn.Module):
    def __init__(self, text_model_name_or_dir, img_feat_dim, hidden=512, dropout=0.2):
        super().__init__()
        # load encoder weights from your fine-tuned text checkpoint directory
        self.text = AutoModel.from_pretrained(text_model_name_or_dir)  # base encoder (no CLS head)
        text_dim = self.text.config.hidden_size  # typically 768 for bert-base
        # project image feats to a comparable scale
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # fuse text[CLS] + img_proj -> MLP head
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, input_ids, attention_mask, img_feat):
        out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token embedding
        if isinstance(out, (tuple, list)):
            text_cls = out[0][:, 0, :]
        else:
            text_cls = out.last_hidden_state[:, 0, :]
        img_h = self.img_proj(img_feat)
        fused = torch.cat([text_cls, img_h], dim=1)
        logits = self.classifier(fused)
        return logits

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        input_ids, attn_mask, img_feat, y = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        img_feat  = img_feat.to(device)
        y = y.to(device)
        logits = model(input_ids, attn_mask, img_feat)
        pred = logits.argmax(dim=1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f1_macro}, y, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/memes.csv")
    ap.add_argument("--image_feats", default="features/image_feats.npy")
    ap.add_argument("--image_index", default="features/image_index.csv")
    ap.add_argument("--text_ckpt", default="outputs/text_bert/best_model")  # your BERT finetuned dir
    ap.add_argument("--out_dir", default="outputs/multimodal")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--text_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load index/csv
    df = pd.read_csv(args.csv).dropna(subset=["image_path","label"]).reset_index(drop=True)
    # load image feats and build path->feat map
    feats = np.load(args.image_feats)  # [N, D]
    idx_df = pd.read_csv(args.image_index)  # columns: image_path,label
    assert len(feats) == len(idx_df), "image_feats and image_index must align"

    img_feat_map = {p: feats[i] for i, p in enumerate(idx_df["image_path"].tolist())}
    img_feat_dim = feats.shape[1]

    # tokenizer & split
    tokenizer = AutoTokenizer.from_pretrained(args.text_ckpt, use_fast=True)
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed, stratify=df["label"])

    # datasets
    train_set = MultiModalDataset(train_df, tokenizer, args.max_len, img_feat_map, img_feat_dim)
    val_set   = MultiModalDataset(val_df,   tokenizer, args.max_len, img_feat_map, img_feat_dim)

    # class weights (optional): use counts on train split
    cnt = Counter(train_df["label"].astype(int).tolist())
    cls_w = torch.tensor([1.0 / max(1, cnt.get(i,1)) for i in range(NUM_CLASSES)], dtype=torch.float32)
    cls_w = cls_w / cls_w.sum() * NUM_CLASSES  # normalize
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_w = cls_w.to(device)

    # criterion (with or without class weights both fine; text already strong)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

    # loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(args.text_ckpt, img_feat_dim, hidden=512, dropout=args.dropout).to(device)

    # optimizer & schedulers
    text_params = list(model.text.parameters())
    head_params = list(model.img_proj.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
    [
        {"params": text_params, "lr": 1e-5},     # 文本分支较小 LR，避免破坏预训练
        {"params": head_params, "lr": args.lr},  # 融合头与图像侧使用原 LR
    ],
    weight_decay=args.weight_decay
    )

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=1)  # 1 warmup epoch
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs-1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_f1 = -1.0
    best_path = Path(args.out_dir) / "best_fusion.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids, attn_mask, img_feat, y = batch
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            img_feat  = img_feat.to(device,  non_blocking=True)
            y = torch.as_tensor(y, device=device)


            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(input_ids, attn_mask, img_feat)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * input_ids.size(0)

        train_loss = epoch_loss / max(1, len(train_set))
        metrics, y_true, y_pred = evaluate(model, val_loader, device)
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_f1={metrics['macro_f1']:.4f} "
            f"lr_text={lrs[0]:.6f} lr_head={lrs[1]:.6f}"
        )
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "text_ckpt": args.text_ckpt,
                "img_feat_dim": img_feat_dim,
                "max_len": args.max_len,
            }, best_path)
            # reports
            rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
            cm  = confusion_matrix(y_true, y_pred)
            with open(Path(args.out_dir)/"val_report.txt","w") as f: f.write(rep)
            with open(Path(args.out_dir)/"val_metrics.json","w") as f: json.dump(metrics, f, indent=2)
            np.savetxt(Path(args.out_dir)/"val_confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")

        scheduler.step()

    print(f"[DONE] Best macro-F1={best_f1:.4f} -> {best_path}")

if __name__ == "__main__":
    main()
