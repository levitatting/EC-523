#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_text_classifier.py
Train a BERT text sentiment classifier on data/memes_text.csv (columns: text,label).

Features:
- Stratified train/val split
- Class-weighted loss (handles label imbalance)
- Mixed precision (fp16) if CUDA available
- Saves best model (by macro F1) + metrics + confusion matrix
- Reproducible (seeded)
"""

import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ----------------------------
# Data utils
# ----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    # drop bad rows
    df = df.dropna(subset=["text", "label"]).copy()
    # coerce label to int
    df["label"] = df["label"].astype(str).str.strip()
    # allow labels like "0","1","2" or integers already
    df["label"] = df["label"].astype(int)
    # clean text lightly
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0]
    return df


class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ----------------------------
# Weighted loss via custom Trainer
# ----------------------------

@dataclass
class WeightedTrainerConfig:
    class_weights: Optional[torch.Tensor] = None  # shape [num_labels]


class WeightedTrainer(Trainer):
    def __init__(self, *args, wt_cfg: Optional[WeightedTrainerConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wt_cfg = wt_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.wt_cfg and self.wt_cfg.class_weights is not None:
            cw = self.wt_cfg.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1}


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/memes_text.csv", help="Input CSV with columns text,label")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--out_dir", default="outputs/text_bert", help="Directory to save checkpoints/metrics")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load data
    df = load_data(args.csv)

    # Expect labels in {0,1,2}
    unique_labels = sorted(df["label"].unique().tolist())
    num_labels = len(unique_labels)
    if not set(unique_labels).issubset({0,1,2}):
        raise ValueError(f"Labels must be in {{0,1,2}}. Got: {unique_labels}")

    # stratified split
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["label"]
    )

    # 2) Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = SimpleTextDataset(train_df["text"], train_df["label"], tokenizer, args.max_len)
    val_ds   = SimpleTextDataset(val_df["text"],   val_df["label"],   tokenizer, args.max_len)

    # 3) Config & model
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label={i: str(i) for i in range(num_labels)},
        label2id={str(i): i for i in range(num_labels)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # 4) Class weights (inverse frequency)
    counts = train_df["label"].value_counts().sort_index()
    freq = counts.values.astype(np.float32)
    invf = 1.0 / np.maximum(freq, 1.0)
    class_weights = torch.tensor(invf / invf.sum() * num_labels, dtype=torch.float32)

    # 5) TrainingArguments
    fp16_ok = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16_ok,
        logging_dir=os.path.join(args.out_dir, "tb"),
        logging_steps=50,
        report_to=["none"],  # set to ["tensorboard"] if you want TB logs
        seed=args.seed,
    )

    # 6) Trainer (weighted loss)
    tcfg = WeightedTrainerConfig(class_weights=class_weights)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        wt_cfg=tcfg,
    )

    # 7) Train
    train_result = trainer.train()
    trainer.save_model(os.path.join(args.out_dir, "best_model"))  # config, tokenizer and model
    tokenizer.save_pretrained(os.path.join(args.out_dir, "best_model"))

    # 8) Eval & reports
    eval_metrics = trainer.evaluate()
    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    y_true = np.array(list(val_df["label"].values))

    cls_report = classification_report(y_true, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, preds)

    # 9) Save artifacts
    with open(os.path.join(args.out_dir, "train_result.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items()}, f, indent=2)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(cls_report)
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm.astype(int), fmt="%d", delimiter=",")

    # 10) Console summary
    print("\n===== Evaluation (Validation set) =====")
    print(cls_report)
    print("Confusion matrix:\n", cm)
    print("\nSaved:")
    print(f"- Best model      : {os.path.join(args.out_dir, 'best_model')}")
    print(f"- Eval metrics    : {os.path.join(args.out_dir, 'eval_metrics.json')}")
    print(f"- Train metrics   : {os.path.join(args.out_dir, 'train_result.json')}")
    print(f"- Report          : {os.path.join(args.out_dir, 'classification_report.txt')}")
    print(f"- Confusion matrix: {os.path.join(args.out_dir, 'confusion_matrix.csv')}")


if __name__ == "__main__":
    main()
