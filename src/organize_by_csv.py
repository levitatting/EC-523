#!/usr/bin/env python3
import argparse, csv, shutil
from pathlib import Path

def normalize_label(lbl: str) -> str:
    s = (lbl or "").strip().lower().replace("-", "_").replace("very ", "very_")
    if s in {"neg","negative","very_negative","vnegative","-1","bad"}:
        return "negative"
    if s in {"pos","positive","very_positive","vpositive","1","good"}:
        return "positive"
    if s in {"neu","neutral","0","mixed","unknown","unsure","not_sure","none","no_sentiment"}:
        return "neutral"
    return "neutral"

def main():
    ap = argparse.ArgumentParser(description="Organize images into data/memes/{negative,neutral,positive} by CSV labels.")
    ap.add_argument("--images", required=True, help="Folder containing images")
    ap.add_argument("--csv", required=True, help="CSV with filename and label columns")
    ap.add_argument("--filename_col", default="filename", help="CSV column for image filename")
    ap.add_argument("--label_col", default="label", help="CSV column for label")
    ap.add_argument("--out_root", default="data/memes", help="Output root")
    args = ap.parse_args()

    images = Path(args.images)
    labels_csv = Path(args.csv)
    out_root = Path(args.out_root)
    (out_root/"negative").mkdir(parents=True, exist_ok=True)
    (out_root/"neutral").mkdir(parents=True, exist_ok=True)
    (out_root/"positive").mkdir(parents=True, exist_ok=True)

    moved = 0
    with labels_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get(args.filename_col, "") or "").strip()
            raw_lbl = (row.get(args.label_col, "") or "")
            lbl = normalize_label(raw_lbl)
            if not fname:
                continue

            src = images / fname
            if not src.exists():
                cands = list(images.glob(fname)) or list(images.glob(fname + ".*"))
                if not cands:
                    continue
                src = cands[0]

            dst = out_root / lbl / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                moved += 1
                if moved % 1000 == 0:
                    print(f"[INFO] organized {moved} images...")

    print(f"[DONE] organized images into {out_root} (moved {moved})")

if __name__ == "__main__":
    main()
