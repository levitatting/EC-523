#!/usr/bin/env python3
"""
prepare_csv.py
Build CSV from folder structure:
    root/
      negative/
      neutral/
      positive/
Outputs: image_path, text, label
Options:
  --skip_ocr  -> do not run Tesseract (fast)
  --lang eng  -> OCR language (ignored if --skip_ocr)
  --progress  -> print progress every N images
"""

import argparse, csv
from pathlib import Path
from PIL import Image
import sys

LABEL_MAP = {"negative": 0, "positive": 1, "neutral": 2}

def extract_text(img_path: Path, lang="eng", skip_ocr=False):
    if skip_ocr:
        return ""
    try:
        import pytesseract
    except Exception:
        # Tesseract not installed or pytesseract missing -> fallback no text
        return ""
    try:
        txt = pytesseract.image_to_string(Image.open(img_path), lang=lang)
        return (txt or "").strip().replace("\n", " ")
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root containing class folders")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    ap.add_argument("--skip_ocr", action="store_true", help="Skip OCR to speed up")
    ap.add_argument("--progress", type=int, default=500, help="Print every N images (0=off)")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # collect files
    files = []
    for name, lid in LABEL_MAP.items():
        d = root / name
        if not d.exists(): 
            continue
        for p in d.iterdir():
            if p.is_file():
                files.append((p, lid))
    total = len(files)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text", "label"])
        for i, (img, lid) in enumerate(files, 1):
            text = extract_text(img, lang=args.lang, skip_ocr=args.skip_ocr)
            w.writerow([str(img), text, lid])
            if args.progress and (i % args.progress == 0 or i == total):
                print(f"[INFO] {i}/{total} written...", flush=True)

    print(f"[DONE] Wrote {total} rows -> {out_csv}")

if __name__ == "__main__":
    main()
