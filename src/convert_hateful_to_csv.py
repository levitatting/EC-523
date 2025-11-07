#!/usr/bin/env python3
import json, csv
from pathlib import Path

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    base = Path("data/memes_hateful/data")
    out_csv = Path("data/memes_hateful/hateful_labels.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fn in ["train.jsonl", "dev.jsonl"]:
        f = base / fn
        if not f.exists():
            continue
        for obj in read_jsonl(f):
            # fields: {"img": "img/00000001.png", "label": 0/1, "text": "...", ...}
            img_rel = obj.get("img", "")
            label = obj.get("label", 0)
            name = Path(img_rel).name  # keep only filename
            # map: 1 -> negative, 0 -> neutral
            overall = "negative" if int(label) == 1 else "neutral"
            rows.append((name, overall))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "overall_sentiment"])
        w.writerows(rows)

    print(f"[DONE] wrote {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
