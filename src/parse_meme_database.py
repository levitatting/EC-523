import argparse
import csv
import io
import re
from pathlib import Path
from typing import List

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon
try:
    _ = nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


CAPTION_PATTERNS = (
    r"\bcaption\b",
    r"\btext\b",
    r"upper[_ ]?text",
    r"lower[_ ]?text",
    r"\btitle\b",
    r"\bname\b",
    r"alternate[_ ]?text",
    r"display[_ ]?name",
    r"base[_ ]?meme[_ ]?name",
    r"meme[_ ]?text",
    r"description",
)

URL_RE = re.compile(r"https?://\S+")
MULTISPACE_RE = re.compile(r"\s+")
NONPRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")  # keep ascii printable + \t\r\n


def detect_delimiter(sample_text: str) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        # This file is usually TSV; default to tab if detection fails
        return "\t"


def normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip())


def pick_text_columns(headers: List[str]) -> List[str]:
    cols = []
    for h in headers:
        nh = normalize_header(h)
        for pat in CAPTION_PATTERNS:
            if re.search(pat, nh, re.IGNORECASE):
                cols.append(h)
                break
    # de-dup but keep order
    seen, out = set(), []
    for c in cols:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    # fallback: if nothing matched and only one header, use it
    return out or (headers[:1] if headers else [])


def clean_text(s: str) -> str:
    s = s or ""
    s = NONPRINTABLE_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def mostly_nonalpha_ratio(s: str) -> float:
    if not s:
        return 1.0
    alphas = sum(ch.isalpha() for ch in s)
    return 1.0 - (alphas / max(1, len(s)))


def map_compound_to_label(compound: float) -> int:
    if compound <= -0.05:
        return 0  # negative
    if compound >= 0.05:
        return 1  # positive
    return 2      # neutral


def main():
    ap = argparse.ArgumentParser(description="Parse MemeGenerator TSV/CSV into clean text+sentiment labels.")
    ap.add_argument("--in", dest="in_path", default="data/meme_database/memegenerator.csv",
                    help="Input TSV/CSV path")
    ap.add_argument("--out", dest="out_path", default="data/memes_text.csv",
                    help="Output CSV path")
    ap.add_argument("--min_len", type=int, default=3, help="Minimum text length to keep")
    ap.add_argument("--max_len", type=int, default=300, help="Maximum text length to keep")
    ap.add_argument("--max_nonalpha_ratio", type=float, default=0.7,
                    help="Drop lines where non-alpha ratio exceeds this")
    args = ap.parse_args()

    in_csv = Path(args.in_path)
    out_csv = Path(args.out_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Cannot find {in_csv}")

    # Read a small sample to detect delimiter (and strip NULs)
    with open(in_csv, "rb") as sf:
        sample = sf.read(20000).replace(b"\x00", b"")
    try:
        sample_text = sample.decode("utf-8", errors="ignore")
    except Exception:
        sample_text = sample.decode("latin-1", errors="ignore")
    delimiter = detect_delimiter(sample_text)

    total, kept = 0, 0
    sia = SentimentIntensityAnalyzer()

    with open(in_csv, "rb") as fbin:
        text_stream = io.TextIOWrapper(fbin, encoding="utf-8", errors="ignore", newline="")
        # Replace NULs on the fly
        def _clean_iter(stream):
            for chunk in stream:
                yield chunk.replace("\x00", "")
        reader = csv.DictReader(_clean_iter(text_stream), delimiter=delimiter)
        headers = reader.fieldnames or []
        text_cols = pick_text_columns(headers)
        print(f"[INFO] Detected delimiter: {repr(delimiter)}")
        print(f"[INFO] Using text columns: {text_cols}")

        with open(out_csv, "w", encoding="utf-8", newline="") as wf:
            writer = csv.writer(wf)
            writer.writerow(["text", "label"])

            for row in reader:
                total += 1
                pieces = [clean_text(row.get(col, "")) for col in text_cols]
                text = clean_text(" ".join([p for p in pieces if p]))
                if not text:
                    continue
                if len(text) < args.min_len or len(text) > args.max_len:
                    continue
                if mostly_nonalpha_ratio(text) > args.max_nonalpha_ratio:
                    continue

                score = sia.polarity_scores(text)["compound"]
                label = map_compound_to_label(score)
                writer.writerow([text, label])
                kept += 1

                if kept % 10000 == 0:
                    print(f"[INFO] kept {kept} (processed {total})")

    print(f"[DONE] kept {kept} / processed {total} -> {out_csv}")


if __name__ == "__main__":
    main()