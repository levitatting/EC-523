# A Multi-modal Framework for Sentimental Analysis of Meme via Transfer Learning

## Overview
This project provides a pipeline for sentiment analysis on memes, supporting text-only, image-only, and multimodal (text + image) classification into 3 classes: negative (0), positive (1), neutral (2).

## Requirements
Python 3.10 + CUDA 12.1 + PyTorch 2.5.1
Libraries: torch, transformers, pytesseract (for OCR), nltk (VADER), pandas, numpy, scikit-learn, Pillow, csv, argparse
Optional: GPU for training (CUDA-enabled)
Data: Download from Kaggle (e.g., utkarshx27/meme-database for metadata, Hateful Memes for images/labels)
Tesseract OCR installed for text extraction

## Usage Steps
Download Data: Use Kaggle API for datasets (see earlier instructions).
Prepare Text Data: python parse_meme_database.py → Outputs data/memes_text.csv
Prepare Hateful Memes Labels: python convert_hateful_to_csv.py → Outputs data/memes_hateful/hateful_labels.csv
Organize Images: python organize_by_csv.py --images <img_dir> --csv <labels.csv> → Populates data/memes/{negative,neutral,positive}
Build Multimodal CSV (with OCR): python prepare_csv.py --root data/memes --out data/memes.csv
Train Text Model: python train_text_classifier.py --csv data/memes_text.csv → Outputs outputs/text_bert/best_model
Train Image Model: python train_image_classifier.py --csv data/memes.csv → Outputs outputs/image_cls/best_model.pt
Extract Image Features: python export_image_features.py --csv data/memes.csv --ckpt outputs/image_cls/best_model.pt --out_npy features/image_feats.npy --out_csv features/image_index.csv
Train Multimodal Model: python train_multimodal.py --csv data/memes.csv --image_feats features/image_feats.npy --text_ckpt outputs/text_bert/best_model → Outputs outputs/multimodal/best_fusion.pt
Smoke Test: python smoke_test.py → Verifies environment (OCR, BERT, VGG19)

## Scripts Summary
parse_meme_database.py: Parse and label text from MemeGenerator CSV.
convert_hateful_to_csv.py: Convert Hateful Memes JSONL to CSV labels.
organize_by_csv.py: Organize images into sentiment folders based on CSV.
prepare_csv.py: Scan images, extract text via OCR, build CSV with paths/labels/text.
train_text_classifier.py: Fine-tune BERT for text sentiment.
train_image_classifier.py: Train CNN (ResNet/VGG) for image sentiment.
export_image_features.py: Extract image embeddings using trained CNN.
train_multimodal.py: Fuse text (BERT) + image features for multimodal classifier.
smoke_test.py: Environment validation with sample image/processing.

# Cite
```
@inproceedings{Pranesh2020MemeSemAMF,
  title={MemeSem:A Multi-modal Framework for Sentimental Analysis of Meme via Transfer Learning},
  author={R. R. Pranesh and Ambesh Shekhar},
  year={2020}
}
```
