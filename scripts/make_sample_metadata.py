"""Utility to generate a starter metadata.jsonl for a folder of images.

Usage:
  python scripts/make_sample_metadata.py --images_dir data/images --out data/metadata.jsonl
Then edit model_std and aliases per row.
"""
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    paths = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p.as_posix())
    paths.sort()

    with open(args.out, "w", encoding="utf-8") as f:
        for i, fp in enumerate(paths, 1):
            image_id = Path(fp).stem
            row = {
                "image_id": image_id,
                "filepath": fp,
                "model_std": "UNKNOWN",
                "aliases": []
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(paths)} rows to {args.out}. Now fill in model_std / aliases.")

if __name__ == "__main__":
    main()
