from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from tqdm import tqdm
import faiss

from .clip_encoder import CLIPEncoder
from .db import open_db, upsert_alias, upsert_image
from .model_normalize import normalize_model, normalize_alias_list

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="metadata jsonl")
    ap.add_argument("--out_dir", required=True, help="output index dir")
    ap.add_argument("--model_name", default=os.getenv("TIR_MODEL_NAME", "ViT-L-14"))
    ap.add_argument("--pretrained", default=os.getenv("TIR_PRETRAINED", "openai"))
    ap.add_argument("--device", default=os.getenv("TIR_DEVICE", "auto"))
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(out_dir / "meta.db")
    session, _ = open_db(db_path)

    print(f"[build_index] Loading metadata: {args.meta}")
    meta_rows = read_jsonl(args.meta)
    if not meta_rows:
        raise SystemExit("No metadata rows found.")

    # Normalize and store into DB
    image_paths = []
    image_ids = []

    print("[build_index] Writing metadata into SQLite...")
    for r in tqdm(meta_rows):
        image_id = str(r.get("image_id") or "").strip()
        filepath = str(r.get("filepath") or "").strip()
        model_std = str(r.get("model_std") or r.get("model") or "").strip()
        aliases = r.get("aliases") or []
        extra = dict(r)
        # remove core keys from extra for cleanliness
        for k in ["image_id", "filepath", "model_std", "model", "aliases"]:
            extra.pop(k, None)

        if not image_id or not filepath or not model_std:
            raise ValueError(f"Missing required fields in row: {r}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        model_std_n = normalize_model(model_std)
        aliases_n = normalize_alias_list([model_std] + list(aliases))

        upsert_image(session, image_id=image_id, filepath=filepath, model_std=model_std_n, extra=extra)
        for a in aliases_n:
            upsert_alias(session, alias=a, model_std=model_std_n)

        image_ids.append(image_id)
        image_paths.append(filepath)

    session.commit()

    print(f"[build_index] Encoding {len(image_paths)} images with OpenCLIP ({args.model_name} / {args.pretrained}) ...")
    encoder = CLIPEncoder(args.model_name, args.pretrained, device=args.device)
    feats = encoder.encode_images(image_paths, batch_size=args.batch_size).astype(np.float32)

    # IndexFlatIP expects inner product; with normalized vectors this equals cosine similarity
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(feats)

    faiss_path = str(out_dir / "index.faiss")
    idmap_path = str(out_dir / "id_map.json")
    print(f"[build_index] Saving FAISS index to {faiss_path}")
    faiss.write_index(index, faiss_path)

    print(f"[build_index] Saving id_map to {idmap_path}")
    with open(idmap_path, "w", encoding="utf-8") as f:
        json.dump(image_ids, f, ensure_ascii=False, indent=2)

    print("[build_index] Done.")

if __name__ == "__main__":
    main()
