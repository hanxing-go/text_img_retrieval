from __future__ import annotations
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
import faiss

from .config import settings
from .schemas import SearchRequest, SearchResponse, SearchHit
from .clip_encoder import CLIPEncoder
from .db import open_db
from .searcher import Searcher

app = FastAPI(title="Text→Image Retrieval (Model+Desc JSON)", version="1.0.0")

_session: Session | None = None
_encoder: CLIPEncoder | None = None
_searcher: Searcher | None = None

def _load():
    global _session, _encoder, _searcher
    idx_dir = Path(settings.INDEX_DIR)
    faiss_path = idx_dir / "index.faiss"
    idmap_path = idx_dir / "id_map.json"
    db_path = idx_dir / "meta.db"

    if not faiss_path.exists() or not idmap_path.exists() or not db_path.exists():
        raise RuntimeError(
            f"Index not found in {idx_dir}. Expected: index.faiss, id_map.json, meta.db. "
            "Run: python -m app.build_index --meta <...> --out_dir data/index"
        )

    _session, _ = open_db(str(db_path))
    index = faiss.read_index(str(faiss_path))
    with open(idmap_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    _encoder = CLIPEncoder(settings.MODEL_NAME, settings.PRETRAINED, device=settings.DEVICE)
    _searcher = Searcher(_session, index=index, id_map=id_map, prompt_templates=settings.PROMPT_TEMPLATES)

@app.on_event("startup")
def startup_event():
    _load()

@app.get("/health")
def health():
    return {
        "ok": True,
        "model": {"name": settings.MODEL_NAME, "pretrained": settings.PRETRAINED, "device": settings.DEVICE},
        "index_dir": settings.INDEX_DIR
    }

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if _session is None or _encoder is None or _searcher is None:
        raise HTTPException(status_code=500, detail="Service not initialized.")

    model = (req.model or "").strip()
    desc = (req.desc or "").strip()

    # 1) model-exact path
    if model:
        model_std = _searcher.resolve_model(model)
        if model_std:
            rows = _searcher.fetch_images_by_model(model_std)
            hits = []
            for r in rows[:req.top_k]:
                hits.append(SearchHit(
                    image_id=r["image_id"],
                    filepath=r["filepath"],
                    model_std=r["model_std"],
                    score=1.0,
                    extra=r.get("extra") or {}
                ))
            q_text = model
            return SearchResponse(mode="model_exact", query_text=q_text, hits=hits)

    # 2) vector search path
    q_text, raw_hits = _searcher.search_vector(_encoder, model=model or None, desc=desc or None, top_k=req.top_k)
    hits = []
    for r in raw_hits:
        hits.append(SearchHit(
            image_id=r["image_id"],
            filepath=r["filepath"],
            model_std=r["model_std"],
            score=float(r["score"]),
            extra=r.get("extra") or {}
        ))
    return SearchResponse(mode="vector", query_text=q_text, hits=hits)
