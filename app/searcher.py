from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import faiss
from sqlalchemy.orm import Session
from sqlalchemy import select
from .db import ImageRow, AliasRow
from .model_normalize import normalize_model

class Searcher:
    def __init__(self, session: Session, index: faiss.Index, id_map: List[str], prompt_templates: tuple[str, ...]):
        self.session = session
        self.index = index
        self.id_map = id_map
        self.prompt_templates = prompt_templates
        # build in-memory alias map for speed
        self.alias_to_model: Dict[str, str] = {}
        for row in session.execute(select(AliasRow)).scalars().all():
            self.alias_to_model[row.alias] = row.model_std

    def resolve_model(self, model: str) -> Optional[str]:
        nm = normalize_model(model)
        if not nm:
            return None
        # direct alias map
        if nm in self.alias_to_model:
            return self.alias_to_model[nm]
        # fallback: sometimes user passes already-normalized std
        # if exists in images table, accept
        r = self.session.execute(select(ImageRow).where(ImageRow.model_std == nm).limit(1)).scalars().first()
        if r:
            return nm
        return None

    def fetch_images_by_model(self, model_std: str) -> List[Dict[str, Any]]:
        rows = self.session.execute(select(ImageRow).where(ImageRow.model_std == model_std)).scalars().all()
        out = []
        for r in rows:
            extra = json.loads(r.extra_json or "{}")
            out.append({
                "image_id": r.image_id,
                "filepath": r.filepath,
                "model_std": r.model_std,
                "extra": extra,
            })
        return out

    def fetch_image_rows_by_ids(self, image_ids: List[str]) -> List[Dict[str, Any]]:
        if not image_ids:
            return []
        rows = self.session.execute(select(ImageRow).where(ImageRow.image_id.in_(image_ids))).scalars().all()
        by_id = {r.image_id: r for r in rows}
        out = []
        for iid in image_ids:
            r = by_id.get(iid)
            if not r:
                continue
            extra = json.loads(r.extra_json or "{}")
            out.append({
                "image_id": r.image_id,
                "filepath": r.filepath,
                "model_std": r.model_std,
                "extra": extra,
            })
        return out

    def build_prompts(self, q_text: str) -> List[str]:
        q_text = q_text.strip()
        if not q_text:
            q_text = "aircraft or vehicle"
        return [t.format(q=q_text) for t in self.prompt_templates]

    def faiss_search(self, query_vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        # query_vec: (d,) normalized float32
        q = query_vec.astype(np.float32)[None, :]
        scores, idxs = self.index.search(q, topk)
        return scores[0], idxs[0]

    def search_vector(self, encoder, model: Optional[str], desc: Optional[str], top_k: int, candidate_k: int = 100):
        parts = []
        if model:
            parts.append(model.strip())
        if desc:
            parts.append(desc.strip())
        q_text = "，".join([p for p in parts if p]) or ""
        prompts = self.build_prompts(q_text)
        text_vecs = encoder.encode_texts(prompts, batch_size=min(64, len(prompts)))
        query_vec = text_vecs.mean(axis=0)
        # L2 normalize again (mean may break unit norm)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        scores, idxs = self.faiss_search(query_vec, topk=min(candidate_k, max(top_k, 1)))
        image_ids = []
        score_map = {}
        for s, ix in zip(scores.tolist(), idxs.tolist()):
            if ix < 0 or ix >= len(self.id_map):
                continue
            iid = self.id_map[ix]
            image_ids.append(iid)
            score_map[iid] = float(s)

        rows = self.fetch_image_rows_by_ids(image_ids)
        # preserve FAISS order
        hits = []
        for iid in image_ids:
            r = next((x for x in rows if x["image_id"] == iid), None)
            if not r:
                continue
            hits.append({
                **r,
                "score": score_map.get(iid, 0.0)
            })
        return q_text, hits[:top_k]
