from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any

class SearchRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model string (may be empty).")
    desc: Optional[str] = Field(default=None, description="Description string (may be empty).")
    top_k: int = Field(default=20, ge=1, le=200)
    rerank: bool = Field(default=False, description="Reserved for future cross-encoder rerank (disabled in this minimal build).")

class SearchHit(BaseModel):
    image_id: str
    filepath: str
    model_std: str
    score: float
    extra: dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    mode: str = Field(description="'model_exact' or 'vector'")
    query_text: str
    hits: list[SearchHit]
