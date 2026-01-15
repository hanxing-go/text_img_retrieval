from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TIR_", case_sensitive=False)

    # OpenCLIP model selection
    MODEL_NAME: str = "ViT-L-14"
    PRETRAINED: str = "openai"

    # Index directory containing index.faiss, id_map.json, meta.db
    INDEX_DIR: str = "data/index"

    # Device: "cuda" / "cpu" / "auto"
    DEVICE: str = "auto"

    # Prompt templates for desc-based retrieval
    PROMPT_TEMPLATES: tuple[str, ...] = (
        "a studio photo of {q}, isolated object, no background",
        "side view of {q}, isolated object, no background",
        "front view of {q}, isolated object, no background",
        "silhouette of {q}, isolated object, no background",
        "{q}, isolated object, no background",
    )

settings = Settings()
