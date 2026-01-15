from __future__ import annotations
import json
from typing import Any, Optional
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()

class ImageRow(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True)
    filepath = Column(String, nullable=False)
    model_std = Column(String, nullable=False)
    extra_json = Column(Text, nullable=False, default="{}")

class AliasRow(Base):
    __tablename__ = "aliases"
    alias = Column(String, primary_key=True)
    model_std = Column(String, nullable=False)

def open_db(db_path: str) -> tuple[Session, Any]:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal(), engine

def upsert_image(session: Session, image_id: str, filepath: str, model_std: str, extra: dict[str, Any]) -> None:
    row = session.get(ImageRow, image_id)
    payload = json.dumps(extra or {}, ensure_ascii=False)
    if row is None:
        session.add(ImageRow(image_id=image_id, filepath=filepath, model_std=model_std, extra_json=payload))
    else:
        row.filepath = filepath
        row.model_std = model_std
        row.extra_json = payload

def upsert_alias(session: Session, alias: str, model_std: str) -> None:
    row = session.get(AliasRow, alias)
    if row is None:
        session.add(AliasRow(alias=alias, model_std=model_std))
    else:
        row.model_std = model_std
