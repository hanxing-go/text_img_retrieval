from __future__ import annotations
import os
from typing import Iterable, List
import numpy as np
import torch
import open_clip
from PIL import Image

class CLIPEncoder:
    def __init__(self, model_name: str, pretrained: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            f = self.model.encode_text(tokens)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.inference_mode()
    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    imgs.append(self.preprocess(im))
            x = torch.stack(imgs, dim=0).to(self.device)
            f = self.model.encode_image(x)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats.append(f.detach().cpu().numpy())
        return np.concatenate(feats, axis=0)
