from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import VectorDB
from .bundle import BundlePaths, load_manifest
from .types import SearchResult


class Encoder:
    """Eagerly loaded local BGE-M3 encoder."""

    def __init__(
        self,
        bundle_dir: str | Path | None = None,
        *,
        device: str | None = None,
        require_complete: bool = True,
    ) -> None:
        paths = BundlePaths.resolve(bundle_dir)
        manifest = load_manifest(paths, require_complete=require_complete)
        revision = manifest.get("model", {}).get("resolved_revision")

        from sentence_transformers import SentenceTransformer

        self.model: Any = SentenceTransformer(
            str(paths.model_dir),
            revision=revision,
            device=device,
            local_files_only=True,
        )

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("search text must be a nonempty string")
        encoded = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        vector = np.asarray(encoded[0], dtype=np.float32)
        if vector.shape != (1024,):
            raise ValueError(f"Encoder returned shape {vector.shape}; expected (1024,)")
        return vector


def search_text(
    database: VectorDB,
    encoder: Encoder,
    text: str,
    *,
    limit: int = 10,
) -> list[SearchResult]:
    return database.search(encoder.encode(text), limit=limit)
