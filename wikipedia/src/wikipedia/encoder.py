from __future__ import annotations

from typing import Any

import numpy as np

from .types import EncoderConfig


class Encoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.model_name_or_path,
                revision=self.config.revision,
                device=self.config.device,
                local_files_only=True,
            )
        return self._model

    def encode_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("search text must be a nonempty string")
        encoded = self._load().encode(
            [text],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )
        vector = np.asarray(encoded[0], dtype=np.float32)
        if vector.shape != (1024,):
            raise ValueError(f"Encoder returned shape {vector.shape}; expected (1024,)")
        return vector
