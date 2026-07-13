from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class WikipediaRecord:
    source_id: str
    url: str
    title: str
    text: str
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    source_id: str
    score: float
    url: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    model_name_or_path: str
    revision: str | None = None
    device: str | None = None
    normalize_embeddings: bool = True

    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path | None = None, device: str | None = None
    ) -> EncoderConfig:
        from .bundle import BundlePaths, load_manifest

        paths = BundlePaths.resolve(bundle_dir)
        manifest = load_manifest(paths, require_complete=True)
        revision = manifest.get("model", {}).get("resolved_revision")
        return cls(str(paths.model_dir), revision=revision, device=device)


@dataclass(frozen=True, slots=True)
class IngestConfig:
    batch_size: int = 256
    checkpoint_path: Path | None = None
    max_shards: int | None = None
    max_records: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_shards is not None and self.max_shards < 1:
            raise ValueError("max_shards must be at least 1")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("max_records must be at least 1")
