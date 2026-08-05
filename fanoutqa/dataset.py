from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEV_REVISION = "4e254877c0d28378e800171ab92781dbba893718"
DEV_URL = (
    "https://raw.githubusercontent.com/zhudotexe/fanoutqa/"
    f"{DEV_REVISION}/fanoutqa/data/fanout-final-dev.json"
)
DEV_BYTES = 1_177_174
DEV_GIT_BLOB_SHA1 = "76ad1feb689b754bfe4e5e24d3ea371b647efa67"
DEV_RECORDS = 310
DEFAULT_DEV_PATH = Path(__file__).with_name("data") / "fanout-final-dev.json"
REQUIRED_FIELDS = frozenset({"id", "question", "categories"})


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    categories: tuple[str, ...]

    def agent_value(self) -> dict[str, Any]:
        """Return the only dataset fields allowed into an agent trace or prompt."""
        return {
            "id": self.id,
            "question": self.question,
            "categories": list(self.categories),
        }


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def validate_dev_bytes(content: bytes) -> list[Question]:
    if len(content) != DEV_BYTES:
        raise DatasetError(
            f"FanOutQA dev size mismatch: got {len(content):,} bytes; "
            f"expected {DEV_BYTES:,}"
        )
    actual_sha = git_blob_sha1(content)
    if actual_sha != DEV_GIT_BLOB_SHA1:
        raise DatasetError(
            f"FanOutQA dev Git blob mismatch: got {actual_sha}; "
            f"expected {DEV_GIT_BLOB_SHA1}"
        )
    try:
        records = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"FanOutQA dev JSON is malformed: {exc}") from exc
    if not isinstance(records, list) or len(records) != DEV_RECORDS:
        count = len(records) if isinstance(records, list) else type(records).__name__
        raise DatasetError(
            f"FanOutQA dev record count mismatch: got {count}; expected {DEV_RECORDS}"
        )

    questions: list[Question] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"FanOutQA dev record {index} is not an object")
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise DatasetError(
                f"FanOutQA dev record {index} missing fields: {sorted(missing)}"
            )
        identifier = record["id"]
        question = record["question"]
        categories = record["categories"]
        if not isinstance(identifier, str) or not identifier:
            raise DatasetError(f"FanOutQA dev record {index} has invalid id")
        if identifier in seen:
            raise DatasetError(f"FanOutQA dev contains duplicate id {identifier!r}")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"FanOutQA dev record {identifier!r} has invalid question")
        if not isinstance(categories, list) or not all(
            isinstance(item, str) and item for item in categories
        ):
            raise DatasetError(f"FanOutQA dev record {identifier!r} has invalid categories")
        seen.add(identifier)
        questions.append(Question(identifier, question, tuple(categories)))
    return questions


def download_dev(
    path: Path = DEFAULT_DEV_PATH,
    *,
    timeout: float = 60,
    client: httpx.Client | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    temporary: str | None = None
    try:
        response = http.get(DEV_URL)
        response.raise_for_status()
        content = response.content
        validate_dev_bytes(content)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except (httpx.HTTPError, OSError) as exc:
        raise DatasetError(f"Could not download pinned FanOutQA dev set: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        if owned_client:
            http.close()
    return path


def load_dev(path: Path = DEFAULT_DEV_PATH, *, download: bool = True) -> list[Question]:
    path = Path(path)
    if not path.is_file():
        if not download:
            raise DatasetError(f"FanOutQA dev set is missing: {path}")
        download_dev(path)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"Could not read FanOutQA dev set {path}: {exc}") from exc
    return validate_dev_bytes(content)


def select_questions(
    questions: list[Question],
    *,
    limit: int = 5,
    seed: int = 42,
) -> list[Question]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    ordered = sorted(questions, key=lambda item: item.id)
    if limit > len(ordered):
        raise ValueError(f"limit {limit} exceeds dataset size {len(ordered)}")
    return random.Random(seed).sample(ordered, limit)


def find_question(questions: list[Question], question_id: str) -> Question:
    for question in questions:
        if question.id == question_id:
            return question
    raise DatasetError(f"Unknown FanOutQA question id: {question_id}")
