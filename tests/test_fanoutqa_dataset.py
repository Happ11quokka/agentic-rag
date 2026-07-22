from pathlib import Path

import pytest

from fanoutqa.dataset import (
    DEV_BYTES,
    DEV_GIT_BLOB_SHA1,
    DEV_RECORDS,
    DEV_REVISION,
    DatasetError,
    Question,
    select_questions,
    validate_dev_bytes,
)


def test_pinned_dev_metadata() -> None:
    assert DEV_REVISION == "4e254877c0d28378e800171ab92781dbba893718"
    assert DEV_BYTES == 1_177_174
    assert DEV_GIT_BLOB_SHA1 == "76ad1feb689b754bfe4e5e24d3ea371b647efa67"
    assert DEV_RECORDS == 310


def test_malformed_download_is_rejected_before_json_use() -> None:
    with pytest.raises(DatasetError, match="size mismatch"):
        validate_dev_bytes(b"[]")


def test_selection_is_id_sorted_then_seeded() -> None:
    questions = [
        Question(str(index), f"q{index}", ("x",)) for index in range(12, -1, -1)
    ]

    first = select_questions(questions, limit=5, seed=42)
    second = select_questions(list(reversed(questions)), limit=5, seed=42)

    assert [item.id for item in first] == [item.id for item in second]
    assert len({item.id for item in first}) == 5


def test_agent_value_excludes_gold_fields() -> None:
    value = Question("id", "question", ("category",)).agent_value()

    assert value == {"id": "id", "question": "question", "categories": ["category"]}
    assert not ({"answer", "decomposition", "evidence"} & value.keys())
