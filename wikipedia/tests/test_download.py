import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from wikipedia import download
from wikipedia.bundle import BundlePaths


class FakeApi:
    def dataset_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha="dataset-sha")

    def model_info(self, repo_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha="model-sha")

    def list_repo_files(self, *args: object, **kwargs: object) -> list[str]:
        return ["README.md", "data/en/part-000.parquet", "data/en/part-001.parquet"]


def test_prepare_bundle_writes_ephemeral_manifest_and_preserves_qdrant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "created_at": "old",
                "qdrant": {"storage_dir": "/data/qdrant"},
            }
        ),
        encoding="utf-8",
    )

    manifest = download.prepare_bundle(
        paths,
        max_shards=1,
        progress_interval=0,
        api=FakeApi(),
    )

    assert manifest["status"] == "incomplete"
    assert manifest["created_at"] == "old"
    assert manifest["partial"] is True
    assert manifest["dataset"]["retention"] == "ephemeral"
    assert manifest["dataset"]["shards"] == ["dataset/data/en/part-000.parquet"]
    assert manifest["qdrant"] == {"storage_dir": "/data/qdrant"}
    assert "dataset: 1/2 shards selected" in capsys.readouterr().err


def test_download_shard_and_model_use_separate_destinations(tmp_path: Path) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.dataset_dir.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"shard")
        return str(destination)

    shard = download.download_dataset_shard(
        paths,
        "data/en/part-000.parquet",
        revision="dataset-sha",
        progress_interval=0,
        download_timeout=30,
        token="token",
        downloader=fake_download,
    )

    def fake_snapshot(**kwargs: object) -> None:
        calls.append(kwargs)
        model = Path(kwargs["local_dir"]) / "config.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("{}", encoding="utf-8")

    download.download_model(
        paths,
        revision="model-sha",
        max_workers=4,
        progress_interval=0,
        download_timeout=30,
        token="token",
        snapshot=fake_snapshot,
    )

    assert shard.read_bytes() == b"shard"
    assert calls[0]["filename"] == "data/en/part-000.parquet"
    assert calls[0]["etag_timeout"] == 30
    assert calls[1]["max_workers"] == 4
    assert (paths.model_dir / "config.json").is_file()


def test_local_progress_includes_resumable_partial_bytes(tmp_path: Path) -> None:
    shard = tmp_path / "data/en/part-000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"a" * 1024)
    partial = tmp_path / ".cache/huggingface/download/chunk.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"b" * 512)

    assert download._local_progress(tmp_path, ["data/en/part-000.parquet"]) == (1, 1536)


def test_progress_heartbeat_reports_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with download.progress_heartbeat("ingest shard 1/2", 0.01, lambda: "3 records"):
        time.sleep(0.03)

    stderr = capsys.readouterr().err
    assert "ingest shard 1/2: still running" in stderr
    assert "3 records" in stderr


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"progress_interval": -1}, "progress_interval"),
        ({"download_timeout": 0}, "download_timeout"),
        (
            {"high_performance": True, "disable_xet": True},
            "high_performance and disable_xet",
        ),
    ],
)
def test_prepare_bundle_rejects_invalid_transfer_options(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        download.prepare_bundle(
            BundlePaths.from_dir(tmp_path),
            api=FakeApi(),
            **kwargs,
        )
