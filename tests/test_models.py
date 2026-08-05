from pathlib import Path

from experiment import common, models


def test_download_models_fetches_and_validates_entire_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    downloads: list[dict[str, object]] = []
    validated: list[str] = []
    monkeypatch.setattr(common, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        models,
        "hf_hub_download",
        lambda **kwargs: downloads.append(kwargs),
    )
    monkeypatch.setattr(
        models,
        "validate_model",
        lambda spec: validated.append(spec.key) or spec.path,
    )

    models.download_models()

    assert [item["repo_id"] for item in downloads] == [
        spec.repo for spec in common.MODEL_SPECS.values()
    ]
    assert [item["filename"] for item in downloads] == [
        spec.filename for spec in common.MODEL_SPECS.values()
    ]
    assert [item["revision"] for item in downloads] == [
        spec.revision for spec in common.MODEL_SPECS.values()
    ]
    assert validated == list(common.MODEL_SPECS)
    assert all(Path(item["local_dir"]).is_dir() for item in downloads)
