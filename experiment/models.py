from __future__ import annotations

from huggingface_hub import hf_hub_download

from .common import MODEL_SPECS, ExperimentError, run_main, validate_model


def download_models() -> None:
    for spec in MODEL_SPECS.values():
        print(f"{spec.key}: downloading {spec.repo}/{spec.filename}", flush=True)
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=spec.path.parent,
        )
        path = validate_model(spec)
        print(f"{spec.key}: ready at {path}", flush=True)


def main() -> None:
    def action() -> None:
        try:
            download_models()
        except ExperimentError:
            raise
        except Exception as exc:
            raise ExperimentError(f"model download failed: {exc}") from exc

    run_main(action)
