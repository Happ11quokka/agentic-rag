"""Keep the unit suite off the real Docker daemon.

`docker_runtime.ensure_docker` shells out to `docker info` and, on macOS, runs
`open -a Docker` when that fails — so a test that misses a patch can launch
Docker Desktop, and a suite run with the daemon down turns into minutes of
timeouts and failures that have nothing to do with the code under test. That
happened during an incident, which is the worst time for the suite to lie.

Any test that genuinely needs to exercise this layer patches
`docker_runtime.subprocess.run` itself; monkeypatch applies that over this
fixture for the duration of the test.
"""

from __future__ import annotations

import pytest

from wikipedia import docker_runtime


@pytest.fixture(autouse=True)
def _no_real_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "a unit test tried to run a real subprocess via docker_runtime "
            f"({args!r}); patch the seam it needs instead — the suite must not "
            "depend on a running Docker daemon"
        )

    monkeypatch.setattr(docker_runtime.subprocess, "run", refuse)
