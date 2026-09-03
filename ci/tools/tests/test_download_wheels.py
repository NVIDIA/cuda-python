# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

DOWNLOAD_WHEELS = Path(__file__).parent.parent / "download-wheels"

FAKE_GH = r"""#!/usr/bin/env python3
import fnmatch
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_GH_LOG"]).open("a", encoding="utf-8") as stream:
    print(json.dumps(args), file=stream)
artifacts = json.loads(os.environ["FAKE_ARTIFACTS"])

if args[:1] == ["api"]:
    print(*artifacts, sep="\n")
    raise SystemExit(0)

if args[:2] == ["run", "download"]:
    if "--name" in args:
        name = args[args.index("--name") + 1]
        selected = [artifact for artifact in artifacts if artifact == name]
    else:
        pattern = args[args.index("-p") + 1]
        selected = [artifact for artifact in artifacts if fnmatch.fnmatchcase(artifact, pattern)]
    if not selected:
        raise SystemExit(1)
    destination = Path(args[args.index("--dir") + 1])
    for artifact in selected:
        artifact_dir = destination / artifact
        artifact_dir.mkdir(parents=True)
        suffix = ".so" if artifact.endswith("-tests") else ".whl"
        (artifact_dir / f"{artifact}{suffix}").touch()
    raise SystemExit(0)

raise SystemExit(2)
"""


@pytest.fixture
def fake_gh(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8")
    gh.chmod(0o755)
    return fake_bin


def resolved_package(*, toolkit_version="13.3.0", origin="tag"):
    return json.dumps({"toolkit_version": toolkit_version, "release_registry_origin": origin})


def run_download(tmp_path, fake_gh, artifacts, component, *, tag="", package=None):
    log = tmp_path / "gh.log"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ARTIFACTS": json.dumps(artifacts),
            "FAKE_GH_LOG": str(log),
            "GH_TOKEN": "test-token",
            "PATH": f"{fake_gh}{os.pathsep}{env['PATH']}",
        }
    )
    args = [str(DOWNLOAD_WHEELS), "123", component, "NVIDIA/cuda-python", str(tmp_path / "dist")]
    if tag or package is not None:
        args.append(tag)
    if package is not None:
        args.append(package)
    result = subprocess.run(  # noqa: S603
        args,
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    commands = [json.loads(item) for item in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, commands


@pytest.mark.agent_authored(model="gpt-5.6-sol")
class TestDownloadWheels:
    @pytest.mark.parametrize(
        ("origin", "artifacts", "expected"),
        (
            ("tag", ["cuda-python-wheel", "cuda-python-wheel-cuda13.3.0"], "cuda-python-wheel-cuda13.3.0"),
            ("control", ["cuda-python-wheel"], "cuda-python-wheel"),
        ),
    )
    def test_python_selects_exact_or_legacy_artifact(self, tmp_path, fake_gh, origin, artifacts, expected):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            artifacts,
            "cuda-python",
            tag="v13.3.1",
            package=resolved_package(origin=origin),
        )

        assert result.returncode == 0, result.stderr
        assert any("--name" in command and expected in command for command in commands)

    def test_tag_registry_rejects_legacy_python_artifact(self, tmp_path, fake_gh):
        result, _ = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel"],
            "cuda-python",
            tag="v13.3.1",
            package=resolved_package(),
        )

        assert result.returncode == 1
        assert "legacy cuda-python-wheel fallback is not allowed" in result.stderr

    def test_bindings_pattern_contains_exact_toolkit_pin(self, tmp_path, fake_gh):
        exact = "cuda-bindings-python312-cuda13.3.0-linux-64-sha"
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-bindings-python312-cuda13.2.0-linux-64-sha", exact, f"{exact}-tests"],
            "cuda-bindings",
            tag="v13.3.1",
            package=resolved_package(),
        )

        assert result.returncode == 0, result.stderr
        downloads = [command[command.index("--name") + 1] for command in commands if "--name" in command]
        assert downloads == [exact]
        assert [path.name for path in (tmp_path / "dist").glob("*.whl")] == [f"{exact}.whl"]

    def test_release_routing_requires_resolved_package(self, tmp_path, fake_gh):
        result, commands = run_download(
            tmp_path,
            fake_gh,
            ["cuda-python-wheel-cuda13.3.0"],
            "cuda-python",
            tag="v13.3.1",
        )

        assert result.returncode == 1
        assert commands == []

    def test_bindings_test_artifact_is_not_releasable(self, tmp_path, fake_gh):
        artifact = "cuda-bindings-python312-cuda13.3.0-linux-64-sha-tests"
        result, _ = run_download(
            tmp_path,
            fake_gh,
            [artifact],
            "cuda-bindings",
            tag="v13.3.1",
            package=resolved_package(),
        )

        assert result.returncode == 1
        assert "no unexpired release artifact" in result.stderr


@pytest.mark.parametrize(
    ("component", "artifacts", "expected"),
    (
        (
            "all",
            ["cuda-core-wheel", "cuda-core-wheel-tests", "cuda-python-wheel", "build-metadata"],
            ["cuda-core-wheel", "cuda-python-wheel"],
        ),
        (
            "cuda-core",
            ["cuda-core-wheel", "cuda-core-wheel-tests", "cuda-python-wheel"],
            ["cuda-core-wheel"],
        ),
    ),
)
@pytest.mark.agent_authored(model="gpt-5.6")
def test_non_release_selection_excludes_test_artifacts(tmp_path, fake_gh, component, artifacts, expected):
    result, commands = run_download(tmp_path, fake_gh, artifacts, component)

    assert result.returncode == 0, result.stderr
    downloads = [command[command.index("--name") + 1] for command in commands if "--name" in command]
    assert downloads == expected
    assert all("-p" not in command for command in commands)
