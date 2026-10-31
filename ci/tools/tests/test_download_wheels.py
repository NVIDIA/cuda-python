# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep unrelated CUDA lines and test artifacts out of release staging."""

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
artifacts = json.loads(os.environ["FAKE_ARTIFACTS"])
if args[:1] == ["api"]:
    print(*artifacts, sep="\n")
    raise SystemExit(0)
if args[:2] == ["run", "download"]:
    selector = "--name" if "--name" in args else "-p"
    pattern = args[args.index(selector) + 1]
    selected = [artifact for artifact in artifacts if fnmatch.fnmatchcase(artifact, pattern)]
    for artifact in selected:
        destination = Path(args[args.index("--dir") + 1]) / artifact
        destination.mkdir(parents=True)
        suffix = ".so" if artifact.endswith("-tests") else ".whl"
        (destination / f"{artifact}{suffix}").touch()
    raise SystemExit(0 if selected else 1)
raise SystemExit(2)
"""


def run_download(tmp_path, artifacts, component, *, package=None, tag=""):
    binary = tmp_path / "gh"
    binary.write_text(FAKE_GH, encoding="utf-8")
    binary.chmod(0o755)
    args = [str(DOWNLOAD_WHEELS), "123", component, "NVIDIA/cuda-python", str(tmp_path / "dist"), tag]
    if package is not None:
        args.append(json.dumps(package))
    return subprocess.run(  # noqa: S603 - the checked-in utility receives separate arguments.
        args,
        cwd=tmp_path,
        env={
            **os.environ,
            "FAKE_ARTIFACTS": json.dumps(artifacts),
            "GH_TOKEN": "test-token",
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        },
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("component", ["cuda-bindings", "cuda-python"])
@pytest.mark.parametrize("major,toolkit", [(12, "12.9.1"), (13, "13.3.0")])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_stages_only_wheels_for_its_exact_toolkit(tmp_path, component, major, toolkit):
    release = f"{toolkit.rsplit('.', 1)[0]}.99"
    package = {"package_root": f"cuda_bindings_{major}", "toolkit_version": toolkit, "release_version": release}
    prefix = "cuda-bindings-python312" if component == "cuda-bindings" else "cuda-python-wheel"
    suffix = "-linux-64-sha" if component == "cuda-bindings" else ""
    expected = f"{prefix}-cuda{toolkit}{suffix}"
    result = run_download(
        tmp_path,
        [f"{prefix}-cuda11.8.0{suffix}", expected, f"{expected}-tests", "cuda-python-wheel"],
        component,
        package=package,
        tag=f"v{package['release_version']}",
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in (tmp_path / "dist").iterdir()} == {f"{expected}.whl"}


@pytest.mark.parametrize("component", ["cuda-bindings", "cuda-python"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_cannot_stage_another_toolkit_or_test_artifacts(tmp_path, component):
    package = {"package_root": "cuda_bindings_13", "toolkit_version": "13.3.0", "release_version": "13.3.1"}
    result = run_download(
        tmp_path,
        ["cuda-bindings-python312-cuda13.3.0-linux-64-sha-tests", "cuda-python-wheel-cuda13.2.0"],
        component,
        package=package,
        tag="v13.3.1",
    )
    assert result.returncode != 0
    assert not list((tmp_path / "dist").glob("*.whl"))


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_routing_requires_a_resolved_package(tmp_path):
    result = run_download(tmp_path, ["cuda-python-wheel-cuda13.3.0"], "cuda-python", tag="v13.3.1")
    assert result.returncode != 0
    assert not list((tmp_path / "dist").glob("*.whl"))


@pytest.mark.parametrize(
    "component,expected",
    [("all", {"cuda-core-wheel", "cuda-python-wheel-cuda13.3.0"}), ("cuda-core", {"cuda-core-wheel"})],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_nonrelease_components_stage_wheels_without_test_artifacts(tmp_path, component, expected):
    result = run_download(
        tmp_path,
        ["cuda-core-wheel", "cuda-core-wheel-tests", "cuda-python-wheel-cuda13.3.0", "build-metadata"],
        component,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in (tmp_path / "dist").iterdir()} == {f"{artifact}.whl" for artifact in expected}


@pytest.mark.parametrize(
    ("toolkit", "artifacts", "expected"),
    [
        ("13.4.2", ["cuda-python-wheel"], "cuda-python-wheel"),
        ("13.4.2", ["cuda-python-wheel", "cuda-python-wheel-cuda13.4.2"], "cuda-python-wheel-cuda13.4.2"),
        ("12.9.1", ["cuda-python-wheel"], None),
    ],
)
@pytest.mark.agent_authored(model="GPT-6")
def test_legacy_metapackage_fallback_is_limited_to_cuda_13(tmp_path, toolkit, artifacts, expected):
    package = {"toolkit_version": toolkit, "allow_legacy_python_artifact": True}
    result = run_download(tmp_path, artifacts, "cuda-python", package=package, tag=f"v{toolkit}")

    assert (result.returncode == 0) == (expected is not None), result.stderr
    assert {path.name for path in (tmp_path / "dist").glob("*.whl")} == (
        set() if expected is None else {f"{expected}.whl"}
    )
