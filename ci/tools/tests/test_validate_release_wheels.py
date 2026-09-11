# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ci.tools.validate_release_wheels import WheelTarget, expected_binary_targets


def resolved_package() -> str:
    return json.dumps(
        {
            "package_root": "alternate_bindings_12_8",
            "toolkit_version": "12.8.0",
            "release_version": "12.8.0",
            "release_registry_origin": "tag",
        },
        separators=(",", ":"),
    )


def write_release_matrix(
    root: Path,
    *,
    python_versions: tuple[str, ...] = ("3.12",),
    platforms: tuple[str, ...] = ("linux-64",),
) -> None:
    build_workflow = root / ".github" / "workflows" / "build-wheel.yml"
    build_workflow.parent.mkdir(parents=True, exist_ok=True)
    build_workflow.write_text(
        yaml.safe_dump({"jobs": {"build": {"strategy": {"matrix": {"python-version": list(python_versions)}}}}}),
        encoding="utf-8",
    )
    ci_workflow = root / ".github" / "workflows" / "ci.yml"
    ci_workflow.write_text(
        yaml.safe_dump(
            {
                "jobs": {
                    f"build-{platform}": {
                        "uses": "./.github/workflows/build-wheel.yml",
                        "with": {"host-platform": platform},
                    }
                    for platform in platforms
                }
            }
        ),
        encoding="utf-8",
    )


def run_validator(
    wheel_dir: Path,
    release_source_root: Path,
    *extra_args: str,
    git_tag: str = "v12.8.0",
    component: str = "cuda-bindings",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [
            sys.executable,
            "-m",
            "ci.tools.validate_release_wheels",
            git_tag,
            component,
            str(wheel_dir),
            "--release-source-root",
            str(release_source_root),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def release_source(tmp_path):
    root = tmp_path / "release-source"
    write_release_matrix(root)
    return root


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_tag_authoritative_package_validates_after_control_registry_moves_on(tmp_path, release_source):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").touch()

    without_resolved_package = run_validator(tmp_path, release_source)
    with_resolved_package = run_validator(
        tmp_path,
        release_source,
        "--bindings-package",
        resolved_package(),
    )

    assert without_resolved_package.returncode == 1
    assert with_resolved_package.returncode == 0, with_resolved_package.stderr


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_resolved_package_must_match_the_release_tag(tmp_path, release_source):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").touch()
    package = json.loads(resolved_package())
    package["release_version"] = "12.9.0"

    result = run_validator(tmp_path, release_source, "--bindings-package", json.dumps(package))

    assert result.returncode == 1
    assert "does not match release tag" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6")
def test_unexpected_distribution_is_rejected(tmp_path, release_source):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").touch()
    unexpected_wheel = "cuda_core-12.8.0-py3-none-any.whl"
    (tmp_path / unexpected_wheel).touch()

    result = run_validator(tmp_path, release_source, "--bindings-package", resolved_package())

    assert result.returncode == 1
    assert f"{unexpected_wheel}: unexpected distribution 'cuda_core'" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_complete_binary_wheel_matrix_is_required(tmp_path):
    release_source = tmp_path / "release-source"
    write_release_matrix(
        release_source,
        python_versions=("3.11", "3.14t", "3.15", "3.15t"),
        platforms=("linux-64", "win-64"),
    )
    wheel_names = (
        "cuda_bindings-12.8.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "cuda_bindings-12.8.0-cp311-cp311-win_amd64.whl",
        "cuda_bindings-12.8.0-cp314-cp314t-manylinux_2_28_x86_64.whl",
        "cuda_bindings-12.8.0-cp314-cp314t-win_amd64.whl",
    )
    for wheel_name in wheel_names:
        (tmp_path / wheel_name).touch()

    complete = run_validator(
        tmp_path,
        release_source,
        "--bindings-package",
        resolved_package(),
    )
    (tmp_path / wheel_names[-1]).unlink()
    partial = run_validator(
        tmp_path,
        release_source,
        "--bindings-package",
        resolved_package(),
    )

    assert complete.returncode == 0, complete.stderr
    assert partial.returncode == 1
    assert "Missing expected cuda_bindings wheel targets: cp314-cp314t-win_amd64" in partial.stderr


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_windows_arm64_targets_follow_the_tagged_toolkit_line(tmp_path):
    release_source = tmp_path / "release-source"
    write_release_matrix(
        release_source,
        python_versions=("3.10", "3.11"),
        platforms=("linux-64", "win-arm64"),
    )

    cuda_12 = expected_binary_targets(release_source, {"toolkit_version": "12.9.1"})
    cuda_13 = expected_binary_targets(release_source, {"toolkit_version": "13.4.1"})

    assert cuda_12 == {
        WheelTarget("cp310", "cp310", "manylinux-x86_64"),
        WheelTarget("cp311", "cp311", "manylinux-x86_64"),
    }
    assert cuda_13 == cuda_12 | {WheelTarget("cp311", "cp311", "win_arm64")}


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_cuda_core_uses_the_complete_binary_wheel_matrix(tmp_path):
    release_source = tmp_path / "release-source"
    write_release_matrix(release_source, platforms=("linux-64", "win-64"))
    (tmp_path / "cuda_core-1.2.0-cp312-cp312-manylinux_2_28_x86_64.whl").touch()
    (tmp_path / "cuda_core-1.2.0-cp312-cp312-win_amd64.whl").touch()

    result = run_validator(
        tmp_path,
        release_source,
        "--bindings-package",
        resolved_package(),
        git_tag="cuda-core-v1.2.0",
        component="cuda-core",
    )

    assert result.returncode == 0, result.stderr
