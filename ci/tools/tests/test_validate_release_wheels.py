# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def run_validator(wheel_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [
            sys.executable,
            "-m",
            "ci.tools.validate_release_wheels",
            "v12.8.0",
            "cuda-bindings",
            str(wheel_dir),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_tag_authoritative_package_validates_after_control_registry_moves_on(tmp_path):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux.whl").touch()

    without_resolved_package = run_validator(tmp_path)
    with_resolved_package = run_validator(
        tmp_path,
        "--bindings-package",
        resolved_package(),
    )

    assert without_resolved_package.returncode == 1
    assert with_resolved_package.returncode == 0, with_resolved_package.stderr


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_resolved_package_must_match_the_release_tag(tmp_path):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux.whl").touch()
    package = json.loads(resolved_package())
    package["release_version"] = "12.9.0"

    result = run_validator(tmp_path, "--bindings-package", json.dumps(package))

    assert result.returncode == 1
    assert "does not match release tag" in result.stderr


@pytest.mark.agent_authored(model="gpt-5.6")
def test_unexpected_distribution_is_rejected(tmp_path):
    (tmp_path / "cuda_bindings-12.8.0-cp312-cp312-manylinux.whl").touch()
    unexpected_wheel = "cuda_core-12.8.0-py3-none-any.whl"
    (tmp_path / unexpected_wheel).touch()

    result = run_validator(tmp_path, "--bindings-package", resolved_package())

    assert result.returncode == 1
    assert f"{unexpected_wheel}: unexpected distribution 'cuda_core'" in result.stderr
