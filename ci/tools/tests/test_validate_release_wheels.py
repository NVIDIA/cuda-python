# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

VALIDATE_RELEASE_WHEELS = Path(__file__).parent.parent / "validate-release-wheels"


def resolved_line() -> str:
    return json.dumps(
        {
            "line_id": "released-13-4",
            "source_dir": "cuda_bindings_13_4",
            "release_source_dir": "cuda_bindings_13_4",
            "release_registry_origin": "tag",
            "ctk_target": "13.4",
            "toolkit_version": "13.4.0",
            "toolkit_channel": "stable",
            "tag_series": "v13.4.",
            "allow_alpha_beta_tags": True,
        },
        separators=(",", ":"),
    )


def run_validator(wheel_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - invokes the repository script under test
        [
            sys.executable,
            str(VALIDATE_RELEASE_WHEELS),
            "v13.4.0",
            "cuda-bindings",
            str(wheel_dir),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_tag_authoritative_line_validates_after_control_registry_moves_on(tmp_path):
    (tmp_path / "cuda_bindings-13.4.0-cp312-cp312-manylinux.whl").touch()

    without_resolved_line = run_validator(tmp_path)
    with_resolved_line = run_validator(
        tmp_path,
        "--bindings-line",
        resolved_line(),
    )

    assert without_resolved_line.returncode == 1
    assert with_resolved_line.returncode == 0, with_resolved_line.stderr


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_resolved_line_must_match_the_release_tag(tmp_path):
    (tmp_path / "cuda_bindings-13.3.0-cp312-cp312-manylinux.whl").touch()
    line = json.loads(resolved_line())
    line.update({"ctk_target": "13.3", "tag_series": "v13.3."})

    result = run_validator(
        tmp_path,
        "--bindings-line",
        json.dumps(line),
    )

    assert result.returncode == 1
    assert "Unsupported git tag format" in result.stderr
