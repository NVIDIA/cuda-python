# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from write_wheel_constraints import WheelConstraintError, write_constraints


def make_wheel(directory: Path, project: str, version: str) -> Path:
    """Create the wheel metadata needed by the constraint writer."""
    directory.mkdir(parents=True, exist_ok=True)
    filename_project = project.replace("-", "_").replace(".", "_")
    wheel_path = directory / f"{filename_project}-{version}-py3-none-any.whl"
    metadata_dir = f"{filename_project}-{version}.dist-info"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            f"{metadata_dir}/METADATA",
            f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n",
        )
    return wheel_path


@pytest.mark.agent_authored(model="gpt-5.6")
def test_writes_exact_linux_file_references(tmp_path: Path) -> None:
    pathfinder_dir = tmp_path / "pathfinder wheels"
    bindings_dir = tmp_path / "bindings wheels"
    pathfinder = make_wheel(pathfinder_dir, "cuda.pathfinder", "1.7.0.dev1")
    bindings = make_wheel(bindings_dir, "cuda_bindings", "13.1.0.dev2")
    output = tmp_path / "constraints" / "cuda-core.txt"

    write_constraints(
        output,
        "linux-64",
        [("cuda-pathfinder", pathfinder_dir), ("cuda-bindings", bindings_dir)],
    )

    expected_pathfinder = Path("/host", *pathfinder.resolve().parts[1:]).as_uri()
    expected_bindings = Path("/host", *bindings.resolve().parts[1:]).as_uri()
    assert output.read_text(encoding="utf-8") == (
        f"cuda-pathfinder @ {expected_pathfinder}\ncuda-bindings @ {expected_bindings}\n"
    )


@pytest.mark.agent_authored(model="gpt-5.6")
def test_rejects_missing_project_wheel(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    make_wheel(wheel_dir, "another-project", "1.0")
    output = tmp_path / "constraints.txt"

    with pytest.raises(WheelConstraintError, match="expected exactly one cuda-pathfinder wheel"):
        write_constraints(output, "linux-64", [("cuda-pathfinder", wheel_dir)])

    assert not output.exists()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_rejects_multiple_project_wheels(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    make_wheel(wheel_dir, "cuda-pathfinder", "1.7.0.dev1")
    make_wheel(wheel_dir, "cuda-pathfinder", "1.7.0.dev2")
    output = tmp_path / "constraints.txt"

    with pytest.raises(WheelConstraintError, match="found 2"):
        write_constraints(output, "linux-64", [("cuda-pathfinder", wheel_dir)])

    assert not output.exists()


@pytest.mark.agent_authored(model="gpt-5.6")
def test_rejects_duplicate_project_constraints(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    make_wheel(wheel_dir, "cuda-pathfinder", "1.7.0.dev1")
    output = tmp_path / "constraints.txt"

    with pytest.raises(WheelConstraintError, match="duplicate project"):
        write_constraints(
            output,
            "linux-64",
            [("cuda-pathfinder", wheel_dir), ("cuda_pathfinder", wheel_dir)],
        )

    assert not output.exists()
