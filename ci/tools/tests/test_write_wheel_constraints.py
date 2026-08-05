# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from write_wheel_constraints import WheelConstraintError, WheelRequirement, main, write_constraints


def _write_wheel(
    directory: Path,
    filename: str,
    *,
    name: str,
    version: str,
    metadata_entries: int = 1,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    wheel_path = directory / filename
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for index in range(metadata_entries):
            suffix = f"-{index}" if metadata_entries > 1 else ""
            wheel.writestr(
                f"{name.replace('-', '_')}-{version}{suffix}.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            )
    return wheel_path


@pytest.mark.agent_authored(model="gpt-5.6-sol")
class TestWriteConstraints:
    def test_writes_canonical_direct_references_and_escapes_spaces(self, tmp_path):
        wheel_dir = tmp_path / "wheel house"
        pathfinder = _write_wheel(
            wheel_dir,
            "cuda_pathfinder-1.5.2-py3-none-any.whl",
            name="cuda.pathfinder",
            version="1.5.2",
        )
        bindings = _write_wheel(
            wheel_dir,
            "cuda_bindings-13.3.0.dev1-py3-none-any.whl",
            name="CUDA_Bindings",
            version="13.3.0.dev1",
        )
        _write_wheel(
            wheel_dir,
            "unrelated-1.0-py3-none-any.whl",
            name="unrelated",
            version="1.0",
        )
        output = tmp_path / "constraints.txt"

        write_constraints(
            output,
            [
                WheelRequirement("cuda-pathfinder", wheel_dir),
                WheelRequirement("cuda.bindings", wheel_dir, expected_major="13"),
            ],
        )

        assert output.read_text(encoding="utf-8").splitlines() == [
            f"cuda-pathfinder @ {pathfinder.resolve().as_uri()}",
            f"cuda-bindings @ {bindings.resolve().as_uri()}",
        ]
        assert "%20" in output.read_text(encoding="utf-8")

    def test_expected_major_selects_from_coexisting_cuda_majors(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_bindings-12.9.1-py3-none-any.whl",
            name="cuda-bindings",
            version="12.9.1",
        )
        selected = _write_wheel(
            wheel_dir,
            "cuda_bindings-13.3.0.dev2-py3-none-any.whl",
            name="cuda-bindings",
            version="13.3.0.dev2",
        )
        output = tmp_path / "constraints.txt"

        write_constraints(output, [WheelRequirement("cuda-bindings", wheel_dir, expected_major="13")])

        assert output.read_text(encoding="utf-8") == f"cuda-bindings @ {selected.resolve().as_uri()}\n"

    @pytest.mark.skipif(os.name != "posix", reason="cibuildwheel /host mapping is POSIX-only")
    def test_maps_wheel_uri_to_container_mount(self, tmp_path):
        wheel = _write_wheel(
            tmp_path / "wheels",
            "cuda_pathfinder-1.5.2-py3-none-any.whl",
            name="cuda-pathfinder",
            version="1.5.2",
        ).resolve()
        output = tmp_path / "constraints.txt"

        write_constraints(
            output,
            [WheelRequirement("cuda-pathfinder", wheel.parent)],
            container_mount=Path("/host"),
        )

        container_wheel = Path("/host") / wheel.relative_to(wheel.anchor)
        assert output.read_text(encoding="utf-8") == f"cuda-pathfinder @ {container_wheel.as_uri()}\n"
        assert stat.S_IMODE(output.stat().st_mode) == 0o644

    def test_rejects_relative_container_mount(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_pathfinder-1.5.2-py3-none-any.whl",
            name="cuda-pathfinder",
            version="1.5.2",
        )

        with pytest.raises(WheelConstraintError, match="Container mount must be an absolute path"):
            write_constraints(
                tmp_path / "constraints.txt",
                [WheelRequirement("cuda-pathfinder", wheel_dir)],
                container_mount=Path("host"),
            )

    @pytest.mark.parametrize("directory_kind", ["missing", "empty"])
    def test_rejects_missing_or_empty_wheel_directory(self, tmp_path, directory_kind):
        wheel_dir = tmp_path / "wheels"
        if directory_kind == "empty":
            wheel_dir.mkdir()

        with pytest.raises(WheelConstraintError, match="does not exist|contains no .whl files"):
            write_constraints(tmp_path / "constraints.txt", [WheelRequirement("cuda-pathfinder", wheel_dir)])

    def test_uses_metadata_name_instead_of_spoofed_filename(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_bindings-13.3.0-py3-none-any.whl",
            name="not-cuda-bindings",
            version="13.3.0",
        )

        with pytest.raises(WheelConstraintError, match="Found no wheel for cuda-bindings"):
            write_constraints(tmp_path / "constraints.txt", [WheelRequirement("cuda-bindings", wheel_dir)])

    def test_rejects_multiple_wheels_for_expected_major(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        for patch in ("0", "1"):
            _write_wheel(
                wheel_dir,
                f"cuda_bindings-13.3.{patch}-py3-none-any.whl",
                name="cuda-bindings",
                version=f"13.3.{patch}",
            )

        with pytest.raises(WheelConstraintError, match="Found multiple wheels.*release major 13"):
            write_constraints(
                tmp_path / "constraints.txt",
                [WheelRequirement("cuda-bindings", wheel_dir, expected_major="13")],
            )

    def test_wrong_major_reports_available_version(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_bindings-12.9.1-py3-none-any.whl",
            name="cuda-bindings",
            version="12.9.1",
        )

        with pytest.raises(WheelConstraintError, match=r"release major 13.*cuda-bindings==12\.9\.1"):
            write_constraints(
                tmp_path / "constraints.txt",
                [WheelRequirement("cuda-bindings", wheel_dir, expected_major="13")],
            )

    def test_rejects_duplicate_canonical_project_requests(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_bindings-13.3.0-py3-none-any.whl",
            name="cuda-bindings",
            version="13.3.0",
        )

        with pytest.raises(WheelConstraintError, match="requested more than once"):
            write_constraints(
                tmp_path / "constraints.txt",
                [WheelRequirement("cuda-bindings", wheel_dir), WheelRequirement("CUDA_Bindings", wheel_dir)],
            )

    @pytest.mark.parametrize("project", ["", "cuda bindings", "---", "cuda-bindings-"])
    def test_rejects_invalid_project_name(self, tmp_path, project):
        with pytest.raises(WheelConstraintError, match="Invalid project name"):
            write_constraints(tmp_path / "constraints.txt", [WheelRequirement(project, tmp_path)])

    def test_rejects_invalid_wheel_archive(self, tmp_path):
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "cuda_bindings-13.3.0-py3-none-any.whl").write_text("not a zip", encoding="utf-8")

        with pytest.raises(WheelConstraintError, match="Cannot read wheel"):
            write_constraints(tmp_path / "constraints.txt", [WheelRequirement("cuda-bindings", wheel_dir)])

    @pytest.mark.parametrize("metadata_entries", [0, 2])
    def test_rejects_missing_or_multiple_metadata_entries(self, tmp_path, metadata_entries):
        wheel_dir = tmp_path / "wheels"
        _write_wheel(
            wheel_dir,
            "cuda_bindings-13.3.0-py3-none-any.whl",
            name="cuda-bindings",
            version="13.3.0",
            metadata_entries=metadata_entries,
        )

        with pytest.raises(WheelConstraintError, match="METADATA entries; expected exactly one"):
            write_constraints(tmp_path / "constraints.txt", [WheelRequirement("cuda-bindings", wheel_dir)])

    def test_failed_resolution_removes_stale_output(self, tmp_path):
        output = tmp_path / "constraints.txt"
        output.write_text("cuda-bindings==0\n", encoding="utf-8")

        with pytest.raises(WheelConstraintError):
            write_constraints(output, [WheelRequirement("cuda-bindings", tmp_path / "missing")])

        assert not output.exists()

    def test_cli_reports_unknown_expected_major_project(self, tmp_path, capsys):
        output = tmp_path / "constraints.txt"
        output.write_text("stale\n", encoding="utf-8")
        rc = main(
            [
                "--output",
                str(output),
                "--wheel",
                "cuda-pathfinder",
                str(tmp_path),
                "--expected-major",
                "cuda-bindings",
                "13",
            ]
        )

        assert rc == 1
        assert "error: Expected major was specified for an unrequested project" in capsys.readouterr().err
        assert not output.exists()

    def test_cli_success_reports_selected_wheel(self, tmp_path, capsys):
        wheel_dir = tmp_path / "wheels"
        wheel = _write_wheel(
            wheel_dir,
            "cuda_pathfinder-1.5.2-py3-none-any.whl",
            name="cuda-pathfinder",
            version="1.5.2",
        )
        output = tmp_path / "constraints.txt"

        rc = main(
            [
                "--output",
                str(output),
                "--wheel",
                "cuda-pathfinder",
                str(wheel_dir),
            ]
        )

        assert rc == 0
        assert output.read_text(encoding="utf-8") == f"cuda-pathfinder @ {wheel.resolve().as_uri()}\n"
        assert "Selected cuda-pathfinder==1.5.2" in capsys.readouterr().out

    def test_cli_reports_corrupt_wheel_without_traceback(self, tmp_path, capsys):
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "cuda_pathfinder-1.5.2-py3-none-any.whl").write_text("not a zip", encoding="utf-8")

        rc = main(
            [
                "--output",
                str(tmp_path / "constraints.txt"),
                "--wheel",
                "cuda-pathfinder",
                str(wheel_dir),
            ]
        )

        assert rc == 1
        captured = capsys.readouterr()
        assert "error: Cannot read wheel" in captured.err
        assert "Traceback" not in captured.err
