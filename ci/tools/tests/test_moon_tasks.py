# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These task and helper tests intentionally use stdlib unittest so Moon's
# contract task does not need a separately managed Python test environment.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ci.tools.merge_cuda_core_wheels import _clean_output_wheels, _wheel_from_directory, _wheel_source_date_epoch


class WheelMergerInputOutputTest(unittest.TestCase):
    def test_derives_reproducible_epoch_from_input_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wheels = []
            for index, date_time in enumerate(((2024, 1, 2, 3, 4, 6), (2025, 6, 7, 8, 9, 10))):
                wheel = root / f"input-{index}.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    info = zipfile.ZipInfo("payload.txt", date_time=date_time)
                    archive.writestr(info, b"payload")
                wheels.append(wheel)

            expected = int(datetime(2025, 6, 7, 8, 9, 10, tzinfo=timezone.utc).timestamp())
            self.assertEqual(_wheel_source_date_epoch(wheels), str(expected))

    def test_selects_exactly_one_wheel_from_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel_dir = Path(temporary_directory)
            wheel = wheel_dir / "cuda_core.whl"
            wheel.touch()

            self.assertEqual(_wheel_from_directory(wheel_dir), wheel)
            (wheel_dir / "another.whl").touch()
            with self.assertRaisesRegex(ValueError, "expected one wheel"):
                _wheel_from_directory(wheel_dir)

    def test_clean_output_only_removes_wheel_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            wheel = output / "stale.whl"
            unrelated = output / "keep.txt"
            wheel.touch()
            unrelated.touch()

            _clean_output_wheels(output)

            self.assertFalse(wheel.exists())
            self.assertTrue(unrelated.exists())

    def test_clean_output_rejects_a_symlinked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target"
            target.mkdir()
            output = root / "output"
            output.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                _clean_output_wheels(output)

    def test_clean_output_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target"
            target.mkdir()
            (target / "outside.whl").touch()
            parent = root / "linked-parent"
            parent.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                _clean_output_wheels(parent / "wheel-merged")

            self.assertTrue((target / "outside.whl").exists())


class MoonTaskCommandTest(unittest.TestCase):
    def test_main_and_nightly_tests_use_canonical_moon_wheel_directories(self) -> None:
        script = (Path(__file__).resolve().parents[1] / "run-tests").read_text(encoding="utf-8")
        for path in (
            "cuda_pathfinder/.moon-out/wheel-pure",
            "cuda_bindings/.moon-out/wheel-current",
            "cuda_bindings/.moon-out/wheel-previous",
            "cuda_bindings/.moon-out/cython-tests",
            "cuda_core/.moon-out/wheel-merged",
            "cuda_core/.moon-out/cython-tests",
            "cuda_core/.moon-out/test-binaries",
            "cuda_python/.moon-out/wheel-pure",
        ):
            self.assertIn(path, script)
        self.assertNotIn("stage_generated", script)
        self.assertNotIn('"${repo_dir}/cuda_pathfinder"', script)
        self.assertNotIn('"${repo_dir}/cuda_core/dist"', script)
        self.assertNotIn('"${repo_dir}/cuda_python"', script)
        self.assertNotIn('"${repo_dir}" "${repo_dir}/cuda_python"', script)

    def test_env_vars_defers_bindings_wheel_directory_selection_to_test_runner(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        for cuda_version, source in (
            ("13.3.0", "main"),
            ("12.9.1", "backport"),
        ):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary_directory:
                temporary = Path(temporary_directory)
                github_env = temporary / "github-env"
                github_path = temporary / "github-path"
                result = subprocess.run(  # noqa: S603
                    [bash, "ci/tools/env-vars", "test"],
                    cwd=Path(__file__).resolve().parents[3],
                    env={
                        **os.environ,
                        "BUILD_CUDA_VER": "13.3.0",
                        "CUDA_VER": cuda_version,
                        "GITHUB_ENV": str(github_env),
                        "GITHUB_PATH": str(github_path),
                        "HOST_PLATFORM": "linux-64",
                        "LOCAL_CTK": "1",
                        "PY_VER": "3.11",
                    },
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                values = github_env.read_text(encoding="utf-8").splitlines()
                self.assertIn(f"BINDINGS_SOURCE={source}", values)
                self.assertFalse(any(value.startswith("CUDA_BINDINGS_ARTIFACTS_DIR=") for value in values))

    def test_declared_unsupported_bindings_lane_skips_before_artifact_lookup(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        result = subprocess.run(  # noqa: S603
            [bash, "ci/tools/run-tests", "bindings"],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "SKIP_CUDA_BINDINGS_TEST": "1"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Skipping cuda.bindings tests", result.stdout)

    def test_non_main_bindings_lane_skips_metapackage_before_artifact_lookup(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        result = subprocess.run(  # noqa: S603
            [bash, "ci/tools/run-tests", "metapackage"],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "BINDINGS_SOURCE": "published"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BINDINGS_SOURCE is not main", result.stdout)

    def test_metapackage_smoke_validates_all_wheels_with_the_resolver(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory)
            script = repo / "ci" / "tools" / "run-tests"
            script.parent.mkdir(parents=True)
            shutil.copy2(Path(__file__).resolve().parents[1] / "run-tests", script)

            wheel_dirs = {
                "cuda_pathfinder/.moon-out/wheel-pure": "pathfinder.whl",
                "cuda_bindings/.moon-out/wheel-current": "bindings.whl",
                "cuda_core/.moon-out/wheel-merged": "core.whl",
                "cuda_python/.moon-out/wheel-pure": "metapackage.whl",
            }
            for relative_directory, wheel_name in wheel_dirs.items():
                directory = repo / relative_directory
                directory.mkdir(parents=True)
                (directory / wheel_name).touch()

            command_log = repo / "commands.txt"
            fake_python = repo / "bin" / "python"
            fake_python.parent.mkdir()
            fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n')
            fake_python.chmod(0o755)
            result = subprocess.run(  # noqa: S603
                [bash, script, "metapackage"],
                cwd=repo,
                env={
                    **os.environ,
                    "BINDINGS_SOURCE": "main",
                    "COMMAND_LOG": str(command_log),
                    "LOCAL_CTK": "0",
                    "PATH": f"{fake_python.parent}{os.pathsep}{os.environ['PATH']}",
                },
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = command_log.read_text().splitlines()
            self.assertEqual(len(commands), 1)
            self.assertIn("bindings.whl", commands[0])
            self.assertIn("metapackage.whl[all]", commands[0])
            self.assertNotIn("--no-deps", commands[0])


if __name__ == "__main__":
    unittest.main()
