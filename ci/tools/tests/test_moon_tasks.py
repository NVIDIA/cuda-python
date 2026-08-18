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
from pathlib import Path

from ci.tools.merge_cuda_core_wheels import _clean_output_wheels, _wheel_from_directory


class WheelMergerInputOutputTest(unittest.TestCase):
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

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _clean_output_wheels(output)


class MoonTaskCommandTest(unittest.TestCase):
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
