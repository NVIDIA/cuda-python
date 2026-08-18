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
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ci.tools.artifacts import output_path
from ci.tools.build_artifacts import _cuda_major
from ci.tools.merge_cuda_core_wheels import _validated_moon_output
from ci.tools.moon_fingerprint import _native_tool_identities, _scm_identity, fingerprint
from ci.tools.run_pixi_test import main as run_pixi_test


class MoonArtifactOutputPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo = Path(self.temporary_directory.name)
        (self.repo / "project").mkdir()
        self.patches = (
            patch("ci.tools.artifacts.REPO_ROOT", self.repo),
            patch.dict(
                "ci.tools.artifacts.PROJECT_PATHS",
                {"pathfinder": Path("project")},
                clear=True,
            ),
        )
        for active_patch in self.patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def test_confines_output_to_the_project_output_root(self) -> None:
        output = output_path("pathfinder", "wheel")

        self.assertEqual(output, self.repo / "project" / ".moon-out" / "wheel")
        with self.assertRaisesRegex(ValueError, "output must be within"):
            output_path("pathfinder", "../dist")
        with self.assertRaisesRegex(ValueError, "output must be within"):
            output_path("pathfinder", "../../outside")

    def test_rejects_symlinked_output_ancestors(self) -> None:
        outside = self.repo / "outside"
        outside.mkdir()
        (self.repo / "project" / ".moon-out").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
            output_path("pathfinder", "wheel")

    def test_rejects_projects_outside_the_workspace(self) -> None:
        with (
            patch.dict("ci.tools.artifacts.PROJECT_PATHS", {"pathfinder": Path("../outside")}),
            self.assertRaisesRegex(ValueError, "project must be within"),
        ):
            output_path("pathfinder", "wheel")


class MoonCleanOutputPathTest(unittest.TestCase):
    def test_core_merger_only_cleans_task_owned_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory)
            output = repo / "cuda_core" / ".moon-out" / "wheel-merged"
            with patch("ci.tools.merge_cuda_core_wheels.REPO_ROOT", repo):
                self.assertEqual(_validated_moon_output(output), output)
                self.assertEqual(
                    _validated_moon_output(Path("cuda_core/.moon-out/wheel-merged")),
                    output,
                )
                with self.assertRaisesRegex(ValueError, "must be below"):
                    _validated_moon_output(repo / "cuda_core" / ".moon-out")
                with self.assertRaisesRegex(ValueError, "must be below"):
                    _validated_moon_output(repo / "cuda_core" / "dist")


class MoonFingerprintTest(unittest.TestCase):
    @patch("ci.tools.moon_fingerprint._git_describe")
    def test_distribution_pretend_version_replaces_git_identity(self, git_describe) -> None:
        with patch.dict(
            "os.environ",
            {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_BINDINGS": "13.2.0"},
            clear=True,
        ):
            identity = _scm_identity("bindings")

        self.assertEqual(identity["describe"], "<pretend-version>")
        self.assertEqual(
            identity["environment"]["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_BINDINGS"],
            "13.2.0",
        )
        git_describe.assert_not_called()

    @patch("ci.tools.moon_fingerprint._git_describe", return_value="v13.2.0-1-gabc")
    def test_portable_fingerprint_includes_ambient_python(self, git_describe) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "ci.tools.moon_fingerprint.platform.python_version",
                side_effect=("3.12.11", "3.13.7"),
            ),
        ):
            first = fingerprint("metapackage", "portable")
            second = fingerprint("metapackage", "portable")

        self.assertNotEqual(first, second)
        self.assertEqual(git_describe.call_count, 2)

    @patch("ci.tools.moon_fingerprint._git_describe", return_value="cuda-core-v1.0.0-1-gabc")
    def test_reproducibility_environment_changes_fingerprint(self, git_describe) -> None:
        with patch.dict("os.environ", {"SOURCE_DATE_EPOCH": "1"}, clear=True):
            first = fingerprint("core", "native")
        with patch.dict("os.environ", {"SOURCE_DATE_EPOCH": "2"}, clear=True):
            second = fingerprint("core", "native")

        self.assertNotEqual(first, second)
        self.assertEqual(git_describe.call_count, 2)

    @patch("ci.tools.moon_fingerprint._git_describe", return_value="v13.2.0-1-gabc")
    @patch("ci.tools.moon_fingerprint._native_tool_identities")
    def test_test_asset_fingerprint_tracks_resolved_build_tools(self, native_tools, git_describe) -> None:
        versions = {"Cython": "3.1.0", "numpy": "2.3.0"}

        def distribution_version(name: str) -> str:
            return versions.get(name, "fixed")

        native_tools.return_value = {"cc": {"output": "cc 1", "returncode": 0}}
        with patch("ci.tools.moon_fingerprint._distribution_version", side_effect=distribution_version):
            first = fingerprint("bindings", "test-assets")
            versions["Cython"] = "3.1.1"
            second = fingerprint("bindings", "test-assets")
            native_tools.return_value = {"cc": {"output": "cc 2", "returncode": 0}}
            third = fingerprint("bindings", "test-assets")

        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(git_describe.call_count, 3)

    @patch("ci.tools.moon_fingerprint.subprocess.run")
    @patch("ci.tools.moon_fingerprint.shutil.which")
    @patch("ci.tools.moon_fingerprint._configured_compilers", return_value={"cc", "missing"})
    def test_native_tool_identity_uses_resolved_executables(self, compilers, which, run) -> None:
        which.side_effect = lambda command: "/tools/cc" if command == "cc" else None
        run.return_value = Namespace(stdout="cc 1.2\n", returncode=0)

        self.assertEqual(
            _native_tool_identities(),
            {"cc": {"output": "cc 1.2", "returncode": 0}},
        )
        run.assert_called_once_with(
            ["/tools/cc", "--version"],
            check=False,
            stdout=-1,
            stderr=-2,
            text=True,
            timeout=10,
        )
        compilers.assert_called_once_with()


class MoonTaskCommandTest(unittest.TestCase):
    def test_focused_tool_modules_are_directly_executable(self) -> None:
        for module in ("ci.tools.build_artifacts", "ci.tools.prepare_test_assets"):
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-m", module, "--help"],
                cwd=Path(__file__).resolve().parents[3],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_pixi_test_forwards_the_selected_environment(self) -> None:
        with (
            patch.dict("os.environ", {"PIXI_ENVIRONMENT_NAME": "cu12"}, clear=True),
            patch("sys.argv", ["run_pixi_test.py", "core"]),
            patch("ci.tools.run_pixi_test.shutil.which", return_value="/tools/pixi"),
            patch("ci.tools.run_pixi_test.subprocess.run") as run,
        ):
            run_pixi_test()

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/tools/pixi")
        self.assertIn("cuda_core/pixi.toml", command[3])
        self.assertEqual(command[-3:], ["--environment", "cu12", "test"])

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

    def test_native_builder_requires_the_lane_cuda_major(self) -> None:
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "BUILD_CUDA_MAJOR"):
            _cuda_major("current")


if __name__ == "__main__":
    unittest.main()
