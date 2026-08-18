# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These tests intentionally use stdlib unittest so Moon's contract task does
# not need a separately managed Python test environment.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ci.tools.moon_ci import (
    _bindings_benchmark_smoke,
    _docs_arguments,
    _installed_test,
    _metapackage_install_test,
    _output_path,
    _pixi_test,
    _prepare_pathfinder_strict,
)
from ci.tools.moon_fingerprint import _native_tool_identities, _scm_identity, fingerprint


class MoonCIOutputPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo = Path(self.temporary_directory.name)
        (self.repo / "project").mkdir()
        self.patches = (
            patch("ci.tools.moon_ci.REPO_ROOT", self.repo),
            patch.dict(
                "ci.tools.moon_ci.PROJECT_PATHS",
                {"pathfinder": Path("project")},
                clear=True,
            ),
        )
        for active_patch in self.patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def test_confines_output_to_the_project_output_root(self) -> None:
        output = _output_path("pathfinder", "wheel")

        self.assertEqual(output, self.repo / "project" / ".moon-out" / "wheel")
        with self.assertRaisesRegex(ValueError, "output must be within"):
            _output_path("pathfinder", "../dist")
        with self.assertRaisesRegex(ValueError, "output must be within"):
            _output_path("pathfinder", "../../outside")

    def test_rejects_symlinked_output_ancestors(self) -> None:
        outside = self.repo / "outside"
        outside.mkdir()
        (self.repo / "project" / ".moon-out").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "must not traverse a symlink"):
            _output_path("pathfinder", "wheel")

    def test_rejects_projects_outside_the_workspace(self) -> None:
        with (
            patch.dict("ci.tools.moon_ci.PROJECT_PATHS", {"pathfinder": Path("../outside")}),
            self.assertRaisesRegex(ValueError, "project must be within"),
        ):
            _output_path("pathfinder", "wheel")

    def test_docs_latest_only_defaults_to_enabled_and_validates_values(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_docs_arguments(), ["latest-only"])
        with patch.dict("os.environ", {"CUDA_PYTHON_DOCS_LATEST_ONLY": "false"}, clear=True):
            self.assertEqual(_docs_arguments(), [])
        with (
            patch.dict("os.environ", {"CUDA_PYTHON_DOCS_LATEST_ONLY": "sometimes"}, clear=True),
            self.assertRaisesRegex(ValueError, "must be true"),
        ):
            _docs_arguments()


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


class MoonCIConditionalTest(unittest.TestCase):
    def test_local_pixi_test_forwards_the_selected_environment(self) -> None:
        with (
            patch.dict("os.environ", {"PIXI_ENVIRONMENT_NAME": "cu12"}, clear=True),
            patch("ci.tools.moon_ci.shutil.which", return_value="/tools/pixi"),
            patch("ci.tools.moon_ci._run") as run,
        ):
            _pixi_test(Namespace(project="core"))

        command = run.call_args.args[0]
        self.assertEqual(command[0], "/tools/pixi")
        self.assertIn("cuda_core/pixi.toml", command[3])
        self.assertEqual(command[-3:], ["--environment", "cu12", "test"])

    def test_declared_unsupported_bindings_lane_skips_benchmark_before_pixi_lookup(self) -> None:
        with (
            patch.dict("os.environ", {"SKIP_CUDA_BINDINGS_TEST": "1"}, clear=True),
            patch("ci.tools.moon_ci._artifact_wheel") as artifact_wheel,
        ):
            _bindings_benchmark_smoke(Namespace())

        artifact_wheel.assert_not_called()

    def test_benchmark_smoke_uses_the_prepared_system_python(self) -> None:
        pathfinder = Path("pathfinder.whl")
        bindings = Path("bindings.whl")
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("ci.tools.moon_ci._artifact_wheel", side_effect=(pathfinder, bindings)),
            patch("ci.tools.moon_ci._run") as run,
        ):
            _bindings_benchmark_smoke(Namespace())

        self.assertEqual(
            run.call_args_list[0].args[0],
            [sys.executable, "-m", "pip", "install", str(pathfinder), str(bindings), "pyperf"],
        )
        self.assertEqual(run.call_args_list[1].args[0][0], sys.executable)
        self.assertEqual(run.call_args_list[1].args[0][-1], "--debug-single-value")

    def test_declared_unsupported_bindings_lane_skips_before_artifact_lookup(self) -> None:
        with (
            patch.dict("os.environ", {"SKIP_CUDA_BINDINGS_TEST": "1"}, clear=True),
            patch("ci.tools.moon_ci._artifact_wheel") as artifact_wheel,
        ):
            _installed_test(Namespace(project="bindings"))

        artifact_wheel.assert_not_called()

    def test_non_main_bindings_lane_skips_metapackage_before_artifact_lookup(self) -> None:
        with (
            patch.dict("os.environ", {"BINDINGS_SOURCE": "published"}, clear=True),
            patch("ci.tools.moon_ci._artifact_wheel") as artifact_wheel,
        ):
            _metapackage_install_test(Namespace())

        artifact_wheel.assert_not_called()

    def test_pathfinder_strict_preparation_requires_numeric_cuda_major(self) -> None:
        with (
            patch.dict("os.environ", {"TEST_CUDA_MAJOR": "latest"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "numeric CUDA major"),
        ):
            _prepare_pathfinder_strict(Namespace())


if __name__ == "__main__":
    unittest.main()
