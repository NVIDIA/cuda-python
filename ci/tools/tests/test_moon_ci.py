# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These tests intentionally use stdlib unittest so the cheap CI planner job has
# no third-party Python dependency.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from ci.tools.moon_ci import _gate, _output_path
from ci.tools.moon_fingerprint import _scm_identity, fingerprint


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
                {"pathfinder": Path("project"), "ci": Path("ci")},
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

    def test_gate_writes_only_a_declared_marker(self) -> None:
        (self.repo / "ci").mkdir()
        _gate(Namespace(marker="build-linux-64"))

        marker = self.repo / "ci" / ".moon-out" / "ci-gates" / "build-linux-64"
        self.assertEqual(marker.read_text(encoding="utf-8"), "true\n")
        with self.assertRaisesRegex(ValueError, "unknown CI gate marker"):
            _gate(Namespace(marker="anything-else"))


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


if __name__ == "__main__":
    unittest.main()
