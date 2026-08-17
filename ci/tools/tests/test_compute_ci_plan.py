# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ci.tools.compute_ci_plan import _expand_linked_paths, compute_workplan

ALL_MODULES = {"pathfinder", "bindings", "core", "python"}
ALL_PLATFORMS = {"linux", "windows"}


def plan_for(
    *paths: str,
    baseline: bool = True,
    linked_paths: set[str] | None = None,
) -> dict[str, object]:
    return compute_workplan(
        list(paths),
        merge_base="base",
        baseline_run_id="123" if baseline else "",
        baseline_sha="base" if baseline else "",
        linked_paths=linked_paths,
    )


def selected(plan: dict[str, object], key: str) -> set[str]:
    modules = plan["modules"]
    assert isinstance(modules, dict)
    return {name for name, decision in modules.items() if decision[key]}


def selected_platforms(plan: dict[str, object]) -> set[str]:
    jobs = plan["jobs"]
    assert isinstance(jobs, dict)
    platforms = jobs["platforms"]
    assert isinstance(platforms, dict)
    assert set(platforms) == ALL_PLATFORMS
    return {name for name, enabled in platforms.items() if enabled}


class ComputeWorkplanTest(unittest.TestCase):
    def test_path_impacts(self) -> None:
        cases = {
            "cuda_pathfinder/cuda/pathfinder/_loader.py": (ALL_MODULES, ALL_MODULES, False),
            "cuda_bindings/cuda/bindings/driver.pyx": (
                {"bindings", "core", "python"},
                {"bindings", "core", "python"},
                False,
            ),
            "cuda_core/cuda/core/_device.py": ({"core"}, {"core", "python"}, True),
            "cuda_core/cuda/core/examples/demo.py": ({"core"}, {"core", "python"}, True),
            "cuda_python/pyproject.toml": ({"bindings", "python"}, {"python"}, False),
            "cuda_pathfinder/tests/test_loader.py": (set(), {"pathfinder"}, False),
            "cuda_bindings/examples/0_Introduction/vectorAddDrv.py": (set(), {"bindings"}, False),
            "cuda_bindings/tests/README.md": (set(), {"bindings"}, False),
            "cuda_core/pytest.ini": (set(), {"core"}, False),
            "cuda_python/tests/test_import.py": (set(), {"python"}, False),
            "cuda_python_test_helpers/cuda_python_test_helpers/cuda_utils.py": (
                set(),
                ALL_MODULES,
                False,
            ),
            "benchmarks/cuda_bindings/run_pyperf.py": (set(), ALL_MODULES, False),
            "benchmarks/cuda_core/runner.py": (set(), ALL_MODULES, False),
            "ci/tools/run-tests": (ALL_MODULES, ALL_MODULES, True),
            "ci/versions.yml": (ALL_MODULES, ALL_MODULES, True),
            "pytest.ini": (ALL_MODULES, ALL_MODULES, True),
        }

        for path, (builds, tests, core_api) in cases.items():
            with self.subTest(path=path):
                plan = plan_for(path)
                assert selected(plan, "needs_build") == builds
                assert selected(plan, "needs_test") == tests
                assert selected_platforms(plan) == ALL_PLATFORMS
                assert plan["jobs"]["sdist_tests"] == bool(builds)
                assert plan["jobs"]["core_api_checks"] == core_api

    def test_test_infrastructure_platforms(self) -> None:
        cases = {
            ".github/workflows/test-wheel-linux.yml": {"linux"},
            ".github/workflows/test-wheel-windows.yml": {"windows"},
            "ci/tools/configure_driver_mode.ps1": {"windows"},
            "ci/tools/guess_latest.sh": {"linux"},
            "ci/tools/install_gpu_driver.ps1": {"windows"},
            "ci/tools/install_gpu_driver.sh": {"linux"},
            "ci/tools/setup-sanitizer": {"linux"},
        }

        for path, platforms in cases.items():
            with self.subTest(path=path):
                plan = plan_for(path)
                assert not selected(plan, "needs_build")
                assert selected(plan, "needs_test") == ALL_MODULES
                assert selected_platforms(plan) == platforms
                assert not plan["jobs"]["sdist_tests"]
                assert not plan["jobs"]["core_api_checks"]

        mixed_plan = plan_for("ci/tools/install_gpu_driver.sh", "ci/tools/install_gpu_driver.ps1")
        assert selected_platforms(mixed_plan) == ALL_PLATFORMS

        source_plan = plan_for("ci/tools/install_gpu_driver.sh", "cuda_python/pyproject.toml")
        assert selected_platforms(source_plan) == ALL_PLATFORMS

    def test_ignored_paths_select_no_work(self) -> None:
        for path in (
            "cuda_core/docs/index.rst",
            "cuda_core/pixi.toml",
            "cuda_core/tests/fixtures/pixi.toml",
            "benchmarks/cuda_bindings/pixi.toml",
            "benchmarks/cuda_bindings/AGENTS.md",
            "cuda_core/cuda/core/_cpp/DESIGN.md",
            "cuda_bindings/README.md",
            "cuda_core/README.md",
            "new-area/pixi.toml",
            "notes.md",
            "diagram.svg",
            ".github/labeler.yml",
            ".github/ISSUE_TEMPLATE/bug.yml",
        ):
            with self.subTest(path=path):
                plan = plan_for(path)
                assert not selected(plan, "needs_build")
                assert not selected(plan, "needs_test")
                assert not selected_platforms(plan)

    def test_unknown_path_and_missing_baseline_force_all(self) -> None:
        for plan in (
            plan_for("new-top-level-file"),
            plan_for("new-area/config.toml"),
            plan_for(".github/workflows/new-main-ci-workflow.yml"),
            plan_for(".github/actions/doc_preview/action.yml"),
            plan_for("ci/ci-pipeline.svg"),
            plan_for("cuda_core/docs/index.rst", baseline=False),
            compute_workplan([], merge_base="base", baseline_run_id="123", baseline_sha=""),
        ):
            assert selected(plan, "needs_build") == ALL_MODULES
            assert selected(plan, "needs_test") == ALL_MODULES
            assert selected_platforms(plan) == ALL_PLATFORMS
            assert plan["jobs"]["core_api_checks"]
            assert plan["baseline"] == {"run_id": "", "sha": ""}

    def test_mixed_changes_are_combined(self) -> None:
        plan = plan_for("cuda_core/tests/test_device.py", "cuda_python/pyproject.toml")
        assert selected(plan, "needs_build") == {"bindings", "python"}
        assert selected(plan, "needs_test") == {"core", "python"}
        assert selected_platforms(plan) == ALL_PLATFORMS
        assert plan["jobs"]["sdist_tests"]

    def test_changed_symlink_targets_include_their_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cuda_python").mkdir()
            (root / "cuda_core").mkdir()
            (root / "README.md").write_text("readme", encoding="utf-8")
            (root / "cuda_python" / "README.md").symlink_to("../README.md")
            (root / "cuda_core" / "README.md").symlink_to("../README.md")

            paths = _expand_linked_paths(
                ["README.md"],
                ["cuda_python/README.md"],
                root=root,
            )

        assert paths == ["README.md", "cuda_python/README.md"]
        plan = plan_for(*paths, linked_paths={"cuda_python/README.md"})
        assert selected(plan, "needs_build") == {"bindings", "python"}
        assert selected(plan, "needs_test") == {"python"}

        removed_link = plan_for("cuda_python/README.md", linked_paths={"cuda_python/README.md"})
        assert selected(removed_link, "needs_build") == {"bindings", "python"}
        assert selected(removed_link, "needs_test") == {"python"}


if __name__ == "__main__":
    unittest.main()
