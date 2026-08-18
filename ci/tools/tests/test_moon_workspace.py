# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These tests intentionally use stdlib unittest so the allocation job has no
# third-party Python dependency.
# ruff: noqa: PT009

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PROJECTS = {
    "root": ".",
    "ci": "ci",
    "pathfinder": "cuda_pathfinder",
    "bindings": "cuda_bindings",
    "core": "cuda_core",
    "metapackage": "cuda_python",
    "test-helpers": "cuda_python_test_helpers",
    "bindings-benchmarks": "benchmarks/cuda_bindings",
}
GATE_MARKERS = {
    "force-all",
    "force-all-unowned",
    "build-portable",
    "build-linux-64",
    "build-linux-aarch64",
    "build-windows",
    "test-sdist-linux",
    "test-sdist-windows",
    "test-linux",
    "test-windows",
    "docs",
    "core-api",
    "build-pathfinder",
    "build-bindings",
    "build-core",
    "build-metapackage",
    "test-pathfinder",
    "test-bindings",
    "test-core",
    "test-metapackage",
}
TAG_TARGETS = {
    "ci-wheel-pure": {"pathfinder:wheel-pure", "metapackage:wheel-pure"},
    "ci-wheel-current": {"bindings:wheel-current", "core:wheel-current"},
    "ci-wheel-previous": {"core:wheel-previous"},
    "ci-wheel-merge": {"core:wheel-merge"},
    "ci-build-test-assets": {
        "bindings:cython-test-assets",
        "core:cython-test-assets",
        "core:test-binaries",
    },
    "ci-sdist": {"pathfinder:sdist", "bindings:sdist", "core:sdist", "metapackage:sdist"},
    "ci-test-linux": {
        "pathfinder:test-installed-linux",
        "pathfinder:test-installed-linux-strict",
        "bindings:test-installed-linux",
        "core:test-installed-linux",
        "metapackage:test-installed-linux",
        "bindings-benchmarks:smoke-linux",
    },
    "ci-test-windows": {
        "pathfinder:test-installed-windows",
        "pathfinder:test-installed-windows-strict",
        "bindings:test-installed-windows",
        "core:test-installed-windows",
        "metapackage:test-installed-windows",
    },
    "ci-docs": {"root:docs-ci"},
}
CACHED_OUTPUTS = {
    "pathfinder:wheel-pure": ".moon-out/wheel-pure",
    "pathfinder:sdist": ".moon-out/sdist",
    "bindings:wheel-current": ".moon-out/wheel-current",
    "bindings:sdist": ".moon-out/sdist",
    "bindings:cython-test-assets": ".moon-out/cython-tests",
    "core:wheel-current": ".moon-out/wheel-current",
    "core:wheel-previous": ".moon-out/wheel-previous",
    "core:wheel-merge": ".moon-out/wheel-merged",
    "core:sdist": ".moon-out/sdist",
    "core:cython-test-assets": ".moon-out/cython-tests",
    "core:test-binaries": ".moon-out/test-binaries",
    "metapackage:wheel-pure": ".moon-out/wheel-pure",
    "metapackage:sdist": ".moon-out/sdist",
}


class MoonWorkspaceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.moon = os.environ.get("MOON_BIN") or shutil.which("moon")
        if not cls.moon:
            raise unittest.SkipTest("Moon is not installed; set MOON_BIN to test the workspace")
        cls.tasks = cls.moon_json("tasks", "--json")
        cls.by_target = {task["target"]: task for task in cls.tasks}

    @classmethod
    def moon_json(cls, *arguments: str) -> Any:
        result = subprocess.run(  # noqa: S603 - the binary is explicitly selected above.
            [cls.moon, *arguments],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return json.loads(result.stdout)

    def test_projects_use_only_the_system_toolchain(self) -> None:
        projects = self.moon_json("projects", "--json")
        by_id = {project["id"]: project for project in projects}
        self.assertEqual(
            {project_id: project["source"] for project_id, project in by_id.items()},
            EXPECTED_PROJECTS,
        )
        for project in by_id.values():
            self.assertEqual(project["language"], "unknown")
            self.assertEqual(project["toolchains"], ["system"])

    def test_ci_gates_are_real_uncached_marker_tasks(self) -> None:
        expected_targets = {f"ci:gate-{marker}" for marker in GATE_MARKERS}
        tagged = {task["target"] for task in self.tasks if "ci-gate" in task.get("tags", [])}
        self.assertEqual(tagged, expected_targets)
        for marker in GATE_MARKERS:
            task = self.by_target[f"ci:gate-{marker}"]
            self.assertEqual(task["command"], "python")
            self.assertEqual(task["args"], ["ci/tools/moon_ci.py", "gate", marker])
            self.assertFalse(task["options"]["cache"])
            self.assertFalse(task["options"]["internal"])
            self.assertTrue(task["options"]["runInCI"])
            self.assertEqual(task["outputs"], [{"file": f".moon-out/ci-gates/{marker}"}])

    def test_ci_tags_select_the_intended_real_tasks(self) -> None:
        for tag, expected in TAG_TARGETS.items():
            selected = {task["target"] for task in self.tasks if tag in task.get("tags", [])}
            self.assertEqual(selected, expected, tag)
            for target in selected:
                self.assertNotEqual(self.by_target[target]["command"], "noop")
                self.assertTrue(self.by_target[target]["options"]["runInCI"])

    def test_cached_producers_have_explicit_non_overlapping_outputs(self) -> None:
        cached = {task["target"] for task in self.tasks if task["options"]["cache"]}
        self.assertEqual(cached, set(CACHED_OUTPUTS))
        destinations: set[tuple[str, str]] = set()
        for target, output in CACHED_OUTPUTS.items():
            task = self.by_target[target]
            self.assertEqual(task["outputs"], [{"file": output}])
            self.assertTrue(task.get("inputs"))
            self.assertTrue(task.get("checks"))
            self.assertNotIn("CUDA_PYTHON_TOOL_VERSIONS", task.get("env") or {})
            self.assertIn({"file": "/ci/tools/moon_fingerprint.py"}, task["inputs"])
            project = target.split(":", maxsplit=1)[0]
            self.assertNotIn((project, output), destinations)
            destinations.add((project, output))

    def test_tests_and_docs_are_uncached(self) -> None:
        ci_test_targets = set().union(
            TAG_TARGETS["ci-test-linux"], TAG_TARGETS["ci-test-windows"], TAG_TARGETS["ci-docs"]
        )
        for target in ci_test_targets:
            self.assertFalse(self.by_target[target]["options"]["cache"])

    def test_cross_runner_tasks_do_not_execute_producer_dependencies(self) -> None:
        for tag in (
            "ci-test-linux",
            "ci-test-windows",
            "ci-build-test-assets",
            "ci-wheel-current",
            "ci-wheel-previous",
        ):
            for target in TAG_TARGETS[tag]:
                self.assertFalse(self.by_target[target].get("deps"), target)
        self.assertFalse(self.by_target["core:wheel-merge"].get("deps"))

    def test_native_producers_hash_downloaded_wheel_bytes(self) -> None:
        required_globs = {
            "bindings:wheel-current": {"/cuda_pathfinder/.moon-out/wheel-pure/*.whl"},
            "core:wheel-current": {
                "/cuda_pathfinder/.moon-out/wheel-pure/*.whl",
                "/cuda_bindings/.moon-out/wheel-current/*.whl",
            },
            "core:wheel-previous": {
                "/cuda_pathfinder/.moon-out/wheel-pure/*.whl",
                "/cuda_bindings/.moon-out/wheel-previous/*.whl",
            },
        }
        for target, expected in required_globs.items():
            configured = {
                item["glob"] for item in self.by_target[target]["inputs"] if isinstance(item, dict) and "glob" in item
            }
            self.assertTrue(expected.issubset(configured), target)

    def test_metapackage_install_smoke_tracks_runtime_inputs(self) -> None:
        for target in ("metapackage:test-installed-linux", "metapackage:test-installed-windows"):
            inputs = self.by_target[target]["inputs"]
            self.assertIn({"project": "pathfinder", "group": "package"}, inputs)
            self.assertIn({"project": "bindings", "group": "package"}, inputs)
            self.assertIn({"project": "core", "group": "package"}, inputs)
            input_globs = {item["glob"] for item in inputs if isinstance(item, dict) and "glob" in item}
            self.assertIn("/cuda_core/.moon-out/wheel-merged/*.whl", input_globs)
        self.assertIn(
            {"project": "core", "group": "package"},
            self.by_target["ci:gate-test-metapackage"]["inputs"],
        )

    def test_cross_runner_producer_inputs_reach_exact_consumers(self) -> None:
        portable_workflow = {"file": "/.github/workflows/build-pure-wheel.yml"}
        portable_consumers = {
            "bindings:wheel-current",
            "core:wheel-current",
            "core:wheel-previous",
            "core:wheel-merge",
            "bindings:cython-test-assets",
            "core:cython-test-assets",
            "pathfinder:test-installed-linux",
            "pathfinder:test-installed-linux-strict",
            "pathfinder:test-installed-windows",
            "pathfinder:test-installed-windows-strict",
            "bindings:test-installed-linux",
            "bindings:test-installed-windows",
            "core:test-installed-linux",
            "core:test-installed-windows",
            "metapackage:test-installed-linux",
            "metapackage:test-installed-windows",
        }
        for target in portable_consumers:
            self.assertIn(portable_workflow, self.by_target[target]["inputs"], target)

        native_workflow = {"file": "/.github/workflows/build-wheel.yml"}
        native_test_consumers = {
            "bindings:test-installed-linux",
            "bindings:test-installed-windows",
            "core:test-installed-linux",
            "core:test-installed-windows",
            "metapackage:test-installed-linux",
            "metapackage:test-installed-windows",
        }
        for target in native_test_consumers:
            self.assertIn(native_workflow, self.by_target[target]["inputs"], target)

        merge_helper = {"file": "/ci/tools/merge_cuda_core_wheels.py"}
        merge_test_consumers = {
            "core:test-installed-linux",
            "core:test-installed-windows",
            "metapackage:test-installed-linux",
            "metapackage:test-installed-windows",
        }
        for target in merge_test_consumers:
            self.assertIn(merge_helper, self.by_target[target]["inputs"], target)

    def test_cross_runner_producer_inputs_reach_matching_gates(self) -> None:
        gate_expectations = {
            "@group(build-portable)": {
                "gate-build-portable",
                "gate-build-linux-64",
                "gate-build-linux-aarch64",
                "gate-build-windows",
                "gate-build-pathfinder",
                "gate-build-bindings",
                "gate-build-core",
                "gate-build-metapackage",
                "gate-test-linux",
                "gate-test-windows",
                "gate-test-pathfinder",
                "gate-test-bindings",
                "gate-test-core",
                "gate-test-metapackage",
            },
            "@group(build-native-common)": {
                "gate-build-linux-64",
                "gate-build-linux-aarch64",
                "gate-build-windows",
                "gate-build-bindings",
                "gate-build-core",
                "gate-test-linux",
                "gate-test-windows",
                "gate-test-bindings",
                "gate-test-core",
                "gate-test-metapackage",
            },
            "@group(build-native-core)": {
                "gate-build-linux-64",
                "gate-build-linux-aarch64",
                "gate-build-windows",
                "gate-build-core",
                "gate-test-linux",
                "gate-test-windows",
                "gate-test-core",
                "gate-test-metapackage",
            },
        }
        for producer_group, expected_gates in gate_expectations.items():
            actual_gates = {
                task["id"]
                for task in self.tasks
                if "ci-gate" in task.get("tags", []) and producer_group in task["inputs"]
            }
            self.assertEqual(actual_gates, expected_gates, producer_group)

    def test_installed_test_runner_does_not_select_metapackage_smoke(self) -> None:
        runner_group = "@group(test-library-runner)"
        for gate in ("gate-test-pathfinder", "gate-test-bindings", "gate-test-core"):
            self.assertIn(runner_group, self.by_target[f"ci:{gate}"]["inputs"])
        self.assertNotIn(runner_group, self.by_target["ci:gate-test-metapackage"]["inputs"])

    def test_platform_test_tasks_track_provider_setup(self) -> None:
        linux_targets = TAG_TARGETS["ci-test-linux"]
        windows_targets = TAG_TARGETS["ci-test-windows"]
        for target in linux_targets:
            inputs = self.by_target[target]["inputs"]
            self.assertIn({"file": "/ci/tools/guess_latest.sh"}, inputs, target)
            self.assertIn({"file": "/ci/tools/install_gpu_driver.sh"}, inputs, target)
        for target in windows_targets:
            inputs = self.by_target[target]["inputs"]
            self.assertIn({"file": "/ci/tools/configure_driver_mode.ps1"}, inputs, target)
            self.assertIn({"file": "/ci/tools/install_gpu_driver.ps1"}, inputs, target)
        for target in ("bindings:test-installed-linux", "core:test-installed-linux"):
            self.assertIn({"file": "/ci/tools/setup-sanitizer"}, self.by_target[target]["inputs"])

    def test_docs_gate_and_task_share_package_owned_groups(self) -> None:
        docs = self.by_target["root:docs-ci"]["inputs"]
        gate = self.by_target["ci:gate-docs"]["inputs"]
        external_groups = [item for item in docs if isinstance(item, dict) and "project" in item]
        for group in external_groups:
            self.assertIn(group, gate)

    def test_core_merge_changes_materialize_all_core_wheel_phases(self) -> None:
        merge_helper = {"file": "/ci/tools/merge_cuda_core_wheels.py"}
        for target in ("core:wheel-current", "core:wheel-previous", "core:wheel-merge"):
            self.assertIn(merge_helper, self.by_target[target]["inputs"])

    def test_local_pixi_tasks_remain_available_and_skip_ci(self) -> None:
        for target in (
            "pathfinder:test",
            "bindings:test",
            "core:test",
            "pathfinder:docs",
            "bindings:docs",
            "core:docs",
            "bindings-benchmarks:bench",
        ):
            task = self.by_target[target]
            self.assertIn("pixi-", " ".join(task.get("args", [])))
            self.assertFalse(task["options"]["runInCI"])

    def test_workspace_disables_python_and_dependency_management(self) -> None:
        workspace = (REPO_ROOT / ".moon" / "workspace.yml").read_text(encoding="utf-8")
        self.assertIn("versionConstraint: '=2.5.1'", workspace)
        self.assertIn("installDependencies: false", workspace)
        self.assertIn("syncProjects: false", workspace)
        self.assertIn("syncWorkspace: false", workspace)
        self.assertIn("verifyIntegrity: true", workspace)
        self.assertFalse((REPO_ROOT / ".moon" / "toolchains.yml").exists())

    def test_generated_cache_and_output_roots_are_ignored(self) -> None:
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".moon/cache/", ignore)
        self.assertIn(".moon-out/", ignore)


if __name__ == "__main__":
    unittest.main()
