# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These tests intentionally use stdlib unittest so Moon's contract task does
# not need a separately managed Python test environment.
# ruff: noqa: PT009

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_PROJECTS = {
    "root": ".",
    "pathfinder": "cuda_pathfinder",
    "bindings": "cuda_bindings",
    "core": "cuda_core",
    "metapackage": "cuda_python",
    "test-helpers": "cuda_python_test_helpers",
}
EXECUTION_TAG_TARGETS = {
    "ci-wheel-current": {"bindings:wheel-current", "core:wheel-current"},
    "ci-build-cython-assets": {
        "bindings:cython-test-assets",
        "core:cython-test-assets",
    },
    "ci-sdist": {"pathfinder:sdist", "bindings:sdist", "core:sdist", "metapackage:sdist"},
    "ci-test-linux": {
        "pathfinder:test-installed-linux",
        "pathfinder:prepare-strict-linux",
        "pathfinder:test-installed-linux-strict",
        "bindings:test-installed-linux",
        "core:test-installed-linux",
        "metapackage:test-installed-linux",
        "bindings:smoke-linux",
    },
    "ci-test-windows": {
        "pathfinder:test-installed-windows",
        "pathfinder:prepare-strict-windows",
        "pathfinder:test-installed-windows-strict",
        "bindings:test-installed-windows",
        "core:test-installed-windows",
        "metapackage:test-installed-windows",
    },
    "ci-docs": {
        "pathfinder:docs-ci",
        "bindings:docs-ci",
        "core:docs-ci",
        "metapackage:docs-ci",
        "root:docs-ci",
    },
    "ci-quality": {
        "root:quality-moon-contracts",
        "core:quality-api-base",
        "core:quality-api-release",
        "bindings:unit-test",
    },
}
RUNNER_TAG_TARGETS = {
    "runner-build-linux-64": {
        "pathfinder:wheel-pure",
        "bindings:wheel-current",
        "core:wheel-current",
        "bindings:cython-test-assets",
        "core:cython-test-assets",
        "core:wheel-previous",
        "core:test-binaries",
        "core:wheel-merge",
    },
    "runner-sdist-linux": EXECUTION_TAG_TARGETS["ci-sdist"],
    "runner-sdist-windows": EXECUTION_TAG_TARGETS["ci-sdist"],
    "runner-test-linux": EXECUTION_TAG_TARGETS["ci-test-linux"],
    "runner-test-windows": EXECUTION_TAG_TARGETS["ci-test-windows"],
    "runner-docs": EXECUTION_TAG_TARGETS["ci-docs"],
    "runner-quality": EXECUTION_TAG_TARGETS["ci-quality"],
}
RUNNER_TAG_TARGETS["runner-build-linux-aarch64"] = RUNNER_TAG_TARGETS["runner-build-linux-64"]
RUNNER_TAG_TARGETS["runner-build-windows"] = RUNNER_TAG_TARGETS["runner-build-linux-64"]

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
FINGERPRINTED_TARGETS = set(CACHED_OUTPUTS)


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
        self.assertEqual({project_id: project["source"] for project_id, project in by_id.items()}, EXPECTED_PROJECTS)
        for project in by_id.values():
            self.assertEqual(project["language"], "unknown")
            self.assertEqual(project["toolchains"], ["system"])

    def test_only_force_all_tasks_are_allocation_only(self) -> None:
        self.assertFalse({task["target"] for task in self.tasks if "ci-gate" in task.get("tags", [])})
        forced = {task["target"] for task in self.tasks if "ci-force-all" in task.get("tags", [])}
        self.assertEqual(forced, {"root:force-all", "root:force-all-unowned"})
        for target in forced:
            task = self.by_target[target]
            self.assertEqual(task["command"], "noop")
            self.assertFalse(task["options"]["cache"])
            self.assertTrue(task["options"]["runInCI"])
            self.assertFalse(task.get("outputs"))

    def test_precise_inputs_are_owned_without_hiding_new_paths(self) -> None:
        def affected(path: str) -> set[str]:
            result = subprocess.run(  # noqa: S603 - the binary is explicitly selected in setUpClass.
                [
                    self.moon,
                    "query",
                    "tasks",
                    "--affected",
                    "stdin",
                    "--upstream",
                    "none",
                    "--downstream",
                    "deep",
                ],
                cwd=REPO_ROOT,
                check=True,
                input=path,
                text=True,
                stdout=subprocess.PIPE,
            )
            queried = json.loads(result.stdout)
            return {task["target"] for project in queried["tasks"].values() for task in project.values()}

        known = affected("benchmarks/cuda_bindings/tests/test_runner.py")
        self.assertIn("bindings:unit-test", known)
        self.assertNotIn("root:force-all-unowned", known)
        self.assertIn(
            "root:force-all-unowned",
            affected("benchmarks/cuda_bindings/new_helper.py"),
        )
        quality = affected("ci/tools/tests/test_moon_tasks.py")
        self.assertIn("root:quality-moon-contracts", quality)
        self.assertNotIn("root:force-all-unowned", quality)

    def test_bindings_benchmark_smoke_uses_materialized_wheels(self) -> None:
        task = self.by_target["bindings:smoke-linux"]
        self.assertIn("printenv SKIP_CUDA_BINDINGS_TEST", task["script"])
        self.assertIn("cuda_pathfinder/.moon-out/wheel-pure/*.whl", task["script"])
        self.assertIn("cuda_bindings/.moon-out/wheel-current/*.whl", task["script"])
        self.assertGreaterEqual(task["script"].count("[[ $# -eq 1 ]]"), 2)
        self.assertIn("benchmarks/cuda_bindings/run_pyperf.py", task["script"])
        self.assertNotIn("moon_ci.py", str(task["inputs"]))

    def test_execution_and_runner_tags_select_real_tasks(self) -> None:
        for tag, expected in {**EXECUTION_TAG_TARGETS, **RUNNER_TAG_TARGETS}.items():
            selected = {task["target"] for task in self.tasks if tag in task.get("tags", [])}
            self.assertEqual(selected, expected, tag)
            for target in selected:
                self.assertTrue(self.by_target[target]["options"]["runInCI"], target)

    def test_cached_producers_have_explicit_non_overlapping_outputs(self) -> None:
        cached = {task["target"] for task in self.tasks if task["options"]["cache"]}
        self.assertEqual(cached, set(CACHED_OUTPUTS))
        destinations: set[tuple[str, str]] = set()
        for target, output in CACHED_OUTPUTS.items():
            task = self.by_target[target]
            self.assertEqual(task["outputs"], [{"file": output}])
            self.assertTrue(task.get("inputs"))
            project = target.split(":", maxsplit=1)[0]
            self.assertNotIn((project, output), destinations)
            destinations.add((project, output))
        for target in FINGERPRINTED_TARGETS:
            task = self.by_target[target]
            self.assertTrue(task.get("checks"), target)
            scripts = [check["script"] for check in task["checks"]]
            self.assertTrue(any("git describe" in script for script in scripts), target)
            self.assertTrue(any("SETUPTOOLS_SCM_" in script for script in scripts), target)
            self.assertTrue(any("python_implementation" in script for script in scripts), target)
            self.assertFalse(task.get("inputEnv"), target)
            self.assertNotIn("moon_fingerprint.py", json.dumps(task), target)
            self.assertNotIn("ACTIONS_RUNTIME", "\n".join(scripts), target)

        for target in ("bindings:wheel-current", "core:wheel-current", "core:wheel-previous"):
            scripts = "\n".join(check["script"] for check in self.by_target[target]["checks"])
            self.assertIn("CUDA_PYTHON_COVERAGE", scripts, target)
            self.assertIn("name.startswith('CIBW_')", scripts, target)
            self.assertIn("ACTIONS_VALUE=<redacted>", scripts, target)
            self.assertIn("hashlib.sha256", scripts, target)

    def test_artifact_commands_are_encoded_in_moon(self) -> None:
        artifact_commands = {
            "pathfinder:wheel-pure": "python -m pip wheel",
            "pathfinder:sdist": "python -m build --sdist",
            "bindings:wheel-current": "python -m cibuildwheel",
            "bindings:sdist": "python -m build --sdist",
            "core:wheel-current": "python -m cibuildwheel",
            "core:wheel-previous": "python -m cibuildwheel",
            "core:sdist": "python -m build --sdist",
            "metapackage:wheel-pure": "python -m pip wheel",
            "metapackage:sdist": "python -m build --sdist",
        }
        for target, expected_command in artifact_commands.items():
            task = self.by_target[target]
            self.assertEqual(task["command"], "bash")
            self.assertFalse(task["options"]["shell"], target)
            self.assertEqual(task["args"][:3], ["-euo", "pipefail", "-c"])
            self.assertIn(expected_command, task["args"][3])
            self.assertIn(".moon-out/", task["args"][3])
            self.assertIn("[[ $# -eq 1 ]]", task["args"][3])

        metapackage = self.by_target["metapackage:wheel-pure"]
        self.assertIn({"glob": "/cuda_bindings/.moon-out/wheel-current/*.whl", "cache": True}, metapackage["inputs"])
        self.assertIn("CUDA_PYTHON_USE_STAGED_BINDINGS_VERSION", metapackage["args"][3])
        self.assertIn("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_PYTHON", metapackage["args"][3])

    def test_metapackage_uses_staged_bindings_version_only_when_requested(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        script = self.by_target["metapackage:wheel-pure"]["args"][3]

        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            bindings = workspace / "cuda_bindings" / ".moon-out" / "wheel-current"
            bindings.mkdir(parents=True)
            (bindings / "stale.whl").touch()

            command_log = workspace / "commands.txt"
            fake_python = workspace / "bin" / "python"
            fake_python.parent.mkdir()
            fake_python.write_text(
                """#!/usr/bin/env bash
printf '%s|%s|%s\n' "$*" "${SETUPTOOLS_SCM_PRETEND_VERSION-}" "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_PYTHON-}" >> "$COMMAND_LOG"
if [[ "$1" == "-c" ]]; then
  printf '%s\n' '13.3.2.dev1'
  exit 0
fi
mkdir -p cuda_python/.moon-out/wheel-pure
touch cuda_python/.moon-out/wheel-pure/cuda_python.whl
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            def run(mode: str | None) -> subprocess.CompletedProcess[str]:
                command_log.unlink(missing_ok=True)
                environment = {
                    **os.environ,
                    "COMMAND_LOG": str(command_log),
                    "PATH": f"{fake_python.parent}{os.pathsep}{os.environ['PATH']}",
                }
                environment.pop("CUDA_PYTHON_USE_STAGED_BINDINGS_VERSION", None)
                if mode is not None:
                    environment["CUDA_PYTHON_USE_STAGED_BINDINGS_VERSION"] = mode
                return subprocess.run(  # noqa: S603
                    [bash, "-euo", "pipefail", "-c", script],
                    cwd=workspace,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )

            local = run(None)
            self.assertEqual(local.returncode, 0, local.stderr)
            self.assertEqual(len(command_log.read_text(encoding="utf-8").splitlines()), 1)

            staged = run("1")
            self.assertEqual(staged.returncode, 0, staged.stderr)
            staged_commands = command_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(staged_commands), 2)
            self.assertTrue(staged_commands[-1].endswith("|13.3.2.dev1|13.3.2.dev1"))

            invalid = run("true")
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("must be unset or 1", invalid.stderr)

    def test_explicit_commands_do_not_use_moons_extra_shell_wrapper(self) -> None:
        for task in self.tasks:
            if task["command"] not in {"noop", "set"}:
                self.assertFalse(task["options"]["shell"], task["target"])

    def test_embedded_bash_preserves_runtime_variables_for_moon(self) -> None:
        scripts: dict[str, str] = {}
        for target, task in self.by_target.items():
            if task.get("script"):
                scripts[target] = task["script"]
            elif task.get("command") == "bash" and task.get("args", [])[:3] == ["-euo", "pipefail", "-c"]:
                scripts[target] = task["args"][3]
        for target, script in scripts.items():
            self.assertNotIn("${#", script, target)
            self.assertNotIn("!}", script, target)
            self.assertNotRegex(script, r"\$\{[^}]*\[[^}]*\}", target)
            self.assertNotRegex(script, r"\$\{[^}]*%[^}]*\}", target)
            self.assertNotRegex(script, r"\$\{[^}]*:-[^}]*\}", target)
            self.assertNotRegex(script, r"\$[a-z_]", target)

        runtime_locals = {
            "pathfinder:sdist": ("$ARCHIVE",),
            "bindings:wheel-current": ("$OUTPUT", "$PATHFINDER_WHEEL"),
            "bindings:sdist": ("$CONSTRAINT_FILE", "$ARCHIVE"),
            "bindings:smoke-linux": ("$SKIP", "$BINDINGS_WHEEL"),
            "core:wheel-current": ("$OUTPUT", "$WHEEL_WITHOUT_SUFFIX"),
            "core:wheel-previous": ("$OUTPUT", "$WHEEL_WITHOUT_SUFFIX"),
            "core:sdist": ("$CONSTRAINT_FILE", "$ARCHIVE"),
            "metapackage:sdist": ("$ARCHIVE",),
            "test-helpers:prepare-test-assets": ("$PATHFINDER_WHEEL", "$CORE_WHEEL"),
        }
        for target, markers in runtime_locals.items():
            for marker in markers:
                self.assertIn(marker, scripts[target], target)

        for target, project in {
            "pathfinder:test-installed-linux": "pathfinder",
            "pathfinder:test-installed-linux-strict": "pathfinder",
            "pathfinder:test-installed-windows": "pathfinder",
            "pathfinder:test-installed-windows-strict": "pathfinder",
            "bindings:test-installed-linux": "bindings",
            "bindings:test-installed-windows": "bindings",
            "core:test-installed-linux": "core",
            "core:test-installed-windows": "core",
            "metapackage:test-installed-linux": "metapackage",
            "metapackage:test-installed-windows": "metapackage",
        }.items():
            task = self.by_target[target]
            if task["command"] != "noop":
                self.assertEqual(task["command"], "bash")
                self.assertEqual(task["args"], ["ci/tools/run-tests", project])

        preparation = self.by_target["test-helpers:prepare-test-assets"]
        self.assertEqual(preparation["command"], "bash")
        self.assertEqual(preparation["args"][:3], ["-euo", "pipefail", "-c"])
        self.assertIn("python -m pip install", preparation["args"][3])
        self.assertIn("--clean-output", self.by_target["core:wheel-merge"]["args"])
        self.assertIn("--output-dir", self.by_target["core:test-binaries"]["args"])
        for target in ("bindings:cython-test-assets", "core:cython-test-assets"):
            self.assertEqual(self.by_target[target]["command"], "bash")
            self.assertIn("--output-dir", self.by_target[target]["args"])

    def test_same_environment_build_dependencies_use_output_bytes(self) -> None:
        expected = {
            "bindings:wheel-current": {"pathfinder:wheel-pure"},
            "core:wheel-current": {"bindings:wheel-current"},
            "core:wheel-previous": {"pathfinder:wheel-pure"},
            "bindings:sdist": {"pathfinder:sdist"},
            "core:sdist": {"pathfinder:sdist", "bindings:sdist"},
            "metapackage:sdist": {"bindings:sdist"},
            "metapackage:test-installed-linux": {"metapackage:wheel-pure"},
            "metapackage:test-installed-windows": {"metapackage:wheel-pure"},
            "metapackage:docs-ci": {"metapackage:wheel-pure"},
            "root:docs-ci": {
                "pathfinder:docs-ci",
                "bindings:docs-ci",
                "core:docs-ci",
                "metapackage:docs-ci",
            },
        }
        for target, dependencies in expected.items():
            configured = {dep["target"] for dep in self.by_target[target]["deps"]}
            self.assertEqual(configured, dependencies, target)
            self.assertTrue(all(dep["cacheStrategy"] == "outputs" for dep in self.by_target[target]["deps"]), target)

    def test_native_asset_preparation_is_shared(self) -> None:
        prep = self.by_target["test-helpers:prepare-test-assets"]
        self.assertFalse(prep["options"]["cache"])
        for target in ("bindings:cython-test-assets", "core:cython-test-assets"):
            deps = {dep["target"] for dep in self.by_target[target]["deps"]}
            self.assertEqual(deps, {"test-helpers:prepare-test-assets"})

    def test_platform_test_tasks_are_serialized_and_os_scoped(self) -> None:
        for tag, operating_system in (("ci-test-linux", "linux"), ("ci-test-windows", "windows")):
            for target in EXECUTION_TAG_TARGETS[tag]:
                options = self.by_target[target]["options"]
                self.assertEqual(options.get("mutex"), "ci-python-gpu", target)
                self.assertEqual(options.get("os"), [operating_system], target)

        # Package tests install their own prerequisites. Cross-package deps
        # would force unaffected test suites to run merely to serialize the
        # shared interpreter; the mutex provides that serialization instead.
        for target in EXECUTION_TAG_TARGETS["ci-test-linux"] | EXECUTION_TAG_TARGETS["ci-test-windows"]:
            if not target.startswith("pathfinder:") and not target.startswith("metapackage:"):
                self.assertFalse(self.by_target[target].get("deps"), target)

    def test_pathfinder_strictness_and_preparation_are_in_the_graph(self) -> None:
        for operating_system in ("linux", "windows"):
            normal = self.by_target[f"pathfinder:test-installed-{operating_system}"]
            prepare = self.by_target[f"pathfinder:prepare-strict-{operating_system}"]
            strict = self.by_target[f"pathfinder:test-installed-{operating_system}-strict"]
            self.assertEqual(normal["env"]["CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS"], "see_what_works")
            self.assertEqual(strict["env"]["CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS"], "all_must_work")
            self.assertEqual({dep["target"] for dep in prepare["deps"]}, {normal["target"]})
            self.assertEqual({dep["target"] for dep in strict["deps"]}, {prepare["target"]})

    def test_platform_test_tasks_track_provider_setup(self) -> None:
        for target in EXECUTION_TAG_TARGETS["ci-test-linux"]:
            inputs = self.by_target[target]["inputs"]
            if "prepare-strict" not in target:
                self.assertIn({"file": "/ci/tools/guess_latest.sh"}, inputs, target)
                self.assertIn({"file": "/ci/tools/install_gpu_driver.sh"}, inputs, target)
        for target in EXECUTION_TAG_TARGETS["ci-test-windows"]:
            inputs = self.by_target[target]["inputs"]
            if "prepare-strict" not in target:
                self.assertIn({"file": "/ci/tools/configure_driver_mode.ps1"}, inputs, target)
                self.assertIn({"file": "/ci/tools/install_gpu_driver.ps1"}, inputs, target)

    def test_docs_components_run_in_parallel_before_assembly(self) -> None:
        docs = self.by_target["root:docs-ci"]
        self.assertFalse(docs["options"]["cache"])
        self.assertTrue(docs["options"]["runDepsInParallel"])
        self.assertEqual(docs["command"], "bash")
        self.assertEqual(docs["args"], ["cuda_python/docs/assemble_moon_docs.sh"])
        root_inputs = docs["inputs"]
        self.assertIn({"project": "core", "group": "package"}, root_inputs)
        self.assertIn({"project": "metapackage", "group": "docs"}, root_inputs)
        self.assertIn({"file": "/.github/workflows/build-wheel.yml"}, root_inputs)
        for target in EXECUTION_TAG_TARGETS["ci-docs"] - {"root:docs-ci"}:
            task = self.by_target[target]
            self.assertFalse(task["options"]["cache"])
            self.assertEqual(task["command"], "bash")
            self.assertEqual(task["args"][-1], "moon-ci")
            self.assertIn({"file": "/cuda_python/docs/environment-docs.yml"}, task["inputs"])

        metapackage_inputs = self.by_target["metapackage:docs-ci"]["inputs"]
        for project in (
            "pathfinder",
            "bindings",
            "core",
        ):
            self.assertIn({"project": project, "group": "package"}, metapackage_inputs)

    def test_quality_tasks_use_external_refs_and_one_selector(self) -> None:
        release = self.by_target["core:quality-api-release"]
        base = self.by_target["core:quality-api-base"]
        self.assertIn("${CUDA_CORE_API_RELEASE_BASE}", release["args"])
        self.assertIn("${CUDA_CORE_API_MERGE_BASE}", base["args"])
        self.assertEqual(self.by_target["root:quality-moon-contracts"]["args"][:2], ["-m", "unittest"])
        for target in (release, base, self.by_target["bindings:unit-test"]):
            self.assertEqual(target["command"], "uvx")
            self.assertIn("--no-managed-python", target["args"])
            self.assertIn("--no-python-downloads", target["args"])

    def test_local_pixi_tasks_remain_available_and_skip_ci(self) -> None:
        for target in ("root:test", "root:docs", "root:pure-wheel"):
            self.assertFalse(self.by_target[target]["options"]["runInCI"])
        for target in ("pathfinder:test", "bindings:test", "core:test"):
            task = self.by_target[target]
            self.assertEqual(task["command"], "bash")
            self.assertEqual(task["args"][:3], ["-euo", "pipefail", "-c"])
            self.assertIn("PIXI_ENVIRONMENT_NAME", task["args"][3])
            self.assertIn("exec pixi", task["args"][3])
            self.assertFalse(task["options"]["runInCI"])
        for target in (
            "pathfinder:docs",
            "bindings:docs",
            "core:docs",
            "bindings:bench",
        ):
            task = self.by_target[target]
            self.assertEqual(task["command"], "pixi")
            self.assertFalse(task["options"]["runInCI"])

    def test_moon_task_helpers_are_removed(self) -> None:
        removed = (
            "artifacts.py",
            "build_artifacts.py",
            "moon_ci.py",
            "moon_fingerprint.py",
            "prepare_test_assets.py",
            "run_pixi_test.py",
        )
        for filename in removed:
            self.assertFalse((REPO_ROOT / "ci" / "tools" / filename).exists(), filename)
        for task in self.tasks:
            serialized = json.dumps(task)
            for filename in removed:
                self.assertNotIn(filename, serialized, task["target"])

    def test_universal_wheels_share_existing_runner_lanes(self) -> None:
        self.assertFalse((REPO_ROOT / ".github" / "workflows" / "build-pure-wheel.yml").exists())
        self.assertFalse({task["target"] for task in self.tasks if "ci-wheel-pure" in task.get("tags", [])})
        self.assertFalse({task["target"] for task in self.tasks if "runner-build-portable" in task.get("tags", [])})
        pathfinder = self.by_target["pathfinder:wheel-pure"]
        self.assertTrue(
            {"runner-build-linux-64", "runner-build-linux-aarch64", "runner-build-windows"} <= set(pathfinder["tags"])
        )
        self.assertFalse(
            {tag for tag in self.by_target["metapackage:wheel-pure"].get("tags", []) if tag.startswith("runner-")}
        )
        self.assertNotIn(
            "build-pure-wheel.yml",
            "\n".join(path.read_text(encoding="utf-8") for path in REPO_ROOT.rglob("moon.yml")),
        )
        native_workflow = (REPO_ROOT / ".github" / "workflows" / "build-wheel.yml").read_text(encoding="utf-8")
        self.assertIn("cuda_pathfinder/.moon-out/wheel-pure", native_workflow)
        self.assertIn("cuda_python/.moon-out/wheel-pure", native_workflow)
        for relative_path in (
            ".github/workflows/build-wheel.yml",
            ".github/workflows/build-docs.yml",
            ".github/workflows/test-wheel-linux.yml",
            ".github/workflows/test-wheel-windows.yml",
        ):
            workflow = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn('CUDA_PYTHON_USE_STAGED_BINDINGS_VERSION: "1"', workflow, relative_path)

    def test_workspace_disables_python_and_dependency_management(self) -> None:
        workspace = (REPO_ROOT / ".moon" / "workspace.yml").read_text(encoding="utf-8")
        self.assertIn("versionConstraint: '=2.5.1'", workspace)
        self.assertIn("installDependencies: false", workspace)
        self.assertIn("syncProjects: false", workspace)
        self.assertIn("syncWorkspace: false", workspace)
        self.assertIn("verifyIntegrity: true", workspace)
        self.assertNotIn("experiments:", workspace)
        self.assertNotIn("remoteCandidates:", workspace)
        self.assertFalse((REPO_ROOT / ".moon" / "toolchains.yml").exists())

    def test_generated_cache_and_output_roots_are_ignored(self) -> None:
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".moon/cache/", ignore)
        self.assertIn(".moon-out/", ignore)


if __name__ == "__main__":
    unittest.main()
