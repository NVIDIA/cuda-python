# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# These tests intentionally use stdlib unittest so Moon's contract task does
# not need a separately managed Python test environment.
# ruff: noqa: PT009, PT027

from __future__ import annotations

import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

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
    "ci-build-native": {
        "pathfinder:wheel-pure",
        "bindings:wheel-current",
        "core:wheel-current",
        "bindings:cython-test-assets",
        "core:cython-test-assets",
        "core:wheel-previous",
        "core:test-binaries",
        "core:wheel-merge",
    },
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

INTERNAL_FINGERPRINT_TARGETS = {
    *(f"{project}:fingerprint-package" for project in ("pathfinder", "bindings", "core", "metapackage")),
    "test-helpers:fingerprint-python-context",
    "test-helpers:fingerprint-python-build",
    "test-helpers:fingerprint-native-context",
    "test-helpers:fingerprint-test-assets",
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
FINGERPRINTED_TARGETS = set(CACHED_OUTPUTS)
SCM_FINGERPRINTED_TARGETS = FINGERPRINTED_TARGETS - {
    "bindings:cython-test-assets",
    "core:cython-test-assets",
    "core:test-binaries",
    "core:wheel-merge",
}
NATIVE_FINGERPRINTED_TARGETS = {
    "bindings:wheel-current",
    "bindings:sdist",
    "bindings:cython-test-assets",
    "core:wheel-current",
    "core:wheel-previous",
    "core:sdist",
    "core:cython-test-assets",
    "core:test-binaries",
}


class MoonWorkspaceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.moon = os.environ.get("MOON_BIN") or shutil.which("moon")
        if not cls.moon:
            raise unittest.SkipTest("Moon is not installed; set MOON_BIN to test the workspace")
        cls.tasks = cls.moon_json("tasks", "--json")
        cls.tasks.extend(cls.moon_json("task", target, "--json") for target in INTERNAL_FINGERPRINT_TARGETS)
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
        def affected(
            path: str,
            *,
            upstream: str = "none",
            downstream: str = "deep",
        ) -> set[str]:
            arguments = [
                self.moon,
                "query",
                "tasks",
                "--affected",
                "stdin",
                "--upstream",
                upstream,
                "--downstream",
                downstream,
            ]
            result = subprocess.run(  # noqa: S603 - the binary is explicitly selected in setUpClass.
                arguments,
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
        nightly = affected(".github/workflows/ci-nightly.yml")
        self.assertIn("root:quality-moon-contracts", nightly)
        self.assertNotIn("root:force-all-unowned", nightly)
        for quality_input in (
            ".gitignore",
            ".github/workflows/build-docs.yml",
            ".github/workflows/build-wheel.yml",
            ".github/workflows/ci-nightly.yml",
            ".github/workflows/ci.yml",
            ".github/workflows/test-sdist-linux.yml",
            ".github/workflows/test-sdist-windows.yml",
            ".github/workflows/test-wheel-linux.yml",
            ".github/workflows/test-wheel-windows.yml",
            "ci/tools/env-vars",
            "ci/build-matrix.yml",
            "ci/test-matrix.yml",
            "ci/tools/merge_cuda_core_wheels.py",
            "ci/tools/run-tests",
            "cuda_bindings/tests/cython/build_tests.py",
            "cuda_bindings/docs/build_docs.sh",
            "cuda_core/tests/cython/build_tests.py",
            "cuda_core/docs/build_docs.sh",
            "cuda_pathfinder/docs/build_docs.sh",
            "cuda_python/docs/assemble_moon_docs.sh",
            "cuda_python/docs/build_component_docs.sh",
            "cuda_python/docs/build_docs.sh",
            "cuda_python/docs/environment-docs.yml",
            "cuda_python_test_helpers/cuda_python_test_helpers/cython_test_builder.py",
        ):
            self.assertIn("root:quality-moon-contracts", affected(quality_input), quality_input)

        metapackage_only = affected("cuda_python/pyproject.toml")
        self.assertTrue(
            {
                "metapackage:wheel-pure",
                "metapackage:sdist",
                "metapackage:test-installed-linux",
                "metapackage:test-installed-windows",
                "metapackage:docs-ci",
            }
            <= metapackage_only,
        )
        self.assertFalse(
            any("ci-build-native" in self.by_target[target].get("tags", []) for target in metapackage_only),
        )

        current_wheels = {"bindings:wheel-current", "core:wheel-current"}
        pathfinder_direct = affected("cuda_pathfinder/cuda/__init__.py", downstream="none")
        bindings_direct = affected("cuda_bindings/cuda/__init__.py", downstream="none")
        self.assertTrue(
            (
                current_wheels
                | {
                    "bindings:sdist",
                    "core:sdist",
                    "core:wheel-previous",
                    "metapackage:sdist",
                    "metapackage:wheel-pure",
                }
            )
            <= pathfinder_direct
        )
        self.assertTrue(
            (current_wheels | {"core:sdist", "metapackage:sdist", "metapackage:wheel-pure"}) <= bindings_direct
        )

        for path, target in (
            ("cuda_pathfinder/docs/source/index.rst", "pathfinder:sdist"),
            ("cuda_bindings/tests/test_basics.py", "bindings:sdist"),
            ("cuda_core/README.md", "core:sdist"),
            ("cuda_python/docs/source/index.rst", "metapackage:sdist"),
        ):
            self.assertIn(target, affected(path, downstream="none"), path)

    def test_native_build_matrix_is_direct_and_covers_test_python_versions(self) -> None:
        matrix_text = (REPO_ROOT / "ci" / "build-matrix.yml").read_text(encoding="utf-8")
        matrix = json.loads("\n".join(line for line in matrix_text.splitlines() if not line.lstrip().startswith("#")))
        self.assertEqual(set(matrix), {"include"})
        rows = matrix["include"]
        self.assertIsInstance(rows, list)
        self.assertTrue(rows)

        versions: set[str] = set()
        formatted_versions: set[str] = set()
        for row in rows:
            self.assertEqual(set(row), {"python-version", "python-version-formatted"})
            version = row["python-version"]
            formatted = row["python-version-formatted"]
            self.assertIsInstance(version, str)
            self.assertIsInstance(formatted, str)
            self.assertRegex(version, r"^3\.(?:0|[1-9][0-9]*)t?$")
            self.assertRegex(formatted, r"^3[0-9]+t?$")
            self.assertEqual(formatted, version.replace(".", ""))
            self.assertNotIn(version, versions)
            self.assertNotIn(formatted, formatted_versions)
            versions.add(version)
            formatted_versions.add(formatted)

        self.assertIn("3.12", versions)
        test_matrix = (REPO_ROOT / "ci" / "test-matrix.yml").read_text(encoding="utf-8")
        test_versions = set(re.findall(r"\bPY_VER:\s*'([^']+)'", test_matrix))
        self.assertTrue(test_versions)
        self.assertLessEqual(test_versions, versions)

    def test_bindings_benchmark_smoke_uses_materialized_wheels(self) -> None:
        task = self.by_target["bindings:smoke-linux"]
        self.assertIn("printenv SKIP_CUDA_BINDINGS_TEST", task["script"])
        self.assertIn("cuda_pathfinder/.moon-out/wheel-pure/*.whl", task["script"])
        self.assertIn("cuda_bindings/.moon-out/wheel-current/*.whl", task["script"])
        self.assertGreaterEqual(task["script"].count("[[ $# -eq 1 ]]"), 2)
        self.assertIn("benchmarks/cuda_bindings/run_pyperf.py", task["script"])
        self.assertNotIn("moon_ci.py", str(task["inputs"]))

    def test_semantic_execution_tags_select_real_tasks(self) -> None:
        for tag, expected in EXECUTION_TAG_TARGETS.items():
            selected = {task["target"] for task in self.tasks if tag in task.get("tags", [])}
            self.assertEqual(selected, expected, tag)
            for target in selected:
                self.assertTrue(self.by_target[target]["options"]["runInCI"], target)
        self.assertFalse({tag for task in self.tasks for tag in task.get("tags", []) if tag.startswith("runner-")})

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
            self.assertFalse(task.get("checks"), target)
            self.assertNotIn("cacheKey", task["options"], target)
            closure: set[str] = set()
            pending = [dependency["target"] for dependency in task.get("deps", [])]
            while pending:
                dependency = pending.pop()
                if dependency in closure:
                    continue
                closure.add(dependency)
                pending.extend(dep["target"] for dep in self.by_target[dependency].get("deps", []))
            fingerprint_tasks = [self.by_target[dependency] for dependency in closure if "fingerprint" in dependency]
            self.assertTrue(fingerprint_tasks, target)
            scripts = [
                check["script"]
                for fingerprint_task in fingerprint_tasks
                for check in fingerprint_task.get("checks", [])
            ]
            self.assertTrue(any("SETUPTOOLS_SCM_" in script for script in scripts), target)
            self.assertTrue(any("python_implementation" in script for script in scripts), target)
            if target in SCM_FINGERPRINTED_TARGETS:
                self.assertTrue(any("'git', 'describe'" in script for script in scripts), target)
            self.assertFalse(task.get("inputEnv"), target)
            self.assertNotIn("moon_fingerprint.py", json.dumps(task), target)
            self.assertNotIn("ACTIONS_RUNTIME", "\n".join(scripts), target)

        for target in NATIVE_FINGERPRINTED_TARGETS:
            closure: set[str] = set()
            pending = [dep["target"] for dep in self.by_target[target]["deps"]]
            while pending:
                dependency = pending.pop()
                if dependency in closure:
                    continue
                closure.add(dependency)
                pending.extend(dep["target"] for dep in self.by_target[dependency].get("deps", []))
            scripts = "\n".join(
                check["script"] for dependency in closure for check in self.by_target[dependency].get("checks", [])
            )
            self.assertIn("CUDA_PYTHON_COVERAGE", scripts, target)
            self.assertIn("CUDA_HOME", scripts, target)
            self.assertIn("CFLAGS", scripts, target)
            self.assertIn("LDFLAGS", scripts, target)
            self.assertIn("name.startswith('CIBW_')", scripts, target)
            self.assertIn("ACTIONS_VALUE=<redacted>", scripts, target)
            self.assertIn("hashlib.sha256", scripts, target)

        fingerprint_tasks = [task for task in self.tasks if "fingerprint" in task["target"]]
        self.assertEqual({task["target"] for task in fingerprint_tasks}, INTERNAL_FINGERPRINT_TARGETS)
        for task in fingerprint_tasks:
            self.assertEqual(task["command"], "noop", task["target"])
            self.assertFalse(task["options"]["cache"], task["target"])
            self.assertTrue(task["options"]["internal"], task["target"])
            self.assertTrue(task["options"]["runInCI"], task["target"])
            self.assertTrue(task.get("checks"), task["target"])

        expected_patterns = {
            "pathfinder": "cuda-pathfinder-v*[0-9]*",
            "bindings": "v*[0-9]*",
            "core": "cuda-core-v*[0-9]*",
            "metapackage": "v*[0-9]*",
        }
        for project, pattern in expected_patterns.items():
            script = self.by_target[f"{project}:fingerprint-package"]["checks"][0]["script"]
            self.assertIn(pattern, script)
            self.assertNotIn("CUDA_PYTHON_SCM_TAG_PATTERN", script)

    def test_fingerprint_checks_execute_without_task_environment(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        scripts = {
            check["script"]
            for target in INTERNAL_FINGERPRINT_TARGETS
            for check in self.by_target[target].get("checks", [])
        }
        for script in scripts:
            result = subprocess.run(  # noqa: S603 - Bash executes checked-in Moon configuration.
                [bash, "-c", script],
                cwd=REPO_ROOT,
                check=False,
                env=os.environ,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, f"{script}\n{result.stdout}")

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
            self.assertIn(".moon-out", task["args"][3])
            self.assertIn("[[ $# -eq 1 ]]", task["args"][3])
            self.assertIn({"file": "/ci/build-constraints.txt"}, task["inputs"])
            self.assertIn("ci/build-constraints.txt", task["args"][3])
            self.assertIn("PIP_BUILD_CONSTRAINT", task["args"][3])
            self.assertIn("PIP_CONSTRAINT", task["args"][3])
            self.assertIn("SOURCE_DATE_EPOCH=$(printenv SOURCE_DATE_EPOCH || true)", task["args"][3])
            self.assertIn("git log -1 --format=%ct HEAD", task["args"][3])
            if target.endswith(":sdist"):
                self.assertIn("@group(sdist)", task["inputs"])
                self.assertNotIn("@group(package)", task["inputs"])

        for target in ("bindings:wheel-current", "core:wheel-current", "core:wheel-previous"):
            self.assertIn("SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH", self.by_target[target]["args"][3])

        metapackage = self.by_target["metapackage:wheel-pure"]
        self.assertIn({"glob": "/cuda_bindings/.moon-out/wheel-current/*.whl", "cache": True}, metapackage["inputs"])
        self.assertIn("CUDA_PYTHON_USE_STAGED_BINDINGS_VERSION", metapackage["args"][3])
        self.assertIn("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_PYTHON", metapackage["args"][3])

        current = self.by_target["core:wheel-current"]
        previous = self.by_target["core:wheel-previous"]
        self.assertEqual(current["args"], previous["args"])
        self.assertEqual(current["env"]["CUDA_PYTHON_WHEEL_VARIANT"], "current")
        self.assertEqual(previous["env"]["CUDA_PYTHON_WHEEL_VARIANT"], "previous")

    def test_metapackage_uses_staged_bindings_version_only_when_requested(self) -> None:
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        assert bash is not None
        script = self.by_target["metapackage:wheel-pure"]["args"][3]

        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            constraints = workspace / "ci" / "build-constraints.txt"
            constraints.parent.mkdir()
            constraints.write_text("setuptools==83.0.0\n", encoding="utf-8")
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
                    "SOURCE_DATE_EPOCH": "1234567890",
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

    def test_cython_asset_builds_isolate_generated_sources_in_moon_output(self) -> None:
        def exercise_builder(project: str) -> None:
            with tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                script_dir = root / project / "tests" / "cython"
                script_dir.mkdir(parents=True)
                script = script_dir / "build_tests.py"
                shutil.copy2(REPO_ROOT / project / "tests" / "cython" / "build_tests.py", script)
                (script_dir / "test_probe.pyx").write_text("# cython probe\n", encoding="utf-8")

                bindings_init = root / "bindings-source" / "cuda" / "bindings" / "__init__.py"
                bindings_init.parent.mkdir(parents=True)
                bindings_init.touch()
                cuda_root = root / "cuda-toolkit"
                (cuda_root / "include").mkdir(parents=True)
                if project == "cuda_core":
                    (root / project / "cuda" / "core" / "_include").mkdir(parents=True)

                cython_calls: list[dict[str, Any]] = []
                setup_calls: list[list[str]] = []

                def fake_cythonize(sources: list[str], **options: Any) -> list[str]:
                    cython_calls.append(dict(options))
                    source = Path(sources[0])
                    generated_dir = Path(options["build_dir"]) if "build_dir" in options else source.parent
                    if not source.is_absolute() and "build_dir" not in options:
                        generated_dir = Path.cwd()
                    generated_dir.mkdir(parents=True, exist_ok=True)
                    (generated_dir / "test_probe.cpp").write_text("// generated\n", encoding="utf-8")
                    return ["fake-extension"]

                def fake_setup(**_: Any) -> None:
                    setup_calls.append(sys.argv.copy())
                    if "--build-temp" in sys.argv:
                        build_temp = Path(sys.argv[sys.argv.index("--build-temp") + 1])
                        build_temp.mkdir(parents=True)
                    if "--build-lib" in sys.argv:
                        build_lib = Path(sys.argv[sys.argv.index("--build-lib") + 1])
                        build_lib.mkdir(parents=True, exist_ok=True)
                        (build_lib / "test_probe.fake.so").touch()

                cython_package = types.ModuleType("Cython")
                cython_build = types.ModuleType("Cython.Build")
                setuptools = types.ModuleType("setuptools")
                cuda = types.ModuleType("cuda")
                cuda_bindings = types.ModuleType("cuda.bindings")
                cython_package.__dict__["__path__"] = []
                cython_build.__dict__["cythonize"] = fake_cythonize
                setuptools.__dict__["setup"] = fake_setup
                cuda.__dict__["__path__"] = []
                cuda.__dict__["bindings"] = cuda_bindings
                cuda_bindings.__dict__["__file__"] = str(bindings_init)
                fake_modules = {
                    "Cython": cython_package,
                    "Cython.Build": cython_build,
                    "setuptools": setuptools,
                    "cuda": cuda,
                    "cuda.bindings": cuda_bindings,
                }

                def run_builder(*arguments: str) -> None:
                    original_argv = sys.argv
                    original_cwd = Path.cwd()
                    helper_source = str(REPO_ROOT / "cuda_python_test_helpers")
                    try:
                        sys.argv = [str(script), *arguments]
                        sys.path.insert(0, helper_source)
                        with (
                            mock.patch.dict(os.environ, {"CUDA_HOME": str(cuda_root)}),
                            mock.patch.dict(sys.modules, fake_modules),
                        ):
                            runpy.run_path(str(script), run_name="__main__")
                    finally:
                        sys.argv = original_argv
                        sys.path.remove(helper_source)
                        os.chdir(original_cwd)

                output = root / project / ".moon-out" / "cython-tests"
                run_builder("--output-dir", str(output))
                self.assertEqual(cython_calls[0]["build_dir"], str(output / ".cython-build"))
                self.assertFalse((output / ".cython-build").exists())
                self.assertFalse((output / ".build-temp").exists())
                self.assertFalse((script_dir / "test_probe.cpp").exists())
                self.assertIn("--build-lib", setup_calls[0])

                run_builder()
                self.assertNotIn("build_dir", cython_calls[1])
                self.assertIn("--inplace", setup_calls[1])
                self.assertTrue((script_dir / "test_probe.cpp").is_file())

        for project in ("cuda_bindings", "cuda_core"):
            with self.subTest(project=project):
                exercise_builder(project)

    def test_shared_cython_builder_confines_outputs_and_preserves_include_flags(self) -> None:
        helper_source = str(REPO_ROOT / "cuda_python_test_helpers")
        sys.path.insert(0, helper_source)
        try:
            from cuda_python_test_helpers.cython_test_builder import (
                _output_directory,
                _set_compiler_include_paths,
            )
        finally:
            sys.path.remove(helper_source)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            script_dir = root / "cuda_bindings" / "tests" / "cython"
            script_dir.mkdir(parents=True)
            output = root / "cuda_bindings" / ".moon-out" / "cython-tests"
            output.mkdir(parents=True)
            (output / "stale.so").touch()

            self.assertEqual(
                _output_directory(script_dir, "cuda_bindings/.moon-out/cython-tests"),
                output,
            )
            self.assertEqual(list(output.iterdir()), [])
            with self.assertRaisesRegex(ValueError, "output must be below"):
                _output_directory(script_dir, str(script_dir))

        posix_environment = {"CPLUS_INCLUDE_PATH": "/existing/include"}
        _set_compiler_include_paths(
            [Path("/core/include"), Path("/cuda/include")],
            environ=posix_environment,
            platform_name="posix",
        )
        self.assertEqual(
            posix_environment,
            {"CPLUS_INCLUDE_PATH": "/core/include:/cuda/include:/existing/include"},
        )

        windows_environment = {"CL": "/D EXISTING"}
        _set_compiler_include_paths(
            [Path("C:/core include"), Path("C:/CUDA/include")],
            environ=windows_environment,
            platform_name="nt",
        )
        self.assertEqual(
            windows_environment,
            {"CL": '/I"C:/core include" /I"C:/CUDA/include" /D EXISTING'},
        )

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
            if target.endswith(("windows", "windows-strict")) and task["command"] == "noop":
                continue
            self.assertEqual(task["command"], "bash")
            self.assertEqual(task["args"], ["ci/tools/run-tests", project])

        preparation = self.by_target["test-helpers:prepare-test-assets"]
        self.assertEqual(preparation["command"], "bash")
        self.assertEqual(preparation["args"][:3], ["-euo", "pipefail", "-c"])
        self.assertIn("python -m pip install", preparation["args"][3])
        self.assertIn("--clean-output", self.by_target["core:wheel-merge"]["args"])
        self.assertIn("--output-dir", self.by_target["core:test-binaries"]["args"])
        for target, driver in (
            ("bindings:cython-test-assets", "cuda_bindings/tests/cython/build_tests.py"),
            ("core:cython-test-assets", "cuda_core/tests/cython/build_tests.py"),
        ):
            task = self.by_target[target]
            self.assertEqual(task["command"], "python")
            self.assertEqual(task["args"][0], driver)
            self.assertIn("--output-dir", task["args"])
            self.assertEqual(task["env"]["PYTHONPATH"], "cuda_python_test_helpers")
            self.assertIn(
                {"file": "/cuda_python_test_helpers/cuda_python_test_helpers/cython_test_builder.py"},
                task["inputs"],
            )

    def test_same_environment_build_dependencies_use_output_bytes(self) -> None:
        expected = {
            "bindings:wheel-current": {"pathfinder:wheel-pure"},
            "core:wheel-current": {"pathfinder:wheel-pure", "bindings:wheel-current"},
            "core:wheel-previous": {"pathfinder:wheel-pure"},
            "core:wheel-merge": {"core:wheel-previous"},
            "bindings:sdist": {"pathfinder:sdist"},
            "core:sdist": {"pathfinder:sdist", "bindings:sdist"},
            "metapackage:sdist": set(),
            "metapackage:test-installed-linux": {"metapackage:wheel-pure"},
            "metapackage:test-installed-windows": {"metapackage:wheel-pure"},
            "root:docs-ci": {
                "pathfinder:docs-ci",
                "bindings:docs-ci",
                "core:docs-ci",
                "metapackage:docs-ci",
            },
        }
        for target, dependencies in expected.items():
            configured = {dep["target"] for dep in self.by_target[target]["deps"] if dep["cacheStrategy"] == "outputs"}
            self.assertEqual(configured, dependencies, target)

    def test_native_asset_preparation_is_shared(self) -> None:
        prep = self.by_target["test-helpers:prepare-test-assets"]
        fingerprint = self.by_target["test-helpers:fingerprint-test-assets"]
        native_context = self.by_target["test-helpers:fingerprint-native-context"]
        self.assertFalse(prep["options"]["cache"])
        self.assertIn("pip install --force-reinstall --no-deps", prep["args"][3])
        self.assertEqual(
            {dep["target"] for dep in fingerprint["deps"]},
            {prep["target"], native_context["target"]},
        )
        self.assertEqual(
            {dep["target"] for dep in native_context["deps"]},
            {"test-helpers:fingerprint-python-context"},
        )
        for target in ("bindings:cython-test-assets", "core:cython-test-assets"):
            deps = {dep["target"] for dep in self.by_target[target]["deps"]}
            self.assertEqual(deps, {fingerprint["target"]})

    def test_cibuildwheel_host_is_restored_after_target_assets(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "build-wheel.yml").read_text(encoding="utf-8")
        target_setup = workflow.index("id: setup-python2")
        target_install = workflow.index("- name: Install target-Python build tools", target_setup)
        target_assets = workflow.index("- name: Build target-Python test assets with Moon", target_install)
        host_restore = workflow.index("- name: Restore cibuildwheel host Python", target_assets)
        previous_build = workflow.index("- name: Build previous-CTK outputs and merge wheels with Moon", host_restore)

        self.assertIn("python-version: ${{ matrix.python-version }}", workflow[target_setup:target_install])
        self.assertNotIn("cibuildwheel", workflow[target_install:target_assets])
        self.assertIn('python-version: "3.12"', workflow[host_restore:previous_build])

    def test_workflows_stage_moon_phases_and_pin_standard_nightly_source(self) -> None:
        build = (REPO_ROOT / ".github" / "workflows" / "build-wheel.yml").read_text(encoding="utf-8")
        self.assertIn('MOON_CACHE: "off"', build)
        current_calls = (
            "moon ci pathfinder:wheel-pure",
            "moon ci bindings:wheel-current",
            'moon ci "${targets[@]}"',
        )
        current_positions = tuple(build.index(call) for call in current_calls)
        self.assertEqual(current_positions, tuple(sorted(current_positions)))
        previous = build.index("moon ci core:wheel-previous core:test-binaries")
        merge = build.index("moon ci core:wheel-merge", previous)
        self.assertLess(previous, merge)
        self.assertIn("cuda_bindings/.moon-out/wheel-previous", build)
        for obsolete_upload in (
            "Upload cuda.bindings Cython tests",
            "Upload cuda.core Cython tests",
            "Upload cuda.core test binaries",
        ):
            self.assertNotIn(obsolete_upload, build)

        for workflow_name in ("test-sdist-linux.yml", "test-sdist-windows.yml"):
            workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertIn('MOON_CACHE: "off"', workflow)
            positions = tuple(
                workflow.index(call)
                for call in (
                    "moon ci pathfinder:sdist",
                    "moon ci bindings:sdist",
                    "moon ci core:sdist metapackage:sdist",
                )
            )
            self.assertEqual(positions, tuple(sorted(positions)), workflow_name)

        for workflow_name in ("test-wheel-linux.yml", "test-wheel-windows.yml"):
            workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertIn("source-ref:", workflow)
            self.assertIn("ref: ${{ inputs.source-ref || github.sha }}", workflow)
            self.assertIn("MOON_HEAD: ${{ inputs.source-ref || github.sha }}", workflow)
            self.assertNotIn("lookup-run-id", workflow)

        nightly = (REPO_ROOT / ".github" / "workflows" / "ci-nightly.yml").read_text(encoding="utf-8")
        self.assertIn("HEAD_SHA: ${{ steps.find.outputs.head_sha }}", nightly)
        self.assertEqual(nightly.count("source-ref:"), 1)
        standard = nightly[nightly.index("test-standard-linux-aarch64:") :]
        self.assertIn("source-ref: ${{ needs.find-wheels.outputs.HEAD_SHA }}", standard)

    def test_cross_phase_assets_keep_only_required_context_and_source_proxies(self) -> None:
        current = self.by_target["core:wheel-current"]
        self.assertNotIn({"file": "/ci/tools/merge_cuda_core_wheels.py"}, current["inputs"])

        previous = self.by_target["core:wheel-previous"]
        for project in ("pathfinder", "bindings"):
            self.assertIn({"project": project, "group": "package"}, previous["inputs"])

        for target in ("metapackage:wheel-pure", "metapackage:sdist"):
            for project in ("pathfinder", "bindings"):
                self.assertIn({"project": project, "group": "package"}, self.by_target[target]["inputs"])

        merge = self.by_target["core:wheel-merge"]
        self.assertEqual(
            {dependency["target"] for dependency in merge["deps"]},
            {"test-helpers:fingerprint-python-build", "core:wheel-previous"},
        )
        for project in ("pathfinder", "bindings"):
            self.assertIn({"project": project, "group": "package"}, merge["inputs"])
        self.assertIn("@group(package)", merge["inputs"])

        binaries = self.by_target["core:test-binaries"]
        self.assertEqual(
            {dependency["target"] for dependency in binaries["deps"]},
            {"test-helpers:fingerprint-native-context"},
        )
        self.assertFalse(any(isinstance(value, dict) and "project" in value for value in binaries["inputs"]))
        self.assertNotIn("@group(package)", binaries["inputs"])

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
        self.assertEqual(docs["outputs"], [{"file": ".moon-out/docs"}])

        preparation = self.by_target["test-helpers:prepare-docs"]
        self.assertFalse(preparation["options"]["cache"])
        self.assertFalse(preparation.get("deps"))
        preparation_script = preparation["args"][3]
        self.assertIn("python -m pip install --force-reinstall", preparation_script)
        metapackage_install = preparation_script.split("python -m pip install --force-reinstall --no-deps", 1)[1]
        self.assertIn('"$METAPACKAGE_WHEEL"', metapackage_install)
        for component_wheel in ("PATHFINDER_WHEEL", "BINDINGS_WHEEL", "CORE_WHEEL"):
            self.assertNotIn(f'"${component_wheel}"', metapackage_install)
        for wheel_input in (
            "/cuda_pathfinder/.moon-out/wheel-pure/*.whl",
            "/cuda_bindings/.moon-out/wheel-current/*.whl",
            "/cuda_core/.moon-out/wheel-merged/*.whl",
            "/cuda_python/.moon-out/wheel-pure/*.whl",
        ):
            self.assertIn({"glob": wheel_input, "cache": True}, preparation["inputs"])

        component_targets = EXECUTION_TAG_TARGETS["ci-docs"] - {"root:docs-ci"}
        for target in component_targets:
            task = self.by_target[target]
            self.assertFalse(task["options"]["cache"])
            self.assertEqual(task["outputs"], [{"file": "docs/build/html"}])
            self.assertEqual({dependency["target"] for dependency in task["deps"]}, {preparation["target"]})
            self.assertIn({"file": "/cuda_python/docs/environment-docs.yml"}, task["inputs"])
            self.assertIn({"file": "/cuda_python/docs/build_component_docs.sh"}, task["inputs"])

        graph = self.moon_json("action-graph", "root:docs-ci", "--json")
        targets_by_node = {
            int(node): action["params"]["target"]
            for node, action in graph["data"].items()
            if action["action"] == "run-task"
        }
        self.assertEqual(
            set(targets_by_node.values()),
            {"root:docs-ci", preparation["target"], *component_targets},
        )
        edges = {
            (targets_by_node[parent], targets_by_node[dependency]) for parent, dependency, _ in graph["graph"]["edges"]
        }
        self.assertEqual(
            edges,
            {
                *(("root:docs-ci", target) for target in component_targets),
                *((target, preparation["target"]) for target in component_targets),
            },
        )

        workflow = (REPO_ROOT / ".github" / "workflows" / "build-docs.yml").read_text(encoding="utf-8")
        assembler = (REPO_ROOT / "cuda_python" / "docs" / "assemble_moon_docs.sh").read_text(encoding="utf-8")
        component_builder = (REPO_ROOT / "cuda_python" / "docs" / "build_component_docs.sh").read_text(encoding="utf-8")
        self.assertIn("moon run metapackage:wheel-pure", workflow)
        self.assertIn("moon ci root:docs-ci", workflow)
        self.assertIn('moon ci "${MOON_PROJECT}:docs-ci"', workflow)
        self.assertIn('cp -aL "${COMPONENT}/docs/build/html/."', workflow)
        self.assertNotIn(".moon-out/docs-ci", workflow)
        for project in ("cuda_pathfinder", "cuda_bindings", "cuda_core", "cuda_python"):
            self.assertIn(f"/{project}/docs/build/html", assembler)
        self.assertNotIn(".moon-out/docs-ci", assembler)
        self.assertNotIn("pip install cuda_pathfinder/.moon-out", workflow)
        for project, component in (
            ("cuda_pathfinder", "cuda-pathfinder"),
            ("cuda_bindings", "cuda-bindings"),
            ("cuda_core", "cuda-core"),
            ("cuda_python", "cuda-python"),
        ):
            wrapper = (REPO_ROOT / project / "docs" / "build_docs.sh").read_text(encoding="utf-8")
            self.assertIn("build_component_docs.sh", wrapper)
            self.assertIn(component, wrapper)
            self.assertIn(f"{component})", component_builder)
        docs_environment = (REPO_ROOT / "cuda_python" / "docs" / "environment-docs.yml").read_text(encoding="utf-8")
        self.assertIn("- python =3.12", docs_environment)

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

    def test_universal_wheels_share_native_build_lanes(self) -> None:
        self.assertFalse((REPO_ROOT / ".github" / "workflows" / "build-pure-wheel.yml").exists())
        self.assertFalse({task["target"] for task in self.tasks if "ci-wheel-pure" in task.get("tags", [])})
        pathfinder = self.by_target["pathfinder:wheel-pure"]
        self.assertIn("ci-build-native", pathfinder["tags"])
        self.assertNotIn("ci-build-native", self.by_target["metapackage:wheel-pure"].get("tags", []))
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

    def test_cross_run_artifacts_transport_canonical_outputs_only(self) -> None:
        for relative_path in (
            ".github/workflows/build-wheel.yml",
            ".github/workflows/test-sdist-linux.yml",
            ".github/workflows/test-sdist-windows.yml",
        ):
            workflow = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(".moon-out/", workflow, relative_path)
            self.assertNotIn(".moon/cache/hashes", workflow, relative_path)
            self.assertNotIn(".moon/cache/outputs", workflow, relative_path)


if __name__ == "__main__":
    unittest.main()
