# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the Moon-owned selective CI graph."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOON = shutil.which("moon")
BASH = shutil.which("bash")

VISIBLE_TASKS = {
    "root": {"test", "docs", "ci-ignore", "ci-fallback"},
    "pathfinder": {"install", "test", "docs", "wheel", "sdist", "ci-test-linux", "ci-test-windows"},
    "bindings": {
        "install",
        "test",
        "docs",
        "wheel",
        "sdist",
        "benchmark",
        "ci-test-linux",
        "ci-test-windows",
        "ci-test-assets",
    },
    "core": {
        "install",
        "test",
        "docs",
        "wheel",
        "wheel-merge",
        "sdist",
        "api-check",
        "ci-test-linux",
        "ci-test-windows",
        "ci-test-assets",
        "ci-test-binaries",
    },
    "metapackage": {"docs", "wheel", "sdist", "ci-test-linux", "ci-test-windows"},
}
INTERNAL_TASKS = {
    "pathfinder:test-installed",
    "bindings:test-installed",
    "bindings:build-cython-tests",
    "bindings:benchmark-smoke",
    "core:test-installed",
    "core:build-cython-tests",
    "core:build-test-binaries",
    "metapackage:test-installed",
}
ALL_ROUTES = {
    f"{project}:ci-test-{os_name}"
    for project in ("pathfinder", "bindings", "core", "metapackage")
    for os_name in ("linux", "windows")
}
LINUX_ROUTES = {target for target in ALL_ROUTES if target.endswith("-linux")}
WINDOWS_ROUTES = {target for target in ALL_ROUTES if target.endswith("-windows")}
ASSET_ROUTES = {"bindings:ci-test-assets", "core:ci-test-assets", "core:ci-test-binaries"}
PRODUCERS = {
    f"{project}:{kind}" for project in ("pathfinder", "bindings", "core", "metapackage") for kind in ("wheel", "sdist")
} | {"core:wheel-merge"}
CI_TASKS = (
    ALL_ROUTES
    | ASSET_ROUTES
    | PRODUCERS
    | {f"{project}:docs" for project in ("root", "pathfinder", "bindings", "core", "metapackage")}
    | {"core:api-check", "root:ci-ignore", "root:ci-fallback"}
)


def run_moon(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    assert MOON is not None
    return subprocess.run(  # noqa: S603 - MOON resolves to the pinned executable.
        [MOON, *args],
        cwd=ROOT,
        input=stdin,
        text=True,
        check=False,
        capture_output=True,
    )


def moon_json(*args: str, stdin: str | None = None) -> dict[str, Any]:
    result = run_moon(*args, stdin=stdin)
    result.check_returncode()
    return json.loads(result.stdout)


def targets(payload: dict[str, Any]) -> set[str]:
    return {f"{project}:{task}" for project, project_tasks in payload["tasks"].items() for task in project_tasks}


def affected(*paths: str) -> set[str]:
    payload = moon_json(
        "query",
        "tasks",
        "--affected",
        "stdin",
        "--upstream",
        "none",
        "--downstream",
        "none",
        stdin="".join(f"{path}\n" for path in paths),
    )
    return targets(payload) & CI_TASKS


def task_graph(target: str) -> dict[str, dict[str, Any]]:
    payload = moon_json("task-graph", target, "--json")
    return {task["target"]: task for task in payload["data"].values()}


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def workflow_step_script(path: str, name: str) -> str:
    lines = read(path).splitlines()
    start = lines.index(f"      - name: {name}")
    run = next(index for index in range(start, len(lines)) if lines[index].strip() == "run: |")
    end = next(
        (index for index in range(run + 1, len(lines)) if lines[index].startswith("      - name:")),
        len(lines),
    )
    return textwrap.dedent("\n".join(lines[run + 1 : end])).replace("${{ github.repository }}", "NVIDIA/cuda-python")


def baseline_artifacts(*, merge_base: str, expired: str | None = None) -> list[dict[str, object]]:
    names = ["cuda-pathfinder-wheel", "cuda-python-wheel"]
    for version in ("3.10", "3.11", "3.12", "3.13", "3.14", "3.14t", "3.15", "3.15t"):
        python = version.replace(".", "")
        for platform in ("linux-64", "linux-aarch64", "win-64"):
            names.append(f"cuda-bindings-python{python}-cuda13.3.0-{platform}-{merge_base}")
            names.append(f"cuda-core-python{python}-{platform}-{merge_base}")
    return [{"name": name, "expired": name == expired} for name in names]


@pytest.mark.skipif(MOON is None, reason="Moon 2.5.1 is required")
@pytest.mark.agent_authored(model="gpt-5")
class TestMoonCi:
    def test_workspace_is_pinned_and_visible_inventory_is_exact(self) -> None:
        assert run_moon("--version").stdout.strip() == "moon 2.5.1"
        payload = moon_json("query", "tasks")
        assert {project: set(project_tasks) for project, project_tasks in payload["tasks"].items()} == VISIBLE_TASKS
        assert not (targets(payload) & INTERNAL_TASKS)

    def test_internal_inventory_is_hidden_and_rejects_direct_execution(self) -> None:
        graph = task_graph("root:ci-fallback")
        internal = {target for target, task in graph.items() if task["options"]["internal"]}
        assert internal == INTERNAL_TASKS
        for target in sorted(INTERNAL_TASKS):
            result = run_moon("run", target, "--upstream", "none", "--downstream", "none")
            assert result.returncode != 0
            assert "Unknown task" in result.stderr

    def test_all_tasks_disable_caching_and_routes_are_commandless(self) -> None:
        visible = moon_json("query", "tasks")["tasks"]
        graph = task_graph("root:ci-fallback")
        assert all(task["options"]["cache"] is False for tasks in visible.values() for task in tasks.values())
        assert all(task["options"]["cache"] is False for task in graph.values())
        for target in (
            ALL_ROUTES
            | ASSET_ROUTES
            | {
                "root:ci-ignore",
                "root:ci-fallback",
                "root:test",
            }
        ):
            project, task = target.split(":")
            assert visible[project][task]["command"] == "noop"

    def test_semantic_tag_inventory_is_exact(self) -> None:
        payload = moon_json("query", "tasks")
        actual = {
            target: set(payload["tasks"][target.split(":")[0]][target.split(":")[1]].get("tags", []))
            for target in targets(payload)
            if payload["tasks"][target.split(":")[0]][target.split(":")[1]].get("tags")
        }
        expected = {
            "pathfinder:wheel": {"ci-wheel-foundation"},
            "bindings:wheel": {"ci-wheel-bindings"},
            "core:wheel": {"ci-wheel-consumers", "ci-wheel-multi-ctk"},
            "metapackage:wheel": {"ci-wheel-consumers"},
            "core:wheel-merge": {"ci-wheel-finalize"},
            "pathfinder:sdist": {"ci-sdist-foundation"},
            "bindings:sdist": {"ci-sdist-bindings"},
            "core:sdist": {"ci-sdist-consumers"},
            "metapackage:sdist": {"ci-sdist-consumers"},
            "bindings:ci-test-assets": {"ci-test-assets-current"},
            "core:ci-test-assets": {"ci-test-assets-current"},
            "core:ci-test-binaries": {"ci-test-assets-previous"},
            "core:api-check": {"ci-api"},
            "root:ci-ignore": {"ci-ignore"},
            "root:ci-fallback": {"ci-force-all"},
        }
        expected.update({target: {"ci-test-linux"} for target in LINUX_ROUTES})
        expected.update({target: {"ci-test-windows"} for target in WINDOWS_ROUTES})
        expected.update({f"{project}:docs": {"ci-docs"} for project in VISIBLE_TASKS})
        assert actual == expected

    def test_package_source_impact_routes(self) -> None:
        cases = {
            "cuda_pathfinder/cuda/pathfinder/__init__.py": PRODUCERS | ALL_ROUTES | ASSET_ROUTES,
            "cuda_bindings/cuda/bindings/__init__.py": {
                "bindings:wheel",
                "bindings:sdist",
                "core:wheel",
                "core:wheel-merge",
                "core:sdist",
                "metapackage:wheel",
                "metapackage:sdist",
                "bindings:ci-test-linux",
                "bindings:ci-test-windows",
                "core:ci-test-linux",
                "core:ci-test-windows",
                "metapackage:ci-test-linux",
                "metapackage:ci-test-windows",
            }
            | ASSET_ROUTES,
            "cuda_core/cuda/core/__init__.py": {
                "core:wheel",
                "core:wheel-merge",
                "core:sdist",
                "core:api-check",
                "core:ci-test-linux",
                "core:ci-test-windows",
                "metapackage:ci-test-linux",
                "metapackage:ci-test-windows",
                "core:ci-test-assets",
                "core:ci-test-binaries",
            },
            "cuda_python/pyproject.toml": {
                "bindings:wheel",
                "bindings:sdist",
                "metapackage:wheel",
                "metapackage:sdist",
                "metapackage:ci-test-linux",
                "metapackage:ci-test-windows",
            },
        }
        cases["cuda_pathfinder/.git_archival.txt"] = cases["cuda_pathfinder/cuda/pathfinder/__init__.py"]
        cases["cuda_bindings/.git_archival.txt"] = cases["cuda_bindings/cuda/bindings/__init__.py"]
        cases["cuda_core/.git_archival.txt"] = cases["cuda_core/cuda/core/__init__.py"]
        for path, expected in cases.items():
            assert affected(path) == expected, path

    def test_tests_helpers_benchmarks_and_os_infrastructure_impact(self) -> None:
        cases = {
            "cuda_pathfinder/tests/test_pathfinder.py": {
                "pathfinder:ci-test-linux",
                "pathfinder:ci-test-windows",
            },
            "cuda_bindings/examples/0_Introduction/vectorAddDrv.py": {
                "bindings:ci-test-linux",
                "bindings:ci-test-windows",
                "bindings:ci-test-assets",
            },
            "cuda_core/tests/test_device.py": {
                "core:ci-test-linux",
                "core:ci-test-windows",
                "core:ci-test-assets",
                "core:ci-test-binaries",
            },
            "cuda_python_test_helpers/pyproject.toml": ALL_ROUTES | ASSET_ROUTES,
            "benchmarks/cuda_bindings/run_pyperf.py": ALL_ROUTES | ASSET_ROUTES,
            "benchmarks/cuda_bindings/compare.py": ALL_ROUTES | ASSET_ROUTES,
            "benchmarks/cuda_core/runtime.py": ALL_ROUTES | ASSET_ROUTES,
            ".github/workflows/test-wheel-linux.yml": LINUX_ROUTES | ASSET_ROUTES,
            ".github/workflows/test-wheel-windows.yml": WINDOWS_ROUTES | ASSET_ROUTES,
            "ci/tools/guess_latest.sh": LINUX_ROUTES | ASSET_ROUTES,
        }
        for path, expected in cases.items():
            assert affected(path) == expected, path

    def test_docs_ignored_unknown_and_fallback_ownership(self) -> None:
        assert affected("cuda_core/docs/source/index.rst") == {"core:docs"}
        for path in (
            ".coveragerc",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/labeler.yml",
            ".pre-commit-config.yaml",
            "CONTRIBUTING.md",
            "context7.json",
            "cuda_core/pixi.toml",
            "cuda_core/tests/AGENTS.md",
            "diagram.svg",
            "greptile.json",
            "new-area/pixi.lock",
            "ruff.toml",
            "toolshed/README.md",
        ):
            assert affected(path) == {"root:ci-ignore"}
        assert affected(".github/workflows/ci.yml") == {"root:ci-fallback"}
        assert affected("an-entirely-new-path.txt") == set()
        gate = read(".github/workflows/ci.yml")
        assert 'length == 0 or any(.[]; .target == "root:ci-fallback")' in gate

        fallback = task_graph("root:ci-fallback")["root:ci-fallback"]
        assert {dep["target"] for dep in fallback["deps"]} == (
            PRODUCERS | ALL_ROUTES | ASSET_ROUTES | {"core:api-check", "root:docs"}
        )
        for path in (".moon/workspace.yml", "moon.yml", "cuda_core/moon.yml", "ci/versions.yml"):
            assert "root:ci-fallback" in affected(path)

    def test_mixed_changes_and_symlink_consumers(self) -> None:
        assert affected("cuda_core/docs/source/index.rst", "cuda_bindings/tests/test_api.py") == {
            "core:docs",
            "bindings:ci-test-linux",
            "bindings:ci-test-windows",
            "bindings:ci-test-assets",
        }
        expected_readme = {
            "bindings:wheel",
            "bindings:sdist",
            "metapackage:wheel",
            "metapackage:sdist",
            "metapackage:ci-test-linux",
            "metapackage:ci-test-windows",
        }
        assert affected("README.md") == expected_readme
        assert affected("cuda_python/README.md") == expected_readme
        assert affected(".git_archival.txt") == PRODUCERS | ALL_ROUTES | ASSET_ROUTES | {"core:api-check"}

    def test_editable_installs_are_first_class_dependencies(self) -> None:
        expected_install_deps = {
            "pathfinder:install": set(),
            "bindings:install": {"pathfinder:install"},
            "core:install": {"bindings:install"},
        }
        graph = task_graph("core:install")
        assert set(graph) == set(expected_install_deps)
        for target, expected in expected_install_deps.items():
            assert {dep["target"] for dep in graph[target].get("deps", [])} == expected
            script = graph[target]["script"]
            assert "pip install -e ." in script
            assert "../cuda_" not in script

        expected_test_graphs = {
            "pathfinder:test": {"pathfinder:install", "pathfinder:test"},
            "bindings:test": {"pathfinder:install", "bindings:install", "bindings:test"},
            "core:test": {
                "pathfinder:install",
                "bindings:install",
                "core:install",
                "core:test",
            },
        }
        for target, expected in expected_test_graphs.items():
            graph = task_graph(target)
            assert set(graph) == expected
            test_script = graph[target]["script"]
            assert "pip install" not in test_script
            assert all("wheel" not in graph_target for graph_target in graph)
            assert all("cibuildwheel" not in task["script"] for task in graph.values())

        root_test_graph = task_graph("root:test")
        assert {dep["target"] for dep in root_test_graph["root:test"]["deps"]} == {
            "pathfinder:test",
            "bindings:test",
            "core:test",
        }
        assert root_test_graph["root:test"]["options"]["runDepsInParallel"] is True

        benchmark_graph = task_graph("bindings:benchmark")
        assert set(benchmark_graph) == {
            "pathfinder:install",
            "bindings:install",
            "bindings:benchmark",
        }
        assert "pip install" not in benchmark_graph["bindings:benchmark"]["script"]

    def test_local_core_wheel_builds_current_dependency_chain(self) -> None:
        assert set(task_graph("core:wheel")) == {
            "pathfinder:wheel",
            "bindings:wheel",
            "core:wheel",
        }

    def test_ci_routes_have_only_hidden_direct_executors(self) -> None:
        expected = {
            "pathfinder:ci-test-linux": {"pathfinder:test-installed"},
            "pathfinder:ci-test-windows": {"pathfinder:test-installed"},
            "bindings:ci-test-linux": {"bindings:test-installed", "bindings:benchmark-smoke"},
            "bindings:ci-test-windows": {"bindings:test-installed"},
            "core:ci-test-linux": {"core:test-installed"},
            "core:ci-test-windows": {"core:test-installed"},
            "metapackage:ci-test-linux": {"metapackage:test-installed"},
            "metapackage:ci-test-windows": {"metapackage:test-installed"},
        }
        fallback = task_graph("root:ci-fallback")
        for route, direct_targets in expected.items():
            actual = {dep["target"] for dep in fallback[route]["deps"]}
            assert actual == direct_targets
            assert all(fallback[target]["options"]["internal"] for target in actual)
            assert all(fallback[target].get("deps") for target in actual)
        for workflow in (".github/workflows/test-wheel-linux.yml", ".github/workflows/test-wheel-windows.yml"):
            assert 'moon run "${target_args[@]}" --upstream direct --downstream none' in read(workflow)

    def test_build_traversal_stages_dependencies_and_runs_exact_targets(self) -> None:
        workflow = read(".github/workflows/build-wheel.yml")
        for phase in (
            "WHEEL_FOUNDATION_TARGETS",
            "WHEEL_BINDINGS_TARGETS",
            "WHEEL_CONSUMER_TARGETS",
            "WHEEL_MULTI_CTK_TARGETS",
            "WHEEL_FINALIZE_TARGETS",
        ):
            assert phase in workflow
        assert workflow.count("--upstream none --downstream none") >= 5
        assert workflow.count("--upstream direct --downstream none") >= 2
        assert "Download reusable cuda.pathfinder wheel" in workflow
        assert "Download reusable cuda.bindings wheel" in workflow
        assert workflow.count("python -m pip install cibuildwheel twine wheel") == 2

    def test_native_assets_follow_the_selected_os(self, tmp_path: Path) -> None:
        assert BASH is not None
        script = workflow_step_script(".github/workflows/build-wheel.yml", "Resolve Moon phase targets")
        common = {
            "WHEEL_FOUNDATION_TARGETS": "[]",
            "WHEEL_BINDINGS_TARGETS": "[]",
            "WHEEL_CONSUMER_TARGETS": "[]",
            "WHEEL_MULTI_CTK_TARGETS": "[]",
            "WHEEL_FINALIZE_TARGETS": "[]",
            "TEST_ASSETS_CURRENT_TARGETS": '["bindings:ci-test-assets","core:ci-test-assets"]',
            "TEST_ASSETS_PREVIOUS_TARGETS": '["core:ci-test-binaries"]',
        }
        cases = {
            "linux-selected": {
                "TEST_LINUX_TARGETS": '["core:ci-test-linux"]',
                "TEST_WINDOWS_TARGETS": "[]",
                "linux-64": "true",
                "win-64": "false",
            },
            "windows-selected": {
                "TEST_LINUX_TARGETS": "[]",
                "TEST_WINDOWS_TARGETS": '["core:ci-test-windows"]',
                "linux-64": "false",
                "win-64": "true",
            },
        }
        for case_name, case in cases.items():
            for platform in ("linux-64", "win-64"):
                output = tmp_path / f"{case_name}-{platform}.env"
                env = (
                    os.environ
                    | common
                    | {
                        "HOST_PLATFORM": platform,
                        "GITHUB_ENV": str(output),
                        "TEST_LINUX_TARGETS": case["TEST_LINUX_TARGETS"],
                        "TEST_WINDOWS_TARGETS": case["TEST_WINDOWS_TARGETS"],
                    }
                )
                result = subprocess.run(  # noqa: S603 - controlled repository script.
                    [BASH, "-c", script],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    check=False,
                    capture_output=True,
                )
                assert result.returncode == 0, (case_name, platform, result.stderr)
                values = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
                assert values["TEST_BINDINGS"] == case[platform]
                assert values["TEST_CORE_CURRENT"] == case[platform]
                assert values["TEST_CORE_PREVIOUS"] == case[platform]

        workflow = read(".github/workflows/ci.yml")
        linux_arm = workflow.split("  build-linux-aarch64:", 1)[1].split("  build-windows:", 1)[0]
        windows = workflow.split("  build-windows:", 1)[1].split("  test-sdist-linux:", 1)[0]
        assert "ci-test-assets" not in linux_arm
        assert "ci-test-assets" not in windows

    def test_core_uses_one_target_in_both_toolkits_then_merges(self) -> None:
        graph = task_graph("root:ci-fallback")
        assert set(graph["core:wheel"]["tags"]) == {"ci-wheel-consumers", "ci-wheel-multi-ctk"}
        assert not graph["core:wheel-merge"].get("deps")
        merger = graph["core:wheel-merge"]["script"]
        assert "dist/cu12/*.whl dist/cu13/*.whl" in merger
        assert "merge_cuda_core_wheels.py" in merger

    def test_only_merged_core_wheel_is_in_baseline_artifact(self) -> None:
        workflow = read(".github/workflows/build-wheel.yml")
        assert "name: ${{ env.CUDA_CORE_ARTIFACT_NAME }}" in workflow
        assert "path: ${{ env.CUDA_CORE_ARTIFACTS_DIR }}/*.whl" in workflow
        assert "path: ${{ env.CUDA_CORE_ARTIFACTS_DIR }}/cu" not in workflow
        for name in ("cuda-pathfinder-wheel", "cuda-python-wheel"):
            assert f"name: {name}" in workflow

    def test_baseline_reuse_requires_one_exact_successful_complete_set(self) -> None:
        workflow = read(".github/workflows/ci.yml")
        for contract in (
            '--commit "${merge_base}"',
            "--event push",
            "--status success",
            "if [[ $(jq 'length' <<< \"$runs\") -ne 1 ]]",
            '"${run_sha}" != "${merge_base}"',
            "length == 1 and .[0].expired == false",
            "if (( ${#missing[@]} != 0 ))",
            'baseline_run_id=""',
            'baseline_sha=""',
        ):
            assert contract in workflow
        assert "cuda-pathfinder-wheel cuda-python-wheel" in workflow
        assert "CUDA_BINDINGS_ARTIFACT_BASENAME" in read(".github/workflows/build-wheel.yml")
        assert "CUDA_CORE_ARTIFACT_BASENAME" in read(".github/workflows/build-wheel.yml")
        assert "uvx --from pytest pytest -q tests/test_moon_ci.py" in workflow

    def test_baseline_reuse_behaviors(self, tmp_path: Path) -> None:
        assert BASH is not None
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        gh = fake_bin / "gh"
        gh.write_text(
            """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$MOCK_GH_LOG"
if [[ "$1 $2" == "run list" ]]; then
  printf '%s\\n' "$MOCK_RUNS"
  exit "$MOCK_RUN_STATUS"
fi
if [[ "$1" == "api" ]]; then
  printf '%s\\n' "$MOCK_ARTIFACTS"
  exit "$MOCK_ARTIFACT_STATUS"
fi
exit 2
""",
            encoding="utf-8",
        )
        yq = fake_bin / "yq"
        yq.write_text(
            """#!/usr/bin/env bash
if [[ "$1" == "-r" ]]; then
  printf '%s\\n' 3.10 3.11 3.12 3.13 3.14 3.14t 3.15 3.15t
else
  printf '%s\\n' 13.3.0
fi
""",
            encoding="utf-8",
        )
        os.chmod(gh, 0o700)
        os.chmod(yq, 0o700)

        merge_base = "exact-base"
        complete = baseline_artifacts(merge_base=merge_base)
        cases = {
            "complete": {
                "runs": [{"databaseId": 42, "headSha": merge_base}],
                "artifacts": complete,
                "accepted": True,
            },
            "incomplete": {
                "runs": [{"databaseId": 42, "headSha": merge_base}],
                "artifacts": complete[:-1],
                "accepted": False,
            },
            "expired": {
                "runs": [{"databaseId": 42, "headSha": merge_base}],
                "artifacts": baseline_artifacts(merge_base=merge_base, expired="cuda-pathfinder-wheel"),
                "accepted": False,
            },
            "failed-run": {"runs": [], "artifacts": complete, "accepted": False},
            "wrong-sha": {
                "runs": [{"databaseId": 42, "headSha": "another-sha"}],
                "artifacts": complete,
                "accepted": False,
            },
            "duplicate": {
                "runs": [{"databaseId": 42, "headSha": merge_base}],
                "artifacts": [*complete, complete[0]],
                "accepted": False,
            },
            "lookup-failure": {
                "runs": [{"databaseId": 42, "headSha": merge_base}],
                "artifacts": complete,
                "accepted": False,
                "run_status": 1,
            },
        }
        script = workflow_step_script(".github/workflows/ci.yml", "Resolve reusable base artifacts")
        for name, case in cases.items():
            output = tmp_path / f"{name}.output"
            summary = tmp_path / f"{name}.summary"
            log = tmp_path / f"{name}.gh.log"
            output.touch()
            summary.touch()
            env = os.environ | {
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "BASE_REF": "main",
                "MERGE_BASE": merge_base,
                "GITHUB_OUTPUT": str(output),
                "GITHUB_STEP_SUMMARY": str(summary),
                "MOCK_GH_LOG": str(log),
                "MOCK_RUNS": json.dumps(case["runs"]),
                "MOCK_ARTIFACTS": "\n".join(json.dumps(item) for item in case["artifacts"]),
                "MOCK_RUN_STATUS": str(case.get("run_status", 0)),
                "MOCK_ARTIFACT_STATUS": "0",
            }
            result = subprocess.run(  # noqa: S603 - controlled script and fake tools.
                [BASH, "-c", script],
                cwd=ROOT,
                env=env,
                text=True,
                check=False,
                capture_output=True,
            )
            assert result.returncode == 0, (name, result.stderr)
            accepted = "run_id=42" in output.read_text(encoding="utf-8")
            assert accepted is case["accepted"], name
            if not case["accepted"]:
                assert "No complete reusable artifact set" in summary.read_text(encoding="utf-8")

        complete_log = (tmp_path / "complete.gh.log").read_text(encoding="utf-8")
        for argument in ("--commit exact-base", "--event push", "--status success"):
            assert argument in complete_log

    def test_docs_select_component_or_parallel_aggregate_layout(self) -> None:
        workflow = read(".github/workflows/build-docs.yml")
        assert "all) targets='[\"root:docs\"]'" in workflow
        for project in ("pathfinder", "bindings", "core", "metapackage"):
            assert f'"{project}:docs"' in workflow
        assert "DOCS_BUILD_ARGS" in workflow
        assert "--upstream deep --downstream none" in workflow
        assert "DOCS_USE_MOON" in workflow
        assert "./build_all_docs.sh latest-only" in workflow
        assert "./build_docs.sh latest-only" in workflow
        root_docs = task_graph("root:docs")
        assert {dep["target"] for dep in root_docs["root:docs"]["deps"]} == {
            "pathfinder:docs",
            "bindings:docs",
            "core:docs",
            "metapackage:docs",
        }
        script = root_docs["root:docs"]["script"]
        for destination in ("cuda-bindings", "cuda-core", "cuda-pathfinder"):
            assert f"cuda_python/docs/build/html/{destination}" in script
        for project in ("pathfinder", "bindings", "core", "metapackage"):
            assert "${DOCS_BUILD_ARGS:-}" in root_docs[f"{project}:docs"]["script"]
