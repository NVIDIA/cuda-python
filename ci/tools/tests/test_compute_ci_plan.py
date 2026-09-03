# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest

from ci.tools.bindings_config import BindingsConfig, BindingsPackage, load_config
from ci.tools.compute_ci_plan import _expand_linked_paths, compute_workplan, main

ALL_MODULES = {"pathfinder", "bindings", "core", "python"}
ALL_PLATFORMS = {"linux", "windows"}
VARIANT_MODULES = {"bindings", "core", "python"}
DEFAULT_BINDINGS_CONFIG = load_config()
CUDA_VARIANTS = {package.cuda_variant for package in DEFAULT_BINDINGS_CONFIG.package_roots}


def plan_for(
    *paths: str,
    baseline: bool = True,
    linked_paths: set[str] | None = None,
    release_tag: str = "",
    bindings_config: BindingsConfig | None = None,
) -> dict[str, object]:
    return compute_workplan(
        list(paths),
        merge_base="base",
        baseline_run_id="123" if baseline else "",
        linked_paths=linked_paths,
        release_tag=release_tag,
        bindings_config=bindings_config,
    )


def as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def enabled(decisions: dict[str, object], key: str | None = None) -> set[str]:
    if key is None:
        return {name for name, value in decisions.items() if value}
    return {name for name, value in decisions.items() if as_dict(value)[key]}


def selected(plan: dict[str, object], key: str) -> set[str]:
    return enabled(as_dict(plan["modules"]), key)


def selected_platforms(plan: dict[str, object]) -> set[str]:
    platforms = as_dict(as_dict(plan["jobs"])["platforms"])
    assert set(platforms) == ALL_PLATFORMS
    return enabled(platforms)


def selected_variants(plan: dict[str, object], module: str, key: str) -> set[str]:
    modules = as_dict(plan["modules"])
    if module == "core":
        return enabled(as_dict(as_dict(modules[module])["cuda_majors"]), key)
    packages = as_dict(as_dict(modules[module])["package_roots"])
    return {str(as_dict(decision)["cuda_variant"]) for decision in packages.values() if as_dict(decision)[key]}


def selected_package_roots(plan: dict[str, object], module: str, key: str) -> set[str]:
    module_plan = as_dict(as_dict(plan["modules"])[module])
    return enabled(as_dict(module_plan["package_roots"]), key)


def selected_core_majors(plan: dict[str, object], key: str) -> set[str]:
    core = as_dict(as_dict(plan["modules"])["core"])
    return enabled(as_dict(core["cuda_majors"]), key)


def selected_cuda_majors(plan: dict[str, object], key: str) -> set[str]:
    majors = as_dict(as_dict(plan["jobs"])[key])
    cuda_majors = as_dict(as_dict(as_dict(plan["modules"])["core"])["cuda_majors"])
    assert set(majors) == set(cuda_majors)
    return enabled(majors)


def selected_sdist_variants(plan: dict[str, object]) -> set[str]:
    jobs = as_dict(plan["jobs"])
    selected_roots = enabled(as_dict(jobs["sdist_package_roots"]))
    packages = as_dict(as_dict(as_dict(plan["modules"])["bindings"])["package_roots"])
    return {str(as_dict(packages[root])["cuda_variant"]) for root in selected_roots}


def synthetic_package(package_root: str, ctk_target: str, release_status: str) -> BindingsPackage:
    return BindingsPackage(
        package_root=package_root,
        toolkit_version=f"{ctk_target}.0",
        release_status=release_status,
        tag_regex=rf"^(?P<version>v{re.escape(ctk_target)}\.\d+)$",
    )


def synthetic_config(
    *packages: BindingsPackage,
) -> BindingsConfig:
    return BindingsConfig(
        schema_version=2,
        package_roots=packages,
    )


class ComputeWorkplanTest(unittest.TestCase):
    def test_path_impacts(self) -> None:
        cases = {
            "cuda_core/cuda/core/examples/demo.py": ({"core"}, {"core", "python"}, True),
            "cuda_pathfinder/tests/test_loader.py": (set(), {"pathfinder"}, False),
            "cuda_bindings/tests/README.md": (set(), {"bindings"}, False),
            "cuda_core/pytest.ini": (set(), {"core"}, False),
            "cuda_python/tests/test_import.py": (set(), {"python"}, False),
            "cuda_python_test_helpers/cuda_python_test_helpers/cuda_utils.py": (
                set(),
                ALL_MODULES,
                False,
            ),
            "benchmarks/cuda_bindings/run_pyperf.py": (set(), ALL_MODULES, False),
        }

        for path, (builds, tests, core_api) in cases.items():
            with self.subTest(path=path):
                plan = plan_for(path)
                assert selected(plan, "needs_build") == builds
                assert selected(plan, "needs_test") == tests
                assert selected_platforms(plan) == ALL_PLATFORMS
                assert plan["jobs"]["sdist_tests"] == bool(builds)
                assert plan["jobs"]["core_api_checks"] == core_api

        self._check_variant_path_impacts()

    def _check_variant_path_impacts(self) -> None:
        all_variant_modules = dict.fromkeys(VARIANT_MODULES, CUDA_VARIANTS)
        cases = (
            (
                ("cuda_bindings_12/cuda/bindings/driver.pyx",),
                {"bindings": {"cu12"}, "core": CUDA_VARIANTS, "python": {"cu12"}},
                {"bindings": {"cu12"}, "core": {"cu12"}, "python": {"cu12"}},
                {"cu12"},
                {"cu12"},
            ),
            (
                ("cuda_bindings/cuda/bindings/driver.pyx",),
                {"bindings": {"cu13"}, "core": CUDA_VARIANTS, "python": {"cu13"}},
                {"bindings": {"cu13"}, "core": {"cu13"}, "python": {"cu13"}},
                {"cu13"},
                {"cu13"},
            ),
            (
                ("cuda_bindings_12/examples/0_Introduction/vectorAddDrv.py",),
                {},
                {"bindings": {"cu12"}},
                {"cu12"},
                set(),
            ),
            (
                ("cuda_bindings/tests/test_driver.py",),
                {},
                {"bindings": {"cu13"}},
                {"cu13"},
                set(),
            ),
            (
                ("cuda_core/cuda/core/_device.py",),
                {"core": CUDA_VARIANTS},
                {"core": CUDA_VARIANTS, "python": CUDA_VARIANTS},
                CUDA_VARIANTS,
                CUDA_VARIANTS,
            ),
            (
                ("cuda_python/pyproject.toml",),
                {"bindings": CUDA_VARIANTS, "python": CUDA_VARIANTS},
                {"python": CUDA_VARIANTS},
                CUDA_VARIANTS,
                CUDA_VARIANTS,
            ),
            (
                ("cuda_pathfinder/cuda/pathfinder/_loader.py",),
                all_variant_modules,
                all_variant_modules,
                CUDA_VARIANTS,
                CUDA_VARIANTS,
            ),
            (
                (
                    "cuda_bindings_12/cuda/bindings/driver.pyx",
                    "cuda_bindings/cuda/bindings/runtime.pyx",
                ),
                all_variant_modules,
                all_variant_modules,
                CUDA_VARIANTS,
                CUDA_VARIANTS,
            ),
        )

        for paths, variant_builds, variant_tests, test_majors, sdist_majors in cases:
            with self.subTest(paths=paths):
                plan = plan_for(*paths)
                for module in VARIANT_MODULES:
                    builds = selected_variants(plan, module, "needs_build")
                    tests = selected_variants(plan, module, "needs_test")
                    assert builds == variant_builds.get(module, set())
                    assert tests == variant_tests.get(module, set())

                    modules = plan["modules"]
                    assert isinstance(modules, dict)
                    decision = modules[module]
                    assert decision["needs_build"] == bool(builds)
                    assert decision["needs_test"] == bool(tests)

                assert selected_cuda_majors(plan, "test_cuda_majors") == test_majors
                assert selected_sdist_variants(plan) == sdist_majors

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_release_tags_select_only_the_matching_package(self) -> None:
        for release_tag, package_root, variant in (
            ("v12.9.9", "cuda_bindings_12", "cu12"),
            ("v13.3.0", "cuda_bindings", "cu13"),
            ("v13.3.0b1", "cuda_bindings", "cu13"),
            ("v13.3.0rc1", "cuda_bindings", "cu13"),
            ("v13.3.0.dev1", "cuda_bindings", "cu13"),
            ("v12.9.9.post1", "cuda_bindings_12", "cu12"),
            ("v13.3.0.post1", "cuda_bindings", "cu13"),
        ):
            with self.subTest(release_tag=release_tag):
                plan = plan_for(baseline=False, release_tag=release_tag)
                assert selected(plan, "needs_build") == {"bindings", "python"}
                assert selected(plan, "needs_test") == {"bindings", "python"}
                for module in ("bindings", "python"):
                    assert selected_package_roots(plan, module, "needs_build") == {package_root}
                    assert selected_package_roots(plan, module, "needs_test") == {package_root}
                    assert selected_variants(plan, module, "needs_build") == {variant}
                    assert selected_variants(plan, module, "needs_test") == {variant}
                assert not selected_variants(plan, "core", "needs_build")
                assert not selected_variants(plan, "core", "needs_test")
                assert selected_cuda_majors(plan, "test_cuda_majors") == {variant}
                assert selected_sdist_variants(plan) == {variant}
                assert selected_platforms(plan) == ALL_PLATFORMS
                assert plan["jobs"]["sdist_tests"]
                assert not plan["jobs"]["core_api_checks"]
                assert plan["baseline"] == {"run_id": "", "sha": ""}

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
                for module in VARIANT_MODULES:
                    assert selected_variants(plan, module, "needs_test") == CUDA_VARIANTS
                assert selected_platforms(plan) == platforms
                assert not plan["jobs"]["sdist_tests"]
                assert not plan["jobs"]["core_api_checks"]
                assert selected_cuda_majors(plan, "test_cuda_majors") == CUDA_VARIANTS
                assert not selected_sdist_variants(plan)

        mixed_plan = plan_for("ci/tools/install_gpu_driver.sh", "ci/tools/install_gpu_driver.ps1")
        assert selected_platforms(mixed_plan) == ALL_PLATFORMS

        source_plan = plan_for("ci/tools/install_gpu_driver.sh", "cuda_python/pyproject.toml")
        assert selected_platforms(source_plan) == ALL_PLATFORMS

    def test_ignored_paths_select_no_work(self) -> None:
        empty_plan = plan_for()
        for path in (
            "cuda_core/docs/index.rst",
            "cuda_core/pixi.toml",
            "cuda_core/tests/fixtures/pixi.toml",
            "benchmarks/cuda_bindings/pixi.toml",
            "benchmarks/cuda_bindings/AGENTS.md",
            "cuda_core/cuda/core/_cpp/DESIGN.md",
            "cuda_bindings/README.md",
            "cuda_bindings_12/README.md",
            "cuda_bindings_12/docs/index.rst",
            "cuda_core/README.md",
            "new-area/pixi.toml",
            "notes.md",
            "diagram.svg",
            ".github/labeler.yml",
            ".github/ISSUE_TEMPLATE/bug.yml",
        ):
            with self.subTest(path=path):
                assert plan_for(path) == empty_plan

    def test_unknown_path_and_missing_baseline_force_all(self) -> None:
        for plan in (
            plan_for("new-top-level-file"),
            plan_for("new-area/config.toml"),
            plan_for(".github/workflows/new-main-ci-workflow.yml"),
            plan_for(".github/actions/doc_preview/action.yml"),
            plan_for("ci/ci-pipeline.svg"),
            plan_for("cuda_core/docs/index.rst", baseline=False),
            compute_workplan([], merge_base="", baseline_run_id="123"),
            plan_for(baseline=False, release_tag="cuda-core-v1.3.0"),
        ):
            assert selected(plan, "needs_build") == ALL_MODULES
            assert selected(plan, "needs_test") == ALL_MODULES
            for module in VARIANT_MODULES:
                assert selected_variants(plan, module, "needs_build") == CUDA_VARIANTS
                assert selected_variants(plan, module, "needs_test") == CUDA_VARIANTS
            assert selected_platforms(plan) == ALL_PLATFORMS
            assert plan["jobs"]["core_api_checks"]
            assert selected_cuda_majors(plan, "test_cuda_majors") == CUDA_VARIANTS
            assert selected_sdist_variants(plan) == CUDA_VARIANTS
            assert plan["baseline"] == {"run_id": "", "sha": ""}

        for release_tag in ("v12.8.1", "v13.4.0", "v14.0.0"):
            with (
                self.subTest(release_tag=release_tag),
                pytest.raises(ValueError, match="no configured CUDA bindings package root"),
            ):
                plan_for(baseline=False, release_tag=release_tag)

    def test_mixed_changes_are_combined(self) -> None:
        plan = plan_for("cuda_core/tests/test_device.py", "cuda_python/pyproject.toml")
        assert selected(plan, "needs_build") == {"bindings", "python"}
        assert selected(plan, "needs_test") == {"core", "python"}
        assert selected_variants(plan, "bindings", "needs_build") == CUDA_VARIANTS
        assert selected_variants(plan, "python", "needs_build") == CUDA_VARIANTS
        assert selected_variants(plan, "core", "needs_test") == CUDA_VARIANTS
        assert selected_variants(plan, "python", "needs_test") == CUDA_VARIANTS
        assert selected_platforms(plan) == ALL_PLATFORMS
        assert plan["jobs"]["sdist_tests"]
        assert selected_cuda_majors(plan, "test_cuda_majors") == CUDA_VARIANTS
        assert selected_sdist_variants(plan) == CUDA_VARIANTS
        assert plan["baseline"] == {"run_id": "123", "sha": "base"}

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
        removed_link = plan_for("cuda_python/README.md", linked_paths={"cuda_python/README.md"})
        for linked_plan in (plan, removed_link):
            assert selected(linked_plan, "needs_build") == {"bindings", "python"}
            assert selected(linked_plan, "needs_test") == {"python"}
            for module in ("bindings", "python"):
                assert selected_variants(linked_plan, module, "needs_build") == CUDA_VARIANTS
            assert selected_variants(linked_plan, "python", "needs_test") == CUDA_VARIANTS

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_same_major_packages_remain_independently_selectable(self) -> None:
        package_11_7 = synthetic_package("cuda_bindings_11_7", "11.7", "maintenance")
        package_11_8 = synthetic_package("cuda_bindings_11_8", "11.8", "current")
        config = synthetic_config(
            package_11_7,
            package_11_8,
        )

        plan = plan_for(
            "cuda_bindings_11_7/cuda/bindings/driver.pyx",
            bindings_config=config,
        )

        assert selected_package_roots(plan, "bindings", "needs_build") == {package_11_7.package_root}
        assert selected_package_roots(plan, "bindings", "needs_test") == {package_11_7.package_root}
        assert selected_package_roots(plan, "python", "needs_build") == {package_11_7.package_root}
        assert selected_package_roots(plan, "python", "needs_test") == {package_11_7.package_root}
        assert selected_core_majors(plan, "needs_build") == {"cu11"}
        assert selected_core_majors(plan, "needs_test") == {"cu11"}
        assert selected_variants(plan, "bindings", "needs_build") == {"cu11"}
        assert selected_sdist_variants(plan) == {"cu11"}
        assert {root for root, enabled in plan["jobs"]["sdist_package_roots"].items() if enabled} == {
            package_11_7.package_root
        }

        package_decision = plan["modules"]["bindings"]["package_roots"][package_11_7.package_root]
        assert package_decision["package_root"] == package_11_7.package_root
        assert package_decision["ctk_target"] == package_11_7.ctk_target
        assert package_decision["cuda_major"] == "11"
        assert package_decision["cuda_variant"] == "cu11"
        assert package_decision["release_status"] == "maintenance"

        release_plan = plan_for(
            baseline=False,
            release_tag="v11.8.1",
            bindings_config=config,
        )
        assert selected_package_roots(release_plan, "bindings", "needs_build") == {package_11_8.package_root}
        assert selected_package_roots(release_plan, "python", "needs_build") == {package_11_8.package_root}

    @pytest.mark.agent_authored(model="gpt-5.6-sol")
    def test_current_package_can_move_to_a_new_cuda_major(self) -> None:
        package_11 = synthetic_package("cuda_bindings_11", "11.8", "maintenance")
        package_12 = synthetic_package("cuda_bindings_12", "12.0", "current")
        config = synthetic_config(
            package_11,
            package_12,
        )

        plan = plan_for(
            "cuda_bindings_12/cuda/bindings/runtime.pyx",
            bindings_config=config,
        )

        assert selected_package_roots(plan, "bindings", "needs_build") == {package_12.package_root}
        assert selected_package_roots(plan, "python", "needs_build") == {package_12.package_root}
        assert selected_core_majors(plan, "needs_build") == {"cu11", "cu12"}
        assert selected_core_majors(plan, "needs_test") == {"cu12"}
        assert selected_cuda_majors(plan, "test_cuda_majors") == {"cu12"}
        assert selected_sdist_variants(plan) == {"cu12"}

        core_plan = plan_for("cuda_core/cuda/core/_device.py", bindings_config=config)
        assert selected_core_majors(core_plan, "needs_build") == {"cu11", "cu12"}
        assert selected_package_roots(core_plan, "python", "needs_test") == {
            package_11.package_root,
            package_12.package_root,
        }


@pytest.mark.agent_authored(model="gpt-5.6")
def test_github_outputs_are_emitted_without_shell_json_transforms(tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    summary = tmp_path / "github-summary"

    main(
        [
            "--release-tag",
            "v13.3.0",
            "--github-output",
            str(output),
            "--github-step-summary",
            str(summary),
        ]
    )

    records = dict(line.split("=", maxsplit=1) for line in output.read_text(encoding="utf-8").splitlines())
    assert set(records) == {"bindings-config", "workplan"}
    registry = json.loads(records["bindings-config"])
    assert [package["package_root"] for package in registry["package_roots"]] == [
        "cuda_bindings_12",
        "cuda_bindings",
    ]
    assert json.loads(records["workplan"])["jobs"]["sdist_package_roots"] == {
        "cuda_bindings_12": False,
        "cuda_bindings": True,
    }
    assert "### CI workplan" in summary.read_text(encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
