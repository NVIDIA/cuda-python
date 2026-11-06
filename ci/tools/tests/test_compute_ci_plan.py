# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect CI selection outcomes that could otherwise silently omit coverage."""

from dataclasses import replace
from pathlib import Path

import pytest

from ci.tools.bindings_config import load_config
from ci.tools.compute_ci_plan import _expand_linked_paths, compute_workplan

CONFIG = load_config()
PACKAGES = CONFIG.package_roots
ROOTS = {package.package_root for package in PACKAGES}
VARIANTS = {package.cuda_variant for package in PACKAGES}
BINDINGS = {("bindings", root) for root in ROOTS}
CORE = {("core", variant) for variant in VARIANTS}
META = {("python", root) for root in ROOTS}
ALL = {("pathfinder", None), *BINDINGS, *CORE, *META}


def plan_for(*paths, baseline=True, release_tag="", config=CONFIG, linked_paths=()):
    return compute_workplan(
        bindings_config=config,
        paths=list(paths),
        linked_paths=set(linked_paths),
        merge_base="base",
        baseline_run_id="123" if baseline else "",
        release_tag=release_tag,
    )


def selected(plan, decision):
    targets = set()
    for component, details in plan["modules"].items():
        if component == "pathfinder":
            if details[decision]:
                targets.add((component, None))
            continue
        selectors = details["cuda_majors" if component == "core" else "package_roots"]
        targets.update((component, selector) for selector, flags in selectors.items() if flags[decision])
    return targets


@pytest.mark.parametrize(
    ("path", "builds", "tests"),
    [
        ("cuda_pathfinder/cuda/pathfinder/_loader.py", ALL, ALL),
        ("cuda_core/cuda/core/_device.py", CORE, CORE | META),
        ("cuda_python/setup.py", BINDINGS | META, META),
        ("cuda_core/tests/test_device.py", set(), CORE),
        ("cuda_pathfinder/tests/test_loader.py", set(), {("pathfinder", None)}),
        ("cuda_python/tests/test_import.py", set(), META),
        ("cuda_python_test_helpers/cuda_python_test_helpers/cuda_utils.py", set(), ALL),
        ("benchmarks/cuda_bindings/run_pyperf.py", set(), ALL),
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_source_and_test_changes_select_their_consumers(path, builds, tests):
    plan = plan_for(path)
    assert selected(plan, "needs_build") == builds
    assert selected(plan, "needs_test") == tests
    assert {platform for platform, enabled in plan["jobs"]["platforms"].items() if enabled} == {"linux", "windows"}


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.package_root)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_bindings_changes_reuse_core_and_test_the_matching_major(package):
    own = {("bindings", package.package_root), ("python", package.package_root)}
    plan = plan_for(f"{package.package_root}/cuda/bindings/driver.pyx")
    assert selected(plan, "needs_build") == own
    assert selected(plan, "needs_test") == own | {("core", package.cuda_variant)}
    assert plan["baseline"] == {"run_id": "123", "sha": "base"}
    assert {root for root, enabled in plan["jobs"]["sdist_package_roots"].items() if enabled} == {package.package_root}

    tests_only = plan_for(f"{package.package_root}/tests/test_driver.py")
    assert selected(tests_only, "needs_build") == set()
    assert selected(tests_only, "needs_test") == {("bindings", package.package_root)}


@pytest.mark.parametrize("path", ["ci/tools/env-vars", ".github/workflows/build-wheel.yml", "unknown-build-input"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_shared_or_unknown_infrastructure_requires_every_component(path):
    plan = plan_for(path)
    assert selected(plan, "needs_build") == ALL
    assert selected(plan, "needs_test") == ALL
    assert plan["baseline"]["run_id"] == ""


@pytest.mark.parametrize("platform", ["linux", "windows"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_platform_test_infrastructure_reuses_wheels_but_runs_every_suite(platform):
    plan = plan_for(f".github/workflows/test-wheel-{platform}.yml")
    assert selected(plan, "needs_build") == set()
    assert selected(plan, "needs_test") == ALL
    assert {name for name, enabled in plan["jobs"]["platforms"].items() if enabled} == {platform}


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "AGENTS.md",
        "cuda_core/docs/source/install.rst",
        "cuda_core/pixi.lock",
        "cuda_core/pixi.toml",
        ".github/workflows/runner-trust-probe.yml",
        "ci/tools/runner_trust_probe.sh",
        "ci/tools/runner_trust_probe.ps1",
        "ci/tools/runner_trust_probe_node.js",
    ],
)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_documentation_and_environment_only_changes_do_not_rebuild_packages(path):
    plan = plan_for(path)
    assert selected(plan, "needs_build") == set()
    assert selected(plan, "needs_test") == set()


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_missing_baselines_and_non_bindings_tags_require_full_coverage():
    cases = [plan_for(baseline=False), plan_for(release_tag="cuda-core-v1.3.0")]
    cases.extend(plan_for(f"{package.package_root}/cuda/bindings/driver.pyx", baseline=False) for package in PACKAGES)
    for plan in cases:
        assert selected(plan, "needs_build") == ALL
        assert selected(plan, "needs_test") == ALL
        assert plan["baseline"] == {"run_id": "", "sha": ""}


@pytest.mark.parametrize("package", PACKAGES, ids=lambda package: package.package_root)
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_release_tags_build_only_their_own_bindings_and_metapackage(package):
    plan = plan_for("ci/tools/env-vars", release_tag=f"v{package.ctk_target}.99")
    expected = {("bindings", package.package_root), ("python", package.package_root)}
    assert selected(plan, "needs_build") == expected
    assert selected(plan, "needs_test") == expected
    assert plan["sources"]["pathfinder"] == "published"
    assert plan["baseline"]["run_id"] == ""
    assert {root for root, enabled in plan["jobs"]["sdist_package_roots"].items() if enabled} == {package.package_root}


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_mixed_changes_combine_coverage_without_rebuilding_unchanged_core():
    package = PACKAGES[0]
    plan = plan_for(f"{package.package_root}/cuda/bindings/driver.pyx", "cuda_core/tests/test_device.py")
    own = {("bindings", package.package_root), ("python", package.package_root)}
    assert selected(plan, "needs_build") == own
    assert selected(plan, "needs_test") == own | CORE

    both = plan_for(*(f"{package.package_root}/cuda/bindings/driver.pyx" for package in PACKAGES))
    assert selected(both, "needs_build") == BINDINGS | META
    assert selected(both, "needs_test") == BINDINGS | CORE | META


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_symlinked_package_inputs_are_not_mistaken_for_documentation_only(tmp_path: Path):
    (tmp_path / "cuda_python").mkdir()
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "cuda_python" / "README.md").symlink_to("../README.md")
    paths = _expand_linked_paths(["README.md"], ["cuda_python/README.md"], root=tmp_path)
    for changes in (paths, ["cuda_python/README.md"]):
        plan = plan_for(*changes, linked_paths={"cuda_python/README.md"})
        assert selected(plan, "needs_build") == BINDINGS | META
        assert selected(plan, "needs_test") == META


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_registered_source_roots_drive_bindings_selection():
    packages = tuple(replace(package, package_root=f"renamed_{index}") for index, package in enumerate(PACKAGES))
    config = replace(CONFIG, package_roots=packages)
    package = packages[0]
    plan = plan_for(f"{package.package_root}/cuda/bindings/driver.pyx", config=config)
    own = {("bindings", package.package_root), ("python", package.package_root)}
    assert selected(plan, "needs_build") == own
    assert selected(plan, "needs_test") == own | {("core", package.cuda_variant)}
