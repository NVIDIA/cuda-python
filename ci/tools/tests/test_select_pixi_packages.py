# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect source-build selection that could otherwise silently omit a root."""

import pytest

from ci.tools.bindings_config import BindingsConfig, BindingsPackage
from ci.tools.select_pixi_packages import select_pixi_packages

MAINTENANCE = BindingsPackage("maintenance_bindings", (12, 9, 1), "maintenance")
CURRENT = BindingsPackage("current_bindings", (13, 4, 2), "current")
CONFIG = BindingsConfig((CURRENT, MAINTENANCE), ("3.13",))


@pytest.mark.parametrize("package", CONFIG.package_roots, ids=lambda package: package.release_status)
@pytest.mark.parametrize("relative", ("cuda/bindings/driver.pyx", "pixi.lock", "docs/source/install.rst"))
@pytest.mark.agent_authored(model="gpt-6")
def test_package_source_manifest_and_docs_changes_select_only_their_root(package, relative):
    selected, docs = select_pixi_packages(CONFIG, paths=[f"{package.package_root}/{relative}"], linked_paths=set())

    assert selected == [package]
    assert docs == ([package] if relative.startswith("docs/") else [])


@pytest.mark.parametrize("path", ("cuda_core/build_hooks.py", "cuda_core/pixi.toml"))
@pytest.mark.agent_authored(model="gpt-6")
def test_shared_abi_or_environment_changes_select_both_roots(path):
    selected, docs = select_pixi_packages(CONFIG, paths=[path], linked_paths=set())

    assert set(selected) == set(CONFIG.package_roots)
    assert docs == []


@pytest.mark.agent_authored(model="gpt-6")
def test_mixed_docs_and_manifest_changes_keep_both_builds_but_only_changed_docs():
    selected, docs = select_pixi_packages(
        CONFIG,
        paths=[f"{MAINTENANCE.package_root}/docs/source/install.rst", f"{CURRENT.package_root}/pixi.toml"],
        linked_paths=set(),
    )

    assert set(selected) == set(CONFIG.package_roots)
    assert docs == [MAINTENANCE]


@pytest.mark.parametrize(("cuda_env", "package"), (("", CURRENT), ("cu12", MAINTENANCE)))
@pytest.mark.agent_authored(model="gpt-6")
def test_manual_runs_select_current_by_default_or_the_requested_registered_abi(cuda_env, package):
    selected, docs = select_pixi_packages(CONFIG, paths=[], linked_paths=set(), cuda_env=cuda_env)

    assert selected == [package]
    assert docs == []
