# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural invariant tests for the authored descriptor catalog.

These verify properties that should always hold for any valid catalog
entry, rather than comparing the catalog against itself.
"""

from __future__ import annotations

import re

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG, DescriptorSpec

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_STRATEGIES = {"ctk", "other", "driver"}
_CATALOG_BY_NAME = {spec.name: spec for spec in DESCRIPTOR_CATALOG}


def test_catalog_names_are_unique():
    names = [spec.name for spec in DESCRIPTOR_CATALOG]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_name_is_valid_identifier(spec: DescriptorSpec):
    assert _VALID_NAME_RE.match(spec.name), f"{spec.name!r} is not a valid Python identifier"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_strategy_is_valid(spec: DescriptorSpec):
    assert spec.strategy in _VALID_STRATEGIES


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_has_at_least_one_soname_or_dll(spec: DescriptorSpec):
    assert spec.linux_sonames or spec.windows_dlls, f"{spec.name} has no sonames or DLLs"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_dependencies_reference_existing_entries(spec: DescriptorSpec):
    for dep in spec.dependencies:
        assert dep in _CATALOG_BY_NAME, f"{spec.name} depends on unknown library {dep!r}"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_no_self_dependency(spec: DescriptorSpec):
    assert spec.name not in spec.dependencies, f"{spec.name} lists itself as a dependency"


@pytest.mark.parametrize(
    "spec",
    [s for s in DESCRIPTOR_CATALOG if s.strategy == "driver"],
    ids=lambda s: s.name,
)
def test_driver_libs_have_no_site_packages(spec: DescriptorSpec):
    """Driver libs are system-search-only; site-packages paths would be unused."""
    assert not spec.site_packages_linux, f"driver lib {spec.name} has site_packages_linux"
    assert not spec.site_packages_windows, f"driver lib {spec.name} has site_packages_windows"


@pytest.mark.parametrize(
    "spec",
    [s for s in DESCRIPTOR_CATALOG if s.strategy == "driver"],
    ids=lambda s: s.name,
)
def test_driver_libs_have_no_dependencies(spec: DescriptorSpec):
    """Driver libs skip the full cascade and shouldn't declare deps."""
    assert not spec.dependencies, f"driver lib {spec.name} has dependencies"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_linux_sonames_look_like_sonames(spec: DescriptorSpec):
    for soname in spec.linux_sonames:
        assert soname.startswith("lib"), f"Unexpected Linux soname format: {soname}"
        assert ".so" in soname, f"Unexpected Linux soname format: {soname}"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_windows_dlls_look_like_dlls(spec: DescriptorSpec):
    for dll in spec.windows_dlls:
        assert dll.endswith(".dll"), f"Unexpected Windows DLL format: {dll}"
