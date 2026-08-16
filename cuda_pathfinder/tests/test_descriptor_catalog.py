# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural invariant tests for the authored descriptor catalog.

These verify properties that should always hold for any valid catalog
entry, rather than comparing the catalog against itself.
"""

from __future__ import annotations

import re

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG, DescriptorSpec, WindowsSearchDirs

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_PACKAGED_WITH_VALUES = {"ctk", "other", "driver"}
_VALID_WINDOWS_ARCHES = ("x64", "arm64")
_CATALOG_BY_NAME = {spec.name: spec for spec in DESCRIPTOR_CATALOG}


def test_catalog_names_are_unique():
    names = [spec.name for spec in DESCRIPTOR_CATALOG]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_name_is_valid_identifier(spec: DescriptorSpec):
    assert _VALID_NAME_RE.match(spec.name), f"{spec.name!r} is not a valid Python identifier"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_packaged_with_is_valid(spec: DescriptorSpec):
    assert spec.packaged_with in _VALID_PACKAGED_WITH_VALUES


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
    [s for s in DESCRIPTOR_CATALOG if s.packaged_with == "driver"],
    ids=lambda s: s.name,
)
def test_driver_libs_have_no_site_packages(spec: DescriptorSpec):
    """Driver libs are system-search-only; site-packages paths would be unused."""
    assert not spec.site_packages_linux, f"driver lib {spec.name} has site_packages_linux"
    assert spec.site_packages_windows == WindowsSearchDirs(), f"driver lib {spec.name} has site_packages_windows"


@pytest.mark.parametrize(
    "spec",
    [s for s in DESCRIPTOR_CATALOG if s.packaged_with == "driver"],
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


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
@pytest.mark.agent_authored(model="gpt-5")
def test_supported_windows_arch_is_explicit_and_canonical(spec: DescriptorSpec):
    expected = tuple(arch for arch in _VALID_WINDOWS_ARCHES if arch in spec.supported_windows_arch)
    assert spec.supported_windows_arch == expected
    assert bool(spec.supported_windows_arch) == bool(spec.windows_dlls)


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
@pytest.mark.agent_authored(model="gpt-5")
def test_windows_search_dirs_do_not_include_unsupported_arches(spec: DescriptorSpec):
    if not spec.windows_dlls:
        return
    for arch in _VALID_WINDOWS_ARCHES:
        if arch not in spec.supported_windows_arch:
            assert not spec.site_packages_windows.for_arch(arch)
            assert not spec.anchor_rel_dirs_windows.for_arch(arch)


@pytest.mark.agent_authored(model="gpt-5")
def test_cusparselt_windows_metadata_matches_wheel_layouts():
    spec = _CATALOG_BY_NAME["cusparseLt"]
    assert spec.supported_windows_arch == ("x64", "arm64")
    assert spec.site_packages_windows == WindowsSearchDirs(
        x64=("nvidia/cu13/bin/x64", "nvidia/cusparselt/bin"),
        arm64=("nvidia/cu13/bin/arm64",),
    )


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_ctk_root_canary_anchors_reference_known_ctk_libs(spec: DescriptorSpec):
    for anchor in spec.ctk_root_canary_anchor_libnames:
        assert anchor in _CATALOG_BY_NAME, f"{spec.name} has unknown canary anchor {anchor!r}"
        assert _CATALOG_BY_NAME[anchor].packaged_with == "ctk", f"{spec.name} has non-CTK canary anchor {anchor!r}"


@pytest.mark.parametrize("spec", DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_only_ctk_libs_define_ctk_root_canary_anchors(spec: DescriptorSpec):
    if spec.ctk_root_canary_anchor_libnames:
        assert spec.packaged_with == "ctk", f"{spec.name} defines canary anchors but is not a CTK lib"


@pytest.mark.agent_authored(model="gpt-5")
def test_only_nvvm_requires_windows_binary_arch_check():
    checked_libs = {spec.name for spec in DESCRIPTOR_CATALOG if spec.requires_windows_binary_arch_check}

    assert checked_libs == {"nvvm"}
