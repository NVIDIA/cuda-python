# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests verifying that the LibDescriptor registry faithfully represents
the current-platform data tables in supported_nvidia_libs.py."""

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import is_supported_on_current_machine
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    DIRECT_DEPENDENCIES,
    LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
    LIBNAMES_REQUIRING_RTLD_DEEPBIND,
    SITE_PACKAGES_LIBDIRS_LINUX,
    SITE_PACKAGES_LIBDIRS_WINDOWS,
    SUPPORTED_LIBNAMES,
    SUPPORTED_LINUX_SONAMES,
    SUPPORTED_WINDOWS_DLLS,
)

# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_covers_all_linux_sonames():
    assert set(SUPPORTED_LINUX_SONAMES) <= set(LIB_DESCRIPTORS)


def test_registry_covers_all_windows_dlls():
    assert set(SUPPORTED_WINDOWS_DLLS) <= set(LIB_DESCRIPTORS)


def test_registry_has_no_extra_entries():
    expected = set(SUPPORTED_LINUX_SONAMES) | set(SUPPORTED_WINDOWS_DLLS)
    extra = set(LIB_DESCRIPTORS) - expected
    assert all(not is_supported_on_current_machine(LIB_DESCRIPTORS[name]) for name in extra)


# ---------------------------------------------------------------------------
# Per-field consistency with source dicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_linux_sonames_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.linux_sonames if is_supported_on_current_machine(desc) else ()
    assert expected == SUPPORTED_LINUX_SONAMES.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_windows_dlls_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.windows_dlls if is_supported_on_current_machine(desc) else ()
    assert expected == SUPPORTED_WINDOWS_DLLS.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_site_packages_linux_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.site_packages_linux if is_supported_on_current_machine(desc) else ()
    assert expected == SITE_PACKAGES_LIBDIRS_LINUX.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_site_packages_windows_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.site_packages_windows if is_supported_on_current_machine(desc) else ()
    assert expected == SITE_PACKAGES_LIBDIRS_WINDOWS.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_dependencies_match(name):
    assert LIB_DESCRIPTORS[name].dependencies == DIRECT_DEPENDENCIES.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_requires_add_dll_directory_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.requires_add_dll_directory if is_supported_on_current_machine(desc) else False
    assert expected == (name in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY)


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_requires_rtld_deepbind_match(name):
    desc = LIB_DESCRIPTORS[name]
    expected = desc.requires_rtld_deepbind if is_supported_on_current_machine(desc) else False
    assert expected == (name in LIBNAMES_REQUIRING_RTLD_DEEPBIND)


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SUPPORTED_LIBNAMES))
def test_ctk_libs_have_ctk_packaging(name):
    assert LIB_DESCRIPTORS[name].packaged_with == "ctk"


def test_other_libs_have_other_packaging():
    # Spot-check a few known "other" libs
    for name in ("nccl", "cutensor", "cusparseLt"):
        if name in LIB_DESCRIPTORS:
            assert LIB_DESCRIPTORS[name].packaged_with == "other", name


# ---------------------------------------------------------------------------
# Descriptor properties
# ---------------------------------------------------------------------------


def test_descriptor_is_frozen():
    desc = LIB_DESCRIPTORS["cudart"]
    with pytest.raises(AttributeError):
        desc.name = "bogus"  # type: ignore[misc]
