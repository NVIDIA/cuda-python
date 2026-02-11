# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests verifying that the LibDescriptor registry faithfully represents
the existing data tables in supported_nvidia_libs.py."""

import pytest

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS, LibDescriptor
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
    assert set(LIB_DESCRIPTORS) == expected


# ---------------------------------------------------------------------------
# Per-field consistency with source dicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_linux_sonames_match(name):
    assert LIB_DESCRIPTORS[name].linux_sonames == SUPPORTED_LINUX_SONAMES.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_windows_dlls_match(name):
    assert LIB_DESCRIPTORS[name].windows_dlls == SUPPORTED_WINDOWS_DLLS.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_site_packages_linux_match(name):
    assert LIB_DESCRIPTORS[name].site_packages_linux == SITE_PACKAGES_LIBDIRS_LINUX.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_site_packages_windows_match(name):
    assert LIB_DESCRIPTORS[name].site_packages_windows == SITE_PACKAGES_LIBDIRS_WINDOWS.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_dependencies_match(name):
    assert LIB_DESCRIPTORS[name].dependencies == DIRECT_DEPENDENCIES.get(name, ())


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_requires_add_dll_directory_match(name):
    assert LIB_DESCRIPTORS[name].requires_add_dll_directory == (name in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY)


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_requires_rtld_deepbind_match(name):
    assert LIB_DESCRIPTORS[name].requires_rtld_deepbind == (name in LIBNAMES_REQUIRING_RTLD_DEEPBIND)


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SUPPORTED_LIBNAMES))
def test_ctk_libs_have_ctk_strategy(name):
    assert LIB_DESCRIPTORS[name].strategy == "ctk"


def test_other_libs_have_other_strategy():
    # Spot-check a few known "other" libs
    for name in ("nccl", "cutensor", "cusparseLt"):
        if name in LIB_DESCRIPTORS:
            assert LIB_DESCRIPTORS[name].strategy == "other", name


# ---------------------------------------------------------------------------
# Descriptor properties
# ---------------------------------------------------------------------------


def test_descriptor_is_frozen():
    desc = LIB_DESCRIPTORS["cudart"]
    with pytest.raises(AttributeError):
        desc.name = "bogus"  # type: ignore[misc]
