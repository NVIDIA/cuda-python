# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the authored descriptor catalog."""

from __future__ import annotations

import pytest

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS


def _catalog_by_name():
    return {spec.name: spec for spec in DESCRIPTOR_CATALOG}


def test_catalog_names_are_unique():
    names = [spec.name for spec in DESCRIPTOR_CATALOG]
    assert len(names) == len(set(names))


def test_catalog_and_registry_cover_same_libs():
    assert set(_catalog_by_name()) == set(LIB_DESCRIPTORS)


@pytest.mark.parametrize("name", sorted(LIB_DESCRIPTORS))
def test_catalog_spec_matches_registry(name):
    spec = _catalog_by_name()[name]
    desc = LIB_DESCRIPTORS[name]
    assert spec.strategy == desc.strategy
    assert spec.linux_sonames == desc.linux_sonames
    assert spec.windows_dlls == desc.windows_dlls
    assert spec.site_packages_linux == desc.site_packages_linux
    assert spec.site_packages_windows == desc.site_packages_windows
    assert spec.dependencies == desc.dependencies
    assert spec.anchor_rel_dirs_linux == desc.anchor_rel_dirs_linux
    assert spec.anchor_rel_dirs_windows == desc.anchor_rel_dirs_windows
    assert spec.requires_add_dll_directory == desc.requires_add_dll_directory
    assert spec.requires_rtld_deepbind == desc.requires_rtld_deepbind
