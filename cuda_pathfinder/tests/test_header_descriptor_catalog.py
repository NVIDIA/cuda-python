#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

from cuda.pathfinder._headers.header_descriptor_catalog import HEADER_DESCRIPTOR_CATALOG, HeaderDescriptorSpec

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALID_RELATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_catalog_names_are_unique():
    names = [spec.name for spec in HEADER_DESCRIPTOR_CATALOG]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("spec", HEADER_DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_name_is_valid_identifier(spec: HeaderDescriptorSpec):
    assert _VALID_NAME_RE.match(spec.name), f"{spec.name!r} is not a valid Python identifier"


@pytest.mark.parametrize("spec", HEADER_DESCRIPTOR_CATALOG, ids=lambda s: s.name)
def test_ctk_companion_tags_are_unique_and_valid(spec: HeaderDescriptorSpec):
    assert len(spec.ctk_companion_tags) == len(set(spec.ctk_companion_tags))
    for tag in spec.ctk_companion_tags:
        assert _VALID_RELATION_NAME_RE.match(tag)


def test_known_toolchain_headers_share_toolchain_tag():
    expected = {"cccl", "cudart", "cutlass", "cute", "nvcc", "nvfatbin", "nvvm"}
    actual = {
        spec.name
        for spec in HEADER_DESCRIPTOR_CATALOG
        if "toolchain_cuda_nvcc" in spec.ctk_companion_tags
    }
    assert actual == expected


def test_supported_runtime_headers_keep_companion_tags():
    expected = {
        "cublas": ("api_cublas",),
        "cudart": ("api_cudart", "toolchain_cuda_nvcc"),
        "cufft": ("api_cufft",),
        "curand": ("api_curand",),
        "cusolver": ("api_cusolver",),
        "cusparse": ("api_cusparse",),
        "npp": ("api_npp",),
        "nvjitlink": ("api_nvjitlink",),
        "nvrtc": ("api_nvrtc",),
        "nvvm": ("api_nvvm", "toolchain_cuda_nvcc"),
    }
    actual = {
        spec.name: spec.ctk_companion_tags
        for spec in HEADER_DESCRIPTOR_CATALOG
        if spec.name in expected
    }
    assert actual == expected
