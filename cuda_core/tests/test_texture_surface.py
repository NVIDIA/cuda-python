# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.core
from cuda.core import (
    AddressMode,
    Array,
    ArrayFormat,
    Device,
    FilterMode,
    ReadMode,
    ResourceDescriptor,
    SurfaceObject,
    TextureDescriptor,
    TextureObject,
)


def test_array_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Array cannot be instantiated directly"):
        cuda.core._array.Array()


def test_texture_object_init_disabled():
    with pytest.raises(RuntimeError, match=r"^TextureObject cannot be instantiated directly"):
        cuda.core._texture.TextureObject()


def test_surface_object_init_disabled():
    with pytest.raises(RuntimeError, match=r"^SurfaceObject cannot be instantiated directly"):
        cuda.core._surface.SurfaceObject()


def test_resource_descriptor_init_disabled():
    with pytest.raises(RuntimeError, match=r"^ResourceDescriptor cannot be instantiated"):
        ResourceDescriptor()


def test_array_2d_create_and_properties(init_cuda):
    arr = Array.from_descriptor(
        shape=(32, 16), format=ArrayFormat.FLOAT32, num_channels=1
    )
    try:
        assert arr.shape == (32, 16)
        assert arr.format == ArrayFormat.FLOAT32
        assert arr.num_channels == 1
        assert arr.element_size == 4
        assert arr.size_bytes == 32 * 16 * 4
        assert arr.surface_load_store is False
        assert arr.handle != 0
        assert isinstance(arr.device, Device)
    finally:
        arr.close()


def test_array_3d_with_surface_flag(init_cuda):
    arr = Array.from_descriptor(
        shape=(8, 8, 4),
        format=ArrayFormat.UINT8,
        num_channels=4,
        surface_load_store=True,
    )
    try:
        assert arr.shape == (8, 8, 4)
        assert arr.surface_load_store is True
        assert arr.element_size == 4
    finally:
        arr.close()


def test_array_rejects_bad_channels(init_cuda):
    with pytest.raises(ValueError, match="num_channels"):
        Array.from_descriptor(shape=(8,), format=ArrayFormat.UINT8, num_channels=3)


def test_array_rejects_bad_rank(init_cuda):
    with pytest.raises(ValueError, match="shape rank"):
        Array.from_descriptor(
            shape=(2, 2, 2, 2), format=ArrayFormat.UINT8, num_channels=1
        )


def test_array_roundtrip_copy(init_cuda):
    import array as _array

    device = Device()
    stream = device.create_stream()
    arr = Array.from_descriptor(
        shape=(16,), format=ArrayFormat.UINT32, num_channels=1
    )
    try:
        src = _array.array("I", list(range(16)))
        dst = _array.array("I", [0] * 16)
        arr.copy_from(src, stream=stream)
        arr.copy_to(dst, stream=stream)
        stream.sync()
        assert list(dst) == list(range(16))
    finally:
        arr.close()
        stream.close()


def test_texture_object_create(init_cuda):
    arr = Array.from_descriptor(
        shape=(32, 16), format=ArrayFormat.FLOAT32, num_channels=1
    )
    try:
        res = ResourceDescriptor.from_array(arr)
        tex_desc = TextureDescriptor(
            address_mode=AddressMode.CLAMP,
            filter_mode=FilterMode.LINEAR,
            read_mode=ReadMode.ELEMENT_TYPE,
            normalized_coords=True,
        )
        tex = TextureObject.from_descriptor(res, tex_desc)
        try:
            assert tex.handle != 0
            assert tex.resource is res
            assert tex.texture_descriptor is tex_desc
        finally:
            tex.close()
    finally:
        arr.close()


def test_surface_object_create(init_cuda):
    arr = Array.from_descriptor(
        shape=(8, 8),
        format=ArrayFormat.UINT8,
        num_channels=4,
        surface_load_store=True,
    )
    try:
        surf = SurfaceObject.from_array(arr)
        try:
            assert surf.handle != 0
            assert isinstance(surf.resource, ResourceDescriptor)
        finally:
            surf.close()
    finally:
        arr.close()


def test_surface_requires_ldst_flag(init_cuda):
    arr = Array.from_descriptor(
        shape=(8, 8), format=ArrayFormat.UINT8, num_channels=4
    )
    try:
        with pytest.raises(ValueError, match="surface_load_store=True"):
            SurfaceObject.from_array(arr)
    finally:
        arr.close()


def test_surface_rejects_non_array_resource(init_cuda):
    # ResourceDescriptor only exposes from_array today, so use a fake kind.
    arr = Array.from_descriptor(
        shape=(8, 8),
        format=ArrayFormat.UINT8,
        num_channels=4,
        surface_load_store=True,
    )
    try:
        res = ResourceDescriptor.from_array(arr)
        res._kind = "linear"  # simulate a future, unsupported resource kind
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(res)
    finally:
        arr.close()


def test_address_mode_normalization(init_cuda):
    arr = Array.from_descriptor(
        shape=(8, 8, 4), format=ArrayFormat.FLOAT32, num_channels=1
    )
    try:
        res = ResourceDescriptor.from_array(arr)
        # Per-axis tuple shorter than 3 should be accepted and padded.
        tex_desc = TextureDescriptor(
            address_mode=(AddressMode.WRAP, AddressMode.CLAMP)
        )
        tex = TextureObject.from_descriptor(res, tex_desc)
        tex.close()
    finally:
        arr.close()
