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


# --- Linear / pitch2D resource descriptors -----------------------------------

def _alloc_device_buffer(device, nbytes):
    """Allocate a device Buffer using the device's default memory resource."""
    return device.memory_resource.allocate(nbytes, stream=device.default_stream)


def test_resource_descriptor_from_linear_defaults_size(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        res = ResourceDescriptor.from_linear(
            buf, format=ArrayFormat.FLOAT32, num_channels=1
        )
        assert res.kind == "linear"
        assert res.format == ArrayFormat.FLOAT32
        assert res.num_channels == 1
        assert res.source is buf
        # repr should include the kind/format hint
        assert "linear" in repr(res)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_size_override(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        res = ResourceDescriptor.from_linear(
            buf, format=ArrayFormat.UINT32, num_channels=1, size_bytes=2048
        )
        assert res._size_bytes == 2048
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_oversize(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="exceeds buffer.size"):
            ResourceDescriptor.from_linear(
                buf, format=ArrayFormat.UINT8, num_channels=1, size_bytes=2048
            )
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_bad_channels(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="num_channels"):
            ResourceDescriptor.from_linear(
                buf, format=ArrayFormat.UINT8, num_channels=3
            )
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_non_buffer():
    with pytest.raises(TypeError, match="Buffer"):
        ResourceDescriptor.from_linear(
            object(), format=ArrayFormat.UINT8, num_channels=1
        )


def test_texture_object_from_linear(init_cuda):
    """A linear-backed texture should bind even though sampling fields are
    effectively ignored by the driver."""
    device = Device()
    # 1024 float elements
    buf = _alloc_device_buffer(device, 1024 * 4)
    try:
        res = ResourceDescriptor.from_linear(
            buf, format=ArrayFormat.FLOAT32, num_channels=1
        )
        tex = TextureObject.from_descriptor(res, TextureDescriptor())
        try:
            assert tex.handle != 0
            assert tex.resource is res
        finally:
            tex.close()
    finally:
        buf.close()


def test_resource_descriptor_from_pitch2d_validates_pitch(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 64 * 1024)
    try:
        # element_size = 4 (UINT32 * 1 channel); width=16 -> min_pitch=64
        with pytest.raises(ValueError, match="pitch_bytes"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormat.UINT32,
                num_channels=1,
                width=16,
                height=8,
                pitch_bytes=32,  # < 64 = width*element_size
            )
    finally:
        buf.close()


def test_resource_descriptor_from_pitch2d_validates_buffer_size(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        with pytest.raises(ValueError, match="exceeds buffer.size"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormat.UINT8,
                num_channels=4,
                width=64,
                height=128,
                pitch_bytes=512,  # 512 * 128 = 65536 > 4096
            )
    finally:
        buf.close()


def test_texture_object_from_pitch2d(init_cuda):
    """A pitch2D-backed texture should bind given driver-aligned pitch."""
    from cuda.bindings import driver

    device = Device()
    # Query the device's required texture pitch alignment (typically 32-512).
    err, align = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT,
        device.device_id,
    )
    assert int(err) == 0
    pitch = max(int(align), 256)
    height = 16
    buf = _alloc_device_buffer(device, pitch * height)
    try:
        res = ResourceDescriptor.from_pitch2d(
            buf,
            format=ArrayFormat.UINT8,
            num_channels=4,
            width=32,
            height=height,
            pitch_bytes=pitch,
        )
        assert res.kind == "pitch2d"
        assert "pitch2d" in repr(res)
        tex = TextureObject.from_descriptor(res, TextureDescriptor())
        try:
            assert tex.handle != 0
        finally:
            tex.close()
    finally:
        buf.close()


def test_surface_rejects_linear_and_pitch2d(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        res_lin = ResourceDescriptor.from_linear(
            buf, format=ArrayFormat.UINT32, num_channels=1
        )
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(res_lin)

        res_p2 = ResourceDescriptor.from_pitch2d(
            buf,
            format=ArrayFormat.UINT8,
            num_channels=4,
            width=8,
            height=8,
            pitch_bytes=64,
        )
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(res_p2)
    finally:
        buf.close()
