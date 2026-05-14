# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import cuda.core
from cuda.core import (
    AddressMode,
    Array,
    ArrayFormat,
    Device,
    FilterMode,
    MipmappedArray,
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


# --- MipmappedArray ----------------------------------------------------------

def test_mipmapped_array_init_disabled():
    with pytest.raises(
        RuntimeError, match=r"^MipmappedArray cannot be instantiated directly"
    ):
        cuda.core._mipmapped_array.MipmappedArray()


def test_mipmapped_array_from_descriptor_2d(init_cuda):
    mip = MipmappedArray.from_descriptor(
        shape=(64, 32),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        num_levels=4,
    )
    try:
        assert mip.shape == (64, 32)
        assert mip.format == ArrayFormat.FLOAT32
        assert mip.num_channels == 1
        assert mip.num_levels == 4
        assert mip.surface_load_store is False
        assert mip.handle != 0
        assert isinstance(mip.device, Device)
    finally:
        mip.close()


def test_mipmapped_array_get_level_zero_matches_shape(init_cuda):
    shape = (64, 32)
    mip = MipmappedArray.from_descriptor(
        shape=shape,
        format=ArrayFormat.UINT8,
        num_channels=4,
        num_levels=4,
    )
    try:
        lvl0 = mip.get_level(0)
        try:
            assert isinstance(lvl0, Array)
            # Level 0 must match the base shape and rank.
            assert lvl0.shape == shape
            assert lvl0.format == ArrayFormat.UINT8
            assert lvl0.num_channels == 4
            assert lvl0.handle != 0
        finally:
            lvl0.close()
    finally:
        mip.close()


def test_mipmapped_array_get_level_halves_dims(init_cuda):
    shape = (64, 32)
    num_levels = 4
    mip = MipmappedArray.from_descriptor(
        shape=shape,
        format=ArrayFormat.UINT8,
        num_channels=1,
        num_levels=num_levels,
    )
    try:
        for level in range(num_levels):
            lvl = mip.get_level(level)
            try:
                # Each dim halves per level, with a floor of 1; rank is preserved.
                expected = tuple(max(1, dim >> level) for dim in shape)
                assert lvl.shape == expected, (
                    f"level={level}: expected {expected}, got {lvl.shape}"
                )
            finally:
                lvl.close()
    finally:
        mip.close()


def test_mipmapped_array_get_level_out_of_range(init_cuda):
    mip = MipmappedArray.from_descriptor(
        shape=(16, 16),
        format=ArrayFormat.UINT8,
        num_channels=1,
        num_levels=2,
    )
    try:
        with pytest.raises(ValueError, match="num_levels"):
            mip.get_level(mip.num_levels)
        with pytest.raises(ValueError, match=">= 0"):
            mip.get_level(-1)
    finally:
        mip.close()


def test_mipmapped_array_rejects_zero_levels(init_cuda):
    with pytest.raises(ValueError, match="num_levels"):
        MipmappedArray.from_descriptor(
            shape=(8, 8),
            format=ArrayFormat.UINT8,
            num_channels=1,
            num_levels=0,
        )


def test_resource_descriptor_from_mipmapped_array(init_cuda):
    mip = MipmappedArray.from_descriptor(
        shape=(32, 16),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        num_levels=3,
    )
    try:
        res = ResourceDescriptor.from_mipmapped_array(mip)
        assert res.kind == "mipmapped_array"
        assert res.source is mip
    finally:
        mip.close()


def test_resource_descriptor_from_mipmapped_array_rejects_non_mipmap():
    with pytest.raises(TypeError, match="MipmappedArray"):
        ResourceDescriptor.from_mipmapped_array(object())


def test_texture_object_from_mipmapped_array(init_cuda):
    mip = MipmappedArray.from_descriptor(
        shape=(32, 32),
        format=ArrayFormat.FLOAT32,
        num_channels=1,
        num_levels=3,
    )
    try:
        res = ResourceDescriptor.from_mipmapped_array(mip)
        # Use non-default mipmap params so the driver exercises that path.
        tex_desc = TextureDescriptor(
            address_mode=AddressMode.CLAMP,
            filter_mode=FilterMode.LINEAR,
            normalized_coords=True,
            mipmap_filter_mode=FilterMode.LINEAR,
            mipmap_level_bias=0.0,
            min_mipmap_level_clamp=0.0,
            max_mipmap_level_clamp=float(mip.num_levels - 1),
        )
        tex = TextureObject.from_descriptor(res, tex_desc)
        try:
            assert tex.handle != 0
            assert tex.resource is res
        finally:
            tex.close()
    finally:
        mip.close()


def test_surface_rejects_mipmapped_array(init_cuda):
    mip = MipmappedArray.from_descriptor(
        shape=(16, 16),
        format=ArrayFormat.UINT8,
        num_channels=4,
        num_levels=2,
        surface_load_store=True,
    )
    try:
        res = ResourceDescriptor.from_mipmapped_array(mip)
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(res)
    finally:
        mip.close()


def test_mipmapped_array_level_keeps_parent_alive(init_cuda):
    """Dropping the local parent reference must not invalidate the level Array;
    the level holds an internal strong ref back to the MipmappedArray.

    cdef classes don't natively support weakref, so we verify the parent
    reference by inspecting the level Array's gc referents.
    """
    mip = MipmappedArray.from_descriptor(
        shape=(16, 16),
        format=ArrayFormat.UINT8,
        num_channels=1,
        num_levels=3,
    )
    parent_id = id(mip)
    lvl = mip.get_level(1)
    # Drop our local reference and force GC; the parent must survive because
    # the level Array holds a strong ref via the internal _parent_ref slot.
    del mip
    gc.collect()

    # The handle is still valid storage; the level still tracks the parent.
    assert lvl.handle != 0
    referents = gc.get_referents(lvl)
    parents = [r for r in referents if isinstance(r, MipmappedArray)]
    assert len(parents) == 1, (
        f"level Array should reference exactly one MipmappedArray parent, got "
        f"{parents!r}"
    )
    assert id(parents[0]) == parent_id, (
        "level Array's parent ref is not the original MipmappedArray"
    )
    # Closing the level drops its parent ref. Don't access the parent past
    # this point; cuMipmappedArrayDestroy may then run.
    lvl.close()
