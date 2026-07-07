# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import numpy as np
import pytest

import cuda.core
from cuda.core import (
    Device,
)
from cuda.core.texture import (
    MipmappedArray,
    MipmappedArrayOptions,
    OpaqueArray,
    OpaqueArrayOptions,
    ResourceDescriptor,
    SurfaceObject,
    TextureObject,
    TextureObjectOptions,
)
from cuda.core.typing import (
    AddressModeType,
    ArrayFormatType,
    FilterModeType,
    ReadModeType,
)


def test_array_init_disabled():
    with pytest.raises(RuntimeError, match=r"^OpaqueArray cannot be instantiated directly"):
        cuda.core.texture._array.OpaqueArray()


def test_texture_object_init_disabled():
    with pytest.raises(RuntimeError, match=r"^TextureObject cannot be instantiated directly"):
        cuda.core.texture._texture.TextureObject()


def test_surface_object_init_disabled():
    with pytest.raises(RuntimeError, match=r"^SurfaceObject cannot be instantiated directly"):
        cuda.core.texture._surface.SurfaceObject()


def test_resource_descriptor_init_disabled():
    with pytest.raises(RuntimeError, match=r"^ResourceDescriptor cannot be instantiated"):
        ResourceDescriptor()


def test_array_2d_create_and_properties(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(32, 16), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        assert arr.shape == (32, 16)
        assert arr.format == ArrayFormatType.FLOAT32
        assert arr.num_channels == 1
        assert arr.element_bytes == 4
        assert arr.size_bytes == 32 * 16 * 4
        assert arr.is_surface_load_store is False
        assert arr.handle != 0
        assert isinstance(arr.device, Device)
    finally:
        arr.close()


def test_array_3d_with_surface_flag(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(
        shape=(8, 8, 4),
        format=ArrayFormatType.UINT8,
        num_channels=4,
        is_surface_load_store=True,
    ))
    try:
        assert arr.shape == (8, 8, 4)
        assert arr.is_surface_load_store is True
        assert arr.element_bytes == 4
    finally:
        arr.close()


@pytest.mark.agent_authored(model="claude-opus-4.8")
@pytest.mark.parametrize(
    "dtype, expected",
    [
        (np.float32, ArrayFormatType.FLOAT32),
        (np.dtype("float16"), ArrayFormatType.FLOAT16),
        (np.uint8, ArrayFormatType.UINT8),
        (np.dtype("i4"), ArrayFormatType.INT32),
    ],
)
def test_array_accepts_numpy_dtype_format(init_cuda, dtype, expected):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=dtype, num_channels=1))
    try:
        assert arr.format == expected
    finally:
        arr.close()


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_array_accepts_str_format(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format="float32", num_channels=1))
    try:
        assert arr.format == ArrayFormatType.FLOAT32
    finally:
        arr.close()


@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_array_rejects_unsupported_dtype_format(init_cuda):
    # float64 has no ArrayFormatType equivalent.
    with pytest.raises(ValueError, match="no ArrayFormatType equivalent"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=(8,), format=np.float64, num_channels=1))


def test_array_rejects_bad_channels(init_cuda):
    with pytest.raises(ValueError, match="num_channels"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=(8,), format=ArrayFormatType.UINT8, num_channels=3))


def test_array_rejects_bad_rank(init_cuda):
    with pytest.raises(ValueError, match="shape rank"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=(2, 2, 2, 2), format=ArrayFormatType.UINT8, num_channels=1))


def test_array_roundtrip_copy(init_cuda):
    import array as _array

    device = Device()
    stream = device.create_stream()
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(16,), format=ArrayFormatType.UINT32, num_channels=1))
    try:
        src = _array.array("I", list(range(16)))
        dst = _array.array("I", [0] * 16)
        arr.copy_from(src, stream=stream)
        arr.copy_to(dst, stream=stream)
        stream.sync()
        # Round-trip recovers data; src must not be mutated by copy_from.
        assert list(dst) == list(range(16))
        assert list(src) == list(range(16))
    finally:
        arr.close()
        stream.close()


def test_array_copy_rejects_undersized_host_buffer(init_cuda):
    import array as _array

    device = Device()
    stream = device.create_stream()
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(16,), format=ArrayFormatType.UINT32, num_channels=1))
    try:
        # arr is 16 * 4 = 64 bytes; pass an 8-element (32-byte) host buffer.
        too_small = _array.array("I", [0] * 8)
        with pytest.raises(ValueError, match="smaller than the array extent"):
            arr.copy_from(too_small, stream=stream)
        with pytest.raises(ValueError, match="smaller than the array extent"):
            arr.copy_to(too_small, stream=stream)
    finally:
        arr.close()
        stream.close()


def test_array_copy_rejects_undersized_device_buffer(init_cuda):
    device = Device()
    stream = device.create_stream()
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(16,), format=ArrayFormatType.UINT32, num_channels=1))
    # arr is 64 bytes; allocate a 32-byte device buffer.
    small_buf = device.memory_resource.allocate(32, stream=device.default_stream)
    try:
        with pytest.raises(ValueError, match="smaller than the array extent"):
            arr.copy_from(small_buf, stream=stream)
        with pytest.raises(ValueError, match="smaller than the array extent"):
            arr.copy_to(small_buf, stream=stream)
    finally:
        small_buf.close()
        arr.close()
        stream.close()


def test_texture_object_create(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(32, 16), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        tex_desc = TextureObjectOptions(
            address_mode=AddressModeType.CLAMP,
            filter_mode=FilterModeType.LINEAR,
            read_mode=ReadModeType.ELEMENT_TYPE,
            normalized_coords=True,
        )
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=tex_desc)
        try:
            assert tex.handle != 0
            assert tex.resource is res
            assert tex.texture_descriptor is tex_desc
        finally:
            tex.close()
    finally:
        arr.close()


def test_surface_object_create(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(
        shape=(8, 8),
        format=ArrayFormatType.UINT8,
        num_channels=4,
        is_surface_load_store=True,
    ))
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
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.UINT8, num_channels=4))
    try:
        with pytest.raises(ValueError, match="is_surface_load_store=True"):
            SurfaceObject.from_array(arr)
    finally:
        arr.close()


def test_address_mode_normalization(init_cuda):
    # Direct unit test of the private normalizer: a scalar should expand to a
    # 3-tuple; a shorter tuple should be padded by repeating the last entry.
    from cuda.core.texture._texture import _normalize_address_modes

    assert _normalize_address_modes(AddressModeType.WRAP) == (
        AddressModeType.WRAP,
        AddressModeType.WRAP,
        AddressModeType.WRAP,
    )
    assert _normalize_address_modes((AddressModeType.WRAP, AddressModeType.CLAMP)) == (
        AddressModeType.WRAP,
        AddressModeType.CLAMP,
        AddressModeType.CLAMP,
    )
    assert _normalize_address_modes((AddressModeType.WRAP, AddressModeType.CLAMP, AddressModeType.MIRROR)) == (
        AddressModeType.WRAP,
        AddressModeType.CLAMP,
        AddressModeType.MIRROR,
    )

    # Smoke test: a 2-entry tuple is also accepted end-to-end.
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8, 4), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        tex_desc = TextureObjectOptions(address_mode=(AddressModeType.WRAP, AddressModeType.CLAMP))
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=tex_desc)
        try:
            assert tex.handle != 0
        finally:
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
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormatType.FLOAT32, num_channels=1)
        assert res.kind == "linear"
        assert res.format == ArrayFormatType.FLOAT32
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
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT32, num_channels=1, size_bytes=2048)
        assert res._size_bytes == 2048
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_oversize(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="exceeds buffer.size"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT8, num_channels=1, size_bytes=2048)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_bad_channels(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="num_channels"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT8, num_channels=3)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_non_buffer():
    with pytest.raises(TypeError, match="Buffer"):
        ResourceDescriptor.from_linear(object(), format=ArrayFormatType.UINT8, num_channels=1)


def test_resource_descriptor_from_linear_rejects_zero_size(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="at least one element"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT32, num_channels=1, size_bytes=0)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_non_multiple(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        # UINT32 x 1 channel = 4 bytes/element; 10 bytes is not a multiple.
        with pytest.raises(ValueError, match="multiple of element size"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT32, num_channels=1, size_bytes=10)
    finally:
        buf.close()


def test_texture_object_from_linear(init_cuda):
    """A linear-backed texture should bind even though sampling fields are
    effectively ignored by the driver."""
    device = Device()
    # 1024 float elements
    buf = _alloc_device_buffer(device, 1024 * 4)
    try:
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormatType.FLOAT32, num_channels=1)
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureObjectOptions())
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
        # element_bytes = 4 (UINT32 * 1 channel); width=16 -> min_pitch=64
        with pytest.raises(ValueError, match="pitch_bytes"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormatType.UINT32,
                num_channels=1,
                width=16,
                height=8,
                pitch_bytes=32,  # < 64 = width*element_bytes
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
                format=ArrayFormatType.UINT8,
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
            format=ArrayFormatType.UINT8,
            num_channels=4,
            width=32,
            height=height,
            pitch_bytes=pitch,
        )
        assert res.kind == "pitch2d"
        assert "pitch2d" in repr(res)
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureObjectOptions())
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
        res_lin = ResourceDescriptor.from_linear(buf, format=ArrayFormatType.UINT32, num_channels=1)
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(resource=res_lin)

        res_p2 = ResourceDescriptor.from_pitch2d(
            buf,
            format=ArrayFormatType.UINT8,
            num_channels=4,
            width=8,
            height=8,
            pitch_bytes=64,
        )
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(resource=res_p2)
    finally:
        buf.close()


# --- MipmappedArray ----------------------------------------------------------


def test_mipmapped_array_init_disabled():
    with pytest.raises(RuntimeError, match=r"^MipmappedArray cannot be instantiated directly"):
        cuda.core.texture._mipmapped_array.MipmappedArray()


def test_mipmapped_array_from_descriptor_2d(init_cuda):
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(64, 32),
        format=ArrayFormatType.FLOAT32,
        num_channels=1,
        num_levels=4,
    ))
    try:
        assert mip.shape == (64, 32)
        assert mip.format == ArrayFormatType.FLOAT32
        assert mip.num_channels == 1
        assert mip.num_levels == 4
        assert mip.is_surface_load_store is False
        assert mip.handle != 0
        assert isinstance(mip.device, Device)
    finally:
        mip.close()


def test_mipmapped_array_get_level_zero_matches_shape(init_cuda):
    shape = (64, 32)
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=shape,
        format=ArrayFormatType.UINT8,
        num_channels=4,
        num_levels=4,
    ))
    try:
        lvl0 = mip.get_level(0)
        try:
            assert isinstance(lvl0, OpaqueArray)
            # Level 0 must match the base shape and rank.
            assert lvl0.shape == shape
            assert lvl0.format == ArrayFormatType.UINT8
            assert lvl0.num_channels == 4
            assert lvl0.handle != 0
        finally:
            lvl0.close()
    finally:
        mip.close()


def test_mipmapped_array_get_level_halves_dims(init_cuda):
    shape = (64, 32)
    num_levels = 4
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=shape,
        format=ArrayFormatType.UINT8,
        num_channels=1,
        num_levels=num_levels,
    ))
    try:
        for level in range(num_levels):
            lvl = mip.get_level(level)
            try:
                # Each dim halves per level, with a floor of 1; rank is preserved.
                expected = tuple(max(1, dim >> level) for dim in shape)
                assert lvl.shape == expected, f"level={level}: expected {expected}, got {lvl.shape}"
            finally:
                lvl.close()
    finally:
        mip.close()


def test_mipmapped_array_get_level_out_of_range(init_cuda):
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(16, 16),
        format=ArrayFormatType.UINT8,
        num_channels=1,
        num_levels=2,
    ))
    try:
        with pytest.raises(ValueError, match="num_levels"):
            mip.get_level(mip.num_levels)
        with pytest.raises(ValueError, match=">= 0"):
            mip.get_level(-1)
    finally:
        mip.close()


def test_mipmapped_array_rejects_zero_levels(init_cuda):
    with pytest.raises(ValueError, match="num_levels"):
        Device().create_mipmapped_array(MipmappedArrayOptions(
            shape=(8, 8),
            format=ArrayFormatType.UINT8,
            num_channels=1,
            num_levels=0,
        ))


def test_resource_descriptor_from_mipmapped_array(init_cuda):
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(32, 16),
        format=ArrayFormatType.FLOAT32,
        num_channels=1,
        num_levels=3,
    ))
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
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(32, 32),
        format=ArrayFormatType.FLOAT32,
        num_channels=1,
        num_levels=3,
    ))
    try:
        res = ResourceDescriptor.from_mipmapped_array(mip)
        # Use non-default mipmap params so the driver exercises that path.
        tex_desc = TextureObjectOptions(
            address_mode=AddressModeType.CLAMP,
            filter_mode=FilterModeType.LINEAR,
            normalized_coords=True,
            mipmap_filter_mode=FilterModeType.LINEAR,
            mipmap_level_bias=0.0,
            min_mipmap_level_clamp=0.0,
            max_mipmap_level_clamp=float(mip.num_levels - 1),
        )
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=tex_desc)
        try:
            assert tex.handle != 0
            assert tex.resource is res
        finally:
            tex.close()
    finally:
        mip.close()


def test_surface_rejects_mipmapped_array(init_cuda):
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(16, 16),
        format=ArrayFormatType.UINT8,
        num_channels=4,
        num_levels=2,
        is_surface_load_store=True,
    ))
    try:
        res = ResourceDescriptor.from_mipmapped_array(mip)
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(resource=res)
    finally:
        mip.close()


def test_mipmapped_array_level_outlives_dropped_parent(init_cuda):
    """A level OpaqueArray must keep the parent mipmap's storage alive structurally
    (via the C++ handle box), so the level stays usable after the Python parent
    is dropped and garbage-collected.

    The level holds no Python reference to the parent (no ``_parent_ref``); the
    only thing keeping the underlying ``CUmipmappedArray`` alive after ``del mip``
    is the parent handle embedded in the level's box. A full round-trip copy on
    the level touches that live storage and would fail with an invalid-handle
    error if the parent had been destroyed.
    """
    import array as _array

    device = Device()
    stream = device.create_stream()
    mip = Device().create_mipmapped_array(MipmappedArrayOptions(
        shape=(16, 16),
        format=ArrayFormatType.UINT32,
        num_channels=1,
        num_levels=3,
    ))
    lvl = mip.get_level(1)  # (8, 8)
    # Drop the only Python reference to the parent and force GC.
    del mip
    gc.collect()
    try:
        assert lvl.handle != 0
        n = lvl.shape[0] * (lvl.shape[1] if len(lvl.shape) > 1 else 1)
        src = _array.array("I", list(range(n)))
        dst = _array.array("I", [0] * n)
        lvl.copy_from(src, stream=stream)
        lvl.copy_to(dst, stream=stream)
        stream.sync()
        assert list(dst) == list(range(n))
    finally:
        lvl.close()
        stream.close()


def test_texture_surface_close_is_idempotent(init_cuda):
    """close() drops this object's handle reference; destruction happens once via
    the handle deleter. A second close() (and the later __dealloc__) must be a
    safe no-op, and handle must read back as 0."""
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.UINT8, num_channels=1))
    arr.close()
    assert arr.handle == 0
    arr.close()  # second close must not raise

    mip = Device().create_mipmapped_array(MipmappedArrayOptions(shape=(8, 8), format=ArrayFormatType.UINT8, num_channels=1, num_levels=2))
    mip.close()
    assert mip.handle == 0
    mip.close()

    surf_arr = Device().create_opaque_array(OpaqueArrayOptions(
        shape=(8, 8), format=ArrayFormatType.UINT8, num_channels=4, is_surface_load_store=True
    ))
    surf = SurfaceObject.from_array(surf_arr)
    surf.close()
    assert surf.handle == 0
    surf.close()
    surf_arr.close()

    tex_arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    tex = TextureObject.from_descriptor(
        resource=ResourceDescriptor.from_opaque_array(tex_arr),
        texture_descriptor=TextureObjectOptions(),
    )
    tex.close()
    assert tex.handle == 0
    tex.close()
    tex_arr.close()


# --- Negative-path validation tests ------------------------------------------


def test_array_from_descriptor_rejects_bad_format(init_cuda):
    with pytest.raises(ValueError, match="format must be an ArrayFormatType"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=(8,), format=0, num_channels=1))


def test_array_from_descriptor_rejects_non_iterable_shape(init_cuda):
    with pytest.raises(TypeError, match="shape must be a tuple"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=8, format=ArrayFormatType.UINT8, num_channels=1))


def test_array_from_descriptor_rejects_zero_dim(init_cuda):
    with pytest.raises(ValueError, match=r"shape\[1\] must be >= 1"):
        Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 0), format=ArrayFormatType.UINT8, num_channels=1))


def test_array_copy_rejects_non_stream(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8,), format=ArrayFormatType.UINT8, num_channels=1))
    try:
        import array as _array

        buf = _array.array("B", [0] * 8)
        with pytest.raises(TypeError, match="Stream or GraphBuilder expected"):
            arr.copy_from(buf, stream="not-a-stream")
        with pytest.raises(TypeError, match="Stream or GraphBuilder expected"):
            arr.copy_to(buf, stream="not-a-stream")
    finally:
        arr.close()


def test_array_copy_to_returns_dst(init_cuda):
    """OpaqueArray.copy_to returns the destination, for parity with Buffer.copy_to."""
    import array as _array

    device = Device()
    stream = device.create_stream()
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(16,), format=ArrayFormatType.UINT32, num_channels=1))
    try:
        dst = _array.array("I", [0] * 16)
        returned = arr.copy_to(dst, stream=stream)
        stream.sync()
        assert returned is dst
    finally:
        arr.close()
        stream.close()


def test_array_copy_accepts_graph_builder(init_cuda):
    """copy_from/copy_to accept a GraphBuilder so the array copy can be captured
    into a CUDA graph (parity with Buffer, which accepts Stream | GraphBuilder)."""
    device = Device()
    stream = device.create_stream()
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(16,), format=ArrayFormatType.UINT32, num_channels=1))
    buf_in = device.memory_resource.allocate(arr.size_bytes, stream=stream)
    buf_out = device.memory_resource.allocate(arr.size_bytes, stream=stream)
    stream.sync()
    try:
        gb = device.create_graph_builder().begin_building()
        # These would raise TypeError before the Stream | GraphBuilder fix.
        arr.copy_from(buf_in, stream=gb)
        arr.copy_to(buf_out, stream=gb)
        graph = gb.end_building().complete()
        graph.launch(stream)
        stream.sync()
    finally:
        buf_in.close()
        buf_out.close()
        arr.close()
        stream.close()


def test_resource_descriptor_from_pitch2d_rejects_non_buffer():
    with pytest.raises(TypeError, match="buffer must be a Buffer"):
        ResourceDescriptor.from_pitch2d(
            object(),
            format=ArrayFormatType.UINT8,
            num_channels=1,
            width=8,
            height=8,
            pitch_bytes=64,
        )


def test_resource_descriptor_from_pitch2d_rejects_bad_format(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        with pytest.raises(ValueError, match="format must be an ArrayFormatType"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=0,
                num_channels=1,
                width=8,
                height=8,
                pitch_bytes=64,
            )
    finally:
        buf.close()


def test_resource_descriptor_from_pitch2d_rejects_bad_channels(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        with pytest.raises(ValueError, match="num_channels"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormatType.UINT8,
                num_channels=3,
                width=8,
                height=8,
                pitch_bytes=64,
            )
    finally:
        buf.close()


def test_resource_descriptor_from_pitch2d_rejects_zero_dims(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        with pytest.raises(ValueError, match="width"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormatType.UINT8,
                num_channels=1,
                width=0,
                height=8,
                pitch_bytes=64,
            )
        with pytest.raises(ValueError, match="height"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormatType.UINT8,
                num_channels=1,
                width=8,
                height=0,
                pitch_bytes=64,
            )
    finally:
        buf.close()


def test_mipmapped_array_rejects_bad_format(init_cuda):
    with pytest.raises(ValueError, match="format must be an ArrayFormatType"):
        Device().create_mipmapped_array(MipmappedArrayOptions(shape=(8, 8), format=0, num_channels=1, num_levels=2))


def test_mipmapped_array_rejects_bad_channels(init_cuda):
    with pytest.raises(ValueError, match="num_channels"):
        Device().create_mipmapped_array(MipmappedArrayOptions(shape=(8, 8), format=ArrayFormatType.UINT8, num_channels=3, num_levels=2))


def test_mipmapped_array_rejects_zero_dim(init_cuda):
    with pytest.raises(ValueError, match=r"shape\[0\] must be >= 1"):
        Device().create_mipmapped_array(MipmappedArrayOptions(shape=(0, 8), format=ArrayFormatType.UINT8, num_channels=1, num_levels=1))


def test_texture_object_rejects_non_resource_descriptor(init_cuda):
    with pytest.raises(TypeError, match="resource must be a ResourceDescriptor"):
        TextureObject.from_descriptor(resource=object(), texture_descriptor=TextureObjectOptions())


def test_texture_object_rejects_non_texture_descriptor(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        with pytest.raises(TypeError, match="texture_descriptor must be a TextureObjectOptions"):
            TextureObject.from_descriptor(resource=res, texture_descriptor="nope")
    finally:
        arr.close()


def test_texture_object_rejects_bad_filter_mode(init_cuda):
    # Invalid enum values are rejected at TextureObjectOptions construction.
    with pytest.raises(ValueError, match="filter_mode must be a FilterModeType"):
        TextureObjectOptions(filter_mode=0)  # int, not FilterModeType


def test_texture_object_rejects_bad_read_mode(init_cuda):
    with pytest.raises(ValueError, match="read_mode must be a ReadModeType"):
        TextureObjectOptions(read_mode=0)  # int, not ReadModeType


def test_texture_object_rejects_bad_mipmap_filter_mode(init_cuda):
    with pytest.raises(ValueError, match="mipmap_filter_mode must be a FilterModeType"):
        TextureObjectOptions(mipmap_filter_mode=0)  # int, not FilterModeType


def test_texture_object_rejects_negative_anisotropy(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(max_anisotropy=-1)
        with pytest.raises(ValueError, match="max_anisotropy"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_rejects_bad_border_color_length(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(border_color=(0.0, 0.0))  # length 2, not 4
        with pytest.raises(ValueError, match="border_color must have 4"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_non_addressmode_scalar(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(address_mode=42)  # int, not AddressModeType / iterable
        with pytest.raises(TypeError, match="address_mode"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_empty_tuple(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(address_mode=())
        with pytest.raises(ValueError, match="address_mode tuple must have 1-3"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_too_long_tuple(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(address_mode=(AddressModeType.WRAP, AddressModeType.WRAP, AddressModeType.WRAP, AddressModeType.WRAP))
        with pytest.raises(ValueError, match="address_mode tuple must have 1-3"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_non_addressmode_entry(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    try:
        res = ResourceDescriptor.from_opaque_array(arr)
        td = TextureObjectOptions(address_mode=(AddressModeType.WRAP, "bad", AddressModeType.CLAMP))
        with pytest.raises(TypeError, match=r"address_mode\[1\]"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_keeps_backing_array_alive(init_cuda):
    """Dropping the local references to the backing OpaqueArray and the
    ResourceDescriptor must NOT invalidate an existing TextureObject. The
    TextureObject holds a strong ref through its _source_ref slot."""
    arr = Device().create_opaque_array(OpaqueArrayOptions(shape=(8, 8), format=ArrayFormatType.FLOAT32, num_channels=1))
    res = ResourceDescriptor.from_opaque_array(arr)
    tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureObjectOptions())
    # Verify the keepalive chain via gc referents: TextureObject -> _source_ref
    # -> ResourceDescriptor -> _source -> OpaqueArray. We can only walk one level
    # at a time, so check tex's referents include the ResourceDescriptor.
    arr_id = id(arr)
    res_id = id(res)
    del arr, res
    gc.collect()

    referents = gc.get_referents(tex)
    res_refs = [r for r in referents if id(r) == res_id]
    assert len(res_refs) == 1, (
        f"TextureObject should still reference the ResourceDescriptor; got referents {referents!r}"
    )
    res_back = res_refs[0]
    arr_refs = [r for r in gc.get_referents(res_back) if id(r) == arr_id]
    assert len(arr_refs) == 1, "ResourceDescriptor should still reference its OpaqueArray"

    # tex.handle should still be valid (non-zero).
    assert tex.handle != 0
    tex.close()


def test_surface_object_keeps_backing_array_alive(init_cuda):
    arr = Device().create_opaque_array(OpaqueArrayOptions(
        shape=(8, 8),
        format=ArrayFormatType.UINT8,
        num_channels=4,
        is_surface_load_store=True,
    ))
    surf = SurfaceObject.from_array(arr)
    arr_id = id(arr)
    del arr
    gc.collect()

    # The surface keeps the ResourceDescriptor alive, which keeps the OpaqueArray
    # alive. We verify the chain end-to-end the same way as the texture case.
    referents = gc.get_referents(surf)
    res_objs = [r for r in referents if isinstance(r, ResourceDescriptor)]
    assert len(res_objs) == 1
    arr_refs = [r for r in gc.get_referents(res_objs[0]) if id(r) == arr_id]
    assert len(arr_refs) == 1, "SurfaceObject should still reference its backing OpaqueArray via the ResourceDescriptor"
    assert surf.handle != 0
    surf.close()
