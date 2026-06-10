# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import cuda.core
from cuda.core import (
    AddressMode,
    ArrayFormat,
    CUDAArray,
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
    with pytest.raises(RuntimeError, match=r"^CUDAArray cannot be instantiated directly"):
        cuda.core._array.CUDAArray()


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
    arr = CUDAArray.from_descriptor(shape=(32, 16), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        assert arr.shape == (32, 16)
        assert arr.format == ArrayFormat.FLOAT32
        assert arr.num_channels == 1
        assert arr.element_size == 4
        assert arr.size_bytes == 32 * 16 * 4
        assert arr.is_surface_load_store is False
        assert arr.handle != 0
        assert isinstance(arr.device, Device)
    finally:
        arr.close()


def test_array_3d_with_surface_flag(init_cuda):
    arr = CUDAArray.from_descriptor(
        shape=(8, 8, 4),
        format=ArrayFormat.UINT8,
        num_channels=4,
        surface_load_store=True,
    )
    try:
        assert arr.shape == (8, 8, 4)
        assert arr.is_surface_load_store is True
        assert arr.element_size == 4
    finally:
        arr.close()


def test_array_rejects_bad_channels(init_cuda):
    with pytest.raises(ValueError, match="num_channels"):
        CUDAArray.from_descriptor(shape=(8,), format=ArrayFormat.UINT8, num_channels=3)


def test_array_rejects_bad_rank(init_cuda):
    with pytest.raises(ValueError, match="shape rank"):
        CUDAArray.from_descriptor(shape=(2, 2, 2, 2), format=ArrayFormat.UINT8, num_channels=1)


def test_array_roundtrip_copy(init_cuda):
    import array as _array

    device = Device()
    stream = device.create_stream()
    arr = CUDAArray.from_descriptor(shape=(16,), format=ArrayFormat.UINT32, num_channels=1)
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
    arr = CUDAArray.from_descriptor(shape=(16,), format=ArrayFormat.UINT32, num_channels=1)
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
    arr = CUDAArray.from_descriptor(shape=(16,), format=ArrayFormat.UINT32, num_channels=1)
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
    arr = CUDAArray.from_descriptor(shape=(32, 16), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        tex_desc = TextureDescriptor(
            address_mode=AddressMode.CLAMP,
            filter_mode=FilterMode.LINEAR,
            read_mode=ReadMode.ELEMENT_TYPE,
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
    arr = CUDAArray.from_descriptor(
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
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.UINT8, num_channels=4)
    try:
        with pytest.raises(ValueError, match="surface_load_store=True"):
            SurfaceObject.from_array(arr)
    finally:
        arr.close()


def test_address_mode_normalization(init_cuda):
    # Direct unit test of the private normalizer: a scalar should expand to a
    # 3-tuple; a shorter tuple should be padded by repeating the last entry.
    from cuda.core._texture import _normalize_address_modes

    assert _normalize_address_modes(AddressMode.WRAP) == (
        AddressMode.WRAP,
        AddressMode.WRAP,
        AddressMode.WRAP,
    )
    assert _normalize_address_modes((AddressMode.WRAP, AddressMode.CLAMP)) == (
        AddressMode.WRAP,
        AddressMode.CLAMP,
        AddressMode.CLAMP,
    )
    assert _normalize_address_modes((AddressMode.WRAP, AddressMode.CLAMP, AddressMode.MIRROR)) == (
        AddressMode.WRAP,
        AddressMode.CLAMP,
        AddressMode.MIRROR,
    )

    # Smoke test: a 2-entry tuple is also accepted end-to-end.
    arr = CUDAArray.from_descriptor(shape=(8, 8, 4), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        tex_desc = TextureDescriptor(address_mode=(AddressMode.WRAP, AddressMode.CLAMP))
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
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormat.FLOAT32, num_channels=1)
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
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT32, num_channels=1, size_bytes=2048)
        assert res._size_bytes == 2048
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_oversize(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="exceeds buffer.size"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT8, num_channels=1, size_bytes=2048)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_bad_channels(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="num_channels"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT8, num_channels=3)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_non_buffer():
    with pytest.raises(TypeError, match="Buffer"):
        ResourceDescriptor.from_linear(object(), format=ArrayFormat.UINT8, num_channels=1)


def test_resource_descriptor_from_linear_rejects_zero_size(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        with pytest.raises(ValueError, match="at least one element"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT32, num_channels=1, size_bytes=0)
    finally:
        buf.close()


def test_resource_descriptor_from_linear_rejects_non_multiple(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 1024)
    try:
        # UINT32 x 1 channel = 4 bytes/element; 10 bytes is not a multiple.
        with pytest.raises(ValueError, match="multiple of element size"):
            ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT32, num_channels=1, size_bytes=10)
    finally:
        buf.close()


def test_texture_object_from_linear(init_cuda):
    """A linear-backed texture should bind even though sampling fields are
    effectively ignored by the driver."""
    device = Device()
    # 1024 float elements
    buf = _alloc_device_buffer(device, 1024 * 4)
    try:
        res = ResourceDescriptor.from_linear(buf, format=ArrayFormat.FLOAT32, num_channels=1)
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureDescriptor())
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
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureDescriptor())
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
        res_lin = ResourceDescriptor.from_linear(buf, format=ArrayFormat.UINT32, num_channels=1)
        with pytest.raises(ValueError, match="array-backed"):
            SurfaceObject.from_descriptor(resource=res_lin)

        res_p2 = ResourceDescriptor.from_pitch2d(
            buf,
            format=ArrayFormat.UINT8,
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
        assert mip.is_surface_load_store is False
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
            assert isinstance(lvl0, CUDAArray)
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
                assert lvl.shape == expected, f"level={level}: expected {expected}, got {lvl.shape}"
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
        tex = TextureObject.from_descriptor(resource=res, texture_descriptor=tex_desc)
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
            SurfaceObject.from_descriptor(resource=res)
    finally:
        mip.close()


def test_mipmapped_array_level_keeps_parent_alive(init_cuda):
    """Dropping the local parent reference must not invalidate the level CUDAArray;
    the level holds an internal strong ref back to the MipmappedArray.

    cdef classes don't natively support weakref, so we verify the parent
    reference by inspecting the level CUDAArray's gc referents.
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
    # the level CUDAArray holds a strong ref via the internal _parent_ref slot.
    del mip
    gc.collect()

    # The handle is still valid storage; the level still tracks the parent.
    assert lvl.handle != 0
    referents = gc.get_referents(lvl)
    parents = [r for r in referents if isinstance(r, MipmappedArray)]
    assert len(parents) == 1, f"level CUDAArray should reference exactly one MipmappedArray parent, got {parents!r}"
    assert id(parents[0]) == parent_id, "level CUDAArray's parent ref is not the original MipmappedArray"
    # Closing the level drops its parent ref. Don't access the parent past
    # this point; cuMipmappedArrayDestroy may then run.
    lvl.close()


# --- Negative-path validation tests ------------------------------------------


def test_array_from_descriptor_rejects_bad_format(init_cuda):
    with pytest.raises(TypeError, match="format must be an ArrayFormat"):
        CUDAArray.from_descriptor(shape=(8,), format=0, num_channels=1)


def test_array_from_descriptor_rejects_non_iterable_shape(init_cuda):
    with pytest.raises(TypeError, match="shape must be a tuple"):
        CUDAArray.from_descriptor(shape=8, format=ArrayFormat.UINT8, num_channels=1)


def test_array_from_descriptor_rejects_zero_dim(init_cuda):
    with pytest.raises(ValueError, match=r"shape\[1\] must be >= 1"):
        CUDAArray.from_descriptor(shape=(8, 0), format=ArrayFormat.UINT8, num_channels=1)


def test_array_copy_rejects_non_stream(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8,), format=ArrayFormat.UINT8, num_channels=1)
    try:
        import array as _array

        buf = _array.array("B", [0] * 8)
        with pytest.raises(TypeError, match="stream must be a Stream"):
            arr.copy_from(buf, stream="not-a-stream")
        with pytest.raises(TypeError, match="stream must be a Stream"):
            arr.copy_to(buf, stream="not-a-stream")
    finally:
        arr.close()


def test_resource_descriptor_from_pitch2d_rejects_non_buffer():
    with pytest.raises(TypeError, match="buffer must be a Buffer"):
        ResourceDescriptor.from_pitch2d(
            object(),
            format=ArrayFormat.UINT8,
            num_channels=1,
            width=8,
            height=8,
            pitch_bytes=64,
        )


def test_resource_descriptor_from_pitch2d_rejects_bad_format(init_cuda):
    device = Device()
    buf = _alloc_device_buffer(device, 4096)
    try:
        with pytest.raises(TypeError, match="format must be an ArrayFormat"):
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
                format=ArrayFormat.UINT8,
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
                format=ArrayFormat.UINT8,
                num_channels=1,
                width=0,
                height=8,
                pitch_bytes=64,
            )
        with pytest.raises(ValueError, match="height"):
            ResourceDescriptor.from_pitch2d(
                buf,
                format=ArrayFormat.UINT8,
                num_channels=1,
                width=8,
                height=0,
                pitch_bytes=64,
            )
    finally:
        buf.close()


def test_mipmapped_array_rejects_bad_format(init_cuda):
    with pytest.raises(TypeError, match="format must be an ArrayFormat"):
        MipmappedArray.from_descriptor(shape=(8, 8), format=0, num_channels=1, num_levels=2)


def test_mipmapped_array_rejects_bad_channels(init_cuda):
    with pytest.raises(ValueError, match="num_channels"):
        MipmappedArray.from_descriptor(shape=(8, 8), format=ArrayFormat.UINT8, num_channels=3, num_levels=2)


def test_mipmapped_array_rejects_zero_dim(init_cuda):
    with pytest.raises(ValueError, match=r"shape\[0\] must be >= 1"):
        MipmappedArray.from_descriptor(shape=(0, 8), format=ArrayFormat.UINT8, num_channels=1, num_levels=1)


def test_texture_object_rejects_non_resource_descriptor(init_cuda):
    with pytest.raises(TypeError, match="resource must be a ResourceDescriptor"):
        TextureObject.from_descriptor(resource=object(), texture_descriptor=TextureDescriptor())


def test_texture_object_rejects_non_texture_descriptor(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        with pytest.raises(TypeError, match="texture_descriptor must be a TextureDescriptor"):
            TextureObject.from_descriptor(resource=res, texture_descriptor="nope")
    finally:
        arr.close()


def test_texture_object_rejects_bad_filter_mode(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(filter_mode=0)  # int, not FilterMode
        with pytest.raises(TypeError, match="filter_mode must be a FilterMode"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_rejects_bad_read_mode(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(read_mode=0)  # int, not ReadMode
        with pytest.raises(TypeError, match="read_mode must be a ReadMode"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_rejects_bad_mipmap_filter_mode(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(mipmap_filter_mode=0)  # int, not FilterMode
        with pytest.raises(TypeError, match="mipmap_filter_mode must be a FilterMode"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_rejects_negative_anisotropy(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(max_anisotropy=-1)
        with pytest.raises(ValueError, match="max_anisotropy"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_rejects_bad_border_color_length(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(border_color=(0.0, 0.0))  # length 2, not 4
        with pytest.raises(ValueError, match="border_color must have 4"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_non_addressmode_scalar(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(address_mode=42)  # int, not AddressMode / iterable
        with pytest.raises(TypeError, match="address_mode"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_empty_tuple(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(address_mode=())
        with pytest.raises(ValueError, match="address_mode tuple must have 1-3"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_too_long_tuple(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(address_mode=(AddressMode.WRAP, AddressMode.WRAP, AddressMode.WRAP, AddressMode.WRAP))
        with pytest.raises(ValueError, match="address_mode tuple must have 1-3"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_address_mode_rejects_non_addressmode_entry(init_cuda):
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    try:
        res = ResourceDescriptor.from_array(arr)
        td = TextureDescriptor(address_mode=(AddressMode.WRAP, "bad", AddressMode.CLAMP))
        with pytest.raises(TypeError, match=r"address_mode\[1\]"):
            TextureObject.from_descriptor(resource=res, texture_descriptor=td)
    finally:
        arr.close()


def test_texture_object_keeps_backing_array_alive(init_cuda):
    """Dropping the local references to the backing CUDAArray and the
    ResourceDescriptor must NOT invalidate an existing TextureObject. The
    TextureObject holds a strong ref through its _source_ref slot."""
    arr = CUDAArray.from_descriptor(shape=(8, 8), format=ArrayFormat.FLOAT32, num_channels=1)
    res = ResourceDescriptor.from_array(arr)
    tex = TextureObject.from_descriptor(resource=res, texture_descriptor=TextureDescriptor())
    # Verify the keepalive chain via gc referents: TextureObject -> _source_ref
    # -> ResourceDescriptor -> _source -> CUDAArray. We can only walk one level
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
    assert len(arr_refs) == 1, "ResourceDescriptor should still reference its CUDAArray"

    # tex.handle should still be valid (non-zero).
    assert tex.handle != 0
    tex.close()


def test_surface_object_keeps_backing_array_alive(init_cuda):
    arr = CUDAArray.from_descriptor(
        shape=(8, 8),
        format=ArrayFormat.UINT8,
        num_channels=4,
        surface_load_store=True,
    )
    surf = SurfaceObject.from_array(arr)
    arr_id = id(arr)
    del arr
    gc.collect()

    # The surface keeps the ResourceDescriptor alive, which keeps the CUDAArray
    # alive. We verify the chain end-to-end the same way as the texture case.
    referents = gc.get_referents(surf)
    res_objs = [r for r in referents if isinstance(r, ResourceDescriptor)]
    assert len(res_objs) == 1
    arr_refs = [r for r in gc.get_referents(res_objs[0]) if id(r) == arr_id]
    assert len(arr_refs) == 1, "SurfaceObject should still reference its backing CUDAArray via the ResourceDescriptor"
    assert surf.handle != 0
    surf.close()
