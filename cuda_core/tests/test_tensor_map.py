# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from cuda.core import (
    Device,
    TensorMapDescriptor,
    TensorMapDataType,
    TensorMapInterleave,
    TensorMapL2Promotion,
    TensorMapOOBFill,
    TensorMapSwizzle,
)


@pytest.fixture
def dev(init_cuda):
    return Device()


@pytest.fixture
def skip_if_no_tma(dev):
    if not dev.properties.tensor_map_access_supported:
        pytest.skip("Device does not support TMA (requires compute capability 9.0+)")


def _alloc_device_tensor(dev, shape, dtype=np.float32, alignment=256):
    """Allocate a device buffer and return it with proper alignment."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    buf = dev.allocate(n_elements * np.dtype(dtype).itemsize + alignment)
    return buf


class TestTensorMapEnums:
    """Test that enum wrappers expose the expected values."""

    def test_data_type_values(self):
        assert TensorMapDataType.UINT8 == 0
        assert TensorMapDataType.FLOAT32 == 7
        assert TensorMapDataType.FLOAT64 == 8
        assert TensorMapDataType.BFLOAT16 == 9

    def test_interleave_values(self):
        assert TensorMapInterleave.NONE == 0
        assert TensorMapInterleave.INTERLEAVE_16B == 1
        assert TensorMapInterleave.INTERLEAVE_32B == 2

    def test_swizzle_values(self):
        assert TensorMapSwizzle.NONE == 0
        assert TensorMapSwizzle.SWIZZLE_32B == 1
        assert TensorMapSwizzle.SWIZZLE_64B == 2
        assert TensorMapSwizzle.SWIZZLE_128B == 3

    def test_l2_promotion_values(self):
        assert TensorMapL2Promotion.NONE == 0
        assert TensorMapL2Promotion.L2_64B == 1
        assert TensorMapL2Promotion.L2_128B == 2
        assert TensorMapL2Promotion.L2_256B == 3

    def test_oob_fill_values(self):
        assert TensorMapOOBFill.NONE == 0
        assert TensorMapOOBFill.NAN_REQUEST_ZERO_FMA == 1


class TestTensorMapDescriptorCreation:
    """Test TensorMapDescriptor factory methods."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
            TensorMapDescriptor()

    def test_from_tiled_1d(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)  # 1024 float32 elements
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None
        assert repr(desc) == "TensorMapDescriptor()"

    def test_from_tiled_2d(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)  # 64x64 float32
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_tiled_3d(self, dev, skip_if_no_tma):
        buf = dev.allocate(16 * 16 * 16 * 4)  # 16x16x16 float32
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(8, 8, 8),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_tiled_with_swizzle(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
        )
        assert desc is not None

    def test_from_tiled_with_l2_promotion(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        )
        assert desc is not None

    def test_from_tiled_with_oob_fill(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf,
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            oob_fill=TensorMapOOBFill.NAN_REQUEST_ZERO_FMA,
        )
        assert desc is not None


class TestTensorMapDescriptorValidation:
    """Test validation in TensorMapDescriptor factory methods."""

    def test_invalid_rank_zero(self, dev, skip_if_no_tma):
        buf = dev.allocate(64)
        with pytest.raises(ValueError, match="rank must be between 1 and 5"):
            TensorMapDescriptor.from_tiled(
                buf,
                box_dim=(),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_box_dim_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="box_dim must have 1 elements"):
            TensorMapDescriptor.from_tiled(
                buf,
                box_dim=(32, 32),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_box_dim_out_of_range(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match=r"box_dim\[0\] must be in \[1, 256\]"):
            TensorMapDescriptor.from_tiled(
                buf,
                box_dim=(512,),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_element_strides_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="element_strides must have 1 elements"):
            TensorMapDescriptor.from_tiled(
                buf,
                box_dim=(64,),
                element_strides=(1, 1),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_invalid_data_type(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(TypeError, match="data_type must be a TensorMapDataType"):
            TensorMapDescriptor.from_tiled(
                buf,
                box_dim=(64,),
                data_type=42,
            )


class TestTensorMapDtypeMapping:
    """Test automatic dtype inference from numpy dtypes."""

    @pytest.mark.parametrize("np_dtype,expected_tma_dt", [
        (np.uint8, TensorMapDataType.UINT8),
        (np.uint16, TensorMapDataType.UINT16),
        (np.uint32, TensorMapDataType.UINT32),
        (np.int32, TensorMapDataType.INT32),
        (np.uint64, TensorMapDataType.UINT64),
        (np.int64, TensorMapDataType.INT64),
        (np.float16, TensorMapDataType.FLOAT16),
        (np.float32, TensorMapDataType.FLOAT32),
        (np.float64, TensorMapDataType.FLOAT64),
    ])
    def test_dtype_mapping(self, np_dtype, expected_tma_dt, dev, skip_if_no_tma):
        from cuda.core._tensor_map import _NUMPY_DTYPE_TO_TMA
        assert _NUMPY_DTYPE_TO_TMA[np.dtype(np_dtype)] == expected_tma_dt

    def test_bfloat16_mapping(self):
        try:
            from ml_dtypes import bfloat16
            from cuda.core._tensor_map import _NUMPY_DTYPE_TO_TMA
            assert _NUMPY_DTYPE_TO_TMA[np.dtype(bfloat16)] == TensorMapDataType.BFLOAT16
        except ImportError:
            pytest.skip("ml_dtypes not installed")


class TestTensorMapReplaceAddress:
    """Test replace_address functionality."""

    def test_replace_address(self, dev, skip_if_no_tma):
        buf1 = dev.allocate(1024 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf1,
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )

        buf2 = dev.allocate(1024 * 4)
        desc.replace_address(buf2)
        # No exception means success

    def test_replace_address_requires_device_accessible(self, dev, skip_if_no_tma):
        buf1 = dev.allocate(1024 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf1,
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        # Create a host-only array (not device-accessible)
        host_arr = np.zeros(1024, dtype=np.float32)
        with pytest.raises(ValueError, match="device-accessible"):
            desc.replace_address(host_arr)


class TestTensorMapIm2col:
    """Test im2col TMA descriptor creation."""

    def test_from_im2col_3d(self, dev, skip_if_no_tma):
        # 3D tensor: batch=1, height=32, channels=64
        buf = dev.allocate(1 * 32 * 64 * 4)
        desc = TensorMapDescriptor.from_im2col(
            buf,
            pixel_box_lower_corner=(0,),
            pixel_box_upper_corner=(4,),
            channels_per_pixel=64,
            pixels_per_column=4,
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_im2col_rank_validation(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="Im2col tensor rank must be between 3 and 5"):
            TensorMapDescriptor.from_im2col(
                buf,
                pixel_box_lower_corner=(),
                pixel_box_upper_corner=(),
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_from_im2col_corner_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1 * 32 * 64 * 4)
        with pytest.raises(ValueError, match="pixel_box_lower_corner must have 1 elements"):
            TensorMapDescriptor.from_im2col(
                buf,
                pixel_box_lower_corner=(0, 0),
                pixel_box_upper_corner=(4,),
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )
