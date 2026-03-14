# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from conftest import create_managed_memory_resource_or_skip, skip_if_managed_memory_unsupported
from cuda.core import (
    Device,
    ManagedMemoryResourceOptions,
    StridedMemoryView,
    TensorMapDescriptor,
    system,
)
from cuda.core._dlpack import DLDeviceType
from cuda.core._tensor_map import (
    TensorMapDataType,
    TensorMapDescriptorOptions,
    TensorMapIm2ColWideMode,
    TensorMapInterleave,
    TensorMapL2Promotion,
    TensorMapOOBFill,
    TensorMapSwizzle,
    _require_view_device,
)


@pytest.fixture
def dev(init_cuda):
    return Device()


@pytest.fixture
def skip_if_no_tma(dev):
    if not dev.properties.tensor_map_access_supported:
        pytest.skip("Device does not support TMA (requires compute capability 9.0+)")


class _DeviceArray:
    """Wrap a Buffer with explicit shape via __cuda_array_interface__.

    dev.allocate() returns a 1D byte buffer. For multi-dimensional TMA tests
    we need the tensor to report a proper shape/dtype so the TMA encoder sees
    the correct rank, dimensions, and strides.
    """

    def __init__(self, buf, shape, dtype=np.float32):
        self._buf = buf  # prevent GC
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": np.dtype(dtype).str,
            "data": (int(buf.handle), False),
            "version": 3,
        }


class _MockTensorMapView:
    def __init__(self, device_type, device_id):
        self._device_type = device_type
        self._device_id = device_id

    def __dlpack_device__(self):
        return (self._device_type, self._device_id)

def _as_view(obj):
    if isinstance(obj, StridedMemoryView):
        return obj
    return StridedMemoryView.from_any_interface(obj, stream_ptr=-1)


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

    def test_im2col_wide_mode_values(self):
        assert TensorMapIm2ColWideMode.W == 0
        assert TensorMapIm2ColWideMode.W128 == 1


class TestTensorMapDescriptorCreation:
    """Test TensorMapDescriptor factory methods."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
            TensorMapDescriptor()

    def test_from_tiled_1d(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)  # 1024 float32 elements
        desc = _as_view(buf).as_tensor_map(
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None
        assert repr(desc) == "TensorMapDescriptor(tiled, rank=1, dtype=FLOAT32, swizzle=NONE)"

    def test_device_property(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        desc = _as_view(buf).as_tensor_map(
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc.device.device_id == dev.device_id

    def test_from_tiled_2d(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)  # 64x64 float32
        tensor = _DeviceArray(buf, (64, 64))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_strided_memory_view_as_tensor_map(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)
        desc = view.as_tensor_map(
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_strided_memory_view_as_tensor_map_options(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)
        desc = view.as_tensor_map(
            options=TensorMapDescriptorOptions(
                box_dim=(32, 32),
                data_type=np.float32,
                swizzle=TensorMapSwizzle.SWIZZLE_128B,
            )
        )
        assert desc is not None

    def test_strided_memory_view_as_tensor_map_options_dict(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        desc = _as_view(buf).as_tensor_map(
            options={
                "box_dim": (64,),
                "data_type": np.float32,
                "element_strides": (1,),
            }
        )
        assert desc is not None

    def test_strided_memory_view_as_tensor_map_rejects_options_with_kwargs(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(TypeError, match="Specify either options or the individual tensor map arguments"):
            _as_view(buf).as_tensor_map(
                box_dim=(64,),
                options=TensorMapDescriptorOptions(box_dim=(64,)),
            )

    def test_from_tiled_3d(self, dev, skip_if_no_tma):
        buf = dev.allocate(16 * 16 * 16 * 4)  # 16x16x16 float32
        tensor = _DeviceArray(buf, (16, 16, 16))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(8, 8, 8),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_tiled_5d(self, dev, skip_if_no_tma):
        # 5D: exercises all 5 c_global_dim / 4 c_global_strides slots
        shape = (2, 4, 4, 4, 8)
        n_bytes = 2 * 4 * 4 * 4 * 8 * 4  # float32
        buf = dev.allocate(n_bytes)
        tensor = _DeviceArray(buf, shape)
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(1, 2, 2, 2, 8),
        )
        assert desc is not None

    def test_from_tiled_with_element_strides_buffer(self, dev, skip_if_no_tma):
        # Use a Buffer input (DLPack path) and explicit element_strides.
        buf = dev.allocate(1024 * 4)
        desc = _as_view(buf).as_tensor_map(
            box_dim=(64,),
            element_strides=(2,),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_tiled_with_element_strides_cai(self, dev, skip_if_no_tma):
        # Use a CAI-style tensor wrapper and explicit element_strides.
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(32, 32),
            element_strides=(2, 1),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_tiled_with_swizzle(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
        )
        assert desc is not None

    def test_from_tiled_with_l2_promotion(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            l2_promotion=TensorMapL2Promotion.L2_128B,
        )
        assert desc is not None

    def test_from_tiled_with_oob_fill(self, dev, skip_if_no_tma):
        buf = dev.allocate(64 * 64 * 4)
        tensor = _DeviceArray(buf, (64, 64))
        desc = _as_view(tensor).as_tensor_map(
            box_dim=(32, 32),
            data_type=TensorMapDataType.FLOAT32,
            oob_fill=TensorMapOOBFill.NAN_REQUEST_ZERO_FMA,
        )
        assert desc is not None


class TestTensorMapDescriptorValidation:
    """Test validation in TensorMapDescriptor factory methods."""

    def test_invalid_rank_zero(self, dev, skip_if_no_tma):
        buf = dev.allocate(64)
        tensor = _DeviceArray(buf, ())  # 0-dim tensor
        with pytest.raises(ValueError, match="rank must be between 1 and 5"):
            _as_view(tensor).as_tensor_map(
                box_dim=(),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_invalid_rank_six(self, dev, skip_if_no_tma):
        shape = (2, 2, 2, 2, 2, 2)
        n_elements = 1
        for s in shape:
            n_elements *= s
        buf = dev.allocate(n_elements * 4)
        arr = _DeviceArray(buf, shape)
        with pytest.raises(ValueError, match="rank must be between 1 and 5"):
            _as_view(arr).as_tensor_map(
                box_dim=(2,) * 6,
            )

    def test_box_dim_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="box_dim must have 1 elements"):
            _as_view(buf).as_tensor_map(
                box_dim=(32, 32),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_box_dim_out_of_range(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match=r"box_dim\[0\] must be in \[1, 256\]"):
            _as_view(buf).as_tensor_map(
                box_dim=(512,),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_element_strides_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="element_strides must have 1 elements"):
            _as_view(buf).as_tensor_map(
                box_dim=(64,),
                element_strides=(1, 1),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_invalid_data_type(self, dev, skip_if_no_tma):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(TypeError, match="data_type must be"):
            _as_view(buf).as_tensor_map(
                box_dim=(64,),
                data_type=42,
            )


class TestTensorMapDtypeMapping:
    """Test automatic dtype inference from numpy dtypes."""

    @pytest.mark.parametrize(
        "np_dtype,expected_tma_dt",
        [
            (np.uint8, TensorMapDataType.UINT8),
            (np.uint16, TensorMapDataType.UINT16),
            (np.uint32, TensorMapDataType.UINT32),
            (np.int32, TensorMapDataType.INT32),
            (np.uint64, TensorMapDataType.UINT64),
            (np.int64, TensorMapDataType.INT64),
            (np.float16, TensorMapDataType.FLOAT16),
            (np.float32, TensorMapDataType.FLOAT32),
            (np.float64, TensorMapDataType.FLOAT64),
        ],
    )
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
        desc = _as_view(buf1).as_tensor_map(
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )

        buf2 = dev.allocate(1024 * 4)
        desc.replace_address(buf2)
        # No exception means success

    def test_replace_address_requires_device_accessible(self, dev, skip_if_no_tma):
        buf1 = dev.allocate(1024 * 4)
        desc = _as_view(buf1).as_tensor_map(
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        # Create a host-only array (not device-accessible)
        host_arr = np.zeros(1024, dtype=np.float32)
        with pytest.raises(ValueError, match="device-accessible"):
            desc.replace_address(host_arr)

    def test_replace_address_rejects_tensor_from_other_device(self, dev, skip_if_no_tma):
        if system.get_num_devices() < 2:
            pytest.skip("requires multi-GPU")

        dev0 = dev
        dev1 = Device(1)

        dev0.set_current()
        buf0 = dev0.allocate(1024 * 4)
        desc = TensorMapDescriptor.from_tiled(
            buf0,
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )

        dev1.set_current()
        buf1 = dev1.allocate(1024 * 4)
        dev0.set_current()

        with pytest.raises(ValueError, match=r"replace_address expects tensor on device 0, got 1"):
            desc.replace_address(buf1)

    def test_replace_address_accepts_managed_buffer_on_nonzero_device(self, init_cuda):
        if system.get_num_devices() < 2:
            pytest.skip("requires multi-GPU")

        dev1 = Device(1)
        if not dev1.properties.tensor_map_access_supported:
            pytest.skip("Device does not support TMA (requires compute capability 9.0+)")
        skip_if_managed_memory_unsupported(dev1)

        dev1.set_current()
        desc = TensorMapDescriptor.from_tiled(
            dev1.allocate(1024 * 4),
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )

        mr = create_managed_memory_resource_or_skip(ManagedMemoryResourceOptions(preferred_location=dev1.device_id))
        managed_buf = mr.allocate(1024 * 4)

        desc.replace_address(managed_buf)


class TestTensorMapMultiDeviceValidation:
    """Test multi-device validation for descriptor creation."""

    def test_from_tiled_rejects_tensor_from_other_device(self, init_cuda):
        if system.get_num_devices() < 2:
            pytest.skip("requires multi-GPU")

        dev0 = Device(0)
        dev1 = Device(1)

        dev1.set_current()
        buf1 = dev1.allocate(1024 * 4)
        dev0.set_current()

        with pytest.raises(
            ValueError,
            match=r"TensorMapDescriptor\.from_tiled expects tensor on device 0, got 1",
        ):
            TensorMapDescriptor.from_tiled(
                buf1,
                box_dim=(64,),
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_from_tiled_accepts_managed_buffer_on_nonzero_device(self, init_cuda):
        if system.get_num_devices() < 2:
            pytest.skip("requires multi-GPU")

        dev1 = Device(1)
        if not dev1.properties.tensor_map_access_supported:
            pytest.skip("Device does not support TMA (requires compute capability 9.0+)")
        skip_if_managed_memory_unsupported(dev1)

        dev1.set_current()
        mr = create_managed_memory_resource_or_skip(ManagedMemoryResourceOptions(preferred_location=dev1.device_id))
        managed_buf = mr.allocate(1024 * 4)

        desc = TensorMapDescriptor.from_tiled(
            managed_buf,
            box_dim=(64,),
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None


class TestTensorMapDeviceValidation:
    """Test device validation behavior for tensor-map-compatible views."""

    def test_require_view_device_accepts_same_cuda_device(self):
        _require_view_device(_MockTensorMapView(DLDeviceType.kDLCUDA, 1), 1, "op")

    def test_require_view_device_rejects_different_cuda_device(self):
        with pytest.raises(ValueError, match=r"op expects tensor on device 0, got 1"):
            _require_view_device(_MockTensorMapView(DLDeviceType.kDLCUDA, 1), 0, "op")

    def test_require_view_device_allows_cuda_host_memory(self):
        _require_view_device(_MockTensorMapView(DLDeviceType.kDLCUDAHost, 0), 1, "op")

    def test_require_view_device_allows_cuda_managed_memory(self):
        _require_view_device(_MockTensorMapView(DLDeviceType.kDLCUDAManaged, 0), 1, "op")


class TestTensorMapIm2col:
    """Test im2col TMA descriptor creation."""

    def test_from_im2col_3d(self, dev, skip_if_no_tma):
        # 3D tensor: batch=1, height=32, channels=64
        buf = dev.allocate(1 * 32 * 64 * 4)
        tensor = _DeviceArray(buf, (1, 32, 64))
        desc = TensorMapDescriptor._from_im2col(
            _as_view(tensor),
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
            TensorMapDescriptor._from_im2col(
                _as_view(buf),
                pixel_box_lower_corner=(),
                pixel_box_upper_corner=(),
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_from_im2col_corner_rank_mismatch(self, dev, skip_if_no_tma):
        buf = dev.allocate(1 * 32 * 64 * 4)
        tensor = _DeviceArray(buf, (1, 32, 64))  # 3D: n_spatial = 1
        with pytest.raises(ValueError, match="pixel_box_lower_corner must have 1 elements"):
            TensorMapDescriptor._from_im2col(
                _as_view(tensor),
                pixel_box_lower_corner=(0, 0),
                pixel_box_upper_corner=(4,),
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )

    def test_from_im2col_4d(self, dev, skip_if_no_tma):
        # NHWC layout: N=1, H=8, W=8, C=64 — 2 spatial dims
        # Exercises spatial corner reversal with n_spatial=2:
        #   Python [H_lower, W_lower] -> driver [W_lower, H_lower]
        shape = (1, 8, 8, 64)
        buf = dev.allocate(1 * 8 * 8 * 64 * 4)
        tensor = _DeviceArray(buf, shape)
        desc = TensorMapDescriptor._from_im2col(
            _as_view(tensor),
            pixel_box_lower_corner=(0, 0),
            pixel_box_upper_corner=(4, 4),
            channels_per_pixel=64,
            pixels_per_column=16,
        )
        assert desc is not None

    def test_from_im2col_5d(self, dev, skip_if_no_tma):
        # NDHWC layout: N=1, D=4, H=8, W=8, C=64 — 3 spatial dims
        # Exercises the full spatial corner reversal:
        #   Python [D, H, W] -> driver [W, H, D]
        shape = (1, 4, 8, 8, 64)
        buf = dev.allocate(1 * 4 * 8 * 8 * 64 * 4)
        tensor = _DeviceArray(buf, shape)
        desc = TensorMapDescriptor._from_im2col(
            _as_view(tensor),
            pixel_box_lower_corner=(0, 0, 0),
            pixel_box_upper_corner=(2, 4, 4),
            channels_per_pixel=64,
            pixels_per_column=32,
        )
        assert desc is not None


class TestTensorMapIm2colWide:
    """Test im2col-wide TMA descriptor creation (compute capability 10.0+)."""

    @pytest.fixture
    def skip_if_no_im2col_wide(self, dev):
        cc = dev.compute_capability
        if cc.major < 10:
            pytest.skip("Device does not support im2col-wide (requires compute capability 10.0+)")

        # Some environments in CI exercise this test module with a cuda.core
        # build that does not include im2col-wide symbols (CUDA < 13 build),
        # or with driver/GPU combinations that reject im2col-wide descriptor
        # encoding for otherwise valid inputs. Probe once per test invocation
        # and skip only for those known unsupported cases.
        buf = dev.allocate(1 * 32 * 64 * 4)
        tensor = _DeviceArray(buf, (1, 32, 64))
        try:
            TensorMapDescriptor._from_im2col_wide(
                _as_view(tensor),
                pixel_box_lower_corner_width=0,
                pixel_box_upper_corner_width=4,
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )
        except RuntimeError as e:
            if "requires a CUDA 13+ build" in str(e):
                pytest.skip("Im2col-wide requires cuda.core built with CUDA 13+")
            raise
        except Exception as e:
            if "CUDA_ERROR_INVALID_VALUE" in str(e):
                pytest.skip("Im2col-wide unsupported on this driver/GPU combination")
            raise

    def test_from_im2col_wide_3d(self, dev, skip_if_no_im2col_wide):
        # 3D tensor: batch=1, width=32, channels=64
        buf = dev.allocate(1 * 32 * 64 * 4)
        tensor = _DeviceArray(buf, (1, 32, 64))
        desc = TensorMapDescriptor._from_im2col_wide(
            _as_view(tensor),
            pixel_box_lower_corner_width=0,
            pixel_box_upper_corner_width=4,
            channels_per_pixel=64,
            pixels_per_column=4,
            data_type=TensorMapDataType.FLOAT32,
        )
        assert desc is not None

    def test_from_im2col_wide_4d(self, dev, skip_if_no_im2col_wide):
        # NHWC layout: N=1, H=8, W=8, C=64
        # Wide mode only uses scalar W corners, even with higher rank
        shape = (1, 8, 8, 64)
        buf = dev.allocate(1 * 8 * 8 * 64 * 4)
        tensor = _DeviceArray(buf, shape)
        desc = TensorMapDescriptor._from_im2col_wide(
            _as_view(tensor),
            pixel_box_lower_corner_width=0,
            pixel_box_upper_corner_width=4,
            channels_per_pixel=64,
            pixels_per_column=16,
        )
        assert desc is not None

    def test_from_im2col_wide_5d(self, dev, skip_if_no_im2col_wide):
        # NDHWC layout: N=1, D=4, H=8, W=8, C=64
        # Max rank boundary — verifies all 5 dim/stride slots are filled
        shape = (1, 4, 8, 8, 64)
        buf = dev.allocate(1 * 4 * 8 * 8 * 64 * 4)
        tensor = _DeviceArray(buf, shape)
        desc = TensorMapDescriptor._from_im2col_wide(
            _as_view(tensor),
            pixel_box_lower_corner_width=0,
            pixel_box_upper_corner_width=4,
            channels_per_pixel=64,
            pixels_per_column=32,
        )
        assert desc is not None

    def test_from_im2col_wide_rank_validation(self, dev, skip_if_no_im2col_wide):
        buf = dev.allocate(1024 * 4)
        with pytest.raises(ValueError, match="Im2col-wide tensor rank must be between 3 and 5"):
            TensorMapDescriptor._from_im2col_wide(
                _as_view(buf),
                pixel_box_lower_corner_width=0,
                pixel_box_upper_corner_width=4,
                channels_per_pixel=64,
                pixels_per_column=4,
                data_type=TensorMapDataType.FLOAT32,
            )
