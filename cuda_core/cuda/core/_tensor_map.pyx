# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, uint32_t, uint64_t
from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import enum

import numpy

from cuda.core._memoryview import StridedMemoryView


try:
    from ml_dtypes import bfloat16 as ml_bfloat16
except ImportError:
    ml_bfloat16 = None


class TensorMapDataType(enum.IntEnum):
    """Data types for tensor map descriptors.

    These correspond to the ``CUtensorMapDataType`` driver enum values.
    """
    UINT8 = cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT8
    UINT16 = cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT16
    UINT32 = cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT32
    INT32 = cydriver.CU_TENSOR_MAP_DATA_TYPE_INT32
    UINT64 = cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT64
    INT64 = cydriver.CU_TENSOR_MAP_DATA_TYPE_INT64
    FLOAT16 = cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
    FLOAT32 = cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
    FLOAT64 = cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT64
    BFLOAT16 = cydriver.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    FLOAT32_FTZ = cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ
    TFLOAT32 = cydriver.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32
    TFLOAT32_FTZ = cydriver.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ


class TensorMapInterleave(enum.IntEnum):
    """Interleave layout for tensor map descriptors.

    These correspond to the ``CUtensorMapInterleave`` driver enum values.
    """
    NONE = cydriver.CU_TENSOR_MAP_INTERLEAVE_NONE
    INTERLEAVE_16B = cydriver.CU_TENSOR_MAP_INTERLEAVE_16B
    INTERLEAVE_32B = cydriver.CU_TENSOR_MAP_INTERLEAVE_32B


class TensorMapSwizzle(enum.IntEnum):
    """Swizzle mode for tensor map descriptors.

    These correspond to the ``CUtensorMapSwizzle`` driver enum values.
    """
    NONE = cydriver.CU_TENSOR_MAP_SWIZZLE_NONE
    SWIZZLE_32B = cydriver.CU_TENSOR_MAP_SWIZZLE_32B
    SWIZZLE_64B = cydriver.CU_TENSOR_MAP_SWIZZLE_64B
    SWIZZLE_128B = cydriver.CU_TENSOR_MAP_SWIZZLE_128B


class TensorMapL2Promotion(enum.IntEnum):
    """L2 promotion mode for tensor map descriptors.

    These correspond to the ``CUtensorMapL2promotion`` driver enum values.
    """
    NONE = cydriver.CU_TENSOR_MAP_L2_PROMOTION_NONE
    L2_64B = cydriver.CU_TENSOR_MAP_L2_PROMOTION_L2_64B
    L2_128B = cydriver.CU_TENSOR_MAP_L2_PROMOTION_L2_128B
    L2_256B = cydriver.CU_TENSOR_MAP_L2_PROMOTION_L2_256B


class TensorMapOOBFill(enum.IntEnum):
    """Out-of-bounds fill mode for tensor map descriptors.

    These correspond to the ``CUtensorMapFloatOOBfill`` driver enum values.
    """
    NONE = cydriver.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    NAN_REQUEST_ZERO_FMA = cydriver.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA


class TensorMapIm2ColWideMode(enum.IntEnum):
    """Im2col wide mode for tensor map descriptors.

    These correspond to the ``CUtensorMapIm2ColWideMode`` driver enum values.
    Supported on compute capability 10.0+.
    """
    W = cydriver.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W
    W128 = cydriver.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128


# Mapping from numpy dtype to TMA data type
_NUMPY_DTYPE_TO_TMA = {
    numpy.dtype(numpy.uint8): TensorMapDataType.UINT8,
    numpy.dtype(numpy.uint16): TensorMapDataType.UINT16,
    numpy.dtype(numpy.uint32): TensorMapDataType.UINT32,
    numpy.dtype(numpy.int32): TensorMapDataType.INT32,
    numpy.dtype(numpy.uint64): TensorMapDataType.UINT64,
    numpy.dtype(numpy.int64): TensorMapDataType.INT64,
    numpy.dtype(numpy.float16): TensorMapDataType.FLOAT16,
    numpy.dtype(numpy.float32): TensorMapDataType.FLOAT32,
    numpy.dtype(numpy.float64): TensorMapDataType.FLOAT64,
}

if ml_bfloat16 is not None:
    _NUMPY_DTYPE_TO_TMA[numpy.dtype(ml_bfloat16)] = TensorMapDataType.BFLOAT16


# Mapping from TMA data type to element size in bytes
_TMA_DATA_TYPE_SIZE = {
    TensorMapDataType.UINT8: 1,
    TensorMapDataType.UINT16: 2,
    TensorMapDataType.UINT32: 4,
    TensorMapDataType.INT32: 4,
    TensorMapDataType.UINT64: 8,
    TensorMapDataType.INT64: 8,
    TensorMapDataType.FLOAT16: 2,
    TensorMapDataType.FLOAT32: 4,
    TensorMapDataType.FLOAT64: 8,
    TensorMapDataType.BFLOAT16: 2,
    TensorMapDataType.FLOAT32_FTZ: 4,
    TensorMapDataType.TFLOAT32: 4,
    TensorMapDataType.TFLOAT32_FTZ: 4,
}


def _resolve_data_type(view, data_type):
    """Resolve the TMA data type from an explicit value or the view's dtype."""

    if data_type is not None:
        if not isinstance(data_type, TensorMapDataType):
            raise TypeError(
                f"data_type must be a TensorMapDataType, got {type(data_type)}")
        return data_type

    dt = view.dtype
    if dt is None:
        raise ValueError(
            "Cannot infer TMA data type from the tensor; "
            "please specify data_type explicitly")

    tma_dt = _NUMPY_DTYPE_TO_TMA.get(dt)
    if tma_dt is None:
        raise ValueError(
            f"Unsupported dtype {dt} for TMA; "
            f"supported dtypes: {list(_NUMPY_DTYPE_TO_TMA.keys())}. "
            "You may also specify data_type explicitly.")

    return tma_dt


cdef class TensorMapDescriptor:
    """Describes a TMA (Tensor Memory Accelerator) tensor map for Hopper+ GPUs.

    A ``TensorMapDescriptor`` wraps the opaque 128-byte ``CUtensorMap`` struct
    used by the hardware TMA unit for efficient bulk data movement between
    global and shared memory.

    Instances are created via the class methods :meth:`from_tiled` and
    :meth:`from_im2col`, and can be passed directly to
    :func:`~cuda.core.launch` as a kernel argument.
    """

    def __init__(self):
        raise RuntimeError(
            "TensorMapDescriptor cannot be instantiated directly. "
            "Use TensorMapDescriptor.from_tiled() or "
            "TensorMapDescriptor.from_im2col().")

    cdef void* _get_data_ptr(self):
        return <void*>&self._tensor_map

    @staticmethod
    def from_tiled(tensor, box_dim, *,
                   element_strides=None,
                   data_type=None,
                   interleave=TensorMapInterleave.NONE,
                   swizzle=TensorMapSwizzle.NONE,
                   l2_promotion=TensorMapL2Promotion.NONE,
                   oob_fill=TensorMapOOBFill.NONE):
        """Create a tiled TMA descriptor from a tensor object.

        Parameters
        ----------
        tensor : object
            Any object supporting DLPack or ``__cuda_array_interface__``,
            or a :obj:`~cuda.core.StridedMemoryView`. Must refer to
            device-accessible memory with a 16-byte-aligned pointer.
        box_dim : tuple of int
            The size of each tile dimension (in elements). Must have the
            same rank as the tensor and each value must be in [1, 256].
            Specified in the same (row-major) order as the tensor shape.
        element_strides : tuple of int, optional
            Per-dimension element traversal strides. Default is all 1s.
            Specified in the same (row-major) order as the tensor shape.
        data_type : TensorMapDataType, optional
            Explicit data type override. If ``None``, inferred from the
            tensor's dtype.
        interleave : TensorMapInterleave
            Interleave layout. Default ``NONE``.
        swizzle : TensorMapSwizzle
            Swizzle mode. Default ``NONE``.
        l2_promotion : TensorMapL2Promotion
            L2 promotion mode. Default ``NONE``.
        oob_fill : TensorMapOOBFill
            Out-of-bounds fill mode. Default ``NONE``.

        Returns
        -------
        TensorMapDescriptor

        Raises
        ------
        ValueError
            If the tensor rank is outside [1, 5], the pointer is not
            16-byte aligned, or dimension/stride constraints are violated.
        """
        cdef TensorMapDescriptor desc = TensorMapDescriptor.__new__(TensorMapDescriptor)

        # Obtain a StridedMemoryView from the tensor
        if isinstance(tensor, StridedMemoryView):
            view = tensor
        else:
            view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)

        if not view.is_device_accessible:
            raise ValueError("The tensor must be device-accessible")

        # Keep a strong reference to prevent GC
        desc._source_ref = tensor

        # Resolve data type
        tma_dt = _resolve_data_type(view, data_type)
        cdef int c_data_type_int = int(tma_dt)
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        # Get tensor metadata
        cdef intptr_t global_address = view.ptr
        shape = view.shape
        strides = view.strides  # in elements, can be None for C-contiguous

        cdef int rank = len(shape)
        if rank < 1 or rank > 5:
            raise ValueError(
                f"Tensor rank must be between 1 and 5, got {rank}")

        if global_address % 16 != 0:
            raise ValueError(
                f"Global memory address must be 16-byte aligned, "
                f"got address 0x{global_address:x} (misaligned by {global_address % 16} bytes)")

        if len(box_dim) != rank:
            raise ValueError(
                f"box_dim must have {rank} elements (same as tensor rank), "
                f"got {len(box_dim)}")

        for i, bd in enumerate(box_dim):
            if bd < 1 or bd > 256:
                raise ValueError(
                    f"box_dim[{i}] must be in [1, 256], got {bd}")

        if element_strides is not None:
            if len(element_strides) != rank:
                raise ValueError(
                    f"element_strides must have {rank} elements, got {len(element_strides)}")
        else:
            element_strides = (1,) * rank

        # Compute byte strides from element strides
        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        if strides is not None:
            byte_strides = tuple(s * elem_size for s in strides)
        else:
            # C-contiguous: strides in bytes, row-major
            byte_strides = []
            stride = elem_size
            for i in range(rank - 1, -1, -1):
                byte_strides.append(stride)
                stride *= shape[i]
            byte_strides.reverse()

        # Reverse dimensions for column-major cuTensorMap convention
        # Python/DLPack: row-major (dim 0 = outermost)
        # cuTensorMap: column-major (dim 0 = innermost)
        cdef uint64_t[5] c_global_dim
        cdef uint64_t[4] c_global_strides  # rank - 1 elements
        cdef uint32_t[5] c_box_dim
        cdef uint32_t[5] c_element_strides
        cdef int i_c

        for i_c in range(rank):
            # Reverse: Python dim i -> cuTensorMap dim (rank - 1 - i)
            c_global_dim[i_c] = <uint64_t>shape[rank - 1 - i_c]
            c_box_dim[i_c] = <uint32_t>box_dim[rank - 1 - i_c]
            c_element_strides[i_c] = <uint32_t>element_strides[rank - 1 - i_c]

        # globalStrides: rank-1 elements (byte strides for dims 1..N-1 in col-major order)
        # The innermost stride (dim 0) is implicit = element size
        for i_c in range(rank - 1):
            c_global_strides[i_c] = <uint64_t>byte_strides[rank - 2 - i_c]

        cdef uint32_t c_rank = <uint32_t>rank
        cdef int c_interleave_int = int(interleave)
        cdef int c_swizzle_int = int(swizzle)
        cdef int c_l2_promotion_int = int(l2_promotion)
        cdef int c_oob_fill_int = int(oob_fill)
        cdef cydriver.CUtensorMapInterleave c_interleave = <cydriver.CUtensorMapInterleave>c_interleave_int
        cdef cydriver.CUtensorMapSwizzle c_swizzle = <cydriver.CUtensorMapSwizzle>c_swizzle_int
        cdef cydriver.CUtensorMapL2promotion c_l2_promotion = <cydriver.CUtensorMapL2promotion>c_l2_promotion_int
        cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = <cydriver.CUtensorMapFloatOOBfill>c_oob_fill_int

        with nogil:
            HANDLE_RETURN(cydriver.cuTensorMapEncodeTiled(
                &desc._tensor_map,
                c_data_type,
                c_rank,
                <void*>global_address,
                c_global_dim,
                c_global_strides,
                c_box_dim,
                c_element_strides,
                c_interleave,
                c_swizzle,
                c_l2_promotion,
                c_oob_fill,
            ))

        desc._repr_info = {
            "method": "tiled",
            "rank": rank,
            "data_type": tma_dt,
            "swizzle": swizzle,
        }

        return desc

    @staticmethod
    def from_im2col(tensor, pixel_box_lower_corner, pixel_box_upper_corner,
                    channels_per_pixel, pixels_per_column, *,
                    element_strides=None,
                    data_type=None,
                    interleave=TensorMapInterleave.NONE,
                    swizzle=TensorMapSwizzle.NONE,
                    l2_promotion=TensorMapL2Promotion.NONE,
                    oob_fill=TensorMapOOBFill.NONE):
        """Create an im2col TMA descriptor from a tensor object.

        Im2col layout is used for convolution-style data access patterns.

        Parameters
        ----------
        tensor : object
            Any object supporting DLPack or ``__cuda_array_interface__``,
            or a :obj:`~cuda.core.StridedMemoryView`. Must refer to
            device-accessible memory with a 16-byte-aligned pointer.
        pixel_box_lower_corner : tuple of int
            Lower corner of the pixel bounding box for each spatial
            dimension (rank - 2 elements). Specified in row-major order
            matching the tensor's spatial dimensions.
        pixel_box_upper_corner : tuple of int
            Upper corner of the pixel bounding box for each spatial
            dimension (rank - 2 elements). Specified in row-major order
            matching the tensor's spatial dimensions.
        channels_per_pixel : int
            Number of channels per pixel.
        pixels_per_column : int
            Number of pixels per column.
        element_strides : tuple of int, optional
            Per-dimension element traversal strides. Default is all 1s.
        data_type : TensorMapDataType, optional
            Explicit data type override. If ``None``, inferred from the
            tensor's dtype.
        interleave : TensorMapInterleave
            Interleave layout. Default ``NONE``.
        swizzle : TensorMapSwizzle
            Swizzle mode. Default ``NONE``.
        l2_promotion : TensorMapL2Promotion
            L2 promotion mode. Default ``NONE``.
        oob_fill : TensorMapOOBFill
            Out-of-bounds fill mode. Default ``NONE``.

        Returns
        -------
        TensorMapDescriptor

        Raises
        ------
        ValueError
            If the tensor rank is outside [3, 5], the pointer is not
            16-byte aligned, or other constraints are violated.
        """
        cdef TensorMapDescriptor desc = TensorMapDescriptor.__new__(TensorMapDescriptor)

        # Obtain a StridedMemoryView from the tensor
        if isinstance(tensor, StridedMemoryView):
            view = tensor
        else:
            view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)

        if not view.is_device_accessible:
            raise ValueError("The tensor must be device-accessible")

        desc._source_ref = tensor

        tma_dt = _resolve_data_type(view, data_type)
        cdef int c_data_type_int = int(tma_dt)
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        shape = view.shape
        strides = view.strides

        cdef int rank = len(shape)
        if rank < 3 or rank > 5:
            raise ValueError(
                f"Im2col tensor rank must be between 3 and 5, got {rank}")

        if global_address % 16 != 0:
            raise ValueError(
                f"Global memory address must be 16-byte aligned, "
                f"got address 0x{global_address:x}")

        cdef int n_spatial = rank - 2
        if len(pixel_box_lower_corner) != n_spatial:
            raise ValueError(
                f"pixel_box_lower_corner must have {n_spatial} elements "
                f"(rank - 2), got {len(pixel_box_lower_corner)}")
        if len(pixel_box_upper_corner) != n_spatial:
            raise ValueError(
                f"pixel_box_upper_corner must have {n_spatial} elements "
                f"(rank - 2), got {len(pixel_box_upper_corner)}")

        if element_strides is not None:
            if len(element_strides) != rank:
                raise ValueError(
                    f"element_strides must have {rank} elements, got {len(element_strides)}")
        else:
            element_strides = (1,) * rank

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        if strides is not None:
            byte_strides = tuple(s * elem_size for s in strides)
        else:
            byte_strides = []
            stride = elem_size
            for i in range(rank - 1, -1, -1):
                byte_strides.append(stride)
                stride *= shape[i]
            byte_strides.reverse()

        # Reverse all dimension arrays for column-major convention
        cdef uint64_t[5] c_global_dim
        cdef uint64_t[4] c_global_strides
        cdef uint32_t[5] c_element_strides
        cdef int[3] c_pixel_box_lower  # max 3 spatial dims (rank 5 - 2)
        cdef int[3] c_pixel_box_upper
        cdef int i_c

        for i_c in range(rank):
            c_global_dim[i_c] = <uint64_t>shape[rank - 1 - i_c]
            c_element_strides[i_c] = <uint32_t>element_strides[rank - 1 - i_c]

        for i_c in range(rank - 1):
            c_global_strides[i_c] = <uint64_t>byte_strides[rank - 2 - i_c]

        # Reverse spatial dimensions for lower/upper corners
        for i_c in range(n_spatial):
            c_pixel_box_lower[i_c] = <int>pixel_box_lower_corner[n_spatial - 1 - i_c]
            c_pixel_box_upper[i_c] = <int>pixel_box_upper_corner[n_spatial - 1 - i_c]

        cdef uint32_t c_rank = <uint32_t>rank
        cdef uint32_t c_channels = <uint32_t>channels_per_pixel
        cdef uint32_t c_pixels = <uint32_t>pixels_per_column
        cdef int c_interleave_int = int(interleave)
        cdef int c_swizzle_int = int(swizzle)
        cdef int c_l2_promotion_int = int(l2_promotion)
        cdef int c_oob_fill_int = int(oob_fill)
        cdef cydriver.CUtensorMapInterleave c_interleave = <cydriver.CUtensorMapInterleave>c_interleave_int
        cdef cydriver.CUtensorMapSwizzle c_swizzle = <cydriver.CUtensorMapSwizzle>c_swizzle_int
        cdef cydriver.CUtensorMapL2promotion c_l2_promotion = <cydriver.CUtensorMapL2promotion>c_l2_promotion_int
        cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = <cydriver.CUtensorMapFloatOOBfill>c_oob_fill_int

        with nogil:
            HANDLE_RETURN(cydriver.cuTensorMapEncodeIm2col(
                &desc._tensor_map,
                c_data_type,
                c_rank,
                <void*>global_address,
                c_global_dim,
                c_global_strides,
                c_pixel_box_lower,
                c_pixel_box_upper,
                c_channels,
                c_pixels,
                c_element_strides,
                c_interleave,
                c_swizzle,
                c_l2_promotion,
                c_oob_fill,
            ))

        desc._repr_info = {
            "method": "im2col",
            "rank": rank,
            "data_type": tma_dt,
            "swizzle": swizzle,
        }

        return desc

    @staticmethod
    def from_im2col_wide(tensor, pixel_box_lower_corner_width, pixel_box_upper_corner_width,
                         channels_per_pixel, pixels_per_column, *,
                         element_strides=None,
                         data_type=None,
                         interleave=TensorMapInterleave.NONE,
                         mode=TensorMapIm2ColWideMode.W,
                         swizzle=TensorMapSwizzle.SWIZZLE_128B,
                         l2_promotion=TensorMapL2Promotion.NONE,
                         oob_fill=TensorMapOOBFill.NONE):
        """Create an im2col-wide TMA descriptor from a tensor object.

        Im2col-wide layout loads elements exclusively along the W (width)
        dimension. This variant is supported on compute capability 10.0+
        (Blackwell and later).

        Parameters
        ----------
        tensor : object
            Any object supporting DLPack or ``__cuda_array_interface__``,
            or a :obj:`~cuda.core.StridedMemoryView`. Must refer to
            device-accessible memory with a 16-byte-aligned pointer.
        pixel_box_lower_corner_width : int
            Lower corner of the pixel bounding box along the W dimension.
        pixel_box_upper_corner_width : int
            Upper corner of the pixel bounding box along the W dimension.
        channels_per_pixel : int
            Number of channels per pixel.
        pixels_per_column : int
            Number of pixels per column.
        element_strides : tuple of int, optional
            Per-dimension element traversal strides. Default is all 1s.
        data_type : TensorMapDataType, optional
            Explicit data type override. If ``None``, inferred from the
            tensor's dtype.
        interleave : TensorMapInterleave
            Interleave layout. Default ``NONE``.
        mode : TensorMapIm2ColWideMode
            Im2col wide mode. Default ``W``.
        swizzle : TensorMapSwizzle
            Swizzle mode. Default ``SWIZZLE_128B``.
        l2_promotion : TensorMapL2Promotion
            L2 promotion mode. Default ``NONE``.
        oob_fill : TensorMapOOBFill
            Out-of-bounds fill mode. Default ``NONE``.

        Returns
        -------
        TensorMapDescriptor

        Raises
        ------
        ValueError
            If the tensor rank is outside [3, 5], the pointer is not
            16-byte aligned, or other constraints are violated.
        """
        cdef TensorMapDescriptor desc = TensorMapDescriptor.__new__(TensorMapDescriptor)

        # Obtain a StridedMemoryView from the tensor
        if isinstance(tensor, StridedMemoryView):
            view = tensor
        else:
            view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)

        if not view.is_device_accessible:
            raise ValueError("The tensor must be device-accessible")

        desc._source_ref = tensor

        tma_dt = _resolve_data_type(view, data_type)
        cdef int c_data_type_int = int(tma_dt)
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        shape = view.shape
        strides = view.strides

        cdef int rank = len(shape)
        if rank < 3 or rank > 5:
            raise ValueError(
                f"Im2col-wide tensor rank must be between 3 and 5, got {rank}")

        if global_address % 16 != 0:
            raise ValueError(
                f"Global memory address must be 16-byte aligned, "
                f"got address 0x{global_address:x}")

        if element_strides is not None:
            if len(element_strides) != rank:
                raise ValueError(
                    f"element_strides must have {rank} elements, got {len(element_strides)}")
        else:
            element_strides = (1,) * rank

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        if strides is not None:
            byte_strides = tuple(s * elem_size for s in strides)
        else:
            byte_strides = []
            stride = elem_size
            for i in range(rank - 1, -1, -1):
                byte_strides.append(stride)
                stride *= shape[i]
            byte_strides.reverse()

        # Reverse all dimension arrays for column-major convention
        cdef uint64_t[5] c_global_dim
        cdef uint64_t[4] c_global_strides
        cdef uint32_t[5] c_element_strides
        cdef int i_c

        for i_c in range(rank):
            c_global_dim[i_c] = <uint64_t>shape[rank - 1 - i_c]
            c_element_strides[i_c] = <uint32_t>element_strides[rank - 1 - i_c]

        for i_c in range(rank - 1):
            c_global_strides[i_c] = <uint64_t>byte_strides[rank - 2 - i_c]

        cdef uint32_t c_rank = <uint32_t>rank
        cdef int c_lower_w = <int>pixel_box_lower_corner_width
        cdef int c_upper_w = <int>pixel_box_upper_corner_width
        cdef uint32_t c_channels = <uint32_t>channels_per_pixel
        cdef uint32_t c_pixels = <uint32_t>pixels_per_column
        cdef int c_interleave_int = int(interleave)
        cdef int c_mode_int = int(mode)
        cdef int c_swizzle_int = int(swizzle)
        cdef int c_l2_promotion_int = int(l2_promotion)
        cdef int c_oob_fill_int = int(oob_fill)
        cdef cydriver.CUtensorMapInterleave c_interleave = <cydriver.CUtensorMapInterleave>c_interleave_int
        cdef cydriver.CUtensorMapIm2ColWideMode c_mode = <cydriver.CUtensorMapIm2ColWideMode>c_mode_int
        cdef cydriver.CUtensorMapSwizzle c_swizzle = <cydriver.CUtensorMapSwizzle>c_swizzle_int
        cdef cydriver.CUtensorMapL2promotion c_l2_promotion = <cydriver.CUtensorMapL2promotion>c_l2_promotion_int
        cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = <cydriver.CUtensorMapFloatOOBfill>c_oob_fill_int

        with nogil:
            HANDLE_RETURN(cydriver.cuTensorMapEncodeIm2colWide(
                &desc._tensor_map,
                c_data_type,
                c_rank,
                <void*>global_address,
                c_global_dim,
                c_global_strides,
                c_lower_w,
                c_upper_w,
                c_channels,
                c_pixels,
                c_element_strides,
                c_interleave,
                c_mode,
                c_swizzle,
                c_l2_promotion,
                c_oob_fill,
            ))

        desc._repr_info = {
            "method": "im2col_wide",
            "rank": rank,
            "data_type": tma_dt,
            "swizzle": swizzle,
        }

        return desc

    def replace_address(self, tensor):
        """Replace the global memory address in this tensor map descriptor.

        This is useful when the tensor data has been reallocated but the
        shape, strides, and other parameters remain the same.

        Parameters
        ----------
        tensor : object
            Any object supporting DLPack or ``__cuda_array_interface__``,
            or a :obj:`~cuda.core.StridedMemoryView`. Must refer to
            device-accessible memory with a 16-byte-aligned pointer.
        """
        if isinstance(tensor, StridedMemoryView):
            view = tensor
        else:
            view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)

        if not view.is_device_accessible:
            raise ValueError("The tensor must be device-accessible")

        cdef intptr_t global_address = view.ptr
        if global_address % 16 != 0:
            raise ValueError(
                f"Global memory address must be 16-byte aligned, "
                f"got address 0x{global_address:x}")

        with nogil:
            HANDLE_RETURN(cydriver.cuTensorMapReplaceAddress(
                &self._tensor_map,
                <void*>global_address,
            ))

        # Update the source reference only after the driver call succeeds,
        # so we don't drop the old tensor (risking a dangling pointer in the
        # CUtensorMap struct) if the call fails.
        self._source_ref = tensor

    def __repr__(self):
        info = self._repr_info
        if info is None:
            return "TensorMapDescriptor()"
        parts = []
        if "method" in info:
            parts.append(info["method"])
        if "rank" in info:
            parts.append(f"rank={info['rank']}")
        if "data_type" in info:
            parts.append(f"dtype={info['data_type'].name}")
        if "swizzle" in info:
            parts.append(f"swizzle={info['swizzle'].name}")
        return f"TensorMapDescriptor({', '.join(parts)})"
