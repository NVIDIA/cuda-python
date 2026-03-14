# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._dlpack cimport kDLInt, kDLUInt, kDLFloat, kDLBfloat, _kDLCUDA

import enum

import numpy

from cuda.core._memoryview import StridedMemoryView

cdef extern from "_cpp/tensor_map_cccl.h":
    int cuda_core_cccl_make_tma_descriptor_tiled(
        void* out_tensor_map,
        void* data,
        int device_type,
        int device_id,
        int ndim,
        const int64_t* shape,
        const int64_t* strides,
        uint8_t dtype_code,
        uint8_t dtype_bits,
        uint16_t dtype_lanes,
        const int* box_sizes,
        const int* elem_strides,
        int interleave_layout,
        int swizzle,
        int l2_fetch_size,
        int oob_fill,
        char* err,
        size_t err_cap) nogil


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


IF CUDA_CORE_BUILD_MAJOR >= 13:
    class TensorMapIm2ColWideMode(enum.IntEnum):
        """Im2col wide mode for tensor map descriptors.

        These correspond to the ``CUtensorMapIm2ColWideMode`` driver enum values.
        Supported on compute capability 10.0+.
        """
        W = cydriver.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W
        W128 = cydriver.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128
ELSE:
    class TensorMapIm2ColWideMode(enum.IntEnum):
        """Im2col wide mode for tensor map descriptors.

        This enum is always defined for API stability, but the
        :meth:`TensorMapDescriptor._from_im2col_wide` factory requires a CUDA 13+
        build and will raise otherwise.
        """
        W = 0
        W128 = 1


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
        if isinstance(data_type, TensorMapDataType):
            return data_type
        try:
            dt = numpy.dtype(data_type)
        except TypeError as e:
            raise TypeError(
                "data_type must be a TensorMapDataType or a numpy/ml_dtypes dtype, "
                f"got {type(data_type)}") from e
        tma_dt = _NUMPY_DTYPE_TO_TMA.get(dt)
        if tma_dt is None:
            raise ValueError(
                f"Unsupported dtype {dt} for TMA; "
                f"supported dtypes: {list(_NUMPY_DTYPE_TO_TMA.keys())}.")
        return tma_dt

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


cdef inline bint _tma_dtype_to_dlpack(
    object tma_dt,
    uint8_t* out_code,
    uint8_t* out_bits,
    uint16_t* out_lanes,
) noexcept:
    if tma_dt == TensorMapDataType.UINT8:
        out_code[0] = <uint8_t>kDLUInt
        out_bits[0] = <uint8_t>8
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.UINT16:
        out_code[0] = <uint8_t>kDLUInt
        out_bits[0] = <uint8_t>16
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.UINT32:
        out_code[0] = <uint8_t>kDLUInt
        out_bits[0] = <uint8_t>32
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.UINT64:
        out_code[0] = <uint8_t>kDLUInt
        out_bits[0] = <uint8_t>64
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.INT32:
        out_code[0] = <uint8_t>kDLInt
        out_bits[0] = <uint8_t>32
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.INT64:
        out_code[0] = <uint8_t>kDLInt
        out_bits[0] = <uint8_t>64
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.FLOAT16:
        out_code[0] = <uint8_t>kDLFloat
        out_bits[0] = <uint8_t>16
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.FLOAT32:
        out_code[0] = <uint8_t>kDLFloat
        out_bits[0] = <uint8_t>32
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.FLOAT64:
        out_code[0] = <uint8_t>kDLFloat
        out_bits[0] = <uint8_t>64
        out_lanes[0] = <uint16_t>1
        return True
    if tma_dt == TensorMapDataType.BFLOAT16:
        out_code[0] = <uint8_t>kDLBfloat
        out_bits[0] = <uint8_t>16
        out_lanes[0] = <uint16_t>1
        return True
    return False


cdef inline int _validate_tensor_map_view(view) except -1:
    if not view.is_device_accessible:
        raise ValueError("The tensor must be device-accessible")

    if view.ptr % 16 != 0:
        raise ValueError(
            f"Global memory address must be 16-byte aligned, "
            f"got address 0x{view.ptr:x}")
    return 0


def _get_validated_view(tensor):
    """Obtain a device-accessible StridedMemoryView with a 16-byte-aligned pointer."""
    if isinstance(tensor, StridedMemoryView):
        view = tensor
    else:
        # stream_ptr=-1: no stream synchronization needed because descriptor
        # creation only reads tensor metadata, it does not move data.
        view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)
    _validate_tensor_map_view(view)
    return view


def _require_view_device(view, expected_device_id, operation):
    """Ensure device-local tensors match the current CUDA device.

    DLPack reports host/managed CUDA memory as ``kDLCUDAHost`` /
    ``kDLCUDAManaged`` with ``device_id=0`` regardless of the current device,
    so only true ``kDLCUDA`` tensors are rejected by device-id mismatch.
    """
    device_type, device_id = view.__dlpack_device__()
    if device_type == _kDLCUDA and device_id != expected_device_id:
        raise ValueError(
            f"{operation} expects tensor on device {expected_device_id}, got {device_id}")


cdef inline intptr_t _get_current_context_ptr() except? 0:
    cdef cydriver.CUcontext ctx
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    if ctx == NULL:
        raise RuntimeError("TensorMapDescriptor requires an active CUDA context")
    return <intptr_t>ctx


cdef inline int _get_current_device_id() except -1:
    cdef cydriver.CUdevice dev
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetDevice(&dev))
    return <int>dev


cdef inline int _require_view_device(
    view,
    int device_id,
    object caller,
) except -1:
    if view.device_id != device_id:
        raise ValueError(
            f"{caller} expects tensor on device {device_id}, got {view.device_id}")
    return 0


def _compute_byte_strides(shape, strides, elem_size):
    """Compute byte strides from element strides or C-contiguous fallback.

    Returns a tuple of byte strides in row-major order.
    """
    if strides is not None:
        return tuple(s * elem_size for s in strides)

    # C-contiguous: compute byte strides from shape, innermost first
    rank = len(shape)
    byte_strides = []
    stride = elem_size
    for i in range(rank - 1, -1, -1):
        byte_strides.append(stride)
        stride *= shape[i]
    byte_strides.reverse()
    return tuple(byte_strides)


def _validate_element_strides(element_strides, rank):
    """Validate or default element_strides to all-ones."""
    if element_strides is not None:
        if len(element_strides) != rank:
            raise ValueError(
                f"element_strides must have {rank} elements, got {len(element_strides)}")
        return element_strides
    return (1,) * rank


cdef class TensorMapDescriptor:
    """Describes a TMA (Tensor Memory Accelerator) tensor map for Hopper+ GPUs.

    A ``TensorMapDescriptor`` wraps the opaque 128-byte ``CUtensorMap`` struct
    used by the hardware TMA unit for efficient bulk data movement between
    global and shared memory.

    Public tiled descriptors are created via
    :meth:`cuda.core.StridedMemoryView.as_tensor_map`. Specialized
    ``_from_*`` helpers remain private while this API surface settles, and
    descriptors can be passed directly to :func:`~cuda.core.launch` as a
    kernel argument.
    """

    def __init__(self):
        raise RuntimeError(
            "TensorMapDescriptor cannot be instantiated directly. "
            "Use StridedMemoryView.as_tensor_map() instead.")

    cdef void* _get_data_ptr(self):
        return <void*>&self._tensor_map

    cdef int _check_context_compat(self) except -1:
        cdef cydriver.CUcontext current_ctx
        cdef cydriver.CUdevice current_dev
        cdef int current_dev_id
        if self._context == 0 and self._device_id < 0:
            return 0
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetCurrent(&current_ctx))
        if current_ctx == NULL:
            raise RuntimeError("TensorMapDescriptor requires an active CUDA context")
        if self._context != 0 and <intptr_t>current_ctx != self._context:
            raise RuntimeError(
                "TensorMapDescriptor was created in a different CUDA context")
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetDevice(&current_dev))
        current_dev_id = <int>current_dev
        if self._device_id >= 0 and current_dev_id != self._device_id:
            raise RuntimeError(
                f"TensorMapDescriptor belongs to device {self._device_id}, "
                f"but current device is {current_dev_id}")
        return 0

    @property
    def device(self):
        """Return the :obj:`~cuda.core.Device` associated with this descriptor."""
        if self._device_id >= 0:
            from cuda.core._device import Device
            return Device(self._device_id)

    @classmethod
    def _from_tiled(cls, view, box_dim, *,
                   element_strides=None,
                   data_type=None,
                   interleave=TensorMapInterleave.NONE,
                   swizzle=TensorMapSwizzle.NONE,
                   l2_promotion=TensorMapL2Promotion.NONE,
                   oob_fill=TensorMapOOBFill.NONE):
        """Create a tiled TMA descriptor from a validated view.

        Parameters
        ----------
        view : StridedMemoryView
            A device-accessible view with a 16-byte-aligned pointer.
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
        cdef TensorMapDescriptor desc = cls.__new__(cls)

        _validate_tensor_map_view(view)
        # Keep both the original tensor object and the validated view alive.
        # For DLPack exporters, the view may hold the owning capsule whose
        # deleter can free the backing allocation when released.
        desc._source_ref = view.exporting_obj
        desc._view_ref = view
        desc._context = _get_current_context_ptr()
        desc._device_id = _get_current_device_id()
        _require_view_device(view, desc._device_id, "TensorMapDescriptor._from_tiled")

        tma_dt = _resolve_data_type(view, data_type)
        cdef int c_data_type_int = int(tma_dt)
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        shape = view.shape

        cdef int rank = len(shape)
        if rank < 1 or rank > 5:
            raise ValueError(
                f"Tensor rank must be between 1 and 5, got {rank}")

        if len(box_dim) != rank:
            raise ValueError(
                f"box_dim must have {rank} elements (same as tensor rank), "
                f"got {len(box_dim)}")

        for i, bd in enumerate(box_dim):
            if bd < 1 or bd > 256:
                raise ValueError(
                    f"box_dim[{i}] must be in [1, 256], got {bd}")

        cdef bint elem_strides_provided = element_strides is not None
        element_strides = _validate_element_strides(element_strides, rank)

        # Reuse CCCL/libcu++'s DLPack -> CUtensorMap conversion when possible.
        # This avoids maintaining a second, independent validation/encoding implementation.
        cdef uint8_t dl_code
        cdef uint8_t dl_bits
        cdef uint16_t dl_lanes
        cdef int64_t c_shape[5]
        cdef int64_t c_strides[5]
        cdef int c_box_sizes[5]
        cdef int c_elem_strides[5]
        cdef const int64_t* c_strides_ptr
        cdef const int* c_elem_strides_ptr
        cdef char errbuf[512]
        cdef int i_cccl
        cdef int device_type
        cdef int c_device_id
        cdef int dl_device_type
        cdef int dl_device_id
        cdef int c_cccl_interleave_int
        cdef int c_cccl_swizzle_int
        cdef int c_cccl_l2_promotion_int
        cdef int c_cccl_oob_fill_int
        cdef int rc
        if _tma_dtype_to_dlpack(tma_dt, &dl_code, &dl_bits, &dl_lanes):
            c_strides_ptr = NULL
            c_elem_strides_ptr = NULL
            errbuf[0] = 0

            for i_cccl in range(rank):
                c_shape[i_cccl] = <int64_t>shape[i_cccl]
                c_box_sizes[i_cccl] = <int>box_dim[i_cccl]
                if elem_strides_provided:
                    c_elem_strides[i_cccl] = <int>element_strides[i_cccl]

            if view.strides is not None:
                for i_cccl in range(rank):
                    c_strides[i_cccl] = <int64_t>view.strides[i_cccl]
                c_strides_ptr = &c_strides[0]

            if elem_strides_provided:
                c_elem_strides_ptr = &c_elem_strides[0]

            dl_device_type, dl_device_id = view.__dlpack_device__()
            device_type = dl_device_type
            c_device_id = dl_device_id
            c_cccl_interleave_int = int(interleave)
            c_cccl_swizzle_int = int(swizzle)
            c_cccl_l2_promotion_int = int(l2_promotion)
            c_cccl_oob_fill_int = int(oob_fill)

            with nogil:
                rc = cuda_core_cccl_make_tma_descriptor_tiled(
                    <void*>&desc._tensor_map,
                    <void*>global_address,
                    device_type,
                    c_device_id,
                    rank,
                    &c_shape[0],
                    c_strides_ptr,
                    dl_code,
                    dl_bits,
                    dl_lanes,
                    &c_box_sizes[0],
                    c_elem_strides_ptr,
                    c_cccl_interleave_int,
                    c_cccl_swizzle_int,
                    c_cccl_l2_promotion_int,
                    c_cccl_oob_fill_int,
                    &errbuf[0],
                    <size_t>sizeof(errbuf),
                )

            if rc == 0:
                desc._repr_info = {
                    "method": "tiled",
                    "rank": rank,
                    "data_type": tma_dt,
                    "swizzle": swizzle,
                }
                return desc

            msg = errbuf[:].split(b"\0", 1)[0].decode("utf-8", errors="replace")
            # If CCCL isn't available at build time, fall back to the direct
            # driver API path to preserve functionality on older toolchains.
            if "not available at build time" not in msg:
                raise ValueError(f"Failed to build TMA descriptor via CCCL: {msg}")

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        byte_strides = _compute_byte_strides(shape, view.strides, elem_size)

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

    @classmethod
    def _from_im2col(cls, view, pixel_box_lower_corner, pixel_box_upper_corner,
                    channels_per_pixel, pixels_per_column, *,
                    element_strides=None,
                    data_type=None,
                    interleave=TensorMapInterleave.NONE,
                    swizzle=TensorMapSwizzle.NONE,
                    l2_promotion=TensorMapL2Promotion.NONE,
                    oob_fill=TensorMapOOBFill.NONE):
        """Create an im2col TMA descriptor from a validated view.

        Im2col layout is used for convolution-style data access patterns.

        Parameters
        ----------
        view : StridedMemoryView
            A device-accessible view with a 16-byte-aligned pointer.
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
        cdef TensorMapDescriptor desc = cls.__new__(cls)

        _validate_tensor_map_view(view)
        desc._source_ref = view.exporting_obj
        desc._view_ref = view
        desc._context = _get_current_context_ptr()
        desc._device_id = _get_current_device_id()
        _require_view_device(view, desc._device_id, "TensorMapDescriptor._from_im2col")

        tma_dt = _resolve_data_type(view, data_type)
        cdef int c_data_type_int = int(tma_dt)
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        shape = view.shape

        cdef int rank = len(shape)
        if rank < 3 or rank > 5:
            raise ValueError(
                f"Im2col tensor rank must be between 3 and 5, got {rank}")

        cdef int n_spatial = rank - 2
        if len(pixel_box_lower_corner) != n_spatial:
            raise ValueError(
                f"pixel_box_lower_corner must have {n_spatial} elements "
                f"(rank - 2), got {len(pixel_box_lower_corner)}")
        if len(pixel_box_upper_corner) != n_spatial:
            raise ValueError(
                f"pixel_box_upper_corner must have {n_spatial} elements "
                f"(rank - 2), got {len(pixel_box_upper_corner)}")

        element_strides = _validate_element_strides(element_strides, rank)

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        byte_strides = _compute_byte_strides(shape, view.strides, elem_size)

        # Reverse all dimension arrays for column-major convention
        cdef uint64_t[5] c_global_dim
        cdef uint64_t[4] c_global_strides
        cdef uint32_t[5] c_element_strides
        cdef int[3] c_pixel_box_lower  # max 3 spatial dims (rank 5 - 2)
        cdef int[3] c_pixel_box_upper
        cdef int i_c

        for i_c in range(3):
            c_pixel_box_lower[i_c] = 0
            c_pixel_box_upper[i_c] = 0

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

    @classmethod
    def _from_im2col_wide(cls, view, pixel_box_lower_corner_width, pixel_box_upper_corner_width,
                         channels_per_pixel, pixels_per_column, *,
                         element_strides=None,
                         data_type=None,
                         interleave=TensorMapInterleave.NONE,
                         mode=TensorMapIm2ColWideMode.W,
                         swizzle=TensorMapSwizzle.SWIZZLE_128B,
                         l2_promotion=TensorMapL2Promotion.NONE,
                         oob_fill=TensorMapOOBFill.NONE):
        """Create an im2col-wide TMA descriptor from a validated view.

        Im2col-wide layout loads elements exclusively along the W (width)
        dimension. This variant is supported on compute capability 10.0+
        (Blackwell and later).

        Parameters
        ----------
        view : StridedMemoryView
            A device-accessible view with a 16-byte-aligned pointer.
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
        IF CUDA_CORE_BUILD_MAJOR < 13:
            raise RuntimeError(
                "TensorMapDescriptor._from_im2col_wide requires a CUDA 13+ build")
        ELSE:
            cdef TensorMapDescriptor desc = cls.__new__(cls)

            _validate_tensor_map_view(view)
            desc._source_ref = view.exporting_obj
            desc._view_ref = view
            desc._context = _get_current_context_ptr()
            desc._device_id = _get_current_device_id()
            _require_view_device(view, desc._device_id, "TensorMapDescriptor._from_im2col_wide")

            tma_dt = _resolve_data_type(view, data_type)
            cdef int c_data_type_int = int(tma_dt)
            cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

            cdef intptr_t global_address = view.ptr
            shape = view.shape

            cdef int rank = len(shape)
            if rank < 3 or rank > 5:
                raise ValueError(
                    f"Im2col-wide tensor rank must be between 3 and 5, got {rank}")

            element_strides = _validate_element_strides(element_strides, rank)

            cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
            byte_strides = _compute_byte_strides(shape, view.strides, elem_size)

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
        self._check_context_compat()
        view = _get_validated_view(tensor)
        _require_view_device(view, self._device_id, "replace_address")

        cdef intptr_t global_address = view.ptr

        with nogil:
            HANDLE_RETURN(cydriver.cuTensorMapReplaceAddress(
                &self._tensor_map,
                <void*>global_address,
            ))

        # Update the source reference only after the driver call succeeds,
        # so we don't drop the old tensor (risking a dangling pointer in the
        # CUtensorMap struct) if the call fails.
        self._source_ref = view.exporting_obj
        self._view_ref = view

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
