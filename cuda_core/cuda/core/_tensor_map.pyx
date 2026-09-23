# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t, int64_t, uint32_t, uint64_t
from libc.stddef cimport size_t
from cuda.bindings cimport cydriver
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core._dlpack cimport (
    DLDataType, DLTensor, kDLInt, kDLUInt, kDLFloat, kDLBfloat, _DLDeviceType, _kDLCUDA
)
from cuda.core._layout cimport _StridedLayout
from cuda.core._memoryview cimport StridedMemoryView

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from cuda.core._utils.cuda_utils import CUDAError, check_or_create_options

if TYPE_CHECKING:
    from cuda.core._device import Device

cdef extern from "<cuda/std/span>" nogil:
    cdef cppclass const_int_span "::cuda::std::span<const int>":
        const_int_span(const int*, size_t)


cdef extern from "<cuda/tma>" namespace "cuda" nogil:
    # Only the `none` enumerators are spelled out: the options arrive as the
    # CU_TENSOR_MAP_* values these scoped enums are defined from (see the enums
    # below), so they are cast rather than mapped name by name.
    cdef enum class tma_interleave_layout(int):
        none
    cdef enum class tma_swizzle(int):
        none
    cdef enum class tma_l2_fetch_size(int):
        none
    cdef enum class tma_oob_fill(int):
        none

    cydriver.CUtensorMap make_tma_descriptor(
        const DLTensor&,
        const_int_span box_sizes,
        const_int_span elem_strides,
        tma_interleave_layout,
        tma_swizzle,
        tma_l2_fetch_size,
        tma_oob_fill) except +


try:
    from ml_dtypes import bfloat16 as ml_bfloat16
except ImportError:
    ml_bfloat16 = None

__all__ = ['TensorMapDescriptor', 'TensorMapDescriptorOptions']


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


_TMA_DT_UINT8: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT8)
_TMA_DT_UINT16: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT16)
_TMA_DT_UINT32: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT32)
_TMA_DT_INT32: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_INT32)
_TMA_DT_UINT64: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_UINT64)
_TMA_DT_INT64: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_INT64)
_TMA_DT_FLOAT16: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT16)
_TMA_DT_FLOAT32: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT32)
_TMA_DT_FLOAT64: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT64)
_TMA_DT_BFLOAT16: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16)
_TMA_DT_FLOAT32_FTZ: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ)
_TMA_DT_TFLOAT32: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32)
_TMA_DT_TFLOAT32_FTZ: int = int(cydriver.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ)


def _normalize_tensor_map_data_type(data_type):
    if data_type is None or isinstance(data_type, TensorMapDataType):
        return data_type
    try:
        return numpy.dtype(data_type)
    except TypeError as e:
        raise TypeError(
            "data_type must be a TensorMapDataType or a numpy/ml_dtypes dtype, "
            f"got {type(data_type)}") from e


def _normalize_tensor_map_sequence(name, values):
    try:
        values = tuple(values)
    except TypeError as e:
        raise TypeError(f"{name} must be a tuple of ints, got {type(values)}") from e
    for i, value in enumerate(values):
        if not isinstance(value, int):
            raise TypeError(f"{name}[{i}] must be an int, got {type(value)}")
    return values


def _require_tensor_map_enum(name, value, enum_type):
    if not isinstance(value, enum_type):
        raise TypeError(f"{name} must be a {enum_type.__name__}, got {type(value)}")
    return value


@dataclass
class TensorMapDescriptorOptions:
    """Options for :meth:`cuda.core.StridedMemoryView.as_tensor_map`.

    Attributes
    ----------
    box_dim : tuple[int, ...]
        Tile size for each tensor dimension, expressed in elements.
    element_strides : tuple[int, ...], optional
        Per-dimension element traversal strides.
    data_type : object, optional
        Explicit dtype override. Prefer NumPy or ``ml_dtypes`` dtype objects;
        :class:`TensorMapDataType` remains accepted for compatibility.
    interleave : TensorMapInterleave, optional
        Interleave layout. Default ``NONE``.
    swizzle : TensorMapSwizzle, optional
        Swizzle mode. Default ``NONE``.
    l2_promotion : TensorMapL2Promotion, optional
        L2 promotion mode. Default ``NONE``.
    oob_fill : TensorMapOOBFill, optional
        Out-of-bounds fill mode. Default ``NONE``.
    """

    box_dim: tuple[int, ...]
    element_strides: tuple[int, ...] | None = None
    data_type: object = None
    interleave: TensorMapInterleave = TensorMapInterleave.NONE
    swizzle: TensorMapSwizzle = TensorMapSwizzle.NONE
    l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE
    oob_fill: TensorMapOOBFill = TensorMapOOBFill.NONE

    def __post_init__(self) -> None:
        self.box_dim = _normalize_tensor_map_sequence("box_dim", self.box_dim)
        if self.element_strides is not None:
            self.element_strides = _normalize_tensor_map_sequence("element_strides", self.element_strides)
        self.data_type = _normalize_tensor_map_data_type(self.data_type)
        self.interleave = _require_tensor_map_enum("interleave", self.interleave, TensorMapInterleave)
        self.swizzle = _require_tensor_map_enum("swizzle", self.swizzle, TensorMapSwizzle)
        self.l2_promotion = _require_tensor_map_enum("l2_promotion", self.l2_promotion, TensorMapL2Promotion)
        self.oob_fill = _require_tensor_map_enum("oob_fill", self.oob_fill, TensorMapOOBFill)


def _coerce_tensor_map_descriptor_options(
    box_dim,
    options,
    *,
    element_strides,
    data_type,
    interleave,
    swizzle,
    l2_promotion,
    oob_fill,
):
    if options is not None:
        if (
            box_dim is not None
            or element_strides is not None
            or data_type is not None
            or interleave != TensorMapInterleave.NONE
            or swizzle != TensorMapSwizzle.NONE
            or l2_promotion != TensorMapL2Promotion.NONE
            or oob_fill != TensorMapOOBFill.NONE
        ):
            raise TypeError(
                "Specify either options or the individual tensor map arguments, not both")
        return check_or_create_options(
            TensorMapDescriptorOptions,
            options,
            "Tensor map descriptor options",
        )

    if box_dim is None:
        raise TypeError("box_dim is required unless options is provided")

    return TensorMapDescriptorOptions(
        box_dim=box_dim,
        element_strides=element_strides,
        data_type=data_type,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
    )


# Mapping from numpy dtype to TMA data type
_NUMPY_DTYPE_TO_TMA = {
    numpy.dtype(numpy.uint8): _TMA_DT_UINT8,
    numpy.dtype(numpy.uint16): _TMA_DT_UINT16,
    numpy.dtype(numpy.uint32): _TMA_DT_UINT32,
    numpy.dtype(numpy.int32): _TMA_DT_INT32,
    numpy.dtype(numpy.uint64): _TMA_DT_UINT64,
    numpy.dtype(numpy.int64): _TMA_DT_INT64,
    numpy.dtype(numpy.float16): _TMA_DT_FLOAT16,
    numpy.dtype(numpy.float32): _TMA_DT_FLOAT32,
    numpy.dtype(numpy.float64): _TMA_DT_FLOAT64,
}

if ml_bfloat16 is not None:
    _NUMPY_DTYPE_TO_TMA[numpy.dtype(ml_bfloat16)] = _TMA_DT_BFLOAT16


# Mapping from TMA data type to element size in bytes
_TMA_DATA_TYPE_SIZE = {
    _TMA_DT_UINT8: 1,
    _TMA_DT_UINT16: 2,
    _TMA_DT_UINT32: 4,
    _TMA_DT_INT32: 4,
    _TMA_DT_UINT64: 8,
    _TMA_DT_INT64: 8,
    _TMA_DT_FLOAT16: 2,
    _TMA_DT_FLOAT32: 4,
    _TMA_DT_FLOAT64: 8,
    _TMA_DT_BFLOAT16: 2,
    _TMA_DT_FLOAT32_FTZ: 4,
    _TMA_DT_TFLOAT32: 4,
    _TMA_DT_TFLOAT32_FTZ: 4,
}

cdef _resolve_data_type(StridedMemoryView view, data_type):
    """Resolve the TMA data type from an explicit value or the view's dtype."""

    if data_type is not None:
        if isinstance(data_type, TensorMapDataType):
            return int(data_type)
        dt = _normalize_tensor_map_data_type(data_type)
        tma_dt = _NUMPY_DTYPE_TO_TMA.get(dt)
        if tma_dt is None:
            raise ValueError(
                f"Unsupported dtype {dt} for TMA; "
                f"supported dtypes: {list(_NUMPY_DTYPE_TO_TMA.keys())}.")
        return tma_dt

    dt = view.get_dtype()
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


cdef inline bint _tma_dtype_to_dlpack(int tma_dt, DLDataType* out) noexcept:
    """Write the DLPack spelling of a TMA data type, or return False if it has none."""
    out.lanes = 1
    if tma_dt == _TMA_DT_UINT8:
        out.code = kDLUInt
        out.bits = 8
    elif tma_dt == _TMA_DT_UINT16:
        out.code = kDLUInt
        out.bits = 16
    elif tma_dt == _TMA_DT_UINT32:
        out.code = kDLUInt
        out.bits = 32
    elif tma_dt == _TMA_DT_UINT64:
        out.code = kDLUInt
        out.bits = 64
    elif tma_dt == _TMA_DT_INT32:
        out.code = kDLInt
        out.bits = 32
    elif tma_dt == _TMA_DT_INT64:
        out.code = kDLInt
        out.bits = 64
    elif tma_dt == _TMA_DT_FLOAT16:
        out.code = kDLFloat
        out.bits = 16
    elif tma_dt == _TMA_DT_FLOAT32:
        out.code = kDLFloat
        out.bits = 32
    elif tma_dt == _TMA_DT_FLOAT64:
        out.code = kDLFloat
        out.bits = 64
    elif tma_dt == _TMA_DT_BFLOAT16:
        out.code = kDLBfloat
        out.bits = 16
    else:
        return False
    return True


cdef inline int _validate_tensor_map_view(StridedMemoryView view) except -1:
    if not view.is_device_accessible:
        raise ValueError("The tensor must be device-accessible")

    if view.ptr % 16 != 0:
        raise ValueError(
            f"Global memory address must be 16-byte aligned, "
            f"got address 0x{view.ptr:x}")
    return 0


cdef StridedMemoryView _get_validated_view(tensor):
    """Obtain a device-accessible StridedMemoryView with a 16-byte-aligned pointer."""
    cdef StridedMemoryView view
    if isinstance(tensor, StridedMemoryView):
        view = <StridedMemoryView>tensor
    else:
        # stream_ptr=-1: no stream synchronization needed because descriptor
        # creation only reads tensor metadata, it does not move data.
        view = StridedMemoryView.from_any_interface(tensor, stream_ptr=-1)
    _validate_tensor_map_view(view)
    return view


cdef int _require_view_device(
        StridedMemoryView view, int expected_device_id, operation) except -1:
    """Ensure device-local tensors match the current CUDA device.

    DLPack reports host/managed CUDA memory as ``kDLCUDAHost`` /
    ``kDLCUDAManaged`` with ``device_id=0`` regardless of the current device,
    so only true ``kDLCUDA`` tensors are rejected by device-id mismatch.
    """
    device_type, device_id = view.__dlpack_device__()
    if device_type == _kDLCUDA and device_id != expected_device_id:
        raise ValueError(
            f"{operation} expects tensor on device {expected_device_id}, got {device_id}")
    return 0


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

cdef void _fill_global_strides(
        uint64_t* out, const int64_t* shape, const int64_t* strides,
        int rank, int elem_size) noexcept nogil:
    """Write cuTensorMap's globalStrides: byte strides in column-major order.

    Only rank - 1 entries are written: the innermost stride is implicitly the
    element size and is not part of the array. ``strides`` is NULL for a
    C-contiguous tensor, matching the layout's own convention.
    """
    cdef int i
    cdef int64_t stride
    if strides != NULL:
        for i in range(rank - 1):
            out[i] = <uint64_t>(strides[rank - 2 - i] * elem_size)
        return

    # C-contiguous: each stride spans everything nested inside that dimension.
    stride = elem_size
    for i in range(rank - 1):
        stride *= shape[rank - 1 - i]
        out[i] = <uint64_t>stride


cdef int _fill_element_strides(int* out, element_strides, int rank) except -1:
    """Write the element traversal strides in the tensor's own (row-major) order.

    Rejecting non-positive strides here keeps the driver paths' narrowing to
    ``uint32_t`` lossless; CCCL and the driver enforce the real TMA limits.
    """
    cdef int i
    cdef int stride
    if element_strides is None:
        for i in range(rank):
            out[i] = 1
        return 0

    if len(element_strides) != rank:
        raise ValueError(
            f"element_strides must have {rank} elements, got {len(element_strides)}")
    for i in range(rank):
        # Cython raises OverflowError if the value does not fit in a C int.
        stride = element_strides[i]
        if stride < 1:
            raise ValueError(f"element_strides[{i}] must be positive, got {stride}")
        out[i] = stride
    return 0


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
        cdef int current_dev_id = <int>current_dev
        if self._device_id >= 0 and current_dev_id != self._device_id:
            raise RuntimeError(
                f"TensorMapDescriptor belongs to device {self._device_id}, "
                f"but current device is {current_dev_id}")
        return 0

    @property
    def device(self) -> Device | None:
        """Return the :obj:`~cuda.core.Device` associated with this descriptor."""
        if self._device_id >= 0:
            from cuda.core._device import Device
            return Device(self._device_id)
        return None

    @classmethod
    def _from_tiled(cls, StridedMemoryView view, box_dim=None, *,
                   options=None,
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
        box_dim : tuple of int, optional
            The size of each tile dimension (in elements). Must have the
            same rank as the tensor and each value must be in [1, 256].
            Specified in the same (row-major) order as the tensor shape.
            Required unless ``options`` is provided.
        options : TensorMapDescriptorOptions or mapping, optional
            Bundled tiled-descriptor options. When provided, do not also pass
            ``box_dim`` or the individual option kwargs.
        element_strides : tuple of int, optional
            Per-dimension element traversal strides. Default is all 1s.
            Specified in the same (row-major) order as the tensor shape.
        data_type : dtype-like or TensorMapDataType, optional
            Explicit dtype override. If ``None``, inferred from the tensor's
            dtype. Prefer NumPy or ``ml_dtypes`` dtype objects; the enum is
            accepted for compatibility.
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
        CUDAError
            If the CUDA driver rejects the encoded descriptor.
        """
        cdef TensorMapDescriptor desc = cls.__new__(cls)

        opts = _coerce_tensor_map_descriptor_options(
            box_dim,
            options,
            element_strides=element_strides,
            data_type=data_type,
            interleave=interleave,
            swizzle=swizzle,
            l2_promotion=l2_promotion,
            oob_fill=oob_fill,
        )
        box_dim = opts.box_dim
        element_strides = opts.element_strides
        data_type = opts.data_type
        interleave = opts.interleave
        swizzle = opts.swizzle
        l2_promotion = opts.l2_promotion
        oob_fill = opts.oob_fill

        # Convert options to driver enum:
        cdef cydriver.CUtensorMapInterleave c_interleave = interleave
        cdef cydriver.CUtensorMapSwizzle c_swizzle = swizzle
        cdef cydriver.CUtensorMapL2promotion c_l2_promotion = l2_promotion
        cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = oob_fill

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
        cdef int c_data_type_int = tma_dt
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        # The layout owns the shape/stride arrays in exactly the form DLPack and
        # the driver want them, so they are read in place instead of going
        # through the tuple-building Python properties. strides is NULL when the
        # tensor is C-contiguous.
        cdef _StridedLayout layout = view.get_layout()
        cdef const int64_t* view_shape = layout.base.shape
        cdef const int64_t* view_strides = layout.base.strides

        cdef int rank = layout.base.ndim
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

        cdef int c_elem_strides[5]
        _fill_element_strides(c_elem_strides, element_strides, rank)

        # Reuse CCCL/libcu++'s DLPack -> CUtensorMap conversion when possible.
        # This avoids maintaining a second, independent validation/encoding implementation.
        cdef DLTensor dl_tensor
        cdef int64_t c_strides[5]
        cdef int64_t c_stride
        cdef int c_box_sizes[5]
        cdef int i_dl
        cdef int dl_device_type
        if _tma_dtype_to_dlpack(tma_dt, &dl_tensor.dtype):
            for i_dl in range(rank):
                c_box_sizes[i_dl] = <int>box_dim[i_dl]

            if view_strides != NULL:
                dl_tensor.strides = <int64_t*>view_strides
            else:
                # DLPack 1.2 and later reject strides=NULL.
                c_stride = 1
                for i_dl in range(rank - 1, -1, -1):
                    c_strides[i_dl] = c_stride
                    c_stride *= view_shape[i_dl]
                dl_tensor.strides = &c_strides[0]

            dl_device_type = view.__dlpack_device__()[0]
            dl_tensor.data = <void*>global_address
            dl_tensor.device.device_type = <_DLDeviceType>dl_device_type
            # DLPack reports kDLCUDAManaged / kDLCUDAHost as device_id=0.
            # CCCL uses this id for compute-capability and shared-memory
            # queries, so pass the current CUDA device instead. kDLCUDA
            # tensors were already required to match this id.
            dl_tensor.device.device_id = desc._device_id
            dl_tensor.ndim = rank
            dl_tensor.shape = <int64_t*>view_shape
            dl_tensor.byte_offset = 0

            try:
                with nogil:
                    desc._tensor_map = make_tma_descriptor(
                        dl_tensor,
                        const_int_span(&c_box_sizes[0], <size_t>rank),
                        const_int_span(&c_elem_strides[0], <size_t>rank),
                        <tma_interleave_layout>c_interleave,
                        <tma_swizzle>c_swizzle,
                        <tma_l2_fetch_size>c_l2_promotion,
                        <tma_oob_fill>c_oob_fill,
                    )
            except RuntimeError as err:
                # RuntimeErrors here should in practice be CUDAErrors.
                # (either driver error or compute capability check).
                raise CUDAError(str(err)) from err
            desc._repr_info = {
                "method": "tiled",
                "rank": rank,
                "data_type": TensorMapDataType(tma_dt),
                "swizzle": swizzle,
            }
            return desc

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]

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
            c_global_dim[i_c] = <uint64_t>view_shape[rank - 1 - i_c]
            c_box_dim[i_c] = <uint32_t>box_dim[rank - 1 - i_c]
            c_element_strides[i_c] = <uint32_t>c_elem_strides[rank - 1 - i_c]

        _fill_global_strides(c_global_strides, view_shape, view_strides, rank, elem_size)

        cdef uint32_t c_rank = <uint32_t>rank

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
            "data_type": TensorMapDataType(tma_dt),
            "swizzle": swizzle,
        }

        return desc

    @classmethod
    def _from_im2col(cls, StridedMemoryView view, pixel_box_lower_corner, pixel_box_upper_corner,
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
        data_type : dtype-like or TensorMapDataType, optional
            Explicit dtype override. If ``None``, inferred from the tensor's
            dtype. Prefer NumPy or ``ml_dtypes`` dtype objects; the enum is
            accepted for compatibility.
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
        cdef int c_data_type_int = tma_dt
        cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

        cdef intptr_t global_address = view.ptr
        cdef _StridedLayout layout = view.get_layout()
        cdef const int64_t* view_shape = layout.base.shape
        cdef const int64_t* view_strides = layout.base.strides

        cdef int rank = layout.base.ndim
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

        cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
        cdef int c_elem_strides[5]
        _fill_element_strides(c_elem_strides, element_strides, rank)

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
            c_global_dim[i_c] = <uint64_t>view_shape[rank - 1 - i_c]
            c_element_strides[i_c] = <uint32_t>c_elem_strides[rank - 1 - i_c]

        _fill_global_strides(c_global_strides, view_shape, view_strides, rank, elem_size)

        # Reverse spatial dimensions for lower/upper corners
        for i_c in range(n_spatial):
            c_pixel_box_lower[i_c] = <int>pixel_box_lower_corner[n_spatial - 1 - i_c]
            c_pixel_box_upper[i_c] = <int>pixel_box_upper_corner[n_spatial - 1 - i_c]

        cdef uint32_t c_rank = <uint32_t>rank
        cdef uint32_t c_channels = <uint32_t>channels_per_pixel
        cdef uint32_t c_pixels = <uint32_t>pixels_per_column
        cdef cydriver.CUtensorMapInterleave c_interleave = <cydriver.CUtensorMapInterleave><int>interleave
        cdef cydriver.CUtensorMapSwizzle c_swizzle = <cydriver.CUtensorMapSwizzle><int>swizzle
        cdef cydriver.CUtensorMapL2promotion c_l2_promotion = <cydriver.CUtensorMapL2promotion><int>l2_promotion
        cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = <cydriver.CUtensorMapFloatOOBfill><int>oob_fill

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
            "data_type": TensorMapDataType(tma_dt),
            "swizzle": swizzle,
        }

        return desc

    @classmethod
    def _from_im2col_wide(cls, StridedMemoryView view,
                         pixel_box_lower_corner_width, pixel_box_upper_corner_width,
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
        data_type : dtype-like or TensorMapDataType, optional
            Explicit dtype override. If ``None``, inferred from the tensor's
            dtype. Prefer NumPy or ``ml_dtypes`` dtype objects; the enum is
            accepted for compatibility.
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
            cdef int c_data_type_int = tma_dt
            cdef cydriver.CUtensorMapDataType c_data_type = <cydriver.CUtensorMapDataType>c_data_type_int

            cdef intptr_t global_address = view.ptr
            cdef _StridedLayout layout = view.get_layout()
            cdef const int64_t* view_shape = layout.base.shape
            cdef const int64_t* view_strides = layout.base.strides

            cdef int rank = layout.base.ndim
            if rank < 3 or rank > 5:
                raise ValueError(
                    f"Im2col-wide tensor rank must be between 3 and 5, got {rank}")

            cdef int elem_size = _TMA_DATA_TYPE_SIZE[tma_dt]
            cdef int c_elem_strides[5]
            _fill_element_strides(c_elem_strides, element_strides, rank)

            # Reverse all dimension arrays for column-major convention
            cdef uint64_t[5] c_global_dim
            cdef uint64_t[4] c_global_strides
            cdef uint32_t[5] c_element_strides
            cdef int i_c

            for i_c in range(rank):
                c_global_dim[i_c] = <uint64_t>view_shape[rank - 1 - i_c]
                c_element_strides[i_c] = <uint32_t>c_elem_strides[rank - 1 - i_c]

            _fill_global_strides(c_global_strides, view_shape, view_strides, rank, elem_size)

            cdef uint32_t c_rank = <uint32_t>rank
            cdef int c_lower_w = <int>pixel_box_lower_corner_width
            cdef int c_upper_w = <int>pixel_box_upper_corner_width
            cdef uint32_t c_channels = <uint32_t>channels_per_pixel
            cdef uint32_t c_pixels = <uint32_t>pixels_per_column
            cdef cydriver.CUtensorMapInterleave c_interleave = <cydriver.CUtensorMapInterleave><int>interleave
            cdef cydriver.CUtensorMapIm2ColWideMode c_mode = <cydriver.CUtensorMapIm2ColWideMode><int>mode
            cdef cydriver.CUtensorMapSwizzle c_swizzle = <cydriver.CUtensorMapSwizzle><int>swizzle
            cdef cydriver.CUtensorMapL2promotion c_l2_promotion = <cydriver.CUtensorMapL2promotion><int>l2_promotion
            cdef cydriver.CUtensorMapFloatOOBfill c_oob_fill = <cydriver.CUtensorMapFloatOOBfill><int>oob_fill

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
                "data_type": TensorMapDataType(tma_dt),
                "swizzle": swizzle,
            }

            return desc

    def replace_address(self, tensor: object) -> None:
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
        cdef StridedMemoryView view = _get_validated_view(tensor)
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

    def __repr__(self) -> str:
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
