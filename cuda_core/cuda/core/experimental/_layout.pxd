# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython
from cython.operator cimport dereference as deref

from libc.stdint cimport int64_t, uint32_t, intptr_t
from libcpp cimport vector

ctypedef int64_t extent_t
ctypedef int64_t stride_t
ctypedef int axis_t

ctypedef uint32_t axes_mask_t  # MUST be exactly STRIDED_LAYOUT_MAX_NDIM bits wide
ctypedef uint32_t property_mask_t

ctypedef vector.vector[stride_t] extents_strides_t
ctypedef vector.vector[axis_t] axis_vec_t

from cuda.core.experimental._utils cimport cuda_utils


ctypedef fused integer_t:
    int64_t
    int


cdef extern from "include/layout.hpp":

    cdef int STRIDED_LAYOUT_MAX_NDIM
    cdef int AXIS_MASK_ALL
    int64_t _c_abs(int64_t x) nogil
    void _order_from_strides(axis_vec_t& indices, extent_t* extent_t, stride_t* stride_t, int ndim) except + nogil
    void _swap(extents_strides_t &a, extents_strides_t &b) noexcept nogil
    void _swap(int64_t* a, int64_t* b) noexcept nogil
    void _swap(int a, int b) noexcept nogil
    void _swap(axis_vec_t &a, axis_vec_t &b) noexcept nogil


cdef enum OrderFlag:
    ORDER_NONE = 0
    ORDER_C = 1
    ORDER_F = 2
    ORDER_PERM = 3


cdef enum Property:
    PROP_IS_UNIQUE = 1 << 0
    PROP_IS_CONTIGUOUS_C = 1 << 1
    PROP_IS_CONTIGUOUS_F = 1 << 2
    PROP_IS_DENSE = 1 << 3
    PROP_OFFSET_BOUNDS = 1 << 4
    PROP_REQUIRED_SIZE_IN_BYTES = 1 << 5
    PROP_SHAPE = 1 << 6
    PROP_STRIDES = 1 << 7
    PROP_STRIDES_IN_BYTES = 1 << 8
    PROP_STRIDE_ORDER = 1 << 9
    PROP_VOLUME = 1 << 10


cdef struct BaseLayout:
    # A struct holding the shape and strides for the layout.
    # Use ``init_base_layout`` to initialize the layout, it will
    # set the ``shape`` and ``strides`` pointers to point to
    # ndim contigious integer arrays.
    # The ``shape`` pointer must not be NULL, the ``strides`` can be
    # set to NULL by the user to indicate C-contiguous layout.
    # Uses single _mem allocation to reduce overhead
    # (allocation and exceptions checks).

    extents_strides_t _mem
    extent_t* shape
    stride_t* strides
    int ndim


@cython.final
cdef class StridedLayout:

    # Definition
    cdef:
        BaseLayout base

        readonly:
            int itemsize
            stride_t slice_offset

    # Lazy properties computed from the defining values.
    cdef:
        # Set to 0 to invalidate all properties,
        # whenever a defining value is changed
        property_mask_t _prop_mask

        # C and Python properties
        property_mask_t _boolean_props
        int64_t _required_size_in_bytes
        stride_t _min_offset
        stride_t _max_offset
        int64_t _volume

        # Python properties
        tuple _py_shape
        tuple _py_strides
        tuple _py_strides_in_bytes
        tuple _py_stride_order

    # ==============================
    # Initialization
    # ==============================

    cdef inline int _init(StridedLayout self, BaseLayout& base, int itemsize, bint strides_in_bytes=False) except -1 nogil:
        _validate_itemsize(itemsize)

        if base.strides != NULL and strides_in_bytes:
            _divide_strides(base, itemsize)

        self.itemsize = itemsize
        self.slice_offset = 0
        _swap_layout(self.base, base)
        return 0

    cdef inline stride_t _init_dense(StridedLayout self, BaseLayout& base, int itemsize, OrderFlag order_flag, axis_vec_t* stride_order=NULL) except -1 nogil:
        _validate_itemsize(itemsize)

        cdef stride_t volume
        if order_flag == ORDER_C:
            volume = _dense_strides_c(base)
        elif order_flag == ORDER_F:
            volume = _dense_strides_f(base)
        elif order_flag == ORDER_PERM:
            if stride_order == NULL:
                raise ValueError("stride_order is required for ORDER_PERM")
            volume = _dense_strides_in_order(base, deref(stride_order))
        else:
            raise ValueError("The stride_order must be 'C', 'F', or a permutation.")

        self.itemsize = itemsize
        self.slice_offset = 0
        _swap_layout(self.base, base)
        self._volume = volume
        _mark_property_valid(self, PROP_VOLUME)
        return 0

    cdef inline int init_from_ptr(StridedLayout self, int ndim, extent_t* shape, stride_t* strides, int itemsize, bint strides_in_bytes=False) except -1 nogil:
        cdef BaseLayout base
        _init_base_layout_from_ptr(base, ndim, shape, strides)
        return self._init(base, itemsize, strides_in_bytes)

    cdef inline int init_dense_from_ptr(StridedLayout self, int ndim, extent_t* shape, int itemsize, OrderFlag order_flag, axis_vec_t* stride_order=NULL) except -1 nogil:
        cdef BaseLayout base
        _init_base_layout_from_ptr(base, ndim, shape, NULL)
        return self._init_dense(base, itemsize, order_flag, stride_order)

    cdef inline int init_from_tuple(StridedLayout self, tuple shape, tuple strides, int itemsize, bint strides_in_bytes=False) except -1:
        cdef BaseLayout base
        _init_base_layout_from_tuple(base, shape, strides)
        return self._init(base, itemsize, strides_in_bytes)

    cdef inline int init_dense_from_tuple(StridedLayout self, tuple shape, int itemsize, object stride_order) except -1:
        cdef axis_vec_t stride_order_vec
        cdef OrderFlag order_flag = _stride_order2vec(stride_order_vec, stride_order)

        if order_flag == ORDER_NONE:
            raise ValueError(f"The stride_order must be 'C', 'F', or a permutation tuple. Got: {stride_order}")

        cdef BaseLayout base
        _init_base_layout_from_tuple(base, shape, None)
        return self._init_dense(base, itemsize, order_flag, &stride_order_vec)

    # ==============================
    # Properties
    # ==============================

    cdef inline tuple get_shape_tuple(StridedLayout self):
        if not _has_valid_property(self, PROP_SHAPE):
            self._py_shape = cuda_utils.carray_integer_t_to_tuple(self.base.shape, self.base.ndim)
            _mark_property_valid(self, PROP_SHAPE)
        return self._py_shape

    cdef inline tuple get_strides_tuple(StridedLayout self):
        if not _has_valid_property(self, PROP_STRIDES):
            if self.base.strides == NULL:
                self._py_strides = None
            else:
                self._py_strides = cuda_utils.carray_integer_t_to_tuple(self.base.strides, self.base.ndim)
            _mark_property_valid(self, PROP_STRIDES)
        return self._py_strides

    cdef inline int get_strides_in_bytes(StridedLayout self, extents_strides_t& strides) except -1 nogil:
        if self.base.strides != NULL:
            strides.resize(self.base.ndim)
            for i in range(self.base.ndim):
                strides[i] = _overflow_checked_mul(self.base.strides[i], self.itemsize)
        return 0

    cdef inline tuple get_strides_in_bytes_tuple(StridedLayout self):
        if _has_valid_property(self, PROP_STRIDES_IN_BYTES):
            return self._py_strides_in_bytes
        cdef extents_strides_t strides
        if self.base.strides == NULL:
            self._py_strides_in_bytes = None
        else:
            self.get_strides_in_bytes(strides)
            self._py_strides_in_bytes = cuda_utils.carray_integer_t_to_tuple(strides.data(), strides.size())
        _mark_property_valid(self, PROP_STRIDES_IN_BYTES)
        return self._py_strides_in_bytes

    cdef inline int64_t get_volume(StridedLayout self) except -1 nogil:
        if not _has_valid_property(self, PROP_VOLUME):
            self._volume = _volume(self.base)
            _mark_property_valid(self, PROP_VOLUME)
        return self._volume

    cdef inline int get_stride_order(StridedLayout self, axis_vec_t& stride_order) except -1 nogil:
        _order_from_strides(stride_order, self.base.shape, self.base.strides, self.base.ndim)
        return 0

    cdef inline tuple get_stride_order_tuple(StridedLayout self):
        if _has_valid_property(self, PROP_STRIDE_ORDER):
            return self._py_stride_order
        cdef axis_vec_t stride_order
        self.get_stride_order(stride_order)
        self._py_stride_order = cuda_utils.carray_integer_t_to_tuple(stride_order.data(), stride_order.size())
        _mark_property_valid(self, PROP_STRIDE_ORDER)
        return self._py_stride_order

    cdef inline bint get_is_unique(StridedLayout self) except -1 nogil:
        if _has_valid_property(self, PROP_IS_UNIQUE):
            return _boolean_property(self, PROP_IS_UNIQUE)
        if self.base.strides == NULL or self.get_volume() == 0:
            return _set_boolean_property(self, PROP_IS_UNIQUE, True)
        cdef axis_vec_t stride_order
        self.get_stride_order(stride_order)
        return _set_boolean_property(self, PROP_IS_UNIQUE, _is_unique(self.base, stride_order))

    cdef inline bint get_is_contiguous_c(StridedLayout self) except -1 nogil:
        if _has_valid_property(self, PROP_IS_CONTIGUOUS_C):
            return _boolean_property(self, PROP_IS_CONTIGUOUS_C)
        cdef bint is_contiguous_c = (
            self.slice_offset == 0 and _is_contiguous_c(self.get_volume(), self.base)
        )
        return _set_boolean_property(self, PROP_IS_CONTIGUOUS_C, is_contiguous_c)

    cdef inline bint get_is_contiguous_f(StridedLayout self) except -1 nogil:
        if _has_valid_property(self, PROP_IS_CONTIGUOUS_F):
            return _boolean_property(self, PROP_IS_CONTIGUOUS_F)
        cdef bint is_contiguous_f = (
            self.slice_offset == 0 and _is_contiguous_f(self.get_volume(), self.base)
        )
        return _set_boolean_property(self, PROP_IS_CONTIGUOUS_F, is_contiguous_f)

    cdef inline bint get_is_dense(StridedLayout self) except -1 nogil:
        if _has_valid_property(self, PROP_IS_DENSE):
            return _boolean_property(self, PROP_IS_DENSE)
        cdef axis_vec_t stride_order
        self.get_stride_order(stride_order)
        cdef bint is_dense = (
            self.slice_offset == 0 and _is_dense(self.get_volume(), self.base, stride_order)
        )
        return _set_boolean_property(self, PROP_IS_DENSE, is_dense)

    cdef inline int get_offset_bounds(StridedLayout self, stride_t& min_offset, stride_t& max_offset) except -1 nogil:
        if _has_valid_property(self, PROP_OFFSET_BOUNDS):
            min_offset = self._min_offset
            max_offset = self._max_offset
            return 0
        cdef int ndim = self.base.ndim
        cdef stride_t stride
        cdef extent_t extent
        min_offset = self.slice_offset
        max_offset = self.slice_offset
        if self.base.strides == NULL:
            max_offset = _overflow_checked_sum(max_offset, self.get_volume() - 1)
        else:
            for i in range(ndim):
                stride = self.base.strides[i]  # can be negative
                extent = self.base.shape[i]  # must be non-negative
                if extent == 0:
                    min_offset = 0
                    max_offset = -1  # empty range
                    return 0
                if stride <= 0:
                    min_offset = _overflow_checked_sum(min_offset, _overflow_checked_mul(stride, (extent - 1)))
                else:
                    max_offset = _overflow_checked_sum(max_offset, _overflow_checked_mul(stride, (extent - 1)))
        self._min_offset = min_offset
        self._max_offset = max_offset
        _mark_property_valid(self, PROP_OFFSET_BOUNDS)
        return 0

    cdef inline int64_t get_required_size_in_bytes(StridedLayout self) except? -1 nogil:
        if _has_valid_property(self, PROP_REQUIRED_SIZE_IN_BYTES):
            return self._required_size_in_bytes
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        if min_offset < 0:
            raise ValueError(
                f"Allocation size for a layout that maps elements "
                f"to negative memory offsets is ambiguous. "
                f"The layout's min_offset is {min_offset}. "
                f"To create a supported layout with the same shape "
                f"please use StridedLayout.to_dense()."
            )
        if max_offset < min_offset:
            return 0
        cdef int64_t required_size_in_bytes = _overflow_checked_sum(max_offset, 1)
        self._required_size_in_bytes = _overflow_checked_mul(required_size_in_bytes, self.itemsize)
        _mark_property_valid(self, PROP_REQUIRED_SIZE_IN_BYTES)
        return self._required_size_in_bytes

    cdef inline int64_t get_slice_offset_in_bytes(StridedLayout self) except? -1 nogil:
        return _overflow_checked_mul(self.slice_offset, self.itemsize)

    cdef axes_mask_t get_flattened_axis_mask(StridedLayout self) except? -1 nogil
    cdef int get_max_compatible_itemsize(StridedLayout self, int max_itemsize, intptr_t data_ptr, int axis=*) except -1 nogil

    # ==============================
    # Layout manipulation
    # ==============================


    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, BaseLayout& new_shape) except -1 nogil
    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_vec_t& axis_order) except -1 nogil

    cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=*) except -1 nogil
    cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil
    cdef int unsqueeze_into(StridedLayout self, StridedLayout out_layout, axis_vec_t& axis_vec) except -1 nogil
    cdef int broadcast_into(StridedLayout self, StridedLayout out_layout, BaseLayout& broadcast) except -1 nogil
    cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=*) except -1 nogil
    cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=*) except -1 nogil
    cdef int slice_into(StridedLayout self, StridedLayout out_layout, tuple slices) except -1

# ==============================
# Base layout helpers
# ==============================


cdef inline int init_base_layout(BaseLayout& layout, int ndim) except -1 nogil:
    if ndim > STRIDED_LAYOUT_MAX_NDIM:
        raise ValueError(f"Unsupported number of dimensions: {ndim}. Max supported ndim is {STRIDED_LAYOUT_MAX_NDIM}")
    # resize(0) is no op, that results in _mem.data() being NULL,
    # which would make it tricky to distinguish between strides == NULL
    # and strides == tuple()
    layout._mem.resize(2 * max(ndim, 1))
    layout.shape = layout._mem.data()
    layout.strides = layout._mem.data() + ndim
    layout.ndim = ndim
    return 0


cdef inline int trim_base_layout(BaseLayout& layout, int ndim) except -1 nogil:
    if ndim > layout.ndim:
        raise AssertionError(f"Cannot trim layout to {ndim} dimensions, it has {layout.ndim} dimensions")
    layout.ndim = ndim
    return 0


cdef inline void _swap_layout(BaseLayout& a, BaseLayout& b) noexcept nogil:
    _swap(a._mem, b._mem)
    _swap(a.shape, b.shape)
    _swap(a.strides, b.strides)
    _swap(a.ndim, b.ndim)


cdef inline void _assure_strides_ptr(BaseLayout& base) noexcept nogil:
    if base.strides == NULL:
        base.strides = base._mem.data() + base._mem.size() // 2


cdef inline stride_t *get_strides_ptr(BaseLayout& base) except? NULL nogil:
    if base.strides != NULL:
        return base.strides
    cdef stride_t* tmp_strides = base._mem.data() + base._mem.size() // 2
    _dense_strides_c_ptrs(base.ndim, base.shape, tmp_strides)
    return tmp_strides


cdef inline bint _base_layout_equal(BaseLayout& a, BaseLayout& b) noexcept nogil:
    if a.ndim != b.ndim:
        return False
    for i in range(a.ndim):
        if a.shape[i] != b.shape[i]:
            return False
    if a.strides != NULL or b.strides != NULL:
        if a.strides == NULL or b.strides == NULL:
            return False
        for i in range(a.ndim):
            if a.strides[i] != b.strides[i]:
                return False
    return True


@cython.overflowcheck(True)
cdef inline int64_t _volume(BaseLayout& base) except? -1 nogil:
    cdef int64_t vol = 1
    for i in range(base.ndim):
        vol *= base.shape[i]
    return vol


cdef inline int _divide_strides(BaseLayout& base, int itemsize) except -1 nogil:
    cdef stride_t stride
    if base.strides == NULL:
        raise ValueError("cannot divide strides, layout has no strides")
    for i in range(base.ndim):
        stride = base.strides[i] // itemsize
        if stride * itemsize != base.strides[i]:
            raise ValueError("strides must be divisible by itemsize")
        base.strides[i] = stride
    return 0


cdef inline void _zero_strides_ptr(int ndim, stride_t* strides) noexcept nogil:
    for i in range(ndim):
        strides[i] = 0


cdef inline void _zero_strides(BaseLayout& base) noexcept nogil:
    _assure_strides_ptr(base)
    _zero_strides_ptr(base.ndim, base.strides)


cdef inline stride_t _dense_strides_c_ptrs(int ndim, extent_t* shape, stride_t* strides) except? -1 nogil:
    cdef stride_t stride = 1
    cdef int i = ndim - 1
    while i >= 0:
        strides[i] = stride
        stride = _overflow_checked_mul(stride, shape[i])
        i -= 1
    if stride == 0:
        _zero_strides_ptr(ndim, strides)
    return stride


cdef inline stride_t _dense_strides_c(BaseLayout& base) except? -1 nogil:
    cdef int ndim = base.ndim
    _assure_strides_ptr(base)
    return _dense_strides_c_ptrs(ndim, base.shape, base.strides)


cdef inline stride_t _dense_strides_f(BaseLayout& base) except? -1 nogil:
    cdef int ndim = base.ndim
    _assure_strides_ptr(base)
    cdef stride_t stride = 1
    cdef int i = 0
    while i < ndim:
        base.strides[i] = stride
        stride = _overflow_checked_mul(stride, base.shape[i])
        i += 1
    if stride == 0:
        _zero_strides(base)
    return stride


cdef inline stride_t _dense_strides_in_order(BaseLayout& base, axis_vec_t& stride_order) except? -1 nogil:
    cdef int ndim = base.ndim
    if <size_t>ndim != stride_order.size():
        raise ValueError(f"stride_order must have the same length as shape. Shape has {ndim} dimensions, but stride_order has {stride_order.size()} elements.")
    _assure_strides_ptr(base)
    cdef stride_t stride = 1
    cdef int i = ndim - 1
    cdef axes_mask_t axis_order_mask = 0
    cdef axes_mask_t axis_mask
    cdef axis_t axis
    while i >= 0:
        axis = stride_order[i]
        if not _normalize_axis(axis, ndim):
            raise ValueError(f"Invalid stride order: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"The stride order must be a permutation. Axis {axis} appears multiple times.")
        axis_order_mask |= axis_mask
        base.strides[axis] = stride
        stride = _overflow_checked_mul(stride, base.shape[axis])
        i -= 1
    if stride == 0:
        _zero_strides(base)
    return stride


cdef inline bint _is_contiguous_c(int64_t volume, BaseLayout& base) except -1 nogil:
    if volume == 0 or base.strides == NULL:
        return True
    cdef int64_t stride = 1
    cdef int64_t j = base.ndim - 1
    cdef extent_t extent
    while j >= 0:
        extent = base.shape[j]
        if extent != 1:
            if base.strides[j] != stride:
                return False
            stride *= extent
        j -= 1
    return True


cdef inline bint _is_contiguous_f(int64_t volume, BaseLayout& base) except -1 nogil:
    if volume == 0:
        return True
    cdef int ndim = base.ndim
    cdef int64_t j = 0
    if base.strides == NULL:
        # find first non-singleton dimension
        while j < ndim and base.shape[j] == 1:
            j += 1
        # if any subsequent dimension is not a singleton, return False
        for i in range(j + 1, ndim):
            if base.shape[i] != 1:
                return False
        return True
    cdef int64_t stride = 1
    cdef extent_t extent
    while j < ndim:
        extent = base.shape[j]
        if extent != 1:
            if base.strides[j] != stride:
                return False
            stride *= extent
        j += 1
    return True


cdef inline bint _is_dense(int64_t volume, BaseLayout& base, axis_vec_t& axis_order) except -1 nogil:
    if volume == 0 or base.strides == NULL:
        return True
    cdef int64_t stride = 1
    cdef int64_t j = base.ndim - 1
    cdef axis_t axis
    cdef extent_t extent
    while j >= 0:
        axis = axis_order[j]
        extent = base.shape[axis]
        if extent != 1:
            if base.strides[axis] != stride:
                return False
            stride *= extent
        j -= 1
    return True


cdef inline int _validate_shape(BaseLayout& base) except -1 nogil:
    for i in range(base.ndim):
        if base.shape[i] < 0:
            raise ValueError("Extents must be non-negative")
    return 0


cdef inline int _init_base_layout_from_tuple(BaseLayout& base, tuple shape, tuple strides) except -1:
    cdef int ndim = len(shape)
    init_base_layout(base, ndim)
    for i in range(ndim):
        base.shape[i] = shape[i]
    _validate_shape(base)

    if strides is None:
        base.strides = NULL
    else:
        if len(strides) != ndim:
            raise ValueError(f"Strides, if provided, must have the same length as shape. Shape has {ndim} dimensions, but strides has {len(strides)} elements.")
        for i in range(ndim):
            base.strides[i] = strides[i]
    return 0


cdef inline int _init_base_layout_from_ptr(BaseLayout& base, int ndim, extent_t* shape, stride_t* strides) except -1 nogil:
    init_base_layout(base, ndim)
    for i in range(ndim):
        base.shape[i] = shape[i]
    _validate_shape(base)

    if strides == NULL:
        base.strides = NULL
    else:
        for i in range(ndim):
            base.strides[i] = strides[i]
    return 0

# ==============================
# Strided layout helpers
# ==============================


cdef inline bint _has_valid_property(StridedLayout self, Property prop) noexcept nogil:
    return self._prop_mask & prop


cdef inline void _mark_property_valid(StridedLayout self, Property prop) noexcept nogil:
    self._prop_mask |= prop


cdef inline bint _boolean_property(StridedLayout self, Property prop) noexcept nogil:
    return self._boolean_props & prop


cdef inline bint _set_boolean_property(StridedLayout self, Property prop, bint value) noexcept nogil:
    if value:
        self._boolean_props |= prop
    else:
        self._boolean_props &= ~prop
    _mark_property_valid(self, prop)
    return value


# ==============================
# Conversion, validation and normalization helpers
# ==============================

cdef inline OrderFlag _stride_order2vec(axis_vec_t& stride_order_vec, object stride_order) except? ORDER_NONE:
    if stride_order == 'C':
        return ORDER_C
    elif stride_order == 'F':
        return ORDER_F
    elif isinstance(stride_order, tuple | list):
        _tuple2axis_vec(stride_order_vec, stride_order)
        return ORDER_PERM
    return ORDER_NONE


cdef inline int _tuple2axis_vec(axis_vec_t& vec, object t) except -1:
    cdef int ndim = len(t)
    vec.resize(ndim)
    for i in range(ndim):
        vec[i] = t[i]
    return 0


cdef inline bint _normalize_axis(integer_t& axis, integer_t extent) except -1 nogil:
    if axis < -extent or axis >= extent:
        return False
    if axis < 0:
        axis += extent
    return True


cdef inline int _validate_itemsize(int itemsize) except -1 nogil:
    if itemsize <= 0:
        raise ValueError("itemsize must be positive")
    if itemsize & (itemsize - 1):
        raise ValueError("itemsize must be a power of two")
    return 0


cdef inline bint _is_unique(BaseLayout& base, axis_vec_t& stride_order) except -1 nogil:
    if base.strides == NULL:
        return True
    cdef int64_t cur_max_offset = 0
    cdef int i = base.ndim - 1
    cdef int64_t stride
    cdef axis_t axis
    cdef extent_t extent
    while i >= 0:
        axis = stride_order[i]
        extent = base.shape[axis]
        if extent != 1:
            stride = _c_abs(base.strides[axis])
            if cur_max_offset >= stride:
                return False
            cur_max_offset = _overflow_checked_sum(cur_max_offset, _overflow_checked_mul(stride, (extent - 1)))
        i -= 1
    return True


@cython.overflowcheck(True)
cdef inline int64_t _overflow_checked_mul(int64_t a, int64_t b) except? -1 nogil:
    return a * b


@cython.overflowcheck(True)
cdef inline int64_t _overflow_checked_diff(int64_t a, int64_t b) except? -1 nogil:
    return a - b


@cython.overflowcheck(True)
cdef inline int64_t _overflow_checked_sum(int64_t a, int64_t b) except? -1 nogil:
    return a + b


@cython.overflowcheck(True)
cdef inline int64_t _overflow_checked_div_ceil(int64_t a, int64_t b) except? -1 nogil:
    return (a + b - 1) // b
