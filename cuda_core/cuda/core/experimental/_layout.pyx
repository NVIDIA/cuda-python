# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport int64_t, intptr_t
from libcpp cimport vector

from cpython.object cimport PyObject


cdef extern from "Python.h":
    int _PySlice_Unpack "PySlice_Unpack" (PyObject *slice, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step) except -1
    Py_ssize_t _PySlice_AdjustIndices "PySlice_AdjustIndices" (Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t step) noexcept nogil


@cython.final
cdef class StridedLayout:

    def __init__(StridedLayout self, object shape, object strides, int itemsize, bint strides_in_bytes=False):
        self.init_from_tuple(shape, strides, itemsize, strides_in_bytes)

    @classmethod
    def dense(cls, object shape, int itemsize, object stride_order='C'):
        cdef StridedLayout new_layout = StridedLayout.__new__(cls)
        new_layout.init_dense_from_tuple(shape, itemsize, stride_order)
        return new_layout

    @classmethod
    def dense_like(cls, StridedLayout other, object stride_order="K"):
        cdef OrderFlag order_flag
        cdef axis_vec_t stride_order_vec

        if stride_order == "K":
            if other.get_is_dense():
                return other
            other.get_stride_order(stride_order_vec)
            order_flag = ORDER_PERM
        else:
            order_flag = _stride_order2vec(stride_order_vec, stride_order)
            if order_flag == ORDER_NONE:
                raise ValueError(
                    f"The stride_order must be 'K', 'C', 'F', "
                    f"or a permutation tuple. Got: {stride_order}"
                )
            elif order_flag == ORDER_C:
                if other.get_is_contiguous_c():
                    return other
            elif order_flag == ORDER_F:
                if other.get_is_contiguous_f():
                    return other

        cdef StridedLayout new_layout = StridedLayout.__new__(cls)
        new_layout.init_dense_from_ptr(
            other.base.ndim,
            other.base.shape,
            other.itemsize,
            order_flag,
            &stride_order_vec
        )
        return new_layout

    def __repr__(StridedLayout self):
        if self.slice_offset == 0:
            return (
                f"StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize})"
            )
        else:
            return (
                f"StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize}, _slice_offset={self.slice_offset})"
            )

    def __eq__(StridedLayout self, StridedLayout other):
        return self.itemsize == other.itemsize and self.slice_offset == other.slice_offset and _base_layout_equal(self.base, other.base)

    @property
    def ndim(StridedLayout self) -> int:
        return self.base.ndim

    @property
    def shape(StridedLayout self) -> tuple:
        return self.get_shape_tuple()

    @property
    def strides(StridedLayout self) -> tuple | None:
        return self.get_strides_tuple()

    @property
    def strides_in_bytes(StridedLayout self) -> tuple | None:
        return self.get_strides_in_bytes_tuple()

    @property
    def stride_order(StridedLayout self) -> tuple:
        return self.get_stride_order_tuple()

    @property
    def volume(StridedLayout self) -> int:
        return self.get_volume()

    @property
    def is_unique(StridedLayout self) -> bool:
        return self.get_is_unique()

    @property
    def is_contiguous_c(StridedLayout self):
        """
        True iff the layout is contiguous in C-order, i.e.
        the rightmost stride is 1 and each subsequent
        stride to the left is the product of the
        next extent and the stride.
        In C-contigious layout, the strides are non-negative,
        increase from the right to the left and the mapping
        from indices to memory offsets is 1 to 1.
        """
        return self.get_is_contiguous_c()

    @property
    def is_contiguous_f(StridedLayout self):
        """
        True iff the layout is contiguous in F-order, i.e.
        the leftmost stride is 1 and each subsequent
        stride to the right is the product of the
        next stride and extent.
        In F-contigious layout, the strides are non-negative,
        increase from the left to the right and the mapping
        from indices to memory offsets is 1 to 1.
        """
        return self.get_is_contiguous_f()

    @property
    def is_dense(StridedLayout self):
        """
        True iff the layout is contiguous in some axis order, i.e.
        there exists a permutation of axes such that the layout
        is C-contiguous.
        In dense layout, the strides are non-negative and the mapping
        from indices to memory offsets is 1 to 1.
        """
        return self.get_is_dense()

    @property
    def offset_bounds(StridedLayout self):
        """
        A tuple of ``(min_offset, max_offset)`` representing the
        minimum and maximum offsets (as a number of elements, not bytes)
        that the layout can map to.
        I.e. there exist two ndim-tuples ``idx_min`` and ``idx_max``,
        where ``0 <= idx[i] < shape[i]`` for ``0 <= i < ndim``,
        such that:
        ``min_offset = sum(idx_min[i] * strides[i] for i in range(ndim))``
        ``max_offset = sum(idx_max[i] * strides[i] for i in range(ndim))``,
        and all other valid ndim-indices are mapped to offsets
        in the range ``[min_offset, max_offset]``.
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return min_offset, max_offset

    @property
    def min_offset(StridedLayout self):
        """
        See ``offset_bounds`` for details.
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return min_offset

    @property
    def max_offset(StridedLayout self):
        """
        See ``offset_bounds`` for details.
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return max_offset

    @property
    def slice_offset_in_bytes(StridedLayout self):
        """
        The memory offset (as a number of bytes)
        of the element at index ``(0,) * ndim``.
        The only way for the index 0 to be mapped to
        non-zero offset in memory is if the layout
        was sliced.
        """
        return self.get_slice_offset_in_bytes()

    def required_size_in_bytes(StridedLayout self):
        """
        The memory allocation size in bytes needed for all
        elements of the ndim-tensor to be mapped to
        offsets within the allocated memory range.
        I.e. for any ndim-tuple ``idx``, such that
        ``0 <= idx[i] < shape[i]`` for ``0 <= i < ndim``,
        the ``sum(idx[i] * strides[i] for i in range(ndim))``
        is in the range ``[0, required_size_in_bytes - 1]``.
        The function raises an error if the layout maps any element
        to a negative memory offset (i.e. layout.offset_bounds[0] < 0).
        """
        return self.get_required_size_in_bytes()

    def flattened_axis_mask(StridedLayout self):
        """
        A mask describing which axes can be merged
        together preserving the index to memory offset mapping
        (see more details in ``flattened`` method documentation).
        The only supported operation is the logical ``&``
        between masks coming from the layouts with equal ndim.
        If such a mask is passed to the
        ``flattened`` method, only the axes that are mergable
        for all the layouts will be flattened.
        """
        return self.get_flattened_axis_mask()

    def to_dense(StridedLayout self, object stride_order="K"):
        return StridedLayout.dense_like(self, stride_order)

    def reshaped(StridedLayout self, object shape):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef BaseLayout new_shape
        init_base_layout(new_shape, len(shape))
        for i in range(len(shape)):
            new_shape.shape[i] = shape[i]
        self.reshape_into(new_layout, new_shape)
        return new_layout

    def permuted(StridedLayout self, object axis_order):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef axis_vec_t axis_order_vec
        _tuple2axis_vec(axis_order_vec, axis_order)
        self.permute_into(new_layout, axis_order_vec)
        return new_layout

    def flattened(StridedLayout self, start_axis=0, end_axis=-1, mask=None):
        """
        Merges consecutive axes into a single axis (where the new extent
        is the product of merged extents) if the mapping of indices to
        memory offsets is preserved (assuming the indices are iterated
        in C-order, i.e. the rightmost axis is incremented first).
        E.g. for ``StridedLayout((2, 2), (4, 2), 1)``
        and the C-ordered indices ``[(0, 0), (0, 1), (1, 0), (1, 1)]`` would
        be mapped to offsets ``[0, 2, 4, 6]``, same as for the
        flattened layout ``StridedLayout((4,), (2,), 1)``
        and the indices ``[0, 1, 2, 3]``.
        If ``start_axis`` and ``end_axis`` are provided, only the axes in the
        inclusive range ``[start_axis, end_axis]`` are considered for flattening.
        Alternatively, a mask specifying which axes to consider can be provided
        (see ``flattened_axis_mask`` method documentation for details).
        """
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef axes_mask_t axis_mask
        if mask is None:
            axis_mask = axis_mask_from_range(self.ndim, start_axis, end_axis)
        else:
            axis_mask = mask
        self.flatten_into(new_layout, axis_mask)
        return new_layout

    def flattened_axis_mask(StridedLayout self):
        return self.get_flattened_axis_mask()

    def squeezed(StridedLayout self):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.squeeze_into(new_layout)
        return new_layout

    def unsqueezed(StridedLayout self, object axis):
        cdef axis_vec_t axis_vec
        if isinstance(axis, int):
            axis_vec.push_back(axis)
        else:
            _tuple2axis_vec(axis_vec, axis)
        if axis_vec.size() == 0:
            return self
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.unsqueeze_into(new_layout, axis_vec)
        return new_layout

    def broadcast_to(StridedLayout self, object shape):
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        cdef BaseLayout new_shape
        cdef int new_ndim = len(shape)
        init_base_layout(new_shape, new_ndim)
        for i in range(new_ndim):
            new_shape.shape[i] = shape[i]
        self.broadcast_into(new_layout, new_shape)
        return new_layout

    def packed(StridedLayout self, int itemsize, intptr_t data_ptr=0, int axis=-1, bint keep_dim=True):
        if itemsize == self.itemsize:
            return self
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.pack_into(new_layout, itemsize, data_ptr, keep_dim, axis)
        return new_layout

    def unpacked(StridedLayout self, int itemsize, int axis=-1):
        if itemsize == self.itemsize:
            return self
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.unpack_into(new_layout, itemsize, axis)
        return new_layout

    def max_compatible_itemsize(StridedLayout self, int max_itemsize=16, intptr_t data_ptr=0, int axis=-1):
        return self.get_max_compatible_itemsize(max_itemsize, data_ptr, axis)

    def sliced(StridedLayout self, object slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        cdef StridedLayout new_layout = StridedLayout.__new__(StridedLayout)
        self.slice_into(new_layout, slices)
        return new_layout

    def __getitem__(StridedLayout self, object slices):
        return self.sliced(slices)

    cdef axes_mask_t get_flattened_axis_mask(StridedLayout self) except? -1 nogil:
        return flattened_strides_in_c_index_order_mask(self.base)

    cdef int get_max_compatible_itemsize(StridedLayout self, int max_itemsize, intptr_t data_ptr, int axis=-1) except -1 nogil:
        return max_compatible_itemsize(self.base, self.slice_offset, self.itemsize, max_itemsize, data_ptr, axis)

    cdef int reshape_into(StridedLayout self, StridedLayout out_layout, BaseLayout& new_shape) except -1 nogil:
        cdef int64_t old_volume = self.get_volume()
        validate_reshaped_shape(new_shape, old_volume)

        cdef int ndim = new_shape.ndim
        _zero_strides(new_shape)

        cdef BaseLayout flattened
        if old_volume != 0:
            flatten_strides_in_c_index_order(flattened, self.base, AXIS_MASK_ALL)
            if not split_strides_in_c_index_order(new_shape, flattened):
                raise ValueError("Layout strides are incompatible with the new shape")

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Copy preserved attributes
        out_layout.slice_offset = self.slice_offset
        out_layout.itemsize = self.itemsize
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, new_shape)
        return 0

    cdef int permute_into(StridedLayout self, StridedLayout out_layout, axis_vec_t& axis_order) except -1 nogil:
        if axis_order.size() != <size_t>self.base.ndim:
            raise ValueError(f"Permutation must have the same length as the number of dimensions, got {axis_order.size()} for {self.ndim}D tensor.")

        cdef BaseLayout permuted
        permute_extents(permuted, self.base, axis_order)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, permuted)
        return 0

    cdef int flatten_into(StridedLayout self, StridedLayout out_layout, axes_mask_t axis_mask=AXIS_MASK_ALL) except -1 nogil:
        cdef BaseLayout flattened
        cdef int ndim = flatten_strides_in_c_index_order(flattened, self.base, axis_mask)

        if out_layout is self and ndim == self.base.ndim:
            return 0

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, flattened)
        return 0

    cdef int squeeze_into(StridedLayout self, StridedLayout out_layout) except -1 nogil:
        cdef BaseLayout squeezed
        squeeze_extents(squeezed, self.base)

        if out_layout is self and squeezed.ndim == self.base.ndim:
            return 0

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, squeezed)
        return 0

    cdef int unsqueeze_into(StridedLayout self, StridedLayout out_layout, axis_vec_t& axis_vec) except -1 nogil:
        if axis_vec.size() == 0 and self is out_layout:
            return 0

        cdef BaseLayout unsqueezed
        unsqueeze_extents(unsqueezed, self.base, axis_vec)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset
        maybe_copy_volume(out_layout, self)

        # Set new attributes
        _swap_layout(out_layout.base, unsqueezed)
        return 0

    cdef int broadcast_into(StridedLayout self, StridedLayout out_layout, BaseLayout& broadcast) except -1 nogil:
        _validate_shape(broadcast)
        broadcast_extents(broadcast, self.base)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize
        out_layout.slice_offset = self.slice_offset

        # Set new attributes
        _swap_layout(out_layout.base, broadcast)
        return 0

    cdef int pack_into(StridedLayout self, StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=-1) except -1 nogil:

        cdef BaseLayout packed
        cdef stride_t new_slice_offset = 0
        cdef int vec_size = pack_extents(
            packed,
            new_slice_offset,
            self.base,
            self.slice_offset,
            self.itemsize,
            itemsize,
            data_ptr,
            keep_dim,
            axis
        )

        if vec_size == 1 and out_layout is self:
            return 0

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Set new attributes
        out_layout.itemsize = itemsize
        out_layout.slice_offset = new_slice_offset
        _swap_layout(out_layout.base, packed)
        return vec_size

    cdef int unpack_into(StridedLayout self, StridedLayout out_layout, int itemsize, int axis=-1) except -1 nogil:
        cdef BaseLayout unpacked
        cdef int vec_size = unpack_extents(
            unpacked,
            self.base,
            self.itemsize,
            itemsize,
            axis
        )
        if vec_size == 1 and out_layout is self:
            return 0

        cdef int64_t new_slice_offset = _overflow_checked_mul(self.slice_offset, vec_size)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Set new attributes
        out_layout.itemsize = itemsize
        out_layout.slice_offset = new_slice_offset
        _swap_layout(out_layout.base, unpacked)
        return vec_size

    cdef int slice_into(StridedLayout self, StridedLayout out_layout, tuple slices) except -1:
        cdef BaseLayout sliced
        cdef stride_t slice_offset = slice_extents(sliced, self.base, slices)
        cdef int64_t new_slice_offset = _overflow_checked_sum(self.slice_offset, slice_offset)

        # Reset all memoized properties
        out_layout._prop_mask = 0

        # Preserved attributes
        out_layout.itemsize = self.itemsize

        # Set new attributes
        _swap_layout(out_layout.base, sliced)
        out_layout.slice_offset = new_slice_offset
        return 0

cdef inline int maybe_copy_volume(StridedLayout out_layout, StridedLayout in_layout) except -1 nogil:
    if _has_valid_property(in_layout, PROP_VOLUME):
        out_layout._volume = in_layout.get_volume()
        _mark_property_valid(out_layout, PROP_VOLUME)
    return 0


cdef inline int validate_reshaped_shape(BaseLayout& new_shape, int64_t old_volume) except -1 nogil:
    cdef int ndim = new_shape.ndim
    cdef int axis = -1
    cdef extent_t extent
    for i in range(ndim):
        extent = new_shape.shape[i]
        if extent < -1:
            raise ValueError("Extents must be non-negative")
        elif extent == -1:
            if axis == -1:
                axis = i
            else:
                raise ValueError("There can be at most one -1 extent in a shape")
    cdef int64_t new_volume = _c_abs(_volume(new_shape))
    if new_volume == 0 and axis != -1:
        raise ValueError("The -1 extent is ambiguous when the volume is 0")
    if new_volume != old_volume:
        if axis == -1:
            raise ValueError(f"The original volume {old_volume} and the new volume {new_volume} must be equal.")
        extent = old_volume // new_volume
        if extent * new_volume != old_volume:
            raise ValueError(f"The original volume {old_volume} must be divisible by the specified sub-volume {new_volume}.")
        new_shape.shape[axis] = extent
    return 0


cdef inline axes_mask_t axis_mask_from_range(int ndim, int start_axis, int end_axis) except? -1 nogil:
    if ndim == 0 and start_axis == 0 and end_axis == -1:
        return AXIS_MASK_ALL
    cdef axes_mask_t axis_mask = AXIS_MASK_ALL
    if not _normalize_axis(start_axis, ndim):
        raise ValueError(f"Invalid start axis: {start_axis} out of range for {ndim}D tensor")
    if not _normalize_axis(end_axis, ndim):
        raise ValueError(f"Invalid end axis: {end_axis} out of range for {ndim}D tensor")
    if start_axis > 0:
        axis_mask &= (AXIS_MASK_ALL << start_axis + 1)
    if end_axis < ndim:
        axis_mask &= (AXIS_MASK_ALL >> (STRIDED_LAYOUT_MAX_NDIM - end_axis - 1))
    return axis_mask


cdef inline int flatten_strides_in_c_index_order(BaseLayout& out_layout, BaseLayout& in_layout, axes_mask_t axis_mask) except -1 nogil:
    cdef int ndim = in_layout.ndim
    init_base_layout(out_layout, ndim)
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    cdef int out_i = 0
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    while group_start < ndim:
        group_vol = in_shape[group_start]
        group_stride = in_strides[group_start]
        group_end = group_start + 1
        while (
            group_end < ndim
            and (axis_mask & (1 << group_end))
            and group_stride == _overflow_checked_mul(in_strides[group_end], in_shape[group_end])
        ):
            group_vol = _overflow_checked_mul(group_vol, in_shape[group_end])
            group_stride = in_strides[group_end]
            group_end += 1
        out_layout.shape[out_i] = group_vol
        out_layout.strides[out_i] = group_stride
        out_i += 1
        group_start = group_end
    if out_i != ndim:
        trim_base_layout(out_layout, out_i)
    return out_i


cdef inline axes_mask_t flattened_strides_in_c_index_order_mask(BaseLayout& layout) except? -1 nogil:
    if layout.strides == NULL:
        return AXIS_MASK_ALL
    cdef axes_mask_t axis_mask = 0
    cdef int ndim = layout.ndim
    cdef int group_start = 0
    cdef int group_end = 0
    cdef int64_t group_vol
    cdef int64_t group_stride
    while group_start < ndim:
        group_vol = layout.shape[group_start]
        group_stride = layout.strides[group_start]
        group_end = group_start + 1
        while group_end < ndim and group_stride == layout.strides[group_end] * layout.shape[group_end]:
            group_vol = _overflow_checked_mul(group_vol, layout.shape[group_end])
            group_stride = layout.strides[group_end]
            axis_mask |= (1 << group_end)
            group_end += 1
        group_start = group_end
    return axis_mask


cdef inline bint split_strides_in_c_index_order(BaseLayout& out_layout, BaseLayout& in_layout) except -1 nogil:
    cdef int i = in_layout.ndim - 1
    cdef int new_i = out_layout.ndim - 1
    cdef extent_t extent
    cdef extent_t new_extent
    cdef extent_t group_vol
    cdef stride_t group_stride
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    if out_layout.strides == NULL:
        _zero_strides(out_layout)
    while i >= 0:
        extent = in_shape[i]
        group_vol = 1
        group_stride = in_strides[i]
        while new_i >= 0 and group_vol < extent:
            new_extent = out_layout.shape[new_i]
            if new_extent == 0:
                return False
            group_vol = _overflow_checked_mul(group_vol, new_extent)
            out_layout.strides[new_i] = group_stride
            group_stride = _overflow_checked_mul(group_stride, new_extent)
            new_i -= 1
        if group_vol != extent:
            return False
        i -= 1
    return True


cdef inline int permute_extents(BaseLayout& out_layout, BaseLayout& in_layout, axis_vec_t& axis_order) except -1 nogil:
    cdef int ndim = in_layout.ndim
    init_base_layout(out_layout, ndim)
    cdef axis_t axis
    cdef axes_mask_t axis_mask
    cdef axes_mask_t axis_order_mask = 0
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)

    for i in range(ndim):
        axis = axis_order[i]
        if not _normalize_axis(axis, ndim):
            raise ValueError(f"Invalid permutation: axis {axis} out of range for {ndim}D tensor")
        axis_mask = 1 << axis
        if axis_order_mask & axis_mask:
            raise ValueError(f"Invalid permutation: axis {axis_order[i]} appears multiple times.")
        axis_order_mask |= axis_mask
        out_layout.shape[i] = in_shape[axis]
        out_layout.strides[i] = in_strides[axis]
    return 0


cdef inline stride_t slice_extents(BaseLayout& out_layout, BaseLayout& in_layout, tuple slices) except? -1:
    cdef int ndim = in_layout.ndim
    cdef int num_slices = len(slices)
    if num_slices > ndim:
        raise ValueError(f"The number of slices ({num_slices}) is greater than the number of dimensions ({ndim}).")
    init_base_layout(out_layout, ndim)
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    cdef stride_t slice_offset = 0
    cdef Py_ssize_t start
    cdef Py_ssize_t stop
    cdef Py_ssize_t step
    cdef extent_t new_extent
    cdef object py_slice
    cdef bint zero_slice = False
    cdef int out_i = 0
    for i in range(num_slices):
        py_slice = slices[i]
        if isinstance(py_slice, int):
            start = py_slice
            if not _normalize_axis(start, in_shape[i]):
                raise ValueError(f"Invalid index: {start} out of range for axis {i} with extent {in_shape[i]}")
            # single element index removes extent from the shape,
            # just increase the offset and skip the shape and stride
            slice_offset = _overflow_checked_sum(slice_offset, _overflow_checked_mul(start, in_strides[i]))
        elif isinstance(py_slice, slice):
            _PySlice_Unpack(<PyObject *>py_slice, &start, &stop, &step)
            new_extent = _PySlice_AdjustIndices(in_shape[i], &start, &stop, step)
            if new_extent > 0:
                # out_extent > 0 implies start is in [0, extent - 1] range
                slice_offset = _overflow_checked_sum(slice_offset, _overflow_checked_mul(start, in_strides[i]))
            else:
                zero_slice = True
            out_layout.shape[out_i] = new_extent
            out_layout.strides[out_i] = _overflow_checked_mul(in_strides[i], step)
            out_i += 1
        else:
            raise TypeError(f"Invalid slice: {py_slice}. Expected slice instance or integer.")
    for i in range(num_slices, ndim):
        out_layout.shape[out_i] = in_shape[i]
        out_layout.strides[out_i] = in_strides[i]
        out_i += 1
    if out_i != ndim:
        trim_base_layout(out_layout, out_i)
    if zero_slice:
        _zero_strides(out_layout)
    return slice_offset


cdef inline int squeeze_extents(BaseLayout& out_layout, BaseLayout& in_layout) except -1 nogil:
    cdef int ndim = in_layout.ndim
    init_base_layout(out_layout, ndim)
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    cdef int out_i = 0
    cdef extent_t extent
    for i in range(ndim):
        extent = in_shape[i]
        if extent == 0:
            trim_base_layout(out_layout, 1)
            out_layout.shape[0] = 0
            out_layout.strides[0] = 0
            return 1
        elif extent != 1:
            out_layout.shape[out_i] = extent
            out_layout.strides[out_i] = in_strides[i]
            out_i += 1
    if out_i != ndim:
        trim_base_layout(out_layout, out_i)
    return out_i


cdef inline int unsqueeze_extents(BaseLayout& out_layout, BaseLayout& in_layout, axis_vec_t& axis_vec) except -1 nogil:
    cdef int ndim = in_layout.ndim
    cdef int num_new_axes = axis_vec.size()
    cdef int out_ndim = ndim + num_new_axes
    # init_base_layout validates out_ndim
    init_base_layout(out_layout, out_ndim)
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    cdef axes_mask_t out_shape_mask = 0
    cdef axes_mask_t axis_mask = 0
    cdef axis_t axis
    for i in range(num_new_axes):
        axis = axis_vec[i]
        if not _normalize_axis(axis, out_ndim):
            raise ValueError(f"Invalid axis: {axis} out of range for {out_ndim}D tensor")
        axis_mask = 1 << axis
        if out_shape_mask & axis_mask:
            raise ValueError(f"Axis {axis} appears multiple times.")
        out_shape_mask |= axis_mask
    cdef int in_i = 0
    for i in range(out_ndim):
        # without the cast, cython has issues with
        # recognizing 1 << i does not require Python interaction
        axis_mask = 1 << <int>i
        if out_shape_mask & axis_mask:
            out_layout.shape[i] = 1
            if in_i < ndim:
                out_layout.strides[i] = _overflow_checked_mul(in_shape[in_i], in_strides[in_i])
            else:
                if ndim > 0:
                    out_layout.strides[i] = in_strides[ndim - 1]
                else:
                    out_layout.strides[i] = 1
        else:
            out_layout.shape[i] = in_shape[in_i]
            out_layout.strides[i] = in_strides[in_i]
            in_i += 1
    assert in_i == ndim
    return 0


cdef inline int broadcast_extents(BaseLayout& broadcast, BaseLayout& in_layout) except -1 nogil:
    if broadcast.ndim < in_layout.ndim:
        raise ValueError(
            f"The broadcast shape ndim ({broadcast.ndim}) must be "
            f"greater than or equal to the input shape "
            f"ndim ({in_layout.ndim})."
        )
    cdef int ndim_diff = broadcast.ndim - in_layout.ndim
    _zero_strides(broadcast)
    cdef extent_t* in_shape = in_layout.shape
    cdef stride_t* in_strides = get_strides_ptr(in_layout)
    cdef extent_t* broadcast_shape = broadcast.shape + ndim_diff
    cdef stride_t* broadcast_strides = broadcast.strides + ndim_diff
    for i in range(in_layout.ndim):
        if in_shape[i] == broadcast_shape[i]:
            broadcast_strides[i] = in_strides[i]
        elif in_shape[i] != 1:
            raise ValueError(
                f"Shapes cannot be broadcast together: "
                f"the original extent must be 1 or be equal to broadcast extent, "
                f"got {in_shape[i]} and {broadcast_shape[i]} for axis {i}."
            )
        # else -> in_extent == 1, the broadcast extent and zero stride are already set
    return 0


cdef inline int64_t gcd(int64_t a, int64_t b) except? -1 nogil:
    while b != 0:
        a, b = b, a % b
    return a


cdef inline int pack_extents(BaseLayout& out_layout, stride_t& out_slice_offset, BaseLayout& in_layout, stride_t slice_offset, int itemsize, int new_itemsize, intptr_t data_ptr, bint keep_dim, int axis) except -1 nogil:
    cdef int ndim = in_layout.ndim
    if new_itemsize <= 0 or new_itemsize & (new_itemsize - 1):
        raise ValueError(f"new itemsize must be a power of two, got {new_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if new_itemsize <= itemsize:
        if new_itemsize == itemsize:
            return 1
        raise ValueError(f"new itemsize ({new_itemsize}) must be greater than or equal to itemsize ({itemsize}).")
    if not _normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    if data_ptr % new_itemsize != 0:
        raise ValueError(f"The data pointer ({data_ptr}) must be aligned to the packed itemsize ({new_itemsize}).")

    cdef extent_t* shape = in_layout.shape
    cdef stride_t* strides = get_strides_ptr(in_layout)
    if strides[axis] != 1:
        raise ValueError(f"The axis {axis} stride must be 1, got {strides[axis]}.")

    cdef int vec_size = new_itemsize // itemsize
    cdef extent_t packed_extent = shape[axis]
    if packed_extent == 0:
        raise ValueError(f"The axis {axis} extent must be non-zero, got {shape[axis]}.")
    packed_extent //= vec_size
    if packed_extent * vec_size != shape[axis]:
        raise ValueError(f"The axis {axis} extent ({shape[axis]}) must be divisible by {vec_size}.")

    cdef stride_t new_slice_offset = slice_offset // vec_size
    if new_slice_offset * vec_size != slice_offset:
        raise ValueError(f"The slice offset ({slice_offset}) must be divisible by {vec_size}.")
    out_slice_offset = new_slice_offset

    init_base_layout(out_layout, ndim)
    cdef stride_t packed_stride
    cdef int out_i = 0
    for i in range(ndim):
        if i == axis:
            if keep_dim or packed_extent != 1:  # omit the packed axis if it is reduced to 1
                out_layout.shape[out_i] = packed_extent
                out_layout.strides[out_i] = 1
                out_i += 1
        else:
            packed_stride = strides[i] // vec_size
            if packed_stride * vec_size != strides[i]:
                raise ValueError(f"The {i} axis stride ({strides[i]}) must be divisible by {vec_size}.")
            out_layout.shape[out_i] = shape[i]
            out_layout.strides[out_i] = packed_stride
            out_i += 1
    if out_i != ndim:
        trim_base_layout(out_layout, out_i)
    return vec_size


cdef inline int unpack_extents(BaseLayout &out_layout, BaseLayout &in_layout, int itemsize, int new_itemsize, int axis) except -1 nogil:
    cdef int ndim = in_layout.ndim
    if not _normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    if new_itemsize <= 0 or new_itemsize & (new_itemsize - 1):
        raise ValueError(f"new itemsize must be a power of two, got {new_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if new_itemsize >= itemsize:
        if new_itemsize == itemsize:
            return 1
        raise ValueError(f"new itemsize ({new_itemsize}) must be less than or equal to itemsize ({itemsize}).")

    cdef extent_t* shape = in_layout.shape
    cdef stride_t* strides = get_strides_ptr(in_layout)
    if shape[axis] == 0:
        raise ValueError(f"The axis {axis} extent must be non-zero, got {shape[axis]}.")
    if strides[axis] != 1:
        raise ValueError(f"The axis {axis} stride must be 1, got {strides[axis]}.")

    cdef int vec_size = itemsize // new_itemsize
    init_base_layout(out_layout, ndim)
    out_layout.shape[axis] = _overflow_checked_mul(shape[axis], vec_size)
    out_layout.strides[axis] = 1

    for i in range(ndim):
        if i == axis:
            continue
        out_layout.shape[i] = shape[i]
        out_layout.strides[i] = _overflow_checked_mul(strides[i], vec_size)
    return vec_size


cdef inline int max_compatible_itemsize(BaseLayout& layout, stride_t slice_offset, int itemsize, int max_itemsize, intptr_t data_ptr, int axis) except? -1 nogil:
    cdef int ndim = layout.ndim
    if max_itemsize <= 0 or max_itemsize & (max_itemsize - 1):
        raise ValueError(f"max_itemsize must be a power of two, got {max_itemsize}.")
    if itemsize <= 0 or itemsize & (itemsize - 1):
        raise ValueError(f"itemsize must be a power of two, got {itemsize}.")
    if not _normalize_axis(axis, ndim):
        raise ValueError(f"Invalid axis: {axis} out of range for {ndim}D tensor")
    max_itemsize = gcd(max_itemsize, _c_abs(data_ptr))
    cdef extent_t* shape = layout.shape
    cdef stride_t* strides = get_strides_ptr(layout)
    if ndim < 1 or strides[axis] != 1 or shape[axis] == 0:
        return min(max_itemsize, itemsize)
    max_itemsize = gcd(max_itemsize, _overflow_checked_mul(slice_offset, itemsize))
    max_itemsize = gcd(max_itemsize, _overflow_checked_mul(shape[axis], itemsize))
    for i in range(ndim):
        if i == axis:
            continue
        max_itemsize = gcd(max_itemsize, _overflow_checked_mul(_c_abs(strides[i]), itemsize))
    return max_itemsize
