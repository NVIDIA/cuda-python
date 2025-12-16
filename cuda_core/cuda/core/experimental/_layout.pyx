# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport int64_t, intptr_t

from cpython.object cimport PyObject


cdef extern from "Python.h":
    int _PySlice_Unpack "PySlice_Unpack" (PyObject *slice, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t *step) except -1
    Py_ssize_t _PySlice_AdjustIndices "PySlice_AdjustIndices" (Py_ssize_t length, Py_ssize_t *start, Py_ssize_t *stop, Py_ssize_t step) noexcept nogil


@cython.final
cdef class _StridedLayout:
    """
    A class describing the layout of a multi-dimensional tensor
    with a shape, strides and itemsize.

    Parameters
    ----------
    shape : tuple
        A tuple of non-negative integers.
    strides : tuple, optional
        If provided, must be a tuple of integers of the same length as ``shape``.
        Otherwise, the strides are assumed to be implicitly C-contiguous and the resulting
        layout's :attr:`strides` will be None.
    itemsize : int
        The number of bytes per single element (dtype size). Must be a power of two.
    divide_strides : bool, optional
        If True, the provided :attr:`strides` will be divided by the :attr:`itemsize`.


    See also :meth:`dense`.


    Attributes
    ----------
    itemsize : int
        The number of bytes per single element (dtype size). Must be a power of two.
    slice_offset : int
        The offset (as a number of elements, not bytes) of the element at
        index ``(0,) * ndim``. See also :attr:`slice_offset_in_bytes`.
    """

    def __init__(
        self : _StridedLayout,
        shape : tuple[int, ...],
        strides : tuple[int, ...] | None,
        itemsize : int,
        divide_strides : bool = False
    ) -> None:
        self.init_from_tuple(shape, strides, itemsize, divide_strides)

    @classmethod
    def dense(
        cls,
        shape : tuple[int],
        itemsize : int,
        stride_order : str | tuple[int] = 'C'
    ) -> _StridedLayout:
        """
        Creates a new _StridedLayout instance with dense strides.

        Parameters
        ----------
        shape : tuple
            A tuple of non-negative integers.
        itemsize : int
            The number of bytes per single element of the tensor.
        stride_order : str or tuple, optional
            The order of the strides:
                * 'C' (default) - the strides are computed in C-order (increasing from the right to the left)
                * 'F' - the strides are computed in F-order (increasing from the left to the right)
                * A tuple - it must be a permutation of ``tuple(range(len(shape)))``.
                  The last element of the tuple is the axis with stride 1.

            See also :attr:`stride_order`.


        .. highlight:: python
        .. code-block:: python

            assert _StridedLayout.dense((5, 3, 7), 1, "C") == _StridedLayout((5, 3, 7), (21, 7, 1), 1)
            assert _StridedLayout.dense((5, 3, 7), 1, "F") == _StridedLayout((5, 3, 7), (1, 5, 15), 1)
            assert _StridedLayout.dense((5, 3, 7), 1, (2, 0, 1)) == _StridedLayout((5, 3, 7), (3, 1, 15), 1)

        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(cls)
        new_layout.init_dense_from_tuple(shape, itemsize, stride_order)
        return new_layout

    @classmethod
    def dense_like(
        cls,
        other : _StridedLayout,
        stride_order : str | tuple[int] = "K"
    ) -> _StridedLayout:
        """
        Creates a _StridedLayout with the same :attr:`shape` and :attr:`itemsize` as the other layout,
        but with contiguous strides in the specified order and no slice offset.

        See also :attr:`is_dense`.

        Parameters
        ----------
        other : _StridedLayout
            The _StridedLayout to copy the :attr:`shape` and :attr:`itemsize` from.
        stride_order : str or tuple, optional
            The order of the strides:
                * 'K' (default) - keeps the order of the strides as in the ``other`` layout.
                * 'C' - the strides are computed in C-order (increasing from the right to the left)
                * 'F' - the strides are computed in F-order (increasing from the left to the right)
                * A tuple - it must be a permutation of ``tuple(range(len(shape)))``.
                  The last element of the tuple is the axis with stride 1.

            See also :attr:`stride_order`.


        .. highlight:: python
        .. code-block:: python

            layout = _StridedLayout.dense((5, 3, 7), 1).permuted((2, 0, 1))
            assert layout == _StridedLayout((7, 5, 3), (1, 21, 7), 1)

            # dense_like with the default "K" stride_order
            # keeps the same order of strides as in the original layout
            assert _StridedLayout.dense_like(layout) == layout
            # "C", "F" recompute the strides accordingly
            assert _StridedLayout.dense_like(layout, "C") == _StridedLayout((7, 5, 3), (15, 3, 1), 1)
            assert _StridedLayout.dense_like(layout, "F") == _StridedLayout((7, 5, 3), (1, 7, 35), 1)
        """
        cdef OrderFlag order_flag
        cdef axis_vec_t stride_order_vec
        cdef bint is_dense = other.get_is_dense()

        if stride_order == "K":
            if is_dense:
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
                if is_dense and other.get_is_contiguous_c():
                    return other
            elif order_flag == ORDER_F:
                if is_dense and other.get_is_contiguous_f():
                    return other

        cdef _StridedLayout new_layout = _StridedLayout.__new__(cls)
        new_layout.init_dense_from_ptr(
            other.base.ndim,
            other.base.shape,
            other.itemsize,
            order_flag,
            &stride_order_vec
        )
        return new_layout

    def __repr__(self : _StridedLayout) -> str:
        if self.slice_offset == 0:
            return (
                f"_StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize})"
            )
        else:
            return (
                f"_StridedLayout(shape={self.shape}, strides={self.strides}, itemsize={self.itemsize}, _slice_offset={self.slice_offset})"
            )

    def __eq__(self : _StridedLayout, other : _StridedLayout) -> bool:
        return self.itemsize == other.itemsize and self.slice_offset == other.slice_offset and _base_layout_equal(self.base, other.base)

    @property
    def ndim(self : _StridedLayout):
        """
        The number of dimensions (length of the shape tuple).

        :type: int
        """
        return self.base.ndim

    @property
    def shape(self : _StridedLayout):
        """
        Shape of the tensor.

        :type: tuple[int]
        """
        return self.get_shape_tuple()

    @property
    def strides(self : _StridedLayout):
        """
        Strides of the tensor (in **counts**, not bytes).
        If _StridedLayout was created with strides=None, the
        returned value is None and layout is implicitly C-contiguous.

        :type: tuple[int] | None
        """
        return self.get_strides_tuple()

    @property
    def strides_in_bytes(self : _StridedLayout):
        """
        Strides of the tensor (in bytes).

        :type: tuple[int] | None
        """
        return self.get_strides_in_bytes_tuple()

    @property
    def stride_order(self : _StridedLayout):
        """
        A permutation of ``tuple(range(ndim))`` describing the
        relative order of the strides.

        .. highlight:: python
        .. code-block:: python

            # C-contiguous layout
            assert _StridedLayout.dense((5, 3, 7), 1).stride_order == (0, 1, 2)
            # F-contiguous layout
            assert _StridedLayout.dense((5, 3, 7), 1, stride_order="F").stride_order == (2, 1, 0)
            # Permuted layout
            assert _StridedLayout.dense((5, 3, 7), 1, stride_order=(2, 0, 1)).stride_order == (2, 0, 1)

        :type: tuple[int]
        """
        return self.get_stride_order_tuple()

    @property
    def volume(self : _StridedLayout):
        """
        The number of elements in the tensor, i.e. the product of the shape tuple.

        :type: int
        """
        return self.get_volume()

    @property
    def is_unique(self : _StridedLayout):
        """
        If True, each element of a tensor with this layout is mapped to
        a unique memory offset.

        All contiguous layouts are unique and so are layouts that can be created
        by permuting, slicing, flattening, squeezing, repacking, or reshaping
        a contiguous layout.
        Conversely, broadcast layouts (layouts with a 0 stride
        for some extent greater than 1) are not unique.

        For layouts resulting from manual stride manipulations
        (such as with ``numpy.lib.stride_tricks``), the check
        may inaccurately report False, as the exact uniqueness
        check may be expensive.

        :type: bool
        """
        return self.get_is_unique()

    @property
    def is_contiguous_c(self : _StridedLayout):
        """
        True iff the layout is contiguous in C-order, i.e.
        the rightmost stride is 1 and each subsequent
        stride to the left is the product of the
        extent and the stride to the right.

        .. highlight:: python
        .. code-block:: python

            layout = _StridedLayout.dense((2, 5, 3), 1, "C")
            assert layout == _StridedLayout((2, 5, 3), (15, 3, 1), 1)
            assert layout.is_contiguous_c

        See also :attr:`is_contiguous_any`.

        :type: bool
        """
        return self.get_is_contiguous_c()

    @property
    def is_contiguous_f(self : _StridedLayout):
        """
        True iff the layout is contiguous in F-order, i.e.
        the leftmost stride is 1 and each subsequent
        stride to the right is the product of the
        stride and extent to the left.

        .. highlight:: python
        .. code-block:: python

            layout = _StridedLayout.dense((2, 5, 3), 1, "F")
            assert layout == _StridedLayout((2, 5, 3), (1, 2, 10), 1)
            assert layout.is_contiguous_f

        See also :attr:`is_contiguous_any`.

        :type: bool
        """
        return self.get_is_contiguous_f()

    @property
    def is_contiguous_any(self : _StridedLayout):
        """
        True iff the layout is contiguous in some axis order, i.e.
        there exists a permutation of axes such that the layout
        is C-contiguous.

        In a contiguous layout, the strides are non-negative and
        the mapping of elements to the memory offset range
        ``[min_offset, max_offset]`` is 1-to-1.

        .. highlight:: python
        .. code-block:: python

            # dense defaults to C-contiguous
            layout = _StridedLayout.dense((5, 3, 7), 1)
            assert layout.is_contiguous_c and not layout.is_contiguous_f
            assert layout.is_contiguous_any

            # reversing the order of axes gives F-contiguous layout
            permuted = layout.permuted((2, 1, 0))
            assert not permuted.is_contiguous_c and permuted.is_contiguous_f
            assert permuted.is_contiguous_any

            # neither C- nor F-order but still contiguous
            permuted = layout.permuted((2, 0, 1))
            assert not permuted.is_contiguous_c and not permuted.is_contiguous_f
            assert permuted.is_contiguous_any

            # slicing the right-most extent creates a gap in the
            # offset_bounds range that is not reachable with any
            # element in the sliced layout
            sliced = layout[:, :, :-1]
            assert not sliced.is_contiguous_c and not sliced.is_contiguous_f
            assert not sliced.is_contiguous_any

        :type: bool
        """
        return self.get_is_contiguous_any()

    @property
    def is_dense(self : _StridedLayout):
        """
        A dense layout is contiguous (:attr:`is_contiguous_any` is True)
        and has no slice offset (:attr:`slice_offset_in_bytes` is 0).

        In a dense layout, elements are mapped 1-to-1 to the ``[0, volume - 1]``
        memory offset range.

        :type: bool
        """
        return self.get_is_dense()

    @property
    def offset_bounds(self : _StridedLayout):
        """
        The memory offset range ``[min_offset, max_offset]`` (in element counts, not bytes)
        that elements of a tensor with this layout are mapped to.

        If the layout is empty (i.e. ``volume == 0``), the returned tuple is ``(0, -1)``.
        Otherwise, ``min_offset <= max_offset`` and all elements of the tensor with
        this layout are mapped within the ``[min_offset, max_offset]`` range.

        .. highlight:: python
        .. code-block:: python

            # Possible implementation of the offset_bounds
            def offset_bounds(layout : _StridedLayout):
                if layout.volume == 0:
                    return 0, -1
                ndim = layout.ndim
                shape = layout.shape
                strides = layout.strides
                idx_min = [shape[i] - 1 if strides[i] < 0 else 0 for i in range(ndim)]
                idx_max = [shape[i] - 1 if strides[i] > 0 else 0 for i in range(ndim)]
                min_offset = sum(strides[i] * idx_min[i] for i in range(ndim)) + layout.slice_offset
                max_offset = sum(strides[i] * idx_max[i] for i in range(ndim)) + layout.slice_offset
                return min_offset, max_offset

        :type: tuple[int, int]
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return min_offset, max_offset

    @property
    def min_offset(self : _StridedLayout):
        """
        See :attr:`offset_bounds` for details.

        :type: int
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return min_offset

    @property
    def max_offset(self : _StridedLayout):
        """
        See :attr:`offset_bounds` for details.

        :type: int
        """
        cdef stride_t min_offset = 0
        cdef stride_t max_offset = 0
        self.get_offset_bounds(min_offset, max_offset)
        return max_offset

    @property
    def slice_offset_in_bytes(self : _StridedLayout):
        """
        The memory offset (as a number of bytes) of the element at index ``(0,) * ndim``.
        Equal to :attr:`itemsize` ``*`` :attr:`slice_offset`.

        .. note::
            The only way for the index ``(0,) * ndim`` to be mapped to a non-zero offset
            is slicing with :meth:`sliced` method (or ``[]`` operator).

        :type: int
        """
        return self.get_slice_offset_in_bytes()

    def required_size_in_bytes(self : _StridedLayout) -> int:
        """
        The memory allocation size (in bytes) needed so that
        all elements of a tensor with this layout can be mapped
        within the allocated memory range.

        The function raises an error if ``min_offset < 0``.
        Otherwise, the returned value is equal to
        ``(max_offset + 1) * itemsize``.

        .. hint::
            For dense layouts, the function always succeeds and the
            ``(max_offset + 1) * itemsize`` is equal to the ``volume * itemsize``.

        .. highlight:: python
        .. code-block:: python

            # Allocating memory on a device to copy a host tensor
            def device_tensor_like(a : numpy.ndarray, device : ccx.Device) -> StridedMemoryView:
                a_view = StridedMemoryView(a, -1)
                # get the original layout of ``a`` and convert it to a dense layout
                # to avoid overallocating memory (e.g. if the ``a`` was sliced)
                layout = a_view.layout.to_dense()
                # get the required size in bytes to fit the tensor
                required_size = layout.required_size_in_bytes()
                # allocate the memory on the device
                device.set_current()
                mem = device.allocate(required_size)
                # create a view on the newly allocated device memory
                b_view = StridedMemoryView.from_buffer(mem, layout, a_view.dtype)
                return b_view
        """
        return self.get_required_size_in_bytes()

    def flattened_axis_mask(self : _StridedLayout) -> axes_mask_t:
        """
        A mask describing which axes of this layout are mergeable
        using the :meth:`flattened` method.
        """
        return self.get_flattened_axis_mask()

    def to_dense(self : _StridedLayout, object stride_order="K") -> _StridedLayout:
        """
        Returns a dense layout with the same shape and itemsize,
        but with dense strides in the specified order.

        See :meth:`dense_like` method documentation for details.
        """
        return _StridedLayout.dense_like(self, stride_order)

    def reshaped(self : _StridedLayout, shape : tuple[int]) -> _StridedLayout:
        """
        Returns a layout with the new shape, if the new shape is compatible
        with the current layout.

        The new shape is compatible if:
            * the new and old shapes have the same volume
            * the old strides can be split or flattened to match the new shape,
              assuming indices are iterated in C-order

        A single extent in the ``shape`` tuple can be set to -1 to indicate
        it should be inferred from the old volume and the other extents.

        .. highlight:: python
        .. code-block:: python

            layout = _StridedLayout.dense((5, 3, 4), 1)
            assert layout.reshaped((20, 3)) == _StridedLayout.dense((20, 3), 1)
            assert layout.reshaped((4, -1)) == _StridedLayout.dense((4, 15), 1)
            assert layout.permuted((2, 0, 1)).reshaped((4, 15,)) == _StridedLayout((4, 15), (1, 4), 1)
            # layout.permuted((2, 0, 1)).reshaped((20, 3)) -> error
        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        cdef BaseLayout new_shape
        init_base_layout(new_shape, len(shape))
        for i in range(len(shape)):
            new_shape.shape[i] = shape[i]
        self.reshape_into(new_layout, new_shape)
        return new_layout

    def permuted(self : _StridedLayout, axis_order : tuple[int]) -> _StridedLayout:
        """
        Returns a new layout where the shape and strides tuples are permuted
        according to the specified permutation of axes.
        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        cdef axis_vec_t axis_order_vec
        _tuple2axis_vec(axis_order_vec, axis_order)
        self.permute_into(new_layout, axis_order_vec)
        return new_layout

    def flattened(self : _StridedLayout, start_axis : int = 0, end_axis : int = -1, mask : int | None = None) -> _StridedLayout:
        """
        Merges consecutive extents into a single extent (equal to the product of merged extents)
        if the corresponding strides can be replaced with a single stride
        (assuming indices are iterated in C-order, i.e. the rightmost
        axis is incremented first).

        .. highlight:: python
        .. code-block:: python

            # the two extents can be merged into a single extent
            # because layout.strides[0] == layout.strides[1] * layout.shape[1]
            layout = _StridedLayout((3, 2), (2, 1), 1)
            assert layout.flattened() == _StridedLayout((6,), (1,), 1)

            # the two extents cannot be merged into a single extent
            # because layout.strides[0] != layout.strides[1] * layout.shape[1]
            layout = _StridedLayout((3, 2), (1, 3), 1)
            assert layout.flattened() == layout

        If ``start_axis`` and ``end_axis`` are provided, only the axes in the
        inclusive range ``[start_axis, end_axis]`` are considered for flattening.

        Alternatively, a mask specifying which axes to consider can be provided.
        A mask of mergeable extents can be obtained using the :meth:`flattened_axis_mask` method.
        Masks for layouts with the same number of dimensions can be combined
        using the logical ``&`` (bitwise AND) operator.

        .. highlight:: python
        .. code-block:: python

            layout = _StridedLayout.dense((4, 5, 3), 4)
            layout2 = _StridedLayout((4, 5, 3), (1, 12, 4), 4)
            # Even though the two layouts have the same shape initially,
            # their shapes differ after flattening.
            assert layout.flattened() == _StridedLayout((60,), (1,), 4)
            assert layout2.flattened() == _StridedLayout((4, 15), (1, 4), 4)
            # With the mask, only extents that are mergeable in both layouts are flattened
            # and the resulting shape is the same for both layouts.
            mask = layout.flattened_axis_mask() & layout2.flattened_axis_mask()
            assert layout.flattened(mask=mask) == _StridedLayout((4, 15), (15, 1), 4)
            assert layout2.flattened(mask=mask) == _StridedLayout((4, 15), (1, 4), 4)
        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        cdef axes_mask_t axis_mask
        if mask is None:
            axis_mask = axis_mask_from_range(self.ndim, start_axis, end_axis)
        else:
            axis_mask = mask
        self.flatten_into(new_layout, axis_mask)
        return new_layout

    def squeezed(self : _StridedLayout) -> _StridedLayout:
        """
        Returns a new layout where all the singleton dimensions (extents equal to 1)
        are removed. Additionally, if the layout volume is 0,
        the returned layout will be reduced to a 1-dim layout
        with shape (0,) and strides (0,).
        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        self.squeeze_into(new_layout)
        return new_layout

    def unsqueezed(self : _StridedLayout, axis : int | tuple[int]) -> _StridedLayout:
        """
        Returns a new layout where the specified axis or axes are added as singleton extents.
        The ``axis`` can be either a single integer in range ``[0, ndim]``
        or a tuple of unique integers in range ``[0, ndim + len(axis) - 1]``.
        """
        cdef axis_vec_t axis_vec
        if isinstance(axis, int):
            axis_vec.push_back(axis)
        else:
            _tuple2axis_vec(axis_vec, axis)
        if axis_vec.size() == 0:
            return self
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        self.unsqueeze_into(new_layout, axis_vec)
        return new_layout

    def broadcast_to(self : _StridedLayout, shape : tuple[int]) -> _StridedLayout:
        """
        Returns a layout with the new shape, if the old shape can be
        broadcast to the new one.

        The shapes are compatible if:
            * the new shape has the same or greater number of dimensions
            * starting from the right, each extent in the old shape must be 1 or
              equal to the corresponding extent in the new shape.

        Strides of the added or modified extents are set to 0, the remaining ones are unchanged.
        If the shapes are not compatible, a ValueError is raised.
        """
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        cdef BaseLayout new_shape
        cdef int new_ndim = len(shape)
        init_base_layout(new_shape, new_ndim)
        for i in range(new_ndim):
            new_shape.shape[i] = shape[i]
        self.broadcast_into(new_layout, new_shape)
        return new_layout

    def repacked(self : _StridedLayout, itemsize : int, data_ptr : intptr_t = 0, axis : int = -1, keep_dim : bool = True) -> _StridedLayout:
        """
        Converts the layout to match the specified itemsize.
        If ``new_itemsize < itemsize``, each element of the tensor is **unpacked** into multiple elements,
        i.e. the extent at ``axis`` increases by the factor ``itemsize // new_itemsize``.
        If ``new_itemsize > itemsize``, the consecutive elements in the tensor are **packed** into a single element,
        i.e. the extent at ``axis`` decreases by the factor ``new_itemsize // itemsize``.
        In either case, the ``volume * itemsize`` of the layout remains the same.

        The conversion is subject to the following constraints:
            * The old and new itemsizes must be powers of two.
            * The extent at ``axis`` must be a positive integer.
            * The stride at ``axis`` must be 1.

        Moreover, if the ``new_itemsize > itemsize``:
            * The extent at ``axis`` must be divisible by ``new_itemsize // itemsize``.
            * All other strides must be divisible by ``new_itemsize // itemsize``.
            * The ``slice_offset`` must be divisible by ``new_itemsize // itemsize``.
            * If ``data_ptr`` is provided, it must be aligned to the new itemsize.

        The maximum itemsize that satisfies all the constraints
        can be obtained using the :meth:`max_compatible_itemsize` method.

        If the ``keep_dim`` is False and the extent at ``axis`` would be reduced to 1,
        it is omitted from the returned layout.

        .. highlight:: python
        .. code-block:: python

            # Repacking the layout with itemsize = 4 bytes as 2, 8, and 16 sized layouts.
            layout = _StridedLayout.dense((5, 4), 4)
            assert layout.repacked(2) == _StridedLayout.dense((5, 8), 2)
            assert layout.repacked(8) == _StridedLayout.dense((5, 2), 8)
            assert layout.repacked(16) == _StridedLayout.dense((5, 1), 16)
            assert layout.repacked(16, keep_dim=False) == _StridedLayout.dense((5,), 16)


        .. highlight:: python
        .. code-block:: python

            # Viewing (5, 6) float array as (5, 3) complex64 array.
            a = numpy.ones((5, 6), dtype=numpy.float32)
            float_view = StridedMemoryView(a, -1)
            layout = float_view.layout
            assert layout.shape == (5, 6)
            assert layout.itemsize == 4
            complex_view = float_view.view(layout.repacked(8), numpy.complex64)
            assert complex_view.layout.shape == (5, 3)
            assert complex_view.layout.itemsize == 8
            b = numpy.from_dlpack(complex_view)
            assert b.shape == (5, 3)
        """

        if itemsize == self.itemsize:
            return self
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        if itemsize > self.itemsize:
            self.pack_into(new_layout, itemsize, data_ptr, keep_dim, axis)
        else:
            self.unpack_into(new_layout, itemsize, axis)
        return new_layout

    def max_compatible_itemsize(self : _StridedLayout, max_itemsize : int = 16, data_ptr : intptr_t = 0, axis : int = -1) -> int:
        """
        Returns the maximum itemsize (but no greater than ``max_itemsize``) that can be used
        with the :meth:`repacked` method for the current layout.
        """
        return self.get_max_compatible_itemsize(max_itemsize, data_ptr, axis)

    def sliced(self : _StridedLayout, slices : int | slice | tuple[int | slice]) -> _StridedLayout:
        """
        Returns a sliced layout.
        The ``slices`` parameter can be a single integer, a single :py:class:`slice` object
        or a tuple of integers/slices.

        .. hint::
            For convenience, instead of calling this method directly, please rely
            on the :py:meth:`~object.__getitem__` operator (i.e. bracket syntax), e.g.:
            ``layout[:, start:end:step]``.

        .. note::
            Slicing is purely a layout transformation and does not involve
            any data access.

        """
        if not isinstance(slices, tuple):
            slices = (slices,)
        cdef _StridedLayout new_layout = _StridedLayout.__new__(_StridedLayout)
        self.slice_into(new_layout, slices)
        return new_layout

    def __getitem__(self : _StridedLayout, slices : int | slice | tuple[int | slice]) -> _StridedLayout:
        return self.sliced(slices)

    cdef axes_mask_t get_flattened_axis_mask(_StridedLayout self) except? -1 nogil:
        return flattened_strides_in_c_index_order_mask(self.base)

    cdef int get_max_compatible_itemsize(_StridedLayout self, int max_itemsize, intptr_t data_ptr, int axis=-1) except -1 nogil:
        return max_compatible_itemsize(self.base, self.slice_offset, self.itemsize, max_itemsize, data_ptr, axis)

    cdef int reshape_into(_StridedLayout self, _StridedLayout out_layout, BaseLayout& new_shape) except -1 nogil:
        cdef int64_t old_volume = self.get_volume()

        validate_reshaped_shape(new_shape, old_volume)
        _zero_strides(new_shape)

        cdef BaseLayout flattened
        if old_volume != 0:
            flatten_strides_in_c_index_order(flattened, self.base, axis_mask_from_range(self.base.ndim, 0, -1))
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

    cdef int permute_into(_StridedLayout self, _StridedLayout out_layout, axis_vec_t& axis_order) except -1 nogil:
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

    cdef int flatten_into(_StridedLayout self, _StridedLayout out_layout, axes_mask_t axis_mask) except -1 nogil:
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

    cdef int squeeze_into(_StridedLayout self, _StridedLayout out_layout) except -1 nogil:
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

    cdef int unsqueeze_into(_StridedLayout self, _StridedLayout out_layout, axis_vec_t& axis_vec) except -1 nogil:
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

    cdef int broadcast_into(_StridedLayout self, _StridedLayout out_layout, BaseLayout& broadcast) except -1 nogil:
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

    cdef int pack_into(_StridedLayout self, _StridedLayout out_layout, int itemsize, intptr_t data_ptr, bint keep_dim, int axis=-1) except -1 nogil:

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

    cdef int unpack_into(_StridedLayout self, _StridedLayout out_layout, int itemsize, int axis=-1) except -1 nogil:
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

    cdef int slice_into(_StridedLayout self, _StridedLayout out_layout, tuple slices) except -1:
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

cdef inline int maybe_copy_volume(_StridedLayout out_layout, _StridedLayout in_layout) except -1 nogil:
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
    if axis == -1:
        if new_volume != old_volume:
            raise ValueError(f"The original volume {old_volume} and the new volume {new_volume} must be equal.")
    else:
        if new_volume == 0:
            raise ValueError("The -1 extent is ambiguous when the specified sub-volume is 0")
        extent = old_volume // new_volume
        if extent * new_volume != old_volume:
            raise ValueError(f"The original volume {old_volume} must be divisible by the specified sub-volume {new_volume}.")
        new_shape.shape[axis] = extent
    return 0


cdef inline axes_mask_t axis_mask_from_range(int ndim, int start_axis, int end_axis) except? -1 nogil:
    cdef axes_mask_t axes_mask = flatten_all_axes_mask(ndim)
    if start_axis == 0 and end_axis == -1:
        return axes_mask
    if not _normalize_axis(start_axis, ndim):
        raise ValueError(f"Invalid start axis: {start_axis} out of range for {ndim}D tensor")
    if not _normalize_axis(end_axis, ndim):
        raise ValueError(f"Invalid end axis: {end_axis} out of range for {ndim}D tensor")
    if start_axis > 0:
        axes_mask &= (AXES_MASK_ALL << start_axis)
    if end_axis < ndim:
        axes_mask &= (AXES_MASK_ALL >> (STRIDED_LAYOUT_MAX_NDIM - end_axis - 1))
    return axes_mask


cdef inline int flatten_strides_in_c_index_order(BaseLayout& out_layout, BaseLayout& in_layout, axes_mask_t axis_mask) except -1 nogil:
    cdef int ndim = in_layout.ndim
    if ndim == 0:
        init_base_layout(out_layout, 1)
        out_layout.shape[0] = 1
        out_layout.strides[0] = 1
        return 1
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
            and (axis_mask & _axis2mask(group_end))
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
        return flatten_all_axes_mask(layout.ndim)
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
            axis_mask |= _axis2mask(group_end)
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
        group_stride = in_strides[i]
        group_vol = 1
        while new_i >= 0:
            new_extent = out_layout.shape[new_i]
            if new_extent == 0:
                return False
            if new_extent == 1 or group_vol < extent:
                out_layout.strides[new_i] = group_stride
                group_stride = _overflow_checked_mul(group_stride, new_extent)
                group_vol = _overflow_checked_mul(group_vol, new_extent)
                new_i -= 1
            else:
                break
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
        axis_mask = _axis2mask(axis)
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
        axis_mask = _axis2mask(axis)
        if out_shape_mask & axis_mask:
            raise ValueError(f"Axis {axis} appears multiple times.")
        out_shape_mask |= axis_mask
    cdef int in_i = 0
    for i in range(out_ndim):
        axis_mask = _axis2mask(<axis_t>i)
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
    if max_itemsize < itemsize:
        raise ValueError(f"max_itemsize ({max_itemsize}) cannot be less than itemsize ({itemsize}).")
    max_itemsize = gcd(max_itemsize, _c_abs(data_ptr))
    cdef extent_t* shape = layout.shape
    cdef stride_t* strides = get_strides_ptr(layout)
    if ndim < 1 or strides[axis] != 1 or shape[axis] == 0:
        return itemsize
    max_itemsize = gcd(max_itemsize, _overflow_checked_mul(slice_offset, itemsize))
    max_itemsize = gcd(max_itemsize, _overflow_checked_mul(shape[axis], itemsize))
    for i in range(ndim):
        if i == axis:
            continue
        max_itemsize = gcd(max_itemsize, _overflow_checked_mul(_c_abs(strides[i]), itemsize))
    return max_itemsize
