# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import random
from enum import Enum

import numpy as np
import pytest
from cuda.core.experimental._layout import StridedLayout
from helpers.layout import (
    _S,
    DenseOrder,
    LayoutSpec,
    NamedParam,
    dtype_from_itemsize,
    flatten_mask2str,
    inv_permutation,
    long_shape,
    permuted,
    pretty_name,
    random_permutations,
)

_ITEMSIZES = [1, 2, 4, 8, 16]

py_rng = random.Random(42)


def _setup_layout_and_np_ref(spec: LayoutSpec):
    np_ref = np.arange(math.prod(spec.shape), dtype=spec.dtype_from_itemsize())

    if isinstance(spec.stride_order, DenseOrder):
        np_ref = np_ref.reshape(spec.shape, order=spec.np_order())
        if spec.stride_order == DenseOrder.IMPLICIT_C:
            layout = StridedLayout(spec.shape, None, spec.itemsize)
        else:
            layout = StridedLayout.dense(spec.shape, spec.itemsize, spec.stride_order.value)
    else:
        assert isinstance(spec.stride_order, tuple)
        assert len(spec.stride_order) == len(spec.shape)
        # numpy does not allow specyfing the tuple order (only C/F)
        np_ref = np_ref.reshape(permuted(spec.shape, spec.stride_order))
        np_ref = np_ref.transpose(inv_permutation(spec.stride_order))
        layout = StridedLayout.dense(spec.shape, spec.itemsize, spec.stride_order)
    return layout, np_ref


def _transform(layout: StridedLayout, np_ref: np.ndarray, spec: LayoutSpec):
    if spec.perm is not None:
        np_ref = np_ref.transpose(spec.perm)
        layout = layout.permuted(spec.perm)
    if spec.slices is not None:
        for sl in spec.slices:
            np_ref = np_ref[sl]
            layout = layout.sliced(sl)
    return layout, np_ref


def _cmp_layout_and_array(layout: StridedLayout, arr: np.ndarray, expect_strides_none: bool):
    """
    Compare StridedLayout and numpy.ndarray.
    Compares shape, strides, itemsize and contiguity flags.
    """
    ndim = len(arr.shape)
    assert layout.ndim == ndim
    assert layout.shape == arr.shape
    volume = math.prod(arr.shape)
    assert layout.volume == volume
    assert layout.itemsize == arr.itemsize
    assert layout.slice_offset * layout.itemsize == layout.slice_offset_in_bytes

    ref_c_contig = arr.flags["C_CONTIGUOUS"]
    ref_f_contig = arr.flags["F_CONTIGUOUS"]
    assert layout.is_contiguous_c == ref_c_contig
    assert layout.is_contiguous_f == ref_f_contig
    ref_any_contig = ref_c_contig or ref_f_contig or arr.transpose(layout.stride_order).flags["C_CONTIGUOUS"]
    assert layout.is_contiguous_any == ref_any_contig
    assert layout.is_dense == (ref_any_contig and layout.slice_offset == 0)

    if expect_strides_none:
        assert layout.strides is None
        assert layout.strides_in_bytes is None
        assert arr.flags["C_CONTIGUOUS"]
    elif math.prod(arr.shape) == 0:
        assert layout.strides_in_bytes == tuple(0 for _ in range(ndim))
    else:
        assert layout.strides_in_bytes == arr.strides


def _cmp_layout_from_dense_vs_from_np(layout: StridedLayout, np_ref: np.ndarray, has_no_strides: bool):
    """
    Compare the layout created through series of transformations vs
    the layout created from numpy.ndarray transformed accordingly.
    """

    layout_from_np = StridedLayout(np_ref.shape, np_ref.strides, np_ref.itemsize, divide_strides=True)
    assert layout_from_np.shape == layout.shape
    assert layout_from_np.itemsize == layout.itemsize
    assert layout_from_np.is_contiguous_c == layout.is_contiguous_c
    assert layout_from_np.is_contiguous_f == layout.is_contiguous_f
    assert layout_from_np.is_contiguous_any == layout.is_contiguous_any
    assert layout_from_np.is_unique == layout.is_unique
    volume = math.prod(np_ref.shape)
    assert layout_from_np.volume == layout.volume == volume

    if volume > 0:
        assert layout_from_np.stride_order == layout.stride_order

        if has_no_strides:
            assert layout_from_np.is_contiguous_c
            assert layout_from_np.is_contiguous_any
            dense_layout = StridedLayout.dense(np_ref.shape, np_ref.itemsize)
            assert layout_from_np.strides == dense_layout.strides
            assert layout_from_np.strides_in_bytes == dense_layout.strides_in_bytes
        else:
            assert layout_from_np.strides == layout.strides
            assert layout_from_np.strides_in_bytes == layout.strides_in_bytes


def _check_envelope(layout: StridedLayout, layout_spec: LayoutSpec):
    orignal_vol = math.prod(layout_spec.shape)
    min_offset, max_offset = layout.offset_bounds
    if layout.volume == 0:
        assert min_offset == 0
        assert max_offset == -1
    else:
        assert min_offset >= 0
        assert min_offset <= max_offset
        assert max_offset <= orignal_vol - 1
        if layout.is_dense:
            assert min_offset == 0
            assert max_offset == math.prod(layout.shape) - 1
        else:
            shape, strides = layout.shape, layout.strides
            ref_min_offset = ref_max_offset = layout.slice_offset
            ref_min_offset += sum(strides[i] * (shape[i] - 1) for i in range(layout.ndim) if strides[i] < 0)
            ref_max_offset += sum(strides[i] * (shape[i] - 1) for i in range(layout.ndim) if strides[i] > 0)
            assert min_offset == ref_min_offset
            assert max_offset == ref_max_offset
    assert 0 <= layout.required_size_in_bytes() <= orignal_vol * layout_spec.itemsize
    assert layout.required_size_in_bytes() == (max_offset + 1) * layout.itemsize


def _cmp_slice_offset(
    layout_0: StridedLayout,
    layout_1: StridedLayout,
    np_ref_0: np.ndarray,
    np_ref_1: np.ndarray,
):
    # cannot access numpy's scalar data pointer
    if layout_1.ndim > 0:
        ref_offset = np_ref_1.ctypes.data - np_ref_0.ctypes.data
        layout_offset = layout_1.slice_offset_in_bytes - layout_0.slice_offset_in_bytes
        assert layout_offset == ref_offset


@pytest.mark.parametrize(
    "layout_spec",
    [
        LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order)
        for shape in [tuple(), (5,), (7, 9), (2, 3, 4)]
        for stride_order in random_permutations(py_rng, len(shape))
    ],
    ids=pretty_name,
)
def test_dense_with_permutation_as_stride_order(layout_spec):
    """
    Test creating StridedLayout with stride_order=tuple(...).
    """
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, False)
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, False)
    _check_envelope(layout, layout_spec)
    assert layout.stride_order == layout_spec.stride_order


@pytest.mark.parametrize(
    "layout_spec",
    [
        LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order, perm=permutation)
        for shape in [
            tuple(),
            (1,),
            (2, 3),
            (5, 6, 7),
            (5, 1, 7),
            (5, 2, 3, 4),
            long_shape(py_rng, 64),
        ]
        for permutation in random_permutations(py_rng, len(shape), sample_size=3)
        for stride_order in list(DenseOrder)
    ],
    ids=pretty_name,
)
def test_permuted(layout_spec):
    """
    Test creating StridedLayout with dense(C/F) order or implict C order
    StridedLayout(strides=None) and calling permuted(perm) on it.
    Tests against numpy transpose and checks stride_order attribute.
    """
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())
    _check_envelope(layout, layout_spec)
    unit_dims_count = sum(dim == 1 for dim in np_ref.shape)
    if unit_dims_count <= 1:
        # stride order with multiple unit dimensions is not unique
        # a simple equality check won't do
        expected_order = inv_permutation(layout_spec.perm)
        if layout_spec.np_order() == "F":
            expected_order = tuple(reversed(expected_order))
        assert layout.stride_order == expected_order


class PermutedErr(Enum):
    REPEATED_AXIS = "axis -?\\d+ appears multiple times"
    OUT_OF_RANGE = "axis -?\\d+ out of range for"
    WRONG_LEN = "the same length as the number of dimensions"


@pytest.mark.parametrize(
    ("layout_spec", "error_msg"),
    [
        (
            LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order, perm=permutation),
            error_msg,
        )
        for shape, permutation, error_msg in [
            (tuple(), (5,), PermutedErr.WRONG_LEN),
            ((1,), (0, 0), PermutedErr.WRONG_LEN),
            ((2, 5, 3), (1, 0, 1), PermutedErr.REPEATED_AXIS),
            ((5, 6, 7), (1, 3, 0), PermutedErr.OUT_OF_RANGE),
            ((5, 6, 7), (1, -2000, 0), PermutedErr.OUT_OF_RANGE),
        ]
        for stride_order in list(DenseOrder)
    ],
    ids=pretty_name,
)
def test_permuted_validation(layout_spec, error_msg):
    layout, _ = _setup_layout_and_np_ref(layout_spec)
    with pytest.raises(ValueError, match=error_msg.value):
        layout.permuted(layout_spec.perm)


class SliceErr(Enum):
    ZERO_STEP = "slice step cannot be zer"
    TOO_MANY_SLICES = "is greater than the number of dimensions"
    OUT_OF_RANGE = "out of range for axis"
    TYPE_ERROR = "Expected slice instance or integer."


@pytest.mark.parametrize(
    ("layout_spec", "error_msg"),
    [
        (
            LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order, slices=slices),
            error_msg,
        )
        for shape, slices, error_msg in [
            (tuple(), _S(), None),
            ((12,), _S()[:], None),
            ((13,), _S()[::-1], None),
            ((13,), _S()[::-1][::-1], None),
            ((13,), _S()[::-1][1:-1][::-1], None),
            ((13,), _S()[2:-3], None),
            ((13,), _S()[2:-3:2], None),
            ((13,), _S()[-3:2:-2], None),
            ((13,), _S()[-3:2:-2][1:3], None),
            ((3, 5), _S()[:2][:, 3:], None),
            ((3, 5), _S()[5:4], None),
            ((3, 5), _S()[:, ::0], SliceErr.ZERO_STEP),
            ((3, 5), _S()[:, :-1, :2], SliceErr.TOO_MANY_SLICES),
            ((11, 12, 3), _S()[:, 0, :-1], None),
            ((11, 12, 3), _S()[0, 1, :-1], None),
            ((11, 12, 3, 5), _S()[0][1], None),
            ((11, 12, 3, 5), _S()[:, 1, :-1], None),
            ((11, 12, 3), _S()[0, 1, 2], None),
            ((11, 12, 3), _S()[0, 1, 5], SliceErr.OUT_OF_RANGE),
            ((11, 12, 3), _S()[-2], None),
            ((11, 12, 3), _S()[-42], SliceErr.OUT_OF_RANGE),
            ((11, 12, 3), _S()["abc"], SliceErr.TYPE_ERROR),
            (long_shape(py_rng, 64), _S([slice(None, None, -1)] * 64), None),
        ]
        for stride_order in list(DenseOrder)
    ],
    ids=pretty_name,
)
def test_slice(layout_spec, error_msg):
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())

    if error_msg is None:
        for sl in layout_spec.slices:
            sliced = layout[sl]
            sliced_ref = np_ref[sl]
            _cmp_layout_and_array(sliced, sliced_ref, False)
            _cmp_layout_from_dense_vs_from_np(sliced, sliced_ref, False)
            _cmp_slice_offset(layout, sliced, np_ref, sliced_ref)
            _check_envelope(sliced, layout_spec)
            layout = sliced
            np_ref = sliced_ref
    else:
        error_cls = TypeError if error_msg == SliceErr.TYPE_ERROR else ValueError
        with pytest.raises(error_cls, match=error_msg.value):
            for sl in layout_spec.slices:
                layout[sl]


class ReshapeErr(Enum):
    VOLUME_MISMATCH = "The original volume \\d+ and the new volume \\d+ must be equal."
    NEG_EXTENT = "Extents must be non-negative"
    MULTI_NEG_EXTENTS = "There can be at most one -1 extent in a shape"
    AMBIGUOUS_NEG_EXTENT = "The -1 extent is ambiguous when the specified sub-volume is 0"
    DIVISIBILITY_VIOLATION = "The original volume \\d+ must be divisible by the specified sub-volume \\d+"
    STRIDE = "Layout strides are incompatible with the new shape"
    TYPE_ERROR = None


@pytest.mark.parametrize(
    ("layout_spec", "new_shape", "error_msg"),
    [
        (
            LayoutSpec(
                shape,
                py_rng.choice(_ITEMSIZES),
                stride_order,
                perm=permutation,
                slices=slices,
            ),
            NamedParam("new_shape", new_shape),
            error_msg,
        )
        for shape, permutation, slices, new_shape, error_msg in [
            (tuple(), None, None, tuple(), None),
            (tuple(), None, None, (1,), None),
            (tuple(), None, None, (-1,), None),
            (tuple(), None, None, (1, -1, 1), None),
            ((1,), None, None, (-1,), None),
            ((1,), None, None, tuple(), None),
            ((12,), None, _S()[:], (12,), None),
            ((12,), None, None, (11,), ReshapeErr.VOLUME_MISMATCH),
            ((12,), None, _S()[1:], (11,), None),
            ((0,), None, None, (0,), None),
            ((0,), None, None, (1, 3), ReshapeErr.VOLUME_MISMATCH),
            ((3,), None, _S()[3:], (3,), ReshapeErr.VOLUME_MISMATCH),
            ((18,), None, None, (0,), ReshapeErr.VOLUME_MISMATCH),
            ((3,), None, _S()[2:-1], (0,), None),
            ((3,), None, _S()[3:], (-1,), None),
            ((0,), None, None, (1, -1), None),
            ((0,), None, None, (0, -1), ReshapeErr.AMBIGUOUS_NEG_EXTENT),
            ((3, 0, 3), None, None, (2, 3, 4, 5, 6, 7, 0, 12), None),
            ((3, 0, 3), None, None, (0,), None),
            ((12,), None, None, (2, 3, 2), None),
            ((12,), None, None, (2, 6), None),
            ((12,), None, None, (4, 3), None),
            ((12,), None, None, (3, 4), None),
            ((7, 12), None, None, (7, 12), None),
            ((7, 12), None, None, (12, 7), None),
            ((12, 11), None, None, (2, 3, 2, 11), None),
            ((12, 11), None, None, (2, 3, 11, 2), None),
            ((12, 11), None, None, (2, 11, 3, 2), None),
            ((12, 11), None, None, (11, 2, 3, 2), None),
            ((12, 11), None, None, (2, 3, 2, -1), None),
            ((12, 11), None, None, (2, 3, -1, 2), None),
            ((12, 11), None, None, (2, -1, 3, 2), None),
            ((12, 11), None, None, (-1, 2, 3, 2), None),
            ((12, 11), None, None, (2, 3, -1, 11), None),
            ((12, 11), None, None, (2, 3, 11, -1), None),
            ((12, 11), None, None, (-1, 11, 3, 2), None),
            ((12, 11), None, None, (11, 2, -1, 2), None),
            ((5, 12), None, None, (2, 5, 6), None),
            ((2, 3, 2), None, None, (12,), None),
            ((2, 3, 2), None, None, (6, 2), None),
            ((2, 3, 2), None, None, (2, 3, 2), None),
            ((2, 3, 2), (1, 2, 0), None, (6, 2), None),
            ((2, 3, 2), (1, 2, 0), None, (2, 6), ReshapeErr.STRIDE),
            ((2, 3, 2), (1, 2, 0), None, (12,), ReshapeErr.STRIDE),
            ((2, 3, 2), (1, 0, 2), None, (3, 2, 2), None),
            ((2, 3, 2), (1, 0, 2), None, (3, 4), ReshapeErr.STRIDE),
            ((2, 3, 2), (1, 0, 2), None, (6, 2), ReshapeErr.STRIDE),
            ((2, 3, 2), (1, 0, 2), None, (12,), ReshapeErr.STRIDE),
            ((10, 10, 10), None, _S()[::-1, ::-1, :], (10, 10, 10), None),
            ((10, 10, 10), None, _S()[::-1, ::-1, ::-1], (1000,), None),
            ((10, 10, 10), None, _S()[::-1, ::-1, :], (100, 10), None),
            ((10, 10, 10), None, _S()[::-1, ::-1, :], (10, 100), ReshapeErr.STRIDE),
            ((10, 10, 10), None, _S()[:, :, ::-1], (100, 10), None),
            ((10, 10, 10), None, _S()[:, :, ::-1], (10, 100), ReshapeErr.STRIDE),
            ((10, 10, 10), None, _S()[::-1, :, ::-1], (1000,), ReshapeErr.STRIDE),
            ((10, 10, 10), (1, 0, 2), _S()[::-1, ::-1], (100, 10), ReshapeErr.STRIDE),
            ((5, 3), None, _S()[:-1, :], (12,), None),
            ((13, 3), None, _S()[1:, :], (6, 6), None),
            ((12, 4), None, _S()[:, :-1], (6, 6), ReshapeErr.STRIDE),
            ((12, 4), None, _S()[:, :-1], (6, 2, 3), None),
            ((7, 6, 5), None, None, (70, -1), None),
            ((7, 6, 5), None, None, (-1, 70), None),
            ((7, 6, 5), None, None, (71, -1), ReshapeErr.DIVISIBILITY_VIOLATION),
            ((7, 6, 5), None, None, (-1, 71), ReshapeErr.DIVISIBILITY_VIOLATION),
            ((7, 6, 5), None, None, (71, -2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), None, None, (-2, 71), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), None, None, (-1, 6, -1), ReshapeErr.MULTI_NEG_EXTENTS),
            ((7, 6, 5), None, None, (-2, -1, -1), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), None, None, (-2, -1, -2), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), None, None, (-7, 6, -5), ReshapeErr.NEG_EXTENT),
            ((7, 6, 5), None, None, (5, 0, -1), ReshapeErr.AMBIGUOUS_NEG_EXTENT),
            ((7, 0, 5), None, None, (5, 0, -1), ReshapeErr.AMBIGUOUS_NEG_EXTENT),
            ((7, 6, 5), None, None, map, ReshapeErr.TYPE_ERROR),
            # random 64-dim shape with 5 non-unit extents 2, 3, 4, 5, 6
            (long_shape(py_rng, 64, 5, 6), None, None, (60, 12), None),
        ]
        for stride_order in [DenseOrder.C, DenseOrder.IMPLICIT_C]
    ],
    ids=pretty_name,
)
def test_reshape(layout_spec, new_shape, error_msg):
    new_shape = new_shape.value
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())

    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    if error_msg is None:
        reshaped = layout.reshaped(new_shape)
        reshaped_ref = np_ref.reshape(new_shape, copy=False)
        _cmp_layout_and_array(reshaped, reshaped_ref, False)
        _cmp_layout_from_dense_vs_from_np(reshaped, reshaped_ref, False)
        _check_envelope(reshaped, layout_spec)
    else:
        # sanity check that numpy is not able to reshape without
        # a copy as well
        if error_msg == ReshapeErr.STRIDE:
            with pytest.raises(ValueError):
                np_ref.reshape(new_shape, copy=False)

        error_cls = TypeError if error_msg == ReshapeErr.TYPE_ERROR else ValueError
        msg = None if error_msg == ReshapeErr.TYPE_ERROR else error_msg.value
        with pytest.raises(error_cls, match=msg):
            layout.reshaped(new_shape)


@pytest.mark.parametrize(
    (
        "layout_spec",
        "expected_shape",
        "expected_strides",
        "expected_axis_mask",
    ),
    [
        (
            LayoutSpec(
                shape,
                py_rng.choice(_ITEMSIZES),
                stride_order,
                perm=permutation,
                slices=slices,
            ),
            NamedParam("expected_shape", expected_shape),
            NamedParam("expected_strides", expected_strides),
            NamedParam("expected_axis_mask", expected_axis_mask),
        )
        for shape, permutation, slices, expected_shape, expected_strides, expected_axis_mask in [
            (tuple(), None, None, (1,), (1,), ""),
            ((12,), None, _S()[:], (12,), (1,), "0"),
            ((1, 2, 3, 4, 5), None, None, (120,), (1,), "01111"),
            ((1, 2, 3, 0, 5), None, None, (0,), (0,), "01111"),
            ((5, 1, 2, 4, 3), None, _S()[:, :, :, :, ::-2], (40, 2), (3, -2), "01110"),
            ((5, 2, 4, 3), None, _S()[:, ::-1, :, :], (5, 2, 12), (24, -12, 1), "0001"),
            ((5, 7, 4, 3), None, _S()[:, ::-1, ::-1], (5, 28, 3), (84, -3, 1), "0010"),
            ((5, 4, 3, 7), (2, 3, 0, 1), _S()[:], (21, 20), (1, 21), "0101"),
            ((5, 4, 3, 7), (3, 2, 0, 1), None, (7, 3, 20), (1, 7, 21), "0001"),
            # random 64-dim shape with 4 non-unit extents 2, 3, 4, 5
            (long_shape(py_rng, 64, 4, 5), None, None, (120,), (1,), "0" + "1" * 63),
        ]
        for stride_order in [DenseOrder.C, DenseOrder.IMPLICIT_C]
    ],
    ids=pretty_name,
)
def test_flatten(
    layout_spec,
    expected_shape,
    expected_strides,
    expected_axis_mask,
):
    expected_shape = expected_shape.value
    expected_strides = expected_strides.value
    expected_axis_mask = expected_axis_mask.value

    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())

    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())
    _check_envelope(layout, layout_spec)

    mask = flatten_mask2str(layout.flattened_axis_mask(), layout.ndim)
    assert mask == expected_axis_mask

    flattened = layout.flattened()
    assert flattened.shape == expected_shape
    assert flattened.strides == expected_strides
    assert flattened.itemsize == layout_spec.itemsize
    assert flattened.slice_offset == layout.slice_offset

    # cannot be flattened any further
    assert flattened.flattened_axis_mask() == 0


@pytest.mark.parametrize(
    (
        "layout_spec_0",
        "layout_spec_1",
        "expected_layout_spec_0",
        "expected_layout_spec_1",
    ),
    [
        (
            layout_spec_0,
            layout_spec_1,
            expected_layout_spec_0,
            expected_layout_spec_1,
        )
        for layout_spec_0, layout_spec_1, expected_layout_spec_0, expected_layout_spec_1 in [
            (
                LayoutSpec(tuple(), 2, DenseOrder.C),
                LayoutSpec(tuple(), 4, DenseOrder.C),
                LayoutSpec((1,), 2, DenseOrder.C),
                LayoutSpec((1,), 4, DenseOrder.C),
            ),
            (
                LayoutSpec(tuple(), 2, DenseOrder.IMPLICIT_C),
                LayoutSpec(tuple(), 4, DenseOrder.IMPLICIT_C),
                LayoutSpec((1,), 2, DenseOrder.C),
                LayoutSpec((1,), 4, DenseOrder.C),
            ),
            (
                LayoutSpec((2, 7, 13, 5), 8, DenseOrder.C),
                LayoutSpec((3, 5, 11, 1), 4, DenseOrder.C),
                LayoutSpec((910,), 8, DenseOrder.C),
                LayoutSpec((165,), 4, DenseOrder.C),
            ),
            (
                LayoutSpec((2, 7, 13, 5), 8, DenseOrder.IMPLICIT_C),
                LayoutSpec((3, 5, 11, 1), 4, DenseOrder.IMPLICIT_C),
                LayoutSpec((910,), 8, DenseOrder.C),
                LayoutSpec((165,), 4, DenseOrder.C),
            ),
            (
                LayoutSpec((5, 7, 13, 2), 4, (3, 1, 2, 0)),
                LayoutSpec((3, 5, 11, 1), 2, DenseOrder.IMPLICIT_C),
                LayoutSpec((5, 91, 2), 4, (2, 1, 0)),
                LayoutSpec((3, 55, 1), 2, DenseOrder.C),
            ),
            (
                LayoutSpec((2, 7, 13, 5), 16, DenseOrder.C),
                LayoutSpec((11, 1, 3, 5), 1, (2, 3, 0, 1)),
                LayoutSpec((14, 65), 16, DenseOrder.C),
                LayoutSpec((11, 15), 1, (1, 0)),
            ),
            (
                LayoutSpec(
                    (4, 5, 11, 2, 3, 7),
                    4,
                    (5, 3, 4, 0, 1, 2),
                ),
                LayoutSpec(
                    (3, 8, 5, 6, 7, 9),
                    4,
                    (0, 1, 3, 4, 5, 2),
                ),
                LayoutSpec((20, 11, 6, 7), 4, (3, 2, 0, 1)),
                LayoutSpec((24, 5, 42, 9), 4, (0, 2, 3, 1)),
            ),
        ]
    ],
    ids=pretty_name,
)
def test_flatten_together(
    layout_spec_0,
    layout_spec_1,
    expected_layout_spec_0,
    expected_layout_spec_1,
):
    layouts = []
    for layout_spec in [
        layout_spec_0,
        layout_spec_1,
        expected_layout_spec_0,
        expected_layout_spec_1,
    ]:
        layout, np_ref = _setup_layout_and_np_ref(layout_spec)
        layout, np_ref = _transform(layout, np_ref, layout_spec)
        _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
        _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())
        layouts.append(layout)

    layout_0, layout_1, expected_layout_0, expected_layout_1 = layouts

    mask_0 = layout_0.flattened_axis_mask()
    mask_1 = layout_1.flattened_axis_mask()
    mask = mask_0 & mask_1

    flattened_0 = layout_0.flattened(mask=mask)
    flattened_1 = layout_1.flattened(mask=mask)
    _check_envelope(flattened_0, layout_spec_0)
    _check_envelope(flattened_1, layout_spec_1)

    for flattened, expected_layout in zip([flattened_0, flattened_1], [expected_layout_0, expected_layout_1]):
        assert flattened == expected_layout
        assert flattened.shape == expected_layout.shape
        assert flattened.strides == expected_layout.strides
        assert flattened.itemsize == expected_layout.itemsize
        assert flattened.slice_offset == expected_layout.slice_offset
        assert flattened.is_contiguous_c == expected_layout.is_contiguous_c
        assert flattened.is_contiguous_f == expected_layout.is_contiguous_f
        assert flattened.is_contiguous_any == expected_layout.is_contiguous_any
        assert flattened.is_unique == expected_layout.is_unique


@pytest.mark.parametrize(
    ("layout_spec",),
    [
        (
            LayoutSpec(
                shape,
                py_rng.choice(_ITEMSIZES),
                stride_order,
                perm=permutation,
                slices=slices,
            ),
        )
        for shape, permutation, slices in [
            (tuple(), None, None),
            ((12,), None, None),
            ((1, 5, 4, 3), None, None),
            ((1, 5, 1, 4, 3), None, _S()[:, -1:, :]),
            ((1, 5, 4, 3), None, _S()[:, -1:, :1, 1:2]),
            ((7, 5, 3), (2, 0, 1), _S()[::-1, 3:2:-1, :]),
            ((7, 5, 3), (2, 0, 1), _S()[:, 3:2, :]),
            (long_shape(py_rng, 64, 1), None, None),
            (long_shape(py_rng, 33, 3), None, None),
        ]
        for stride_order in list(DenseOrder)
    ],
    ids=pretty_name,
)
def test_squeezed(layout_spec):
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    squeezed = layout.squeezed()
    squeezed_ref = np_ref.squeeze()
    if math.prod(np_ref.shape) != 0:
        _cmp_layout_and_array(squeezed, squeezed_ref, False)
        _cmp_layout_from_dense_vs_from_np(squeezed, squeezed_ref, False)
    else:
        assert squeezed.shape == (0,)
        assert squeezed.strides == (0,)
    assert squeezed.slice_offset == layout.slice_offset
    _check_envelope(squeezed, layout_spec)


@pytest.mark.parametrize(
    (
        "layout_spec",
        "axes",
    ),
    [
        (
            LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order, slices=slices),
            NamedParam("axes", axes),
        )
        for shape, slices in [
            (tuple(), None),
            ((7,), None),
            ((4, 5, 7, 11), _S()[1:-1, ::-1, 2:-1, ::3]),
        ]
        for stride_order in list(DenseOrder)
        for num_axes in range(3)
        for axes in itertools.combinations(list(range(len(shape) + num_axes)), num_axes)
    ],
    ids=pretty_name,
)
def test_unsqueezed_layout(layout_spec, axes):
    axes = tuple(axes.value)

    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    unsqueezed = layout.unsqueezed(axes)
    unsqueezed_ref = np.expand_dims(np_ref, axis=axes)
    # the implicit C layout is kept if the original layout has such strides
    # and there are no actual transformations along the way: no slices
    # and unsqueezing with empty axes tuple
    has_no_strides = layout_spec.has_no_strides_transformed() and len(axes) == 0
    _cmp_layout_and_array(unsqueezed, unsqueezed_ref, has_no_strides)
    _cmp_layout_from_dense_vs_from_np(unsqueezed, unsqueezed_ref, has_no_strides)
    _check_envelope(unsqueezed, layout_spec)


@pytest.mark.parametrize(
    (
        "layout_spec",
        "axis",
        "expected_max_itemsize",
        "new_itemsize",
    ),
    [
        (
            LayoutSpec(shape, itemsize, stride_order, perm=permutation, slices=slices),
            NamedParam("axis", axis),
            NamedParam("expected_max_itemsize", expected_max_itemsize),
            NamedParam("new_itemsize", new_itemsize),
        )
        for shape, permutation, slices, stride_order, itemsize, axis, expected_max_itemsize, new_itemsize in [
            ((12,), None, None, DenseOrder.C, 1, -1, 4, 1),
            ((12,), None, None, DenseOrder.IMPLICIT_C, 1, -1, 4, 1),
            ((12,), None, None, DenseOrder.F, 1, 0, 4, 1),
            ((12,), None, None, DenseOrder.C, 4, -1, 16, 8),
            ((12,), None, None, DenseOrder.IMPLICIT_C, 4, -1, 16, 8),
            ((12,), None, None, DenseOrder.F, 4, 0, 16, 8),
            ((16, 5, 4, 6), None, None, DenseOrder.C, 2, -1, 4, 4),
            ((16, 5, 4, 6), None, None, DenseOrder.IMPLICIT_C, 2, -1, 4, 4),
            ((16, 5, 4, 6), None, None, DenseOrder.F, 2, 0, 16, 4),
            ((11, 5, 9), None, _S()[:, :, -1:], DenseOrder.C, 2, 2, 2, 2),
            ((11, 5, 9), None, _S()[:, :, -1:], DenseOrder.IMPLICIT_C, 2, 2, 2, 2),
            ((11, 5, 9), None, _S()[:, :, -1:], DenseOrder.F, 2, 0, 2, 2),
            ((12, 3, 24), (1, 2, 0), _S()[::-1, 20:, 1:], DenseOrder.C, 2, 1, 8, 8),
            ((12, 3, 24), (1, 2, 0), _S()[1:, ::-1, 10:], DenseOrder.F, 2, 2, 4, 4),
            ((1, 3) + (1,) * 61 + (4,), None, None, DenseOrder.C, 2, -1, 8, 8),
            ((1, 3) + (1,) * 61 + (4,), None, None, DenseOrder.IMPLICIT_C, 2, -1, 8, 4),
            ((4, 3) + (1,) * 61 + (3,), None, None, DenseOrder.F, 2, 0, 8, 4),
        ]
    ],
    ids=pretty_name,
)
def test_packed_unpacked(
    layout_spec,
    axis,
    expected_max_itemsize,
    new_itemsize,
):
    axis = axis.value
    expected_max_itemsize = expected_max_itemsize.value
    new_itemsize = new_itemsize.value

    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    assert layout.max_compatible_itemsize(axis=axis) == expected_max_itemsize
    packed = layout.repacked(new_itemsize, axis=axis)
    # numpy does not allow specifying the axis to repack,
    # so we need to transpose the array
    packed_ref = (
        np_ref.transpose(layout.stride_order)
        .view(dtype=dtype_from_itemsize(new_itemsize))
        .transpose(inv_permutation(layout.stride_order))
    )
    has_no_strides = layout_spec.has_no_strides_transformed() and layout.itemsize == new_itemsize
    _cmp_layout_and_array(packed, packed_ref, has_no_strides)
    _cmp_layout_from_dense_vs_from_np(packed, packed_ref, has_no_strides)
    _check_envelope(packed, layout_spec)
    vec_size = new_itemsize // layout.itemsize
    assert packed.slice_offset * vec_size == layout.slice_offset
    unpacked = packed.repacked(layout.itemsize, axis=axis)
    _cmp_layout_and_array(unpacked, np_ref, has_no_strides)
    _cmp_layout_from_dense_vs_from_np(unpacked, np_ref, has_no_strides)
    _check_envelope(unpacked, layout_spec)


@pytest.mark.parametrize(
    (
        "layout_spec",
        "new_shape",
    ),
    [
        (
            LayoutSpec(shape, py_rng.choice(_ITEMSIZES), stride_order, slices=slices),
            NamedParam("new_shape", new_shape),
        )
        for shape, slices, new_shape in [
            (tuple(), None, tuple()),
            (tuple(), None, (1,)),
            (tuple(), None, (17, 1, 5)),
            ((1,), None, (5,)),
            ((1,), None, (3, 5, 2)),
            ((7,), None, (7,)),
            ((7,), None, (2, 7)),
            ((5, 11), _S()[1:-1, ::-1], (3, 11)),
            ((5, 11), _S()[1:-1, ::-1], (7, 3, 11)),
            ((5, 11), _S()[::-1, 3:4], (5, 7)),
            ((5, 11), _S()[::-1, 3:4], (5, 30)),
            ((5, 11), _S()[::-1, 3:4], (4, 5, 12)),
            ((5, 11), _S()[-1:,], (4, 13, 11)),
            ((2, 3, 3), _S()[:, 1:2], (401, 3) + (1,) * 59 + (2, 4, 3)),
        ]
        for stride_order in list(DenseOrder)
    ],
    ids=pretty_name,
)
def test_broadcast_layout(
    layout_spec,
    new_shape,
):
    new_shape = new_shape.value
    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    broadcasted = layout.broadcast_to(new_shape)
    broadcasted_ref = np.broadcast_to(np_ref, new_shape)
    _cmp_layout_and_array(broadcasted, broadcasted_ref, False)
    _cmp_layout_from_dense_vs_from_np(broadcasted, broadcasted_ref, False)
    _check_envelope(broadcasted, layout_spec)
    assert layout.is_unique
    ndim_diff = len(broadcasted_ref.shape) - len(np_ref.shape)
    expect_unique = all(broadcasted_ref.shape[i] == 1 for i in range(ndim_diff))
    expect_unique = expect_unique and all(
        broadcasted_ref.shape[i + ndim_diff] == np_ref.shape[i] for i in range(len(np_ref.shape))
    )
    assert broadcasted.is_unique is expect_unique


@pytest.mark.parametrize(
    (
        "layout_spec",
        "new_stride_order",
    ),
    [
        (
            LayoutSpec(
                shape,
                py_rng.choice(_ITEMSIZES),
                stride_order,
                perm=permutation,
                slices=slices,
            ),
            NamedParam("new_stride_order", new_stride_order),
        )
        for shape, permutation, slices in [
            (tuple(), None, None),
            ((1,), None, None),
            ((7,), None, None),
            ((7,), None, _S()[3:6]),
            ((7,), None, _S()[::-1]),
            ((5, 11), None, None),
            ((5, 11), None, _S()[1:-1]),
            ((5, 11), None, _S()[::-1, 3:10]),
            ((5, 11), None, _S()[1:4, ::-1]),
            ((5, 11), None, _S()[-1:,]),
            ((3, 5, 7), (1, 0, 2), None),
        ]
        for stride_order in list(DenseOrder)
        for new_stride_order in ["C", "F", "K"] + random_permutations(py_rng, len(shape))
    ],
    ids=pretty_name,
)
def test_to_dense(layout_spec, new_stride_order):
    new_stride_order = new_stride_order.value

    layout, np_ref = _setup_layout_and_np_ref(layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides())
    layout, np_ref = _transform(layout, np_ref, layout_spec)
    _cmp_layout_and_array(layout, np_ref, layout_spec.has_no_strides_transformed())
    _cmp_layout_from_dense_vs_from_np(layout, np_ref, layout_spec.has_no_strides_transformed())

    if isinstance(new_stride_order, str):
        if new_stride_order == "K":
            is_noop = layout.slice_offset == 0 and layout.is_contiguous_any
        elif new_stride_order == "C":
            is_noop = layout.slice_offset == 0 and layout.is_contiguous_c
        elif new_stride_order == "F":
            is_noop = layout.slice_offset == 0 and layout.is_contiguous_f
        else:
            raise AssertionError(f"Invalid new_stride_order: {new_stride_order}")
        has_no_strides = layout_spec.has_no_strides_transformed() and is_noop
        dense = layout.to_dense(new_stride_order)
        dense_ref = np_ref.copy(order=new_stride_order)
        _cmp_layout_and_array(dense, dense_ref, has_no_strides)
        _cmp_layout_from_dense_vs_from_np(dense, dense_ref, has_no_strides)
    else:
        assert isinstance(new_stride_order, tuple)
        assert len(new_stride_order) == len(layout.shape)
        dense = layout.to_dense(new_stride_order)
        dense_ref = np_ref.transpose(new_stride_order).copy(order="C").transpose(inv_permutation(new_stride_order))
        _cmp_layout_and_array(dense, dense_ref, False)
        _cmp_layout_from_dense_vs_from_np(dense, dense_ref, False)

    assert dense.is_dense
    assert dense.required_size_in_bytes() == np_ref.size * layout.itemsize
    assert dense.offset_bounds == (0, np_ref.size - 1)
