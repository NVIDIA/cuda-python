# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from enum import Enum

import numpy as np


class NamedParam:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __bool__(self):
        return bool(self.value)

    def pretty_name(self):
        if isinstance(self.value, Enum):
            value_str = self.value.name
        else:
            value_str = str(self.value)
        return f"{self.name}.{value_str}"


class DenseOrder(Enum):
    """
    Whether to initialize the dense layout in C or F order.
    For C, the strides can be explicit or implicit (None).
    """

    C = "C"
    IMPLICIT_C = "implicit_c"
    F = "F"


class _S:
    """
    SliceSpec
    """

    def __init__(self, slices=None):
        if slices is None:
            slices = []
        else:
            assert isinstance(slices, list)
        self.slices = slices

    def __getitem__(self, value):
        self.slices.append(value)
        return self


class LayoutSpec:
    """
    Pretty printable specification of a layout in a test case.
    """

    def __init__(
        self,
        shape,
        itemsize,
        stride_order=DenseOrder.C,
        perm=None,
        slices=None,
        np_ref=None,
    ):
        self.shape = shape
        self.itemsize = itemsize
        self.stride_order = stride_order
        self.perm = perm
        if slices is not None:
            assert isinstance(slices, _S)
            slices = slices.slices
        self.slices = slices
        self.np_ref = np_ref

    def pretty_name(self):
        desc = [
            f"ndim.{len(self.shape)}",
            f"shape.{self.shape}",
            f"itemsize.{self.itemsize}",
        ]
        if self.stride_order is not None:
            if isinstance(self.stride_order, DenseOrder):
                desc.append(f"stride_order.{self.stride_order.value}")
            else:
                assert isinstance(self.stride_order, tuple)
                assert len(self.stride_order) == len(self.shape)
                desc.append(f"stride_order.{self.stride_order}")
        if self.perm is not None:
            desc.append(f"perm.{self.perm}")
        if self.slices is not None:
            desc.append(f"slices.{self.slices}")
        return "-".join(desc)

    def dtype_from_itemsize(self):
        return dtype_from_itemsize(self.itemsize)

    def np_order(self):
        return "F" if self.stride_order == DenseOrder.F else "C"

    def has_no_strides(self):
        return self.stride_order == DenseOrder.IMPLICIT_C

    def has_no_strides_transformed(self):
        return self.stride_order == DenseOrder.IMPLICIT_C and self.perm is None and self.slices is None


def dtype_from_itemsize(itemsize):
    if itemsize <= 8:
        return np.dtype(f"int{itemsize * 8}")
    elif itemsize == 16:
        return np.dtype("complex128")
    else:
        raise ValueError(f"Unsupported itemsize: {itemsize}")


def pretty_name(val):
    """
    Pytest does not pretty print (repr/str) parameters of custom types.
    Use this function as the `ids` argument of `pytest.mark.parametrize`, e.g.:
    ``@pytest.mark.parametrize(..., ids=pretty_name)``
    """
    if hasattr(val, "pretty_name"):
        return val.pretty_name()
    # use default pytest pretty printing
    return None


def flatten_mask2str(mask, ndim):
    return "".join("1" if mask & (1 << i) else "0" for i in range(ndim))


def random_permutations(rng, perm_len, cutoff_len=3, sample_size=6):
    if perm_len <= cutoff_len:
        return [perm for perm in itertools.permutations(range(perm_len))]
    perms = []
    for _ in range(sample_size):
        perm = list(range(perm_len))
        rng.shuffle(perm)
        perms.append(tuple(perm))
    return perms


def inv_permutation(perm):
    inv = [None] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def permuted(t, perm):
    return tuple(t[i] for i in perm)


def long_shape(rng, ndim, num_non_unit_dims=5, max_dim_size=6):
    dims = [min(i + 2, max_dim_size) for i in range(num_non_unit_dims)]
    dims.extend(1 for i in range(ndim - num_non_unit_dims))
    rng.shuffle(dims)
    return tuple(dims)
