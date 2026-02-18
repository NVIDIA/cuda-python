# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy


cdef Py_ssize_t _get_cuda_scalar_alignment(object descr):
    """Return the CUDA alignment for a scalar (non-structured) NumPy dtype.

    Uses standard C/CUDA alignment rules: alignment equals the itemsize
    for types up to 16 bytes, capped at 16.  The result is always a
    power of two.
    """
    cdef Py_ssize_t size = descr.itemsize
    # Cap at 16 (e.g. complex128 is 16 bytes, no CUDA type exceeds 16-byte alignment)
    if size > 16:
        return 16
    # Round down to nearest power of 2
    cdef Py_ssize_t alignment = 1
    while alignment * 2 <= size:
        alignment *= 2
    return alignment


def make_aligned_dtype(dtype, *, int alignment=-1, int recurse=False):
    """Create a new structured dtype with sufficient alignment for GPU use.

    Many CUDA kernels require structure members to be naturally aligned
    (each field's offset is a multiple of its own alignment) and the
    overall structure size to be a multiple of the largest member
    alignment.  :func:`make_aligned_dtype` recomputes field offsets and
    the total ``itemsize`` so that these constraints are satisfied.

    Parameters
    ----------
    dtype : numpy.dtype or dtype-like
        A NumPy dtype or anything accepted by :class:`numpy.dtype`.
        Typically a structured dtype such as
        ``numpy.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])``.
    alignment : int, optional
        Desired minimum alignment (in bytes) of the resulting dtype.
        Must be a positive power of two.  When ``-1`` (the default) the
        alignment is inferred from the fields.

        If the requested alignment is *smaller* than the minimum
        alignment inferred from the fields, a :exc:`ValueError` is
        raised.  If it is *larger*, the extra alignment is recorded in
        the dtype's ``metadata`` under the key ``"__cuda_alignment__"``
        and the ``itemsize`` is padded accordingly.

    recurse : bool, optional
        When ``True``, nested structured dtypes are recursively
        re-aligned as if ``alignment=-1`` were passed for each
        sub-structure.  When ``False`` (the default), the alignment
        of nested dtypes is taken as-is.

    Returns
    -------
    numpy.dtype
        A new structured dtype whose field offsets, ``itemsize``, and
        (optionally) metadata satisfy GPU alignment requirements.

    Raises
    ------
    ValueError
        If *alignment* is not a positive power of two, or if *alignment*
        is smaller than the minimum alignment inferred from the fields.

    Notes
    -----
    * By default this function does **not** recurse into nested
      structures.  You can nest a dtype with a specific alignment by
      creating it with :func:`make_aligned_dtype` first and then
      embedding it in an outer dtype.
    * NumPy promotion (e.g. in :func:`numpy.concatenate`) may
      "canonicalize" the dtype and drop the struct layout and alignment
      metadata.

    Examples
    --------
    >>> import numpy as np
    >>> from cuda.core.utils import make_aligned_dtype
    >>> dt = np.dtype([("x", "f4"), ("y", "i1")])
    >>> dt.itemsize
    5
    >>> aligned = make_aligned_dtype(dt)
    >>> aligned.itemsize
    8

    Requesting a specific alignment:

    >>> aligned16 = make_aligned_dtype(dt, alignment=16)
    >>> aligned16.itemsize
    16
    >>> aligned16.metadata
    {'__cuda_alignment__': 16}
    """
    cdef Py_ssize_t final_alignment = 1
    cdef Py_ssize_t itemsize = 0
    cdef Py_ssize_t curr_offset = 0
    cdef Py_ssize_t min_offset = 0
    cdef Py_ssize_t subalignment
    descr = numpy.dtype(dtype, align=True)

    if alignment != -1 and (alignment <= 0 or (alignment & (alignment - 1)) != 0):
        raise ValueError("Alignment must be a positive power of 2.")

    if descr.names is None:
        # Non-structured dtype: alignment is just the scalar alignment
        final_alignment = descr.alignment
        curr_offset = descr.itemsize
    else:
        names = []
        offsets = []
        subdtypes = []

        for name in descr.names:
            field_info = descr.fields[name]
            subdtype = field_info[0]
            offset = field_info[1]

            if offset < min_offset:
                raise ValueError(
                    "make_aligned_dtype() only supports well-behaved "
                    "in-order fields (it ignores field offsets).")

            min_offset = offset + subdtype.itemsize

            if subdtype.names is None:
                subalignment = _get_cuda_scalar_alignment(subdtype)
            elif not recurse:
                subalignment = subdtype.alignment
                if subdtype.metadata:
                    subalignment = subdtype.metadata.get(
                        "__cuda_alignment__", subalignment)
            else:
                subdtype = make_aligned_dtype(subdtype, recurse=recurse)
                subalignment = subdtype.alignment
                if subdtype.metadata:
                    subalignment = subdtype.metadata.get(
                        "__cuda_alignment__", subalignment)

            if curr_offset % subalignment != 0:
                curr_offset += subalignment - (curr_offset % subalignment)

            final_alignment = max(final_alignment, subalignment)

            names.append(name)
            subdtypes.append(subdtype)
            offsets.append(curr_offset)
            curr_offset += subdtype.itemsize

        dtype_info = dict(names=names, offsets=offsets, formats=subdtypes,
                          itemsize=itemsize)

    metadata = {}
    if alignment != -1:
        if alignment >= final_alignment:
            final_alignment = alignment
            metadata = {"metadata": {"__cuda_alignment__": alignment}}
        else:
            raise ValueError(
                f"make_aligned_dtype(): given alignment={alignment} "
                f"is smaller than minimum alignment {final_alignment}"
            )

    itemsize = (
        (curr_offset + final_alignment - 1) // final_alignment
        * final_alignment)

    if descr.names is None:
        if descr.itemsize != itemsize:
            raise ValueError(
                "Alignment larger than itemsize for non-structured dtype.")
        return numpy.dtype(descr, **metadata)
    else:
        if descr.itemsize > itemsize:
            raise ValueError(
                "Input descriptor had larger itemsize than inferred.")
        dtype_info["itemsize"] = itemsize

    return numpy.dtype(dtype_info, align=True, **metadata)
