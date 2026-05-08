# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def python_property(fget=None, fset=None, fdel=None, doc=None):
    """Create a Python property without Cython's cdef-class @property lowering."""
    return property(fget, fset, fdel, doc)
