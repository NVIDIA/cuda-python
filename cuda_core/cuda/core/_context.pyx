# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.core._utils.cuda_utils import driver


@dataclass
class ContextOptions:
    pass  # TODO


cdef class Context:

    cdef:
        readonly object _handle
        int _device_id

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @classmethod
    def _from_ctx(cls, handle: driver.CUcontext, int device_id):
        cdef Context ctx = Context.__new__(Context)
        ctx._handle = handle
        ctx._device_id = device_id
        return ctx

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        return int(self._handle) == int(_other._handle)

    def __hash__(self) -> int:
        return hash(int(self._handle))
