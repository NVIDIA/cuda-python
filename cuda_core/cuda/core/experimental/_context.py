# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass

from cuda.core.experimental._utils import driver


@dataclass
class ContextOptions:
    pass  # TODO


class Context:
    __slots__ = ("_handle", "_id")

    def __new__(self, *args, **kwargs):
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @classmethod
    def _from_ctx(cls, obj, dev_id):
        assert isinstance(obj, driver.CUcontext)
        ctx = super().__new__(cls)
        ctx._handle = obj
        ctx._id = dev_id
        return ctx
