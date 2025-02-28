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

    def __init__(self):
        raise NotImplementedError("TODO") # ACTNBL do not instantiate directly? FN_NOT_CALLED

    @staticmethod
    def _from_ctx(obj, dev_id):
        assert isinstance(obj, driver.CUcontext) # ACTNBL show type(obj) HAPPY_ONLY_EXERCISED
        ctx = Context.__new__(Context)
        ctx._handle = obj
        ctx._id = dev_id
        return ctx
