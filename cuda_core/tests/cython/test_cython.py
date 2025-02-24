# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import importlib
import sys


def py_func(func):
    """
    Wraps func in a plain Python function.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


cython_test_modules = [
    "test_get_cuda_native_handle",
]


for mod in cython_test_modules:
    try:
        # For each callable in `mod` with name `test_*`,
        # wrap the callable in a plain Python function
        # and set the result as an attribute of this module.
        mod = importlib.import_module(mod)
        for name in dir(mod):
            item = getattr(mod, name)
            if callable(item) and name.startswith("test_"):
                item = py_func(item)
                setattr(sys.modules[__name__], name, item)
    except ImportError:
        raise
