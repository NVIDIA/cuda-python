# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
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


cython_test_modules = ["test_ccuda", "test_ccudart", "test_interoperability_cython"]


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
