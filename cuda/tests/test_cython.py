# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
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


cython_test_modules = ["cuda.tests.test_ccuda",
                       "cuda.tests.test_ccudart",
                       "cuda.tests.test_interoperability_cython"]


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
