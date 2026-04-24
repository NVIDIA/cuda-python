# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import os
from pathlib import Path

import pytest
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

TESTS_DIR = Path(__file__).resolve().parent
CYTHON_TEST_MODULES = [
    "test_get_cuda_native_handle",
]


def _get_include_dirs():
    include_dirs = [str((TESTS_DIR / "../../cuda/core/_include").resolve())]
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home:
        include_dirs.append(str((Path(cuda_home) / "include").resolve()))
    return include_dirs


def build_cython_test_modules():
    extensions = [
        Extension(
            name=module_name,
            sources=[str(TESTS_DIR / f"{module_name}.pyx")],
            include_dirs=_get_include_dirs(),
        )
        for module_name in CYTHON_TEST_MODULES
    ]

    ext_modules = cythonize(
        extensions,
        compiler_directives={"language_level": "3", "freethreading_compatible": True},
        nthreads=1,
    )

    distribution = Distribution(
        {
            "name": "cuda-core-cython-tests",
            "ext_modules": ext_modules,
        }
    )
    build_ext = distribution.get_command_obj("build_ext")
    build_ext.inplace = True
    build_ext.build_temp = str(TESTS_DIR / "build" / "temp")

    # Ensure in-place extension outputs are written into tests/cython.
    with contextlib.chdir(TESTS_DIR):
        distribution.run_command("build_ext")


def _import_cython_test_modules():
    imported_modules = {}
    build_attempted = False

    for module_name in CYTHON_TEST_MODULES:
        try:
            imported_modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            if not build_attempted:
                build_cython_test_modules()
                importlib.invalidate_caches()
                build_attempted = True

            imported_modules[module_name] = importlib.import_module(module_name)

    return imported_modules


@pytest.fixture(scope="session")
def cython_test_modules():
    return _import_cython_test_modules()


def _make_wrapped_test(module_name, test_name):
    def wrapped(cython_test_modules):
        test_func = getattr(cython_test_modules[module_name], test_name)
        return test_func()

    wrapped.__name__ = test_name
    wrapped.__module__ = __name__
    return wrapped


registered_tests = set()
for module_name, module in _import_cython_test_modules().items():
    for test_name in dir(module):
        item = getattr(module, test_name)
        if not callable(item) or not test_name.startswith("test_"):
            continue

        if test_name in registered_tests:
            raise RuntimeError(f"duplicate cython test name discovered: {test_name}")
        registered_tests.add(test_name)
        globals()[test_name] = _make_wrapped_test(module_name, test_name)


def main():
    build_cython_test_modules()


if __name__ == "__main__":
    main()
