# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import importlib
import os
import re
from pathlib import Path

import pytest
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

TESTS_DIR = Path(__file__).resolve().parent
CYTHON_TEST_MODULES = ["test_ccuda", "test_ccudart", "test_interoperability_cython"]
TEST_NAME_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(")


def _get_cuda_include_dir():
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_home:
        return Path(cuda_home) / "include"
    return None


def build_cython_test_modules():
    include_dirs = []
    cuda_include_dir = _get_cuda_include_dir()
    if cuda_include_dir:
        include_dirs.append(str(cuda_include_dir))

    extensions = [
        Extension(
            name=module_name,
            sources=[str(TESTS_DIR / f"{module_name}.pyx")],
            include_dirs=include_dirs,
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
            "name": "cuda-bindings-cython-tests",
            "ext_modules": ext_modules,
        }
    )
    build_ext = distribution.get_command_obj("build_ext")
    build_ext.inplace = True
    build_ext.build_temp = str(TESTS_DIR / "build" / "temp")
    distribution.run_command("build_ext")


def _import_cython_test_modules(rebuild_if_needed):
    imported_modules = {}
    build_attempted = False

    for module_name in CYTHON_TEST_MODULES:
        try:
            imported_modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            if not rebuild_if_needed:
                raise

            if not build_attempted:
                build_cython_test_modules()
                importlib.invalidate_caches()
                build_attempted = True

            imported_modules[module_name] = importlib.import_module(module_name)

    return imported_modules


def _discover_test_function_names(module_name):
    module_source = TESTS_DIR / f"{module_name}.pyx"
    test_names = []

    with module_source.open(encoding="utf-8") as f:
        for line in f:
            match = TEST_NAME_RE.match(line)
            if match:
                test_names.append(match.group(1))

    return test_names


@pytest.fixture(scope="session")
def cython_test_modules():
    return _import_cython_test_modules(rebuild_if_needed=True)


def _make_wrapped_test(module_name, test_name):
    def wrapped(cython_test_modules):
        test_func = getattr(cython_test_modules[module_name], test_name)
        return test_func()

    wrapped.__name__ = test_name
    wrapped.__module__ = __name__
    return wrapped


registered_tests = set()
for module_name in CYTHON_TEST_MODULES:
    for test_name in _discover_test_function_names(module_name):
        if test_name in registered_tests:
            raise RuntimeError(f"duplicate cython test name discovered: {test_name}")
        registered_tests.add(test_name)
        globals()[test_name] = _make_wrapped_test(module_name, test_name)


def main():
    build_cython_test_modules()


if __name__ == "__main__":
    main()
