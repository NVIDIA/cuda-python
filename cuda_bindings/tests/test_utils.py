# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

from cuda.bindings import driver, runtime
from cuda.bindings._internal.utils import get_c_compiler
from cuda.bindings.utils import (
    check_nvvm_compiler_options,
    get_cuda_native_handle,
    get_minimal_required_cuda_ver_from_ptx_ver,
    get_ptx_ver,
)

have_cufile = importlib.util.find_spec("cuda.bindings.cufile") is not None


def _is_libnvvm_available() -> bool:
    from cuda.bindings._internal.nvvm import _inspect_function_pointer
    from cuda.pathfinder import DynamicLibNotFoundError

    try:
        return _inspect_function_pointer("__nvvmCreateProgram") != 0
    except DynamicLibNotFoundError:
        return False


_libnvvm_available = _is_libnvvm_available()
_skip_no_libnvvm = pytest.mark.skipif(not _libnvvm_available, reason="libNVVM not available")

ptx_88_kernel = r"""
.version 8.8
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


ptx_72_kernel = r"""
.version  7.2
.target sm_75
.address_size 64

	// .globl	empty_kernel

.visible .entry empty_kernel()
{
	ret;
}
"""


@pytest.mark.parametrize(
    "kernel,actual_ptx_ver,min_cuda_ver", ((ptx_88_kernel, "8.8", 12090), (ptx_72_kernel, "7.2", 11020))
)
def test_ptx_utils(kernel, actual_ptx_ver, min_cuda_ver):
    ptx_ver = get_ptx_ver(kernel)
    assert ptx_ver == actual_ptx_ver
    cuda_ver = get_minimal_required_cuda_ver_from_ptx_ver(ptx_ver)
    assert cuda_ver == min_cuda_ver


@pytest.mark.parametrize(
    "target",
    (
        driver.CUcontext,
        driver.CUstream,
        driver.CUevent,
        driver.CUmodule,
        driver.CUlibrary,
        driver.CUfunction,
        driver.CUkernel,
        driver.CUgraph,
        driver.CUgraphNode,
        driver.CUgraphExec,
        driver.CUmemoryPool,
        runtime.cudaStream_t,
        runtime.cudaEvent_t,
        runtime.cudaGraph_t,
        runtime.cudaGraphNode_t,
        runtime.cudaGraphExec_t,
        runtime.cudaMemPool_t,
    ),
)
def test_get_handle(target):
    ptr = random.randint(1, 1024)
    obj = target(ptr)
    handle = get_cuda_native_handle(obj)
    assert handle == ptr


@pytest.mark.parametrize(
    "target",
    (
        (1, 2, 3, 4),
        [5, 6],
        {},
        None,
    ),
)
def test_get_handle_error(target):
    with pytest.raises(TypeError) as e:
        handle = get_cuda_native_handle(target)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_get_handle_does_not_report_a_registered_type_as_unknown(monkeypatch):
    """A KeyError from inside a handle getter is a bug in that getter.

    Reporting it as "Unknown type" is wrong twice over: the type *is*
    registered, and `from None` hides the traceback that would say otherwise.
    """
    from cuda.bindings.utils import _handle_getters

    class Registered:
        pass

    def getter(_obj):
        raise KeyError("lookup inside the getter failed")

    monkeypatch.setitem(_handle_getters, Registered, getter)

    with pytest.raises(KeyError, match="lookup inside the getter failed"):
        get_cuda_native_handle(Registered())


@pytest.mark.parametrize(
    "module",
    # Top-level modules for external Python use
    [
        "driver",
        "nvjitlink",
        "nvrtc",
        "nvvm",
        "runtime",
        *(["cufile"] if have_cufile else []),
    ],
)
def test_cyclical_imports(module):
    subprocess.check_call(  # noqa: S603
        [sys.executable, Path(__file__).parent / "utils" / "check_cyclical_import.py", f"cuda.bindings.{module}"],
    )


def test_get_c_compiler():
    c_compiler = get_c_compiler()
    prefix = ("GCC", "Clang", "MSVC", "Unknown")
    assert sum(c_compiler.startswith(p) for p in prefix) == 1


@_skip_no_libnvvm
def test_check_nvvm_compiler_options_valid():
    assert check_nvvm_compiler_options(["-arch=compute_90"]) is True


@_skip_no_libnvvm
def test_check_nvvm_compiler_options_invalid():
    assert check_nvvm_compiler_options(["--this-is-not-a-valid-option"]) is False


@_skip_no_libnvvm
def test_check_nvvm_compiler_options_empty():
    assert check_nvvm_compiler_options([]) is True


@_skip_no_libnvvm
def test_check_nvvm_compiler_options_multiple_valid():
    assert check_nvvm_compiler_options(["-arch=compute_90", "-opt=3", "-g"]) is True


@_skip_no_libnvvm
def test_check_nvvm_compiler_options_arch_detection():
    assert check_nvvm_compiler_options(["-arch=compute_90"]) is True
    assert check_nvvm_compiler_options(["-arch=compute_99999"]) is False


def test_check_nvvm_compiler_options_no_libnvvm():
    if _libnvvm_available:
        pytest.skip("libNVVM is available; this test targets the fallback path")
    assert check_nvvm_compiler_options(["-arch=compute_90"]) is False


class _RaiseOnImport:
    """A sys.meta_path finder that makes one module fail to import."""

    def __init__(self, fullname: str, exc: BaseException):
        self._fullname = fullname
        self._exc = exc

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._fullname:
            raise self._exc
        return None


def _simulate_import_failure(monkeypatch, exc: BaseException) -> None:
    """Make ``cuda.bindings.nvvm`` unimportable for the duration of a test."""
    import cuda.bindings

    # Both the parent-package attribute and the sys.modules entry short-circuit
    # the import system, so a previously imported nvvm has to be hidden too.
    monkeypatch.delattr(cuda.bindings, "nvvm", raising=False)
    monkeypatch.delitem(sys.modules, "cuda.bindings.nvvm", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_RaiseOnImport("cuda.bindings.nvvm", exc), *sys.meta_path])


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_nvvm_compiler_options_without_the_nvvm_binding(monkeypatch):
    """An absent cuda.bindings.nvvm means "options unsupported", not a crash.

    This is reachable from a source checkout in which the nvvm extension has
    not been built yet.
    """
    _simulate_import_failure(
        monkeypatch,
        ModuleNotFoundError("No module named 'cuda.bindings.nvvm'", name="cuda.bindings.nvvm"),
    )
    assert check_nvvm_compiler_options(["-arch=compute_90"]) is False


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_nvvm_compiler_options_does_not_mask_a_missing_dependency(monkeypatch):
    """Only the nvvm module itself is optional; a broken dependency must surface."""
    _simulate_import_failure(
        monkeypatch,
        ModuleNotFoundError("No module named 'not_a_real_dependency'", name="not_a_real_dependency"),
    )
    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency"):
        check_nvvm_compiler_options(["-arch=compute_90"])


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_nvvm_compiler_options_without_libnvvm(monkeypatch):
    """This is what test_check_nvvm_compiler_options_no_libnvvm above hits for real.

    That test only runs on a machine without libNVVM, so it never runs in CI
    (see #2077). Simulate the same condition here: _inspect_function_pointer()
    loads libNVVM lazily and raises DynamicLibNotFoundError when it is absent.
    """
    import cuda.bindings._internal as internal_pkg
    from cuda.pathfinder import DynamicLibNotFoundError

    def raise_not_found(_name):
        raise DynamicLibNotFoundError("libnvvm not found (simulated)")

    fake = types.ModuleType("cuda.bindings._internal.nvvm")
    fake._inspect_function_pointer = raise_not_found
    # Cover both routes a `from ... import ...` can take to the submodule.
    monkeypatch.setitem(sys.modules, "cuda.bindings._internal.nvvm", fake)
    monkeypatch.setattr(internal_pkg, "nvvm", fake, raising=False)

    assert check_nvvm_compiler_options(["-arch=compute_90"]) is False
