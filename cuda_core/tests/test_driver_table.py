# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the C++ driver function table in cuda/core/_cpp/rt/py_driver_fns.cpp when its fill fails.

The first driver call fills the table from ``cuda.bindings._internal.driver._inspect_function_pointers()``.
A child interpreter replaces that function so that the fill fails in a controlled way. It then
makes driver calls through ``cuda.core`` and reports what happened. The expected outcome:

- The fill reports the failure once as a :class:`CUDAWarning`.
- Every affected call raises :class:`CUDAError` with the reason attached as a note.
- The failure latches: there is no second warning and no retry.

The child needs a loadable CUDA driver and a visible device, so this module skips without them.
The module runs with ``--noconftest``.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "cuda" / "core"
LOADER = Path(__file__).resolve().parents[2] / "cuda_bindings" / "cuda" / "bindings" / "_internal" / "driver_linux.pyx"


def _gpu_available() -> bool:
    """The child calls cuInit and cuDeviceGetCount through cuda-bindings before it reaches
    the C++ table, so it needs a loadable driver and a visible device."""
    try:
        from cuda.bindings import driver

        (status,) = driver.cuInit(0)
        if int(status) != 0:
            return False
        status, count = driver.cuDeviceGetCount()
        return int(status) == 0 and count > 0
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _gpu_available(), reason="the child needs a CUDA driver and a visible device"),
    pytest.mark.thread_unsafe(reason="spawns child interpreters"),
]


def _table_keys() -> list[str]:
    """The keys the fill looks up: "__" + the symbol that the cuda-bindings loader requests for
    each name in driver_api.hpp. For example, cuStreamDestroy maps to __cuStreamDestroy_v2."""
    if not LOADER.is_file():
        pytest.skip("needs the cuda_bindings source tree next to cuda_core")
    names = re.findall(
        r"^\s*X\((cu\w+), \d+\)", (CORE / "_cpp" / "rt" / "driver_api.hpp").read_text(encoding="utf-8"), re.M
    )
    loader = LOADER.read_text(encoding="utf-8")
    keys = []
    for name in names:
        m = re.search(rf"cuGetProcAddress_v2\('{name}', <void \*\*>&(__\w+),", loader)
        assert m is not None, name
        keys.append(m.group(1))
    return keys


_CHILD = textwrap.dedent("""
    import warnings
    import cuda.bindings._internal.driver as loader

    KEYS = {keys!r}

    def fake_inspect_function_pointers():
        table = {{key: 1 for key in KEYS}}  # placeholder addresses: the fill fails before any call
        {mutation}
        return table

    loader._inspect_function_pointers = fake_inspect_function_pointers

    from cuda.core import CUDAError, CUDAWarning, Device

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for attempt in range(2):
            try:
                # cuInit and the device query are Cython calls. The primary context
                # retain is the first call through the C++ table.
                Device(0).set_current()
            except CUDAError as exc:
                print(f"ATTEMPT {{attempt}} CUDAError: {{exc}} NOTES: {{getattr(exc, '__notes__', [])}}")
            else:
                print(f"ATTEMPT {{attempt}} no error")
    cuda_warnings = [w for w in caught if issubclass(w.category, CUDAWarning)]
    print(f"CUDAWARNINGS {{len(cuda_warnings)}}")
    for w in cuda_warnings:
        print("WARNING:", str(w.message))
""")


def _run_child(mutation: str, tmp_path: Path) -> str:
    code = _CHILD.format(keys=_table_keys(), mutation=mutation)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _build_major() -> int:
    from cuda.core import _build_info

    return _build_info.CUDA_MAJOR


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_null_baseline_entry_fails_the_fill_once_and_latches(tmp_path):
    out = _run_child('table["__cuGetErrorName"] = 0', tmp_path)
    lines = out.splitlines()
    attempts = [line for line in lines if line.startswith("ATTEMPT")]
    assert len(attempts) == 2
    for line in attempts:
        assert "CUDAError: CUDA_ERROR_NOT_INITIALIZED" in line
        assert "could not call cu" in line  # the reason rides on every raised error
        assert f"lacks cuGetErrorName, which every CUDA {_build_major()} driver provides" in line
    assert "CUDAWARNINGS 1" in lines  # reported once, then latched
    warning = next(line for line in lines if line.startswith("WARNING:"))
    assert "lacks cuGetErrorName" in warning


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_missing_table_entry_names_the_mismatch(tmp_path):
    out = _run_child('table.pop("__cuDevicePrimaryCtxRetain", None)', tmp_path)
    lines = out.splitlines()
    attempts = [line for line in lines if line.startswith("ATTEMPT")]
    assert len(attempts) == 2
    for line in attempts:
        assert "CUDAError: CUDA_ERROR_NOT_INITIALIZED" in line
        assert "has no entry for cuDevicePrimaryCtxRetain" in line
        assert "Install the cuda-bindings this cuda.core requires" in line
    assert "CUDAWARNINGS 1" in lines
