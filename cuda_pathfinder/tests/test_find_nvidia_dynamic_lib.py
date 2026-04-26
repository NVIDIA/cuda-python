# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import platform
import subprocess
import sys
import textwrap

import pytest
from child_load_nvidia_dynamic_lib_helper import run_load_nvidia_dynamic_lib_in_subprocess

from cuda.pathfinder import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
    find_nvidia_dynamic_lib,
)
from cuda.pathfinder._dynamic_libs import find_nvidia_dynamic_lib as find_nvidia_dynamic_lib_module
from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib_module
from cuda.pathfinder._dynamic_libs import supported_nvidia_libs
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    STATUS_NOT_FOUND,
    STATUS_OK,
    DynamicLibSubprocessPayload,
    parse_dynamic_lib_subprocess_payload,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, quote_for_shell

STRICTNESS = os.environ.get("CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS", "see_what_works")
assert STRICTNESS in ("see_what_works", "all_must_work")


def test_unknown_libname_raises_dynamic_lib_unknown_error():
    with pytest.raises(DynamicLibUnknownError, match=r"Unknown library name: 'not_a_real_lib'.*cudart"):
        find_nvidia_dynamic_lib("not_a_real_lib")


def test_known_but_platform_unavailable_libname_raises_dynamic_lib_not_available_error(monkeypatch):
    find_nvidia_dynamic_lib.cache_clear()
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_ALL_KNOWN_LIBNAMES", frozenset(("known_but_unavailable",)))
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_ALL_SUPPORTED_LIBNAMES", frozenset())
    monkeypatch.setattr(load_nvidia_dynamic_lib_module, "_PLATFORM_NAME", "TestOS")
    with pytest.raises(
        DynamicLibNotAvailableError,
        match=r"known_but_unavailable.*not available on TestOS",
    ):
        find_nvidia_dynamic_lib("known_but_unavailable")


def _is_expected_find_failure(libname: str) -> bool:
    # Mirror load-side strictness: libnames known to fail loading on this
    # platform are also allowed to fail finding.
    return libname == "nvpl_fftw" and platform.machine().lower() != "aarch64"


@pytest.mark.parametrize(
    "libname",
    supported_nvidia_libs.SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else supported_nvidia_libs.SUPPORTED_LINUX_SONAMES,
)
def test_find_nvidia_dynamic_lib_returns_existing_path_without_loading(info_summary_append, libname):
    find_nvidia_dynamic_lib.cache_clear()
    try:
        abs_path = find_nvidia_dynamic_lib(libname)
    except DynamicLibNotFoundError:
        if STRICTNESS == "all_must_work" and not _is_expected_find_failure(libname):
            raise
        info_summary_append(f"Not found: {libname=!r}")
        return

    info_summary_append(f"abs_path={quote_for_shell(abs_path)}")
    assert os.path.isabs(abs_path)
    assert os.path.isfile(abs_path)


def test_find_matches_load_in_subprocess(info_summary_append):
    # Single representative libname is enough to exercise the consistency
    # claim (see issue #757); per-libname coverage is provided by the
    # parametrized find/load tests independently.
    libname = "cudart"
    find_nvidia_dynamic_lib.cache_clear()
    timeout = 120 if IS_WINDOWS else 30
    load_result = run_load_nvidia_dynamic_lib_in_subprocess(libname, timeout=timeout)
    if load_result.returncode != 0:
        pytest.skip(f"load subprocess failed for {libname!r}; consistency comparison N/A")

    load_payload = parse_dynamic_lib_subprocess_payload(
        load_result.stdout,
        libname=libname,
        error_label="Load subprocess child process",
    )
    if load_payload.status != STATUS_OK:
        pytest.skip(f"{libname} not loadable on this host; nothing to compare against")

    find_abs_path = find_nvidia_dynamic_lib(libname)
    assert load_payload.abs_path is not None
    info_summary_append(
        f"{libname}: load={quote_for_shell(load_payload.abs_path)} find={quote_for_shell(find_abs_path)}"
    )
    assert os.path.samefile(find_abs_path, load_payload.abs_path)


def test_find_nvidia_dynamic_lib_propagates_subprocess_not_found_message(monkeypatch):
    # End-to-end of the structured-error path: the subprocess child encodes
    # DynamicLibNotFoundError into the JSON payload; the parent re-raises
    # with the original message preserved.
    find_nvidia_dynamic_lib.cache_clear()
    expected = "child loader said: cudart could not be located"

    def fake_run(mode, libname, *, timeout, error_label):
        return DynamicLibSubprocessPayload(
            status=STATUS_NOT_FOUND,
            abs_path=None,
            error={"type": "DynamicLibNotFoundError", "message": expected},
        )

    monkeypatch.setattr(find_nvidia_dynamic_lib_module, "run_dynamic_lib_subprocess", fake_run)
    with pytest.raises(DynamicLibNotFoundError, match=expected):
        find_nvidia_dynamic_lib("cudart")


def test_find_nvidia_dynamic_lib_falls_back_when_subprocess_not_found_omits_message(monkeypatch):
    find_nvidia_dynamic_lib.cache_clear()

    def fake_run(mode, libname, *, timeout, error_label):
        return DynamicLibSubprocessPayload(status=STATUS_NOT_FOUND, abs_path=None, error=None)

    monkeypatch.setattr(find_nvidia_dynamic_lib_module, "run_dynamic_lib_subprocess", fake_run)
    with pytest.raises(DynamicLibNotFoundError, match=r"could not locate 'cudart'"):
        find_nvidia_dynamic_lib("cudart")


_DOES_NOT_LOAD_PROBE = textwrap.dedent(
    """
    import json
    import os
    import sys

    from cuda.pathfinder import DynamicLibNotFoundError, find_nvidia_dynamic_lib

    libname = sys.argv[1]
    try:
        find_nvidia_dynamic_lib(libname)
        find_status = "found"
    except DynamicLibNotFoundError:
        find_status = "not-found"

    with open("/proc/self/maps") as f:
        maps = f.read()
    print(json.dumps({"find_status": find_status, "loaded": ("lib" + libname) in maps}))
    """
).strip()


def test_find_nvidia_dynamic_lib_does_not_load_in_caller_process():
    if IS_WINDOWS or not os.path.exists("/proc/self/maps"):
        pytest.skip("Requires /proc/self/maps for in-process load detection")

    libname = "cudart"
    # Run in a fresh interpreter so test ordering / other pathfinder tests
    # in the same process can't have pre-loaded the library.
    result = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + inline probe
        [sys.executable, "-c", _DOES_NOT_LOAD_PROBE, libname],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload["find_status"] == "not-found":
        pytest.skip(f"{libname} not available on this host")
    assert payload["loaded"] is False, "find_nvidia_dynamic_lib must not load the library into the caller process"
