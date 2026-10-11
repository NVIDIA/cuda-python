# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the rlcompleter monkeypatch installed by `cuda.core` in interactive
sessions.

These tests reproduce the original bug report (NVIDIA/cuda-python#2053): tab
completion on a non-IPC-enabled DeviceMemoryResource crashes because the
Cython @property `allocation_handle` raises RuntimeError, and rlcompleter's
narrow `isinstance(..., property)` check misses C-level getset_descriptor
types and therefore invokes the descriptor.

The patch only installs in interactive mode, so each scenario is exercised in
a fresh subprocess with a controlled combination of `PYTHONINSPECT` and
`CUDA_CORE_DONT_FIX_TAB_COMPLETION`.
"""

import subprocess
import sys
import textwrap

import pytest
from cuda_python_test_helpers.subprocess_runner import run_python_snippet

from cuda.core import Device


def _gpu_with_mempool_or_skip():
    """Skip when no GPU or no mempool support — test mirrors the bug repro."""
    if len(Device.get_all_devices()) == 0:
        pytest.skip("Test requires a CUDA device")
    dev = Device(0)
    if not dev.properties.memory_pools_supported:
        pytest.skip("Device 0 does not support mempool operations")


# Probe script: reproduces the bug-report repro literally, then runs
# rlcompleter against `mr` and reports the outcome.
_PROBE_SCRIPT = textwrap.dedent("""
    import rlcompleter
    from cuda.core import Device, DeviceMemoryResource

    dev = Device(0)
    dev.set_current()
    mr = DeviceMemoryResource(dev)
    assert not mr.is_ipc_enabled, "test setup: mr should not be IPC-enabled"

    completer = rlcompleter.Completer({"mr": mr})
    try:
        matches = completer.attr_matches("mr.")
    except Exception as exc:
        print(f"crash: {type(exc).__name__}: {exc}")
    else:
        print(f"ok: {len(matches)} matches")
        print(f"allocation_handle: {'mr.allocation_handle' in matches}")
""")


def _run_probe(*, pythoninspect: bool, opt_out: bool = False) -> subprocess.CompletedProcess:
    return run_python_snippet(
        _PROBE_SCRIPT,
        # The child must import the installed wheel, not a source tree
        # reachable through an inherited PYTHONPATH.
        unset_env=("CUDA_CORE_DONT_FIX_TAB_COMPLETION", "PYTHONPATH"),
        extra_env={"CUDA_CORE_DONT_FIX_TAB_COMPLETION": "1"} if opt_out else None,
        check=False,
    )


def test_patched_completion_succeeds_on_non_ipc_resource():
    """With the patch installed (PYTHONINSPECT=1), tab completion must not
    crash and `mr.allocation_handle` must appear in the matches."""
    _gpu_with_mempool_or_skip()

    result = _run_probe(pythoninspect=True)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert result.stdout.startswith("ok:"), result.stdout
    assert "allocation_handle: True" in result.stdout, result.stdout


@pytest.mark.skipif(
    sys.version_info >= (3, 13), reason="Python 3.13.13, 3.14.6 and 3.15 fixed the rlcompleter bug upstream"
)
def test_opt_out_env_var_disables_patch_even_when_interactive():
    """`CUDA_CORE_DONT_FIX_TAB_COMPLETION=1` must short-circuit before the
    interactive check, so the bug reproduces again even under PYTHONINSPECT."""
    _gpu_with_mempool_or_skip()

    result = _run_probe(pythoninspect=True, opt_out=True)
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "crash: RuntimeError" in result.stdout, result.stdout


# Imports cuda.core and reports whether the rlcompleter patch was installed.
# No CUDA device is needed: the opt-out is evaluated at import time. The
# stdlib rlcompleter module has no `property` attribute of its own, so its
# presence is exactly the signal that the patch ran.
_OPT_OUT_PROBE_SCRIPT = textwrap.dedent("""
    import rlcompleter

    import cuda.core  # noqa: F401

    print(f"patched: {hasattr(rlcompleter, 'property')}")
""")


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    ("value", "expect_patched"),
    [
        # Empty / whitespace-only means "not set": `export VAR=` is the usual
        # way to neutralize a variable in a shell profile or container spec.
        ("", True),
        ("   ", True),
        # Integer values keep their long-standing meaning.
        ("0", True),
        ("00", True),
        ("1", False),
        ("2", False),
        # Non-integer values are honored as an opt-out.
        ("true", False),
        ("yes", False),
    ],
)
def test_opt_out_env_var_values(value, expect_patched):
    """`CUDA_CORE_DONT_FIX_TAB_COMPLETION` must never break `import cuda.core`.

    The opt-out used to be read with a bare `int(...)` at import time, so any
    value that is not a base-10 integer -- including the empty string -- raised
    `ValueError: invalid literal for int() with base 10: ''` out of
    `cuda/core/__init__.py` and made the package unimportable.
    """
    result = run_python_snippet(
        _OPT_OUT_PROBE_SCRIPT,
        unset_env=("PYTHONPATH",),
        extra_env={"CUDA_CORE_DONT_FIX_TAB_COMPLETION": value},
    )
    assert result.stdout.strip() == f"patched: {expect_patched}", result.stdout
