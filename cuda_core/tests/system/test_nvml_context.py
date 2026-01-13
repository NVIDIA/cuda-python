# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from .conftest import skip_if_nvml_unsupported

pytestmark = skip_if_nvml_unsupported

import multiprocessing as mp
from platform import uname

import pytest

UNINITIALIZED = 0
INITIALIZED = 1
DISABLED_LIBRARY_NOT_FOUND = 2


def _run_process(target):
    p = mp.get_context("spawn").Process(target=target)
    p.start()
    p.join()
    assert not p.exitcode


def _test_uninitialized():
    from cuda.core.system import _nvml_context

    assert _nvml_context._get_nvml_state() == UNINITIALIZED


def test_uninitialized():
    _run_process(_test_uninitialized)


def _test_is_initialized():
    from cuda.core.system import _nvml_context

    _nvml_context.initialize()
    assert _nvml_context._get_nvml_state() == INITIALIZED
    assert _nvml_context.is_initialized() is True


def test_is_initialized():
    _run_process(_test_is_initialized)


@pytest.mark.skipif("microsoft-standard" in uname().release, reason="Probably a WSL system")
def test_no_wsl():
    assert "microsoft-standard" not in uname().release


@pytest.mark.skipif("microsoft-standard" not in uname().release, reason="Probably a non-WSL system")
def test_wsl():
    assert "microsoft-standard" in uname().release


def _test_validate():
    from cuda.core.system import _nvml_context

    _nvml_context.initialize()

    assert _nvml_context.validate() is None


def test_validate():
    _run_process(_test_validate)
