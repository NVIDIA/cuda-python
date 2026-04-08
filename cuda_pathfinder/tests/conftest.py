# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


def pytest_configure(config):
    config.custom_info = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("verbose"):
        return
    if hasattr(config.option, "iterations"):  # pytest-freethreaded runs all tests at least twice
        return
    if getattr(config.option, "count", 1) > 1:  # pytest-repeat
        return

    if config.custom_info:
        terminalreporter.write_sep("=", "INFO summary")
        for msg in config.custom_info:
            terminalreporter.line(f"INFO {msg}")


@pytest.fixture
def info_summary_append(request):
    def _append(message):
        request.config.custom_info.append(f"{request.node.name}: {message}")

    return _append


def skip_if_missing_libnvcudla_so(libname: str, *, timeout: float) -> None:
    if libname not in ("cudla", "nvcudla"):
        return
    # Keep the import inside the helper so unrelated import issues do not fail
    # pytest collection for the whole test suite.
    from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib_module

    if load_nvidia_dynamic_lib_module._loadable_via_canary_subprocess("nvcudla", timeout=timeout):
        return
    pytest.skip("libnvcudla.so is not loadable via canary subprocess on this host.")
