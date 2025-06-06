# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

import pytest


def pytest_configure(config):
    config.custom_info = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.custom_info:
        terminalreporter.write_sep("=", "INFO summary")
        for msg in config.custom_info:
            terminalreporter.line(f"INFO {msg}")


@pytest.fixture
def info_summary_append(request):
    def _append(message):
        request.config.custom_info.append(f"{request.node.name}: {message}")

    return _append
