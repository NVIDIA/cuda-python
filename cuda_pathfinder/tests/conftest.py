# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest


def pytest_configure(config):
    config.custom_info = []


@pytest.fixture
def info_summary_append(request):
    def _append(message):
        request.config.custom_info.append(f"{request.node.name}: {message}")

    return _append
