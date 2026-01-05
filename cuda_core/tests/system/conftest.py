# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from cuda.core import system

skip_if_nvml_unsupported = pytest.mark.skipif(
    not system.HAS_WORKING_NVML, reason="NVML support requires cuda.bindings version 12.9.6+ or 13.1.2+"
)


@pytest.fixture(autouse=True, scope="session")
def initialize_nvml():
    if system.HAS_WORKING_NVML:
        from cuda.core.system._nvml_context import initialize

        initialize()
