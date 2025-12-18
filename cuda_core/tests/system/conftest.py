# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from cuda.core import system

NVML_SUPPORTED = system._HAS_WORKING_NVML


skip_if_nvml_unsupported = pytest.mark.skipif(
    not NVML_SUPPORTED, reason="NVML support requires cuda.bindings version 12.9.5+ or 13.1.1+"
)
