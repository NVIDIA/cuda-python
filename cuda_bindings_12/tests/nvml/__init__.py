# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from cuda.bindings._test_helpers.arch_check import hardware_supports_nvml

if not hardware_supports_nvml():
    pytest.skip("NVML not supported on this platform", allow_module_level=True)
