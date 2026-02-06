# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from cuda.core import system

SHOULD_SKIP_NVML_TESTS = not system.CUDA_BINDINGS_NVML_IS_COMPATIBLE


if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from cuda.bindings._test_helpers.arch_check import hardware_supports_nvml

    SHOULD_SKIP_NVML_TESTS &= not hardware_supports_nvml()


skip_if_nvml_unsupported = pytest.mark.skipif(
    SHOULD_SKIP_NVML_TESTS,
    reason="NVML support requires cuda.bindings version 12.9.6+ or 13.1.2+, and hardware that supports NVML",
)


def unsupported_before(device, expected_device_arch):
    from cuda.bindings._test_helpers.arch_check import unsupported_before as nvml_unsupported_before

    return nvml_unsupported_before(device._handle, expected_device_arch)
