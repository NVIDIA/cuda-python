# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import pytest

skip_if_nvml_unsupported = pytest.mark.usefixtures("require_nvml_runtime_or_skip_local")


def unsupported_before(device, expected_device_arch):
    from cuda.bindings._test_helpers.arch_check import unsupported_before as nvml_unsupported_before

    return nvml_unsupported_before(device._handle, expected_device_arch)
