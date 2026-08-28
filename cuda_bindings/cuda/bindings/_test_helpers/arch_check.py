# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated shim. Moved to cuda_python_test_helpers.arch_check in #2384 (2026-08-04).
Kept for one release cycle so released cuda-core test trees keep importing.
Remove after cuda-core >= 1.2 is the oldest supported release.
"""

from cuda.bindings._test_helpers import _import_current

_arch_check = _import_current("arch_check")
hardware_supports_nvml = _arch_check.hardware_supports_nvml
unsupported_before = _arch_check.unsupported_before

__all__ = ["hardware_supports_nvml", "unsupported_before"]
