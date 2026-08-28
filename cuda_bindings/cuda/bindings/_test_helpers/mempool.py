# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated shim. Moved to cuda_python_test_helpers.mempool in #2384 (2026-08-04).
Kept for one release cycle so released cuda-core test trees keep importing.
Remove after cuda-core >= 1.2 is the oldest supported release.
"""

from cuda.bindings._test_helpers import _import_current

_mempool = _import_current("mempool")
is_windows_mcdm_device = _mempool.is_windows_mcdm_device
xfail_if_mempool_oom = _mempool.xfail_if_mempool_oom

__all__ = ["is_windows_mcdm_device", "xfail_if_mempool_oom"]
