# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated shim. Moved to cuda_python_test_helpers.pep723 in #2384 (2026-08-04).
Kept for one release cycle so released cuda-core test trees keep importing.
Remove after cuda-core >= 1.2 is the oldest supported release.
"""

from cuda_python_test_helpers.pep723 import has_package_requirements_or_skip

__all__ = ["has_package_requirements_or_skip"]
