# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants shared across the cuda_core test suite."""

# Cap for memory pools created by tests; see cuda_python_test_helpers.memory
# for why it matters (issue #2381) and AGENTS.md in the tests directory.
from cuda_python_test_helpers.memory import POOL_SIZE

__all__ = ["POOL_SIZE"]
