# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

try:
    from cuda_python_test_helpers import *  # noqa: F403
except ModuleNotFoundError:
    # Import shared platform helpers for tests across repos
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "cuda_python_test_helpers"))
    from cuda_python_test_helpers import *  # noqa: F403
