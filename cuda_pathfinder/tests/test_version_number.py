# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from helpers import validate_version_number

import cuda.pathfinder


def test_version_number():
    validate_version_number(cuda.pathfinder.__version__, "cuda-pathfinder")
