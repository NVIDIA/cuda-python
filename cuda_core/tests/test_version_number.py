# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import cuda.core
from helpers import validate_version_number


def test_version_number():
    validate_version_number(cuda.core.__version__, "cuda-core")
