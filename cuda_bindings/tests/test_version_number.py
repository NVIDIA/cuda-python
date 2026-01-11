# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import cuda.bindings
from helpers import validate_version_number


def test_version_number():
    validate_version_number(cuda.bindings.__version__, "cuda-bindings")
