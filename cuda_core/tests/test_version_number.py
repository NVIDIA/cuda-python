# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import cuda.bindings
import cuda.core
import cuda.pathfinder
from helpers import validate_version_number


def test_bindings_version():
    validate_version_number(cuda.bindings.__version__, "cuda-bindings")


def test_core_version():
    validate_version_number(cuda.core.__version__, "cuda-core")


def test_pathfinder_version():
    validate_version_number(cuda.pathfinder.__version__, "cuda-pathfinder")
