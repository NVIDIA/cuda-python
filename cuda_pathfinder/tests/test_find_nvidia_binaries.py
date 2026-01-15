# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cuda.pathfinder import find_nvidia_binary
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES


def test_unknown_binary():
    with pytest.raises(ValueError, match=r"Unknown binary: 'unknown-binary'"):
        find_nvidia_binary("unknown-binary")


@pytest.mark.parametrize("binary_name", SUPPORTED_BINARIES)
def test_find_binaries(info_summary_append, binary_name):
    binary_path = find_nvidia_binary(binary_name)
    info_summary_append(f"{binary_path=!r}")
    if binary_path:
        assert os.path.isfile(binary_path)
        # Verify the binary name is in the path
        assert binary_name in os.path.basename(binary_path)
