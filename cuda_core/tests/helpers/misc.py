# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def try_create_condition(g, default_value=1):
    """Create a Condition on graph *g*, skipping the test if unsupported."""
    from cuda.core._utils.cuda_utils import CUDAError

    try:
        return g.create_condition(default_value=default_value)
    except CUDAError:
        pytest.skip("Conditional nodes not supported (requires CC >= 9.0)")


class StreamWrapper:
    """
    A wrapper around Stream for testing IsStreamT conversions.
    """

    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return self._stream.__cuda_stream__()
