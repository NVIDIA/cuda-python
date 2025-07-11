# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.core.experimental


def test_context_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Context objects cannot be instantiated directly\."):
        cuda.core.experimental._context.Context()  # Ensure back door is locked.
