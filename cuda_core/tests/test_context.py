# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import pytest

import cuda.core.experimental


def test_context_init_disabled():
    with pytest.raises(RuntimeError, match=r"^Context objects cannot be instantiated directly\."):
        cuda.core.experimental._context.Context()  # Ensure back door is locked.
