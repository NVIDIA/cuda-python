# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from cuda.core.system.utils import format_bytes, unpack_bitmask


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1) == "1 B"
    assert format_bytes(1023) == "1023 B"
    assert format_bytes(1024) == "1.00 KiB"
    assert format_bytes(1024**2) == "1.00 MiB"
    assert format_bytes(1024**3) == "1.00 GiB"
    assert format_bytes(1024**4) == "1.00 TiB"
    assert format_bytes(1024**5) == "1024.00 TiB"
    assert format_bytes(1024**6) == "1048576.00 TiB"


@pytest.mark.parametrize(
    "params",
    [
        {
            "input": [1152920405096267775, 0],
            "output": [i for i in range(20)] + [i + 40 for i in range(20)],
        },
        {
            "input": [17293823668613283840, 65535],
            "output": [i + 20 for i in range(20)] + [i + 60 for i in range(20)],
        },
        {"input": [18446744073709551615, 0], "output": [i for i in range(64)]},
        {"input": [0, 18446744073709551615], "output": [i + 64 for i in range(64)]},
    ],
)
def test_unpack_bitmask(params):
    assert unpack_bitmask(params["input"]) == params["output"]


def test_unpack_bitmask_single_value():
    with pytest.raises(TypeError):
        unpack_bitmask(1)
