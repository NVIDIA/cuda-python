# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata
import re


@functools.cache
def have_distribution(name_pattern: str) -> bool:
    re_name_pattern = re.compile(name_pattern)
    return any(
        re_name_pattern.match(dist.metadata["Name"])
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata
    )
