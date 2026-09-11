# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata
import re

from packaging.version import Version


@functools.cache
def have_distribution(name_pattern: str, *, minimum_version: str | None = None) -> bool:
    re_name_pattern = re.compile(name_pattern)
    parsed_minimum_version = Version(minimum_version) if minimum_version is not None else None
    return any(
        re_name_pattern.match(dist.metadata["Name"])
        and (parsed_minimum_version is None or Version(dist.version) >= parsed_minimum_version)
        for dist in importlib.metadata.distributions()
        if "Name" in dist.metadata
    )
