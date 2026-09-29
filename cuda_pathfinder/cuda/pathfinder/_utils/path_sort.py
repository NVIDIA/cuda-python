# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic path ordering helpers."""

from __future__ import annotations

import os
import re

_DIGIT_RUN = re.compile(r"(\d+)")


def numeric_aware_path_sort_key(path: str) -> tuple[tuple[int, int, str], ...]:
    """Return a key that compares embedded digit runs by numeric value."""
    key: list[tuple[int, int, str]] = []
    for part in _DIGIT_RUN.split(os.path.normcase(path)):
        if part.isdigit():
            normalized = part.lstrip("0") or "0"
            key.append((1, len(normalized), normalized))
        else:
            key.append((0, 0, part))
    return tuple(key)
