# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def envvar_bool(name: str, default: bool = False) -> bool:
    """Read a bool-like environment variable.

    Unset, empty, or whitespace-only means ``default``. ``1/true/yes/on`` and
    ``0/false/no/off`` are recognised case-insensitively, and any other integer
    follows C truthiness, so ``2`` is true and ``-0`` is false.

    A value that is none of those keeps the historical set-means-true
    behaviour rather than raising, because these variables are read during
    import and a raise would turn a typo into an import failure.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    try:
        return int(raw, 0) != 0
    except ValueError:
        return True
