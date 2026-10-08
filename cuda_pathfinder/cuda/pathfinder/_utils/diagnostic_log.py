# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in logging of where :func:`load_nvidia_dynamic_lib` found each library.

Disabled by default. Set ``CUDA_PATHFINDER_LOG_LEVEL`` to a level name (e.g.
``INFO``) or number to enable it. ``logging`` is only imported when the variable
is set, and the root logger is never configured.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

ENV_VAR_NAME = "CUDA_PATHFINDER_LOG_LEVEL"
LOGGER_NAME = "cuda.pathfinder"
_LEVEL_NAMES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _make_logger() -> logging.Logger | None:
    raw = os.environ.get(ENV_VAR_NAME, "").strip()
    if not raw:
        return None

    import logging

    name = raw.upper()
    level = int(raw) if raw.isdigit() else getattr(logging, name) if name in _LEVEL_NAMES else None
    if level is None:
        warnings.warn(f"{ENV_VAR_NAME}={raw!r} is not a valid logging level; logging stays disabled.", stacklevel=2)
        return None

    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(level)
    return logger


#: ``None`` when logging is disabled, so call sites can skip all work.
LOGGER: logging.Logger | None = _make_logger()
