# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in diagnostic logging for cuda.pathfinder.

Disabled by default. Set ``CUDA_PATHFINDER_LOG_LEVEL`` to a standard level name
(``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``) or to a numeric
level to enable it::

    CUDA_PATHFINDER_LOG_LEVEL=DEBUG python -c "import cuda.pathfinder as p; p.load_nvidia_dynamic_lib('cudart')"

Design notes
------------
``logging`` is **not** imported unless the environment variable is set. Importing
it pulls in seven additional modules (``logging``, ``atexit``, ``string``,
``_string``, ``textwrap``, ``traceback``, ``_colorize``) and measurably slows
``import cuda.pathfinder``, which sits on the import hot path of every consumer.
Deferring the import keeps the disabled path free rather than merely cheap.

Call sites guard on ``LOGGER is not None`` so that a disabled logger costs one
module-global lookup and an identity check, and so that no message string or
``extra`` dict is built when logging is off.

The environment variable is read exactly once, at import. This matches
:func:`cuda.pathfinder._utils.env_vars.get_cuda_path_or_home`, which is
``functools.cache``-d and documents the same read-once policy.

This module never configures the root logger, never calls ``basicConfig``, and
attaches only a ``NullHandler``. Consumers remain in full control of handlers
and formatting.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

#: Environment variable that enables logging. Matches the existing
#: ``CUDA_PATHFINDER_*`` prefix used elsewhere in this package.
ENV_VAR_NAME = "CUDA_PATHFINDER_LOG_LEVEL"

#: Logger name. Mirrors the import path so consumers can filter on it.
LOGGER_NAME = "cuda.pathfinder"

_VALID_LEVEL_NAMES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _resolve_level(raw: str) -> int | None:
    """Map an environment-variable value to a logging level, or None if invalid."""
    import logging

    name = raw.strip().upper()
    if name in _VALID_LEVEL_NAMES:
        return int(getattr(logging, name))
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _make_logger() -> logging.Logger | None:
    """Build the package logger, or return None when logging is disabled.

    ``logging`` is imported only on the enabled path.
    """
    raw = os.environ.get(ENV_VAR_NAME)
    if not raw or not raw.strip():
        return None

    level = _resolve_level(raw)
    if level is None:
        warnings.warn(
            f"{ENV_VAR_NAME}={raw!r} is not a valid logging level; "
            f"expected one of {', '.join(_VALID_LEVEL_NAMES)} or an integer. "
            "cuda.pathfinder logging stays disabled.",
            UserWarning,
            stacklevel=2,
        )
        return None

    import logging

    logger = logging.getLogger(LOGGER_NAME)
    # NullHandler keeps "No handlers could be found" quiet without imposing a
    # destination; consumers attach their own handler if they want output.
    logger.addHandler(logging.NullHandler())
    logger.setLevel(level)
    return logger


#: The package logger, or ``None`` when logging is disabled. Guard every call
#: site with ``if LOGGER is not None:`` so the disabled path builds nothing.
LOGGER: logging.Logger | None = _make_logger()


def search_extra(libname: str, **fields: Any) -> dict[str, Any]:
    """Build the ``extra`` mapping shared by pathfinder log records.

    Only ever called from inside a ``LOGGER is not None`` guard, so it costs
    nothing when logging is disabled. Consumers can filter on these fields
    instead of parsing the message text.
    """
    return {"pathfinder_libname": libname, **fields}
