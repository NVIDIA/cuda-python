# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdviseOptions:
    """Per-call options for :func:`cuda.core.utils.advise`.

    Reserved for future advise flags. Currently has no fields; pass
    ``AdviseOptions()`` or ``None`` to use driver defaults.
    """


@dataclass(frozen=True)
class PrefetchOptions:
    """Per-call options for :func:`cuda.core.utils.prefetch`.

    Reserved for future prefetch flags. Currently has no fields; pass
    ``PrefetchOptions()`` or ``None`` to use driver defaults.
    """


@dataclass(frozen=True)
class DiscardOptions:
    """Per-call options for :func:`cuda.core.utils.discard`.

    Reserved for future discard flags. Currently has no fields; pass
    ``DiscardOptions()`` or ``None`` to use driver defaults.
    """


@dataclass(frozen=True)
class DiscardPrefetchOptions:
    """Per-call options for :func:`cuda.core.utils.discard_prefetch`.

    Reserved for future discard-and-prefetch flags. Currently has no
    fields; pass ``DiscardPrefetchOptions()`` or ``None`` to use driver
    defaults.
    """
