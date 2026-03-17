# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Managed-memory range operations."""

from cuda.core._memory._buffer import advise, discard_prefetch, prefetch

__all__ = ["advise", "discard_prefetch", "prefetch"]
