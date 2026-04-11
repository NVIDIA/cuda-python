# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-library descriptor and registry.

The canonical authored data lives in :mod:`descriptor_catalog`. This module
provides a name-keyed registry consumed by the runtime search/load path.
"""

from __future__ import annotations

from typing import TypeAlias

from cuda.pathfinder._dynamic_libs.descriptor_catalog import (
    DESCRIPTOR_CATALOG,
    DescriptorSpec,
)

# Keep the historical type name for downstream imports.
LibDescriptor: TypeAlias = DescriptorSpec


#: Canonical registry of all known libraries.
LIB_DESCRIPTORS: dict[str, LibDescriptor] = {desc.name: desc for desc in DESCRIPTOR_CATALOG}
