# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Texture and surface APIs for ``cuda.core``.

This subpackage groups the CUDA "Texture and Surface" object model — opaque
array storage and the bindless texture/surface handles built on it — under a
single namespace, mirroring how the CUDA driver documentation organizes them.

Import these types from here, e.g.::

    from cuda.core.textures import CUDAArray, TextureObject, TextureDescriptor
"""

from cuda.core._array import ArrayFormat, CUDAArray
from cuda.core._mipmapped_array import MipmappedArray
from cuda.core._surface import SurfaceObject
from cuda.core._texture import (
    AddressMode,
    FilterMode,
    ReadMode,
    ResourceDescriptor,
    TextureDescriptor,
    TextureObject,
)

__all__ = [
    "AddressMode",
    "ArrayFormat",
    "CUDAArray",
    "FilterMode",
    "MipmappedArray",
    "ReadMode",
    "ResourceDescriptor",
    "SurfaceObject",
    "TextureDescriptor",
    "TextureObject",
]
