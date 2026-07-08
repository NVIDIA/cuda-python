# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Texture and surface APIs for ``cuda.core``.

This subpackage groups the CUDA "Texture and Surface" object model — opaque
array storage and the bindless texture/surface handles built on it — under a
single namespace, mirroring how the CUDA driver documentation organizes them.

Import these types from here, e.g.::

    from cuda.core.texture import OpaqueArray, TextureObject, TextureObjectOptions

The associated enumerations (:class:`~cuda.core.typing.ArrayFormatType`,
:class:`~cuda.core.typing.AddressModeType`,
:class:`~cuda.core.typing.FilterModeType`,
:class:`~cuda.core.typing.ReadModeType`) live in :mod:`cuda.core.typing`
alongside the other ``cuda.core`` enumerations.
"""

from cuda.core.texture._array import OpaqueArray, OpaqueArrayOptions
from cuda.core.texture._mipmapped_array import MipmappedArray, MipmappedArrayOptions
from cuda.core.texture._surface import SurfaceObject
from cuda.core.texture._texture import (
    ResourceDescriptor,
    TextureObject,
    TextureObjectOptions,
)

__all__ = [
    "MipmappedArray",
    "MipmappedArrayOptions",
    "OpaqueArray",
    "OpaqueArrayOptions",
    "ResourceDescriptor",
    "SurfaceObject",
    "TextureObject",
    "TextureObjectOptions",
]
