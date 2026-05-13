# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core._array cimport Array
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import enum
from dataclasses import dataclass, field


# Driver texture-descriptor flag bits (CU_TRSF_*).
_TRSF_READ_AS_INTEGER = 0x01
_TRSF_NORMALIZED_COORDINATES = 0x02
_TRSF_SRGB = 0x10
_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 0x20
_TRSF_SEAMLESS_CUBEMAP = 0x40


class AddressMode(enum.IntEnum):
    """Boundary behavior for out-of-range texture coordinates."""
    WRAP   = cydriver.CU_TR_ADDRESS_MODE_WRAP
    CLAMP  = cydriver.CU_TR_ADDRESS_MODE_CLAMP
    MIRROR = cydriver.CU_TR_ADDRESS_MODE_MIRROR
    BORDER = cydriver.CU_TR_ADDRESS_MODE_BORDER


class FilterMode(enum.IntEnum):
    """Texel sampling mode."""
    POINT  = cydriver.CU_TR_FILTER_MODE_POINT
    LINEAR = cydriver.CU_TR_FILTER_MODE_LINEAR


class ReadMode(enum.IntEnum):
    """How sampled values are returned to the kernel.

    - ``ELEMENT_TYPE``: return the raw element value (integer formats stay
      integer, float stays float).
    - ``NORMALIZED_FLOAT``: integer formats are promoted to a normalized
      ``float`` in ``[0, 1]`` (unsigned) or ``[-1, 1]`` (signed).
      Float formats are unaffected.
    """
    ELEMENT_TYPE     = 0
    NORMALIZED_FLOAT = 1


class ResourceDescriptor:
    """Describes the memory backing a :class:`TextureObject`.

    Construct via the ``from_*`` classmethods. Only the ``from_array`` path is
    implemented in this initial version; ``from_linear`` and ``from_pitch2d``
    will follow once their metadata story (format/channel count on
    :class:`Buffer`) is settled.
    """

    __slots__ = ("_kind", "_source")

    def __init__(self):
        raise RuntimeError(
            "ResourceDescriptor cannot be instantiated directly. "
            "Use ResourceDescriptor.from_* factories."
        )

    @classmethod
    def from_array(cls, array):
        """Build a resource descriptor backed by a :class:`Array`."""
        if not isinstance(array, Array):
            raise TypeError(f"array must be an Array, got {type(array).__name__}")
        self = cls.__new__(cls)
        self._kind = "array"
        self._source = array
        return self

    @property
    def kind(self):
        return self._kind

    @property
    def source(self):
        return self._source

    def __repr__(self):
        return f"ResourceDescriptor(kind={self._kind!r})"


@dataclass
class TextureDescriptor:
    """Sampling state for a :class:`TextureObject` (mirrors ``CUDA_TEXTURE_DESC``).

    Attributes
    ----------
    address_mode : tuple of AddressMode
        Boundary behavior per axis. May be a single :class:`AddressMode` (applied
        to all axes) or a tuple of 1-3 entries (one per dimension).
    filter_mode : FilterMode
        Texel sampling mode. Default ``POINT``.
    read_mode : ReadMode
        How sampled integer values are returned. Default ``ELEMENT_TYPE``.
    normalized_coords : bool
        If True, coordinates are in ``[0, 1]`` instead of pixel indices.
    srgb : bool
        If True, perform sRGB → linear conversion on read (8-bit formats only).
    disable_trilinear_optimization : bool
        If True, request exact trilinear filtering.
    seamless_cubemap : bool
        If True, enable seamless cubemap edge filtering.
    max_anisotropy : int
        Maximum anisotropy; 0 disables anisotropic filtering.
    mipmap_filter_mode : FilterMode
        Filtering between mipmap levels. Default ``POINT``.
    mipmap_level_bias : float
    min_mipmap_level_clamp : float
    max_mipmap_level_clamp : float
    border_color : tuple of float or None
        4-tuple used when ``address_mode`` includes ``BORDER``; ``None`` means
        zero.
    """

    address_mode: object = AddressMode.CLAMP
    filter_mode: FilterMode = FilterMode.POINT
    read_mode: ReadMode = ReadMode.ELEMENT_TYPE
    normalized_coords: bool = False
    srgb: bool = False
    disable_trilinear_optimization: bool = False
    seamless_cubemap: bool = False
    max_anisotropy: int = 0
    mipmap_filter_mode: FilterMode = FilterMode.POINT
    mipmap_level_bias: float = 0.0
    min_mipmap_level_clamp: float = 0.0
    max_mipmap_level_clamp: float = 0.0
    border_color: tuple | None = None


cdef inline intptr_t _get_current_context_ptr() except? 0:
    cdef cydriver.CUcontext ctx
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    if ctx == NULL:
        raise RuntimeError("TextureObject requires an active CUDA context")
    return <intptr_t>ctx


cdef inline int _get_current_device_id() except -1:
    cdef cydriver.CUdevice dev
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxGetDevice(&dev))
    return <int>dev


cdef _normalize_address_modes(address_mode):
    """Return a 3-tuple of AddressMode values from a scalar or 1-3 tuple."""
    if isinstance(address_mode, AddressMode):
        return (address_mode, address_mode, address_mode)
    try:
        modes = tuple(address_mode)
    except TypeError as e:
        raise TypeError(
            "address_mode must be an AddressMode or a tuple of AddressMode"
        ) from e
    if not 1 <= len(modes) <= 3:
        raise ValueError(
            f"address_mode tuple must have 1-3 entries, got {len(modes)}"
        )
    for i, m in enumerate(modes):
        if not isinstance(m, AddressMode):
            raise TypeError(
                f"address_mode[{i}] must be an AddressMode, got {type(m).__name__}"
            )
    # Pad to 3 entries by repeating the last one.
    padded = list(modes) + [modes[-1]] * (3 - len(modes))
    return tuple(padded)


cdef class TextureObject:
    """A bindless texture handle for kernel-side sampled reads.

    Wraps ``cuTexObjectCreate``. The underlying memory resource (e.g. the
    :class:`Array` referenced by the descriptor) is kept alive for the
    lifetime of this object to prevent dangling handles.

    Construct via :meth:`from_descriptor`. Passes to kernels as a 64-bit
    handle (via the ``handle`` property).
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TextureObject cannot be instantiated directly. "
            "Use TextureObject.from_descriptor()."
        )

    @classmethod
    def from_descriptor(cls, resource_desc, texture_desc):
        """Create a texture object from a resource + sampling descriptor.

        Parameters
        ----------
        resource_desc : ResourceDescriptor
        texture_desc : TextureDescriptor
        """
        if not isinstance(resource_desc, ResourceDescriptor):
            raise TypeError(
                f"resource_desc must be a ResourceDescriptor, got "
                f"{type(resource_desc).__name__}"
            )
        if not isinstance(texture_desc, TextureDescriptor):
            raise TypeError(
                f"texture_desc must be a TextureDescriptor, got "
                f"{type(texture_desc).__name__}"
            )

        cdef cydriver.CUDA_RESOURCE_DESC res_desc
        cdef cydriver.CUDA_TEXTURE_DESC tex_desc
        memset(&res_desc, 0, sizeof(res_desc))
        memset(&tex_desc, 0, sizeof(tex_desc))

        # --- Resource descriptor ---
        cdef Array arr
        if resource_desc.kind == "array":
            arr = <Array>resource_desc.source
            res_desc.resType = cydriver.CU_RESOURCE_TYPE_ARRAY
            res_desc.res.array.hArray = arr._handle
        else:
            raise NotImplementedError(
                f"ResourceDescriptor kind {resource_desc.kind!r} is not yet supported"
            )

        # --- Texture descriptor ---
        modes = _normalize_address_modes(texture_desc.address_mode)
        tex_desc.addressMode[0] = <cydriver.CUaddress_mode><int>modes[0]
        tex_desc.addressMode[1] = <cydriver.CUaddress_mode><int>modes[1]
        tex_desc.addressMode[2] = <cydriver.CUaddress_mode><int>modes[2]

        if not isinstance(texture_desc.filter_mode, FilterMode):
            raise TypeError("filter_mode must be a FilterMode")
        tex_desc.filterMode = <cydriver.CUfilter_mode><int>texture_desc.filter_mode

        if not isinstance(texture_desc.read_mode, ReadMode):
            raise TypeError("read_mode must be a ReadMode")

        cdef unsigned int flags = 0
        # CU_TRSF_READ_AS_INTEGER suppresses normalization, so it maps to
        # ReadMode.ELEMENT_TYPE.
        if texture_desc.read_mode == ReadMode.ELEMENT_TYPE:
            flags |= _TRSF_READ_AS_INTEGER
        if texture_desc.normalized_coords:
            flags |= _TRSF_NORMALIZED_COORDINATES
        if texture_desc.srgb:
            flags |= _TRSF_SRGB
        if texture_desc.disable_trilinear_optimization:
            flags |= _TRSF_DISABLE_TRILINEAR_OPTIMIZATION
        if texture_desc.seamless_cubemap:
            flags |= _TRSF_SEAMLESS_CUBEMAP
        tex_desc.flags = flags

        if texture_desc.max_anisotropy < 0:
            raise ValueError("max_anisotropy must be >= 0")
        tex_desc.maxAnisotropy = <unsigned int>texture_desc.max_anisotropy

        if not isinstance(texture_desc.mipmap_filter_mode, FilterMode):
            raise TypeError("mipmap_filter_mode must be a FilterMode")
        tex_desc.mipmapFilterMode = <cydriver.CUfilter_mode><int>texture_desc.mipmap_filter_mode
        tex_desc.mipmapLevelBias = <float>texture_desc.mipmap_level_bias
        tex_desc.minMipmapLevelClamp = <float>texture_desc.min_mipmap_level_clamp
        tex_desc.maxMipmapLevelClamp = <float>texture_desc.max_mipmap_level_clamp

        cdef int i
        if texture_desc.border_color is None:
            for i in range(4):
                tex_desc.borderColor[i] = 0.0
        else:
            bc = tuple(texture_desc.border_color)
            if len(bc) != 4:
                raise ValueError(
                    f"border_color must have 4 elements, got {len(bc)}"
                )
            for i in range(4):
                tex_desc.borderColor[i] = <float>bc[i]

        cdef TextureObject self = cls.__new__(cls)
        self._source_ref = resource_desc
        self._texture_desc = texture_desc
        self._context = _get_current_context_ptr()
        self._device_id = _get_current_device_id()

        with nogil:
            HANDLE_RETURN(
                cydriver.cuTexObjectCreate(&self._handle, &res_desc, &tex_desc, NULL)
            )
        return self

    @property
    def handle(self):
        """The underlying ``CUtexObject`` as an integer (64-bit kernel arg)."""
        return <intptr_t>self._handle

    @property
    def resource(self):
        """The :class:`ResourceDescriptor` this texture was built from."""
        return self._source_ref

    @property
    def texture_descriptor(self):
        """The :class:`TextureDescriptor` this texture was built from."""
        return self._texture_desc

    @property
    def device(self):
        from cuda.core._device import Device
        return Device(self._device_id)

    cpdef close(self):
        """Destroy the underlying ``CUtexObject``."""
        if self._handle != 0:
            HANDLE_RETURN(cydriver.cuTexObjectDestroy(self._handle))
        self._handle = 0
        self._source_ref = None

    def __dealloc__(self):
        # Cython destructors cannot raise; any cuTexObjectDestroy error is
        # silently dropped. Callers needing visibility should use close().
        if self._handle != 0:
            cydriver.cuTexObjectDestroy(self._handle)
            self._handle = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return f"TextureObject(handle=0x{<intptr_t>self._handle:x})"
