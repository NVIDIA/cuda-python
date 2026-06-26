# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport intptr_t
from libc.string cimport memset

from cuda.bindings cimport cydriver
from cuda.core.texture._array cimport OpaqueArray
from cuda.core.texture._array import ArrayFormat, _FORMAT_ELEM_SIZE, _validate_format_channels
from cuda.core._memory._buffer cimport Buffer
from cuda.core.texture._mipmapped_array cimport MipmappedArray
from cuda.core.texture._mipmapped_array import MipmappedArray as _PyMipmappedArray
from cuda.core._resource_handles cimport (
    TexObjectHandle,
    as_cu,
    as_intptr,
    create_tex_object_handle_array,
    create_tex_object_handle_linear,
    create_tex_object_handle_mipmap,
    get_last_error,
)
from cuda.core._utils.cuda_utils cimport (
    HANDLE_RETURN,
    _get_current_device_id,
)

from dataclasses import dataclass
from enum import IntEnum


# Driver texture-descriptor flag bits (CU_TRSF_*).
_TRSF_READ_AS_INTEGER = 0x01
_TRSF_NORMALIZED_COORDINATES = 0x02
_TRSF_SRGB = 0x10
_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 0x20
_TRSF_SEAMLESS_CUBEMAP = 0x40


class AddressMode(IntEnum):
    """Boundary behavior for out-of-range texture coordinates."""
    WRAP   = cydriver.CU_TR_ADDRESS_MODE_WRAP
    CLAMP  = cydriver.CU_TR_ADDRESS_MODE_CLAMP
    MIRROR = cydriver.CU_TR_ADDRESS_MODE_MIRROR
    BORDER = cydriver.CU_TR_ADDRESS_MODE_BORDER


class FilterMode(IntEnum):
    """Texel sampling mode."""
    POINT  = cydriver.CU_TR_FILTER_MODE_POINT
    LINEAR = cydriver.CU_TR_FILTER_MODE_LINEAR


class ReadMode(IntEnum):
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

    Construct via the ``from_*`` classmethods:

    - :meth:`from_array` wraps a :class:`OpaqueArray` (works for both
      :class:`TextureObject` and :class:`SurfaceObject`).
    - :meth:`from_mipmapped_array` wraps a :class:`MipmappedArray` for mipmapped
      sampling (texture only, not surface).
    - :meth:`from_linear` wraps a :class:`Buffer` as a typed 1D fetch. Texture
      objects built from a linear resource do not support filtering,
      normalized coordinates, or addressing modes.
    - :meth:`from_pitch2d` wraps a :class:`Buffer` as a row-pitched 2D image.
      Supports filtering and 2D addressing, but only 2D access.

    Linear and pitch2D resources cannot back a :class:`SurfaceObject` — those
    require an :class:`OpaqueArray` allocated with ``is_surface_load_store=True``.
    """

    __slots__ = (
        "_kind", "_source",
        "_format", "_num_channels",
        "_size_bytes",
        "_width", "_height", "_pitch_bytes",
    )

    def __init__(self):
        raise RuntimeError(
            "ResourceDescriptor cannot be instantiated directly. "
            "Use ResourceDescriptor.from_* factories."
        )

    @classmethod
    def from_array(cls, array):
        """Build a resource descriptor backed by a :class:`OpaqueArray`."""
        if not isinstance(array, OpaqueArray):
            raise TypeError(f"array must be a OpaqueArray, got {type(array).__name__}")
        self = cls.__new__(cls)
        self._kind = "array"
        self._source = array
        self._format = None
        self._num_channels = None
        self._size_bytes = None
        self._width = None
        self._height = None
        self._pitch_bytes = None
        return self

    @classmethod
    def from_mipmapped_array(cls, mipmapped_array):
        """Build a resource descriptor backed by a :class:`MipmappedArray`.

        Suitable for binding to a :class:`TextureObject` for mipmapped
        sampling. Not valid as a :class:`SurfaceObject` backing: surfaces
        require a single :class:`OpaqueArray` level (obtain via
        :meth:`MipmappedArray.get_level`).
        """
        if not isinstance(mipmapped_array, _PyMipmappedArray):
            raise TypeError(
                f"mipmapped_array must be a MipmappedArray, got "
                f"{type(mipmapped_array).__name__}"
            )
        self = cls.__new__(cls)
        self._kind = "mipmapped_array"
        self._source = mipmapped_array
        self._format = None
        self._num_channels = None
        self._size_bytes = None
        self._width = None
        self._height = None
        self._pitch_bytes = None
        return self

    @classmethod
    def from_linear(cls, buffer, *, format, num_channels, size_bytes=None):
        """Build a resource descriptor for a linear (typed 1D) texture fetch.

        Parameters
        ----------
        buffer : Buffer
            Device-memory backing. Must remain alive for the lifetime of any
            :class:`TextureObject` built from this descriptor.
        format : ArrayFormat
            Element format.
        num_channels : int
            Channels per element. Must be 1, 2, or 4.
        size_bytes : int, optional
            Bytes of ``buffer`` to bind. Defaults to ``buffer.size``. Must not
            exceed it.

        Notes
        -----
        Texture objects built from a linear resource ignore the
        :class:`TextureDescriptor` addressing/filtering fields — kernels read
        through a typed 1D fetch with bounds checking only.
        """
        if not isinstance(buffer, Buffer):
            raise TypeError(f"buffer must be a Buffer, got {type(buffer).__name__}")
        _validate_format_channels(format, num_channels)

        buf_size = int(buffer.size)
        elem = _FORMAT_ELEM_SIZE[int(format)] * int(num_channels)
        if size_bytes is None:
            size = buf_size
        else:
            size = int(size_bytes)
            if size > buf_size:
                raise ValueError(
                    f"size_bytes ({size}) exceeds buffer.size ({buf_size})"
                )
        if size < elem:
            raise ValueError(
                f"size_bytes ({size}) must be at least one element ({elem} bytes)"
            )
        if size % elem != 0:
            raise ValueError(
                f"size_bytes ({size}) must be a multiple of element size "
                f"({elem} bytes for {format.name} x {num_channels})"
            )

        self = cls.__new__(cls)
        self._kind = "linear"
        self._source = buffer
        self._format = int(format)
        self._num_channels = int(num_channels)
        self._size_bytes = size
        self._width = None
        self._height = None
        self._pitch_bytes = None
        return self

    @classmethod
    def from_pitch2d(
        cls, buffer, *, format, num_channels, width, height, pitch_bytes
    ):
        """Build a resource descriptor for a row-pitched 2D image.

        Parameters
        ----------
        buffer : Buffer
            Device-memory backing. Must remain alive for the lifetime of any
            :class:`TextureObject` built from this descriptor.
        format : ArrayFormat
            Element format.
        num_channels : int
            Channels per element. Must be 1, 2, or 4.
        width : int
            Image width, in elements.
        height : int
            Image height, in rows.
        pitch_bytes : int
            Distance between consecutive rows, in bytes. Must be at least
            ``width * format_size * num_channels`` and meet the driver's
            ``CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT``.
        """
        if not isinstance(buffer, Buffer):
            raise TypeError(f"buffer must be a Buffer, got {type(buffer).__name__}")
        _validate_format_channels(format, num_channels)

        w = int(width)
        h = int(height)
        p = int(pitch_bytes)
        if w < 1:
            raise ValueError(f"width must be >= 1, got {w}")
        if h < 1:
            raise ValueError(f"height must be >= 1, got {h}")
        elem = _FORMAT_ELEM_SIZE[int(format)] * int(num_channels)
        min_pitch = w * elem
        if p < min_pitch:
            raise ValueError(
                f"pitch_bytes ({p}) must be >= width * element_size ({min_pitch})"
            )
        if p * h > int(buffer.size):
            raise ValueError(
                f"pitch_bytes * height ({p * h}) exceeds buffer.size ({int(buffer.size)})"
            )

        self = cls.__new__(cls)
        self._kind = "pitch2d"
        self._source = buffer
        self._format = int(format)
        self._num_channels = int(num_channels)
        self._size_bytes = None
        self._width = w
        self._height = h
        self._pitch_bytes = p
        return self

    @property
    def kind(self):
        return self._kind

    @property
    def source(self):
        return self._source

    @property
    def format(self):
        """The element :class:`ArrayFormat` (``None`` for array-backed)."""
        return None if self._format is None else ArrayFormat(self._format)

    @property
    def num_channels(self):
        """Channels per element (``None`` for array-backed)."""
        return self._num_channels

    @property
    def size_bytes(self):
        """Bytes bound for a linear resource (``None`` for other kinds)."""
        return self._size_bytes

    @property
    def width(self):
        """Pitch2D image width, in elements (``None`` for other kinds)."""
        return self._width

    @property
    def height(self):
        """Pitch2D image height, in rows (``None`` for other kinds)."""
        return self._height

    @property
    def pitch_bytes(self):
        """Pitch2D row pitch, in bytes (``None`` for other kinds)."""
        return self._pitch_bytes

    def __repr__(self):
        if self._kind == "linear":
            return (
                f"ResourceDescriptor(kind='linear', format={self.format.name}, "
                f"num_channels={self._num_channels}, size_bytes={self._size_bytes})"
            )
        if self._kind == "pitch2d":
            return (
                f"ResourceDescriptor(kind='pitch2d', format={self.format.name}, "
                f"num_channels={self._num_channels}, "
                f"width={self._width}, height={self._height}, "
                f"pitch_bytes={self._pitch_bytes})"
            )
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

    address_mode: AddressMode | tuple[AddressMode, ...] = AddressMode.CLAMP
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
    border_color: tuple[float, ...] | None = None


def _normalize_address_modes(address_mode):
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
    :class:`OpaqueArray` referenced by the descriptor) is kept alive for the
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
    def from_descriptor(cls, *, resource, texture_descriptor):
        """Create a texture object from a resource + sampling descriptor.

        Parameters
        ----------
        resource : ResourceDescriptor
        texture_descriptor : TextureDescriptor
        """
        if not isinstance(resource, ResourceDescriptor):
            raise TypeError(
                f"resource must be a ResourceDescriptor, got "
                f"{type(resource).__name__}"
            )
        if not isinstance(texture_descriptor, TextureDescriptor):
            raise TypeError(
                f"texture_descriptor must be a TextureDescriptor, got "
                f"{type(texture_descriptor).__name__}"
            )

        cdef cydriver.CUDA_RESOURCE_DESC res_desc
        cdef cydriver.CUDA_TEXTURE_DESC tex_desc
        memset(&res_desc, 0, sizeof(res_desc))
        memset(&tex_desc, 0, sizeof(tex_desc))

        # --- Resource descriptor ---
        cdef OpaqueArray arr
        cdef MipmappedArray mip
        cdef Buffer buf
        cdef intptr_t devptr
        if resource.kind == "array":
            arr = <OpaqueArray>resource.source
            res_desc.resType = cydriver.CU_RESOURCE_TYPE_ARRAY
            res_desc.res.array.hArray = as_cu(arr._handle)
        elif resource.kind == "mipmapped_array":
            mip = <MipmappedArray>resource.source
            res_desc.resType = cydriver.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
            res_desc.res.mipmap.hMipmappedArray = as_cu(mip._handle)
        elif resource.kind == "linear":
            buf = <Buffer>resource.source
            devptr = int(buf.handle)
            res_desc.resType = cydriver.CU_RESOURCE_TYPE_LINEAR
            res_desc.res.linear.devPtr = <cydriver.CUdeviceptr>devptr
            res_desc.res.linear.format = <cydriver.CUarray_format><int>resource._format
            res_desc.res.linear.numChannels = <unsigned int>resource._num_channels
            res_desc.res.linear.sizeInBytes = <size_t>resource._size_bytes
        elif resource.kind == "pitch2d":
            buf = <Buffer>resource.source
            devptr = int(buf.handle)
            res_desc.resType = cydriver.CU_RESOURCE_TYPE_PITCH2D
            res_desc.res.pitch2D.devPtr = <cydriver.CUdeviceptr>devptr
            res_desc.res.pitch2D.format = <cydriver.CUarray_format><int>resource._format
            res_desc.res.pitch2D.numChannels = <unsigned int>resource._num_channels
            res_desc.res.pitch2D.width = <size_t>resource._width
            res_desc.res.pitch2D.height = <size_t>resource._height
            res_desc.res.pitch2D.pitchInBytes = <size_t>resource._pitch_bytes
        else:
            raise NotImplementedError(
                f"ResourceDescriptor kind {resource.kind!r} is not yet supported"
            )

        # --- Texture descriptor ---
        modes = _normalize_address_modes(texture_descriptor.address_mode)
        tex_desc.addressMode[0] = <cydriver.CUaddress_mode><int>modes[0]
        tex_desc.addressMode[1] = <cydriver.CUaddress_mode><int>modes[1]
        tex_desc.addressMode[2] = <cydriver.CUaddress_mode><int>modes[2]

        if not isinstance(texture_descriptor.filter_mode, FilterMode):
            raise TypeError(
                f"filter_mode must be a FilterMode, got "
                f"{type(texture_descriptor.filter_mode).__name__}"
            )
        tex_desc.filterMode = <cydriver.CUfilter_mode><int>texture_descriptor.filter_mode

        if not isinstance(texture_descriptor.read_mode, ReadMode):
            raise TypeError(
                f"read_mode must be a ReadMode, got "
                f"{type(texture_descriptor.read_mode).__name__}"
            )

        cdef unsigned int flags = 0
        # CU_TRSF_READ_AS_INTEGER suppresses normalization, so it maps to
        # ReadMode.ELEMENT_TYPE.
        if texture_descriptor.read_mode == ReadMode.ELEMENT_TYPE:
            flags |= _TRSF_READ_AS_INTEGER
        if texture_descriptor.normalized_coords:
            flags |= _TRSF_NORMALIZED_COORDINATES
        if texture_descriptor.srgb:
            flags |= _TRSF_SRGB
        if texture_descriptor.disable_trilinear_optimization:
            flags |= _TRSF_DISABLE_TRILINEAR_OPTIMIZATION
        if texture_descriptor.seamless_cubemap:
            flags |= _TRSF_SEAMLESS_CUBEMAP
        tex_desc.flags = flags

        if texture_descriptor.max_anisotropy < 0:
            raise ValueError("max_anisotropy must be >= 0")
        tex_desc.maxAnisotropy = <unsigned int>texture_descriptor.max_anisotropy

        if not isinstance(texture_descriptor.mipmap_filter_mode, FilterMode):
            raise TypeError(
                f"mipmap_filter_mode must be a FilterMode, got "
                f"{type(texture_descriptor.mipmap_filter_mode).__name__}"
            )
        tex_desc.mipmapFilterMode = <cydriver.CUfilter_mode><int>texture_descriptor.mipmap_filter_mode
        tex_desc.mipmapLevelBias = <float>texture_descriptor.mipmap_level_bias
        tex_desc.minMipmapLevelClamp = <float>texture_descriptor.min_mipmap_level_clamp
        tex_desc.maxMipmapLevelClamp = <float>texture_descriptor.max_mipmap_level_clamp

        cdef int i
        if texture_descriptor.border_color is None:
            for i in range(4):
                tex_desc.borderColor[i] = 0.0
        else:
            bc = tuple(texture_descriptor.border_color)
            if len(bc) != 4:
                raise ValueError(
                    f"border_color must have 4 elements, got {len(bc)}"
                )
            for i in range(4):
                tex_desc.borderColor[i] = <float>bc[i]

        cdef TexObjectHandle h
        if resource.kind == "array":
            h = create_tex_object_handle_array(res_desc, tex_desc, arr._handle)
        elif resource.kind == "mipmapped_array":
            h = create_tex_object_handle_mipmap(res_desc, tex_desc, mip._handle)
        else:  # linear or pitch2d — both backed by a device Buffer
            h = create_tex_object_handle_linear(res_desc, tex_desc, buf._h_ptr)
        if not h:
            HANDLE_RETURN(get_last_error())

        cdef TextureObject self = cls.__new__(cls)
        self._handle = h
        self._source_ref = resource
        self._texture_desc = texture_descriptor
        self._device_id = _get_current_device_id()
        return self

    @property
    def handle(self):
        """The underlying ``CUtexObject`` as an integer (64-bit kernel arg)."""
        return as_intptr(self._handle)

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
        """Release this object's reference to the underlying ``CUtexObject``.

        Destruction (``cuTexObjectDestroy``) and release of the backing resource
        happen via the handle's deleter when the last reference is dropped.
        Idempotent.
        """
        self._handle.reset()
        self._source_ref = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return f"TextureObject(handle=0x{as_intptr(self._handle):x})"
