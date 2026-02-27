# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    create_graphics_resource_handle,
    deviceptr_create_with_owner,
    as_cu,
    as_intptr,
)
from cuda.core._memory._buffer cimport Buffer
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

__all__ = ['GraphicsResource']

_REGISTER_FLAGS = {
    "none": cydriver.CU_GRAPHICS_REGISTER_FLAGS_NONE,
    "read_only": cydriver.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
    "write_discard": cydriver.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
    "surface_load_store": cydriver.CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST,
    "texture_gather": cydriver.CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER,
}


def _parse_register_flags(flags):
    if flags is None:
        return 0
    if isinstance(flags, str):
        flags = (flags,)
    result = 0
    for f in flags:
        try:
            result |= _REGISTER_FLAGS[f]
        except KeyError:
            raise ValueError(
                f"Unknown register flag {f!r}. "
                f"Valid flags: {', '.join(sorted(_REGISTER_FLAGS))}"
            ) from None
    return result


cdef class GraphicsResource(Buffer):
    """RAII wrapper for a CUDA graphics resource (``CUgraphicsResource``).

    A :class:`GraphicsResource` represents an OpenGL buffer or image that has
    been registered for access by CUDA. This enables zero-copy sharing of GPU
    data between CUDA compute kernels and graphics renderers.

    :class:`GraphicsResource` inherits from :class:`~cuda.core.Buffer`, so when
    mapped it can be used directly anywhere a :class:`~cuda.core.Buffer` is
    expected. The buffer properties (:attr:`handle`, :attr:`size`) are only
    valid while the resource is mapped.

    The resource is automatically unregistered when :meth:`close` is called or
    when the object is garbage collected.

    :class:`GraphicsResource` objects should not be instantiated directly.
    Use the factory classmethods :meth:`from_gl_buffer` or :meth:`from_gl_image`.

    Examples
    --------
    Register an OpenGL VBO, map it to get a buffer, and write to it from CUDA:

    .. code-block:: python

        with GraphicsResource.from_gl_buffer(vbo, stream=s) as buf:
            view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)
            # view.ptr is a CUDA device pointer into the GL buffer

    Or use explicit map/unmap for render loops:

    .. code-block:: python

        resource.map(stream=s)
        # ... launch kernels using resource.handle, resource.size ...
        resource.unmap(stream=s)
    """

    def __init__(self):
        raise RuntimeError(
            "GraphicsResource objects cannot be instantiated directly. "
            "Use GraphicsResource.from_gl_buffer() or GraphicsResource.from_gl_image()."
        )

    @classmethod
    def from_gl_buffer(cls, int gl_buffer, *, flags=None, stream=None) -> GraphicsResource:
        """Register an OpenGL buffer object for CUDA access.

        Parameters
        ----------
        gl_buffer : int
            The OpenGL buffer name (``GLuint``) to register.
        flags : str or sequence of str, optional
            Registration flags specifying intended usage. Accepted values:
            ``"none"``, ``"read_only"``, ``"write_discard"``,
            ``"surface_load_store"``, ``"texture_gather"``.
            Multiple flags can be combined by passing a sequence
            (e.g., ``("surface_load_store", "read_only")``).
            Defaults to ``None`` (no flags).
        stream : :class:`~cuda.core.Stream`, optional
            If provided, the resource is immediately mapped on this stream
            so it can be used directly as a context manager::

                with GraphicsResource.from_gl_buffer(vbo, stream=s) as buf:
                    view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)

        Returns
        -------
        GraphicsResource
            A new graphics resource wrapping the registered GL buffer.
            If *stream* was given, the resource is already mapped.

        Raises
        ------
        CUDAError
            If the registration fails (e.g., no current GL context, invalid
            buffer name, or operating system error).
        ValueError
            If an unknown flag string is provided.
        """
        cdef GraphicsResource self = GraphicsResource.__new__(cls)
        cdef cydriver.CUgraphicsResource resource
        cdef cydriver.GLuint cy_buffer = <cydriver.GLuint>gl_buffer
        cdef unsigned int cy_flags = _parse_register_flags(flags)
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsGLRegisterBuffer(&resource, cy_buffer, cy_flags)
            )
        self._handle = create_graphics_resource_handle(resource)
        self._mapped = False
        self._map_stream = None
        if stream is not None:
            self.map(stream=stream)
        return self

    @classmethod
    def from_gl_image(
        cls, int image, int target, *, flags=None
    ) -> GraphicsResource:
        """Register an OpenGL texture or renderbuffer for CUDA access.

        Parameters
        ----------
        image : int
            The OpenGL texture or renderbuffer name (``GLuint``) to register.
        target : int
            The OpenGL target type (e.g., ``GL_TEXTURE_2D``).
        flags : str or sequence of str, optional
            Registration flags specifying intended usage. Accepted values:
            ``"none"``, ``"read_only"``, ``"write_discard"``,
            ``"surface_load_store"``, ``"texture_gather"``.
            Multiple flags can be combined by passing a sequence
            (e.g., ``("surface_load_store", "read_only")``).
            Defaults to ``None`` (no flags).

        Returns
        -------
        GraphicsResource
            A new graphics resource wrapping the registered GL image.

        Raises
        ------
        CUDAError
            If the registration fails.
        ValueError
            If an unknown flag string is provided.
        """
        cdef GraphicsResource self = GraphicsResource.__new__(cls)
        cdef cydriver.CUgraphicsResource resource
        cdef cydriver.GLuint cy_image = <cydriver.GLuint>image
        cdef cydriver.GLenum cy_target = <cydriver.GLenum>target
        cdef unsigned int cy_flags = _parse_register_flags(flags)
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsGLRegisterImage(&resource, cy_image, cy_target, cy_flags)
            )
        self._handle = create_graphics_resource_handle(resource)
        self._mapped = False
        self._map_stream = None
        return self

    def map(self, *, stream: Stream):
        """Map this graphics resource for CUDA access.

        After mapping, the CUDA device pointer and size are available via
        the inherited :attr:`~cuda.core.Buffer.handle` and
        :attr:`~cuda.core.Buffer.size` properties.

        Can be used as a context manager for automatic unmapping::

            with resource.map(stream=s) as buf:
                # buf IS the GraphicsResource, which IS-A Buffer
                # use buf.handle, buf.size, etc.
            # automatically unmapped here

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`
            The CUDA stream on which to perform the mapping.

        Returns
        -------
        GraphicsResource
            Returns ``self`` (which is a :class:`~cuda.core.Buffer`).

        Raises
        ------
        RuntimeError
            If the resource is already mapped or has been closed.
        CUDAError
            If the mapping fails.
        """
        if not self._handle:
            raise RuntimeError("GraphicsResource has been closed")
        if self._mapped:
            raise RuntimeError("GraphicsResource is already mapped")

        cdef Stream s_obj = Stream_accept(stream)
        cdef cydriver.CUgraphicsResource raw = as_cu(self._handle)
        cdef cydriver.CUstream cy_stream = as_cu(s_obj._h_stream)

        cdef cydriver.CUdeviceptr dev_ptr = 0
        cdef size_t size = 0
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsMapResources(1, &raw, cy_stream)
            )
            HANDLE_RETURN(
                cydriver.cuGraphicsResourceGetMappedPointer(&dev_ptr, &size, raw)
            )
        self._mapped = True
        # Populate Buffer internals with the mapped device pointer
        self._h_ptr = deviceptr_create_with_owner(dev_ptr, None)
        self._size = size
        self._owner = None
        self._mem_attrs_inited = False
        self._map_stream = stream
        return self

    def unmap(self, *, stream: Stream):
        """Unmap this graphics resource, releasing it back to the graphics API.

        After unmapping, the buffer properties (:attr:`handle`, :attr:`size`)
        are no longer valid.

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`
            The CUDA stream on which to perform the unmapping.

        Raises
        ------
        RuntimeError
            If the resource is not currently mapped or has been closed.
        CUDAError
            If the unmapping fails.
        """
        if not self._handle:
            raise RuntimeError("GraphicsResource has been closed")
        if not self._mapped:
            raise RuntimeError("GraphicsResource is not mapped")

        cdef Stream s_obj = Stream_accept(stream)
        cdef cydriver.CUgraphicsResource raw = as_cu(self._handle)
        cdef cydriver.CUstream cy_stream = as_cu(s_obj._h_stream)
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsUnmapResources(1, &raw, cy_stream)
            )
        self._mapped = False
        # Clear Buffer fields
        self._h_ptr.reset()
        self._size = 0
        self._map_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mapped:
            self.unmap(stream=self._map_stream)
        return False

    cpdef close(self, stream=None):
        """Unregister this graphics resource from CUDA.

        If the resource is currently mapped, it is unmapped first (on the
        default stream). After closing, the resource cannot be used again.

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            Accepted for compatibility with :meth:`Buffer.close` but not
            used for the graphics unmap/unregister operations.
        """
        cdef cydriver.CUgraphicsResource raw
        cdef cydriver.CUstream cy_stream
        if not self._handle:
            return
        if self._mapped:
            # Best-effort unmap before unregister (use stream 0 as fallback)
            raw = as_cu(self._handle)
            cy_stream = <cydriver.CUstream>0
            with nogil:
                cydriver.cuGraphicsUnmapResources(1, &raw, cy_stream)
            self._mapped = False
        self._handle.reset()
        # Clear Buffer fields
        self._h_ptr.reset()
        self._size = 0
        self._map_stream = None

    @property
    def is_mapped(self) -> bool:
        """Whether the resource is currently mapped for CUDA access."""
        return self._mapped

    @property
    def resource_handle(self) -> int:
        """The raw ``CUgraphicsResource`` handle as a Python int."""
        return as_intptr(self._handle)

    def __repr__(self):
        mapped_str = " mapped" if self._mapped else ""
        closed_str = " closed" if not self._handle else ""
        return f"<GraphicsResource handle={as_intptr(self._handle):#x}{mapped_str}{closed_str}>"
