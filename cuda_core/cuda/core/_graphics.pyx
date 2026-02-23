# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    create_graphics_resource_handle,
    as_cu,
    as_intptr,
)
from cuda.core._stream cimport Stream, Stream_accept
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

from cuda.core._memory import Buffer

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


class _MappedBufferContext:
    """Context manager returned by :meth:`GraphicsResource.map`.

    Wraps a :class:`~cuda.core.Buffer` and ensures the graphics resource
    is unmapped when the context exits. Can also be used without ``with``
    by calling :meth:`GraphicsResource.unmap` explicitly.
    """
    __slots__ = ('_buffer', '_resource', '_stream')

    def __init__(self, buffer, resource, stream):
        self._buffer = buffer
        self._resource = resource
        self._stream = stream

    def __enter__(self):
        return self._buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._resource.unmap(stream=self._stream)
        return False

    # Delegate Buffer attributes so the return value of map() is directly usable
    @property
    def handle(self):
        return self._buffer.handle

    @property
    def size(self):
        return self._buffer.size

    def __repr__(self):
        return repr(self._buffer)


cdef class GraphicsResource:
    """RAII wrapper for a CUDA graphics resource (``CUgraphicsResource``).

    A :class:`GraphicsResource` represents an OpenGL buffer or image that has
    been registered for access by CUDA. This enables zero-copy sharing of GPU
    data between CUDA compute kernels and graphics renderers.

    The resource is automatically unregistered when :meth:`close` is called or
    when the object is garbage collected.

    :class:`GraphicsResource` objects should not be instantiated directly.
    Use the factory classmethods :meth:`from_gl_buffer` or :meth:`from_gl_image`.

    Examples
    --------
    Register an OpenGL VBO, map it to get a :class:`~cuda.core.Buffer`, and
    write to it from CUDA:

    .. code-block:: python

        resource = GraphicsResource.from_gl_buffer(vbo)

        with resource.map(stream=s) as buf:
            view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)
            # view.ptr is a CUDA device pointer into the GL buffer

    Or use explicit map/unmap for render loops:

    .. code-block:: python

        buf = resource.map(stream=s)
        # ... launch kernels using buf ...
        resource.unmap(stream=s)
    """

    def __init__(self):
        raise RuntimeError(
            "GraphicsResource objects cannot be instantiated directly. "
            "Use GraphicsResource.from_gl_buffer() or GraphicsResource.from_gl_image()."
        )

    @classmethod
    def from_gl_buffer(cls, int gl_buffer, *, flags=None) -> GraphicsResource:
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

        Returns
        -------
        GraphicsResource
            A new graphics resource wrapping the registered GL buffer.

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
        return self

    def map(self, *, stream: Stream | None = None):
        """Map this graphics resource for CUDA access.

        After mapping, a CUDA device pointer into the underlying graphics
        memory is available as a :class:`~cuda.core.Buffer`.

        Can be used as a context manager for automatic unmapping::

            with resource.map(stream=s) as buf:
                # use buf.handle, buf.size, etc.
            # automatically unmapped here

        Or called directly for explicit control::

            mapped = resource.map(stream=s)
            buf = mapped._buffer  # or use mapped.handle, mapped.size
            # ... do work ...
            resource.unmap(stream=s)

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            The CUDA stream on which to perform the mapping. If ``None``,
            the default stream (``0``) is used.

        Returns
        -------
        _MappedBufferContext
            An object that is both a context manager and provides access
            to the underlying :class:`~cuda.core.Buffer`. When used with
            ``with``, the resource is unmapped on exit.

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

        cdef cydriver.CUgraphicsResource raw = as_cu(self._handle)
        cdef cydriver.CUstream cy_stream = <cydriver.CUstream>0
        cdef Stream s_obj = None
        if stream is not None:
            s_obj = Stream_accept(stream)
            cy_stream = as_cu(s_obj._h_stream)

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
        buf = Buffer.from_handle(int(dev_ptr), size, owner=self)
        return _MappedBufferContext(buf, self, stream)

    def unmap(self, *, stream: Stream | None = None):
        """Unmap this graphics resource, releasing it back to the graphics API.

        After unmapping, the :class:`~cuda.core.Buffer` previously returned
        by :meth:`map` must not be used.

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            The CUDA stream on which to perform the unmapping. If ``None``,
            the default stream (``0``) is used.

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

        cdef cydriver.CUgraphicsResource raw = as_cu(self._handle)
        cdef cydriver.CUstream cy_stream = <cydriver.CUstream>0
        if stream is not None:
            cy_stream = as_cu((<Stream>Stream_accept(stream))._h_stream)
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsUnmapResources(1, &raw, cy_stream)
            )
        self._mapped = False

    cpdef close(self):
        """Unregister this graphics resource from CUDA.

        If the resource is currently mapped, it is unmapped first (on the
        default stream). After closing, the resource cannot be used again.
        """
        cdef cydriver.CUgraphicsResource raw
        cdef cydriver.CUstream cy_stream
        if not self._handle:
            return
        if self._mapped:
            # Best-effort unmap before unregister
            raw = as_cu(self._handle)
            cy_stream = <cydriver.CUstream>0
            with nogil:
                cydriver.cuGraphicsUnmapResources(1, &raw, cy_stream)
            self._mapped = False
        self._handle.reset()

    @property
    def is_mapped(self) -> bool:
        """Whether the resource is currently mapped for CUDA access."""
        return self._mapped

    @property
    def handle(self) -> int:
        """The raw ``CUgraphicsResource`` handle as a Python int."""
        return as_intptr(self._handle)

    def __repr__(self):
        mapped_str = " mapped" if self._mapped else ""
        closed_str = " closed" if not self._handle else ""
        return f"<GraphicsResource handle={as_intptr(self._handle):#x}{mapped_str}{closed_str}>"
