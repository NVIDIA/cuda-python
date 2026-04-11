# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    create_graphics_resource_handle,
    deviceptr_create_mapped_graphics,
    as_cu,
    as_intptr,
)
from cuda.core._memory._buffer cimport Buffer, Buffer_from_deviceptr_handle
from cuda.core._stream cimport Stream, Stream_accept, default_stream
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


cdef class GraphicsResource:
    """RAII wrapper for a CUDA graphics resource (``CUgraphicsResource``).

    A :class:`GraphicsResource` represents an OpenGL buffer or image that has
    been registered for access by CUDA. This enables zero-copy sharing of GPU
    data between CUDA compute kernels and graphics renderers.

    Mapping the resource returns a :class:`~cuda.core.Buffer` whose lifetime
    controls when the graphics resource is unmapped. This keeps stream-ordered
    cleanup tied to the mapped pointer itself rather than to mutable state on
    the :class:`GraphicsResource` object.

    The resource is automatically unregistered when :meth:`close` is called or
    when the object is garbage collected.

    :class:`GraphicsResource` objects should not be instantiated directly.
    Use the factory classmethods :meth:`from_gl_buffer` or :meth:`from_gl_image`.

    Examples
    --------
    Register an OpenGL VBO, map it to get a buffer, and write to it from CUDA:

    .. code-block:: python

        resource = GraphicsResource.from_gl_buffer(vbo)

        with resource.map(stream=s) as buf:
            view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)
            # view.ptr is a CUDA device pointer into the GL buffer

    Or scope registration separately from mapping:

    .. code-block:: python

        with GraphicsResource.from_gl_buffer(vbo) as resource:
            with resource.map(stream=s) as buf:
                # ... launch kernels using buf.handle, buf.size ...
                pass
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
            If provided, the resource can be used directly as a context manager
            and it will be mapped on entry::

                with GraphicsResource.from_gl_buffer(vbo, stream=s) as buf:
                    view = StridedMemoryView.from_buffer(buf, shape=(256,), dtype=np.float32)

            If omitted, the returned resource can still be used as a context
            manager to scope registration and automatic cleanup::

                with GraphicsResource.from_gl_buffer(vbo) as resource:
                    with resource.map(stream=s) as buf:
                        ...

        Returns
        -------
        GraphicsResource
            A new graphics resource wrapping the registered GL buffer.
            The returned resource can be used as a context manager. If
            *stream* was given, entering maps the resource and yields a
            :class:`~cuda.core.Buffer`; otherwise entering yields the
            :class:`GraphicsResource` itself and closes it on exit.

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
        self._mapped_buffer = None
        self._context_manager_stream = stream
        self._entered_buffer = None
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
        self._mapped_buffer = None
        self._context_manager_stream = None
        self._entered_buffer = None
        return self

    def _get_mapped_buffer(self):
        cdef Buffer buf
        if self._mapped_buffer is None:
            return None
        buf = <Buffer>self._mapped_buffer
        if not buf._h_ptr:
            self._mapped_buffer = None
            return None
        return self._mapped_buffer

    def map(self, *, stream: Stream | None = None) -> Buffer:
        """Map this graphics resource for CUDA access.

        After mapping, a CUDA device pointer into the underlying graphics
        memory is available as a :class:`~cuda.core.Buffer`.

        Can be used as a context manager for automatic unmapping::

            with resource.map(stream=s) as buf:
                # use buf.handle, buf.size, etc.
            # automatically unmapped here

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            The CUDA stream on which to perform the mapping. If ``None``,
            the current default stream is used.

        Returns
        -------
        Buffer
            A buffer whose lifetime controls when the graphics resource is
            unmapped.

        Raises
        ------
        RuntimeError
            If the resource is already mapped or has been closed.
        CUDAError
            If the mapping fails.
        """
        cdef Stream s_obj
        cdef cydriver.CUgraphicsResource raw
        cdef cydriver.CUstream cy_stream
        cdef cydriver.CUdeviceptr dev_ptr = 0
        cdef size_t size = 0
        cdef Buffer buf
        if not self._handle:
            raise RuntimeError("GraphicsResource has been closed")
        if self._get_mapped_buffer() is not None:
            raise RuntimeError("GraphicsResource is already mapped")

        s_obj = default_stream() if stream is None else Stream_accept(stream)
        raw = as_cu(self._handle)
        cy_stream = as_cu(s_obj._h_stream)
        with nogil:
            HANDLE_RETURN(
                cydriver.cuGraphicsMapResources(1, &raw, cy_stream)
            )
            HANDLE_RETURN(
                cydriver.cuGraphicsResourceGetMappedPointer(&dev_ptr, &size, raw)
            )
        buf = Buffer_from_deviceptr_handle(
            deviceptr_create_mapped_graphics(dev_ptr, self._handle, s_obj._h_stream),
            size,
            None,
            None,
        )
        self._mapped_buffer = buf
        return buf

    def unmap(self, *, stream: Stream | None = None):
        """Unmap this graphics resource, releasing it back to the graphics API.

        After unmapping, the :class:`~cuda.core.Buffer` previously returned
        by :meth:`map` must not be used.

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            If provided, overrides the stream that will be used when the
            mapped buffer is closed. Otherwise the mapping stream is reused.

        Raises
        ------
        RuntimeError
            If the resource is not currently mapped or has been closed.
        CUDAError
            If the unmapping fails.
        """
        cdef object buf_obj
        cdef Buffer buf
        if not self._handle:
            raise RuntimeError("GraphicsResource has been closed")
        buf_obj = self._get_mapped_buffer()
        if buf_obj is None:
            raise RuntimeError("GraphicsResource is not mapped")
        buf = <Buffer>buf_obj
        buf.close(stream=stream)
        self._mapped_buffer = None

    def __enter__(self):
        if self._context_manager_stream is None:
            return self
        self._entered_buffer = self.map(stream=self._context_manager_stream)
        return self._entered_buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    cpdef close(self, stream=None):
        """Unregister this graphics resource from CUDA.

        If the resource is currently mapped, it is unmapped first. After
        closing, the resource cannot be used again.

        Parameters
        ----------
        stream : :class:`~cuda.core.Stream`, optional
            Optional override for the stream used to close the currently
            mapped buffer, if one exists.
        """
        cdef object buf_obj
        cdef Buffer buf
        if not self._handle:
            return
        buf_obj = self._get_mapped_buffer()
        if buf_obj is not None:
            buf = <Buffer>buf_obj
            buf.close(stream=stream)
            self._mapped_buffer = None
        self._handle.reset()
        self._context_manager_stream = None
        self._entered_buffer = None

    @property
    def is_mapped(self) -> bool:
        """Whether the resource is currently mapped for CUDA access."""
        return self._get_mapped_buffer() is not None

    @property
    def handle(self) -> int:
        """The raw ``CUgraphicsResource`` handle as a Python int."""
        return as_intptr(self._handle)

    @property
    def resource_handle(self) -> int:
        """Alias for :attr:`handle`."""
        return self.handle

    def __repr__(self):
        mapped_str = " mapped" if self.is_mapped else ""
        closed_str = " closed" if not self._handle else ""
        return f"<GraphicsResource handle={as_intptr(self._handle):#x}{mapped_str}{closed_str}>"
