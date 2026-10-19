.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.core

10 minutes to ``cuda.core``
===========================

Why ``cuda.core``?
------------------

``cuda.core`` gives you a **Pythonic interface to the CUDA runtime**: you can
compile a CUDA C++ kernel at runtime, launch it, move memory, time it, and
capture it into a CUDA graph, all without writing a single raw CUDA driver or
runtime call. It is part of `CUDA Python
<https://nvidia.github.io/cuda-python/>`_, providing high-level, Pythonic access
to the CUDA runtime on top of the low-level
:doc:`cuda.bindings <cuda.bindings:index>`, so you get clean, idiomatic Python:

- **Write GPU kernels from Python.** Compile CUDA C++ to a runnable kernel in a
  few lines, with no separate build step, no ``Makefile``, and no ``nvcc``
  invocation.
- **Stay in the Python GPU ecosystem.** It shares CUDA context, streams, and
  memory with `CuPy <https://cupy.dev/>`_ and `PyTorch <https://pytorch.org/>`_,
  so you can launch your own kernels directly on their arrays, with zero copies.
- **Safe by design.** Resources are real Python objects that you release with
  ``close()``, and errors raise Python exceptions instead of return codes you
  have to check.

If you have ever wanted to drop a custom CUDA kernel into a NumPy/CuPy/PyTorch
workflow without leaving Python, this is the fastest way to do it.

Before you begin
----------------

This is a short, hands-on introduction to ``cuda.core``, geared mainly toward
new users. It walks from "talk to a GPU" to "compile, launch, time, and capture
a kernel" using small, runnable snippets. For the full reference, see the
:doc:`API reference <api>`; for more complete programs, see the
:doc:`examples page <examples>`.

.. note::

   These snippets target CUDA 13 and need a working CUDA 13 installation plus
   ``cuda.core`` and a matching ``cuda.bindings`` (13.x). The quickest way to get
   both from PyPI is:

   .. code-block:: console

      $ pip install "cuda-core[cu13]"

   The :doc:`install page <install>` covers driver requirements, conda, CUDA 12,
   and installing from source.

Customarily, we import as follows:

.. code-block:: python

   from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch


Selecting a device
------------------

:class:`Device` is your entry point. Creating one does **not** initialize the
GPU; calling :meth:`Device.set_current` is what sets up a `CUDA context
<https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#context>`_
on the current host thread. Always call it before doing GPU work.

.. code-block:: python

   dev = Device()        # current device (device 0 if none is current); Device(1) picks a specific GPU
   dev.set_current()     # initialize CUDA and make this device current

   print(dev.name)                 # e.g. 'NVIDIA GB10'
   print(dev.compute_capability)   # ComputeCapability(major=12, minor=1)
   print(dev.arch)                 # '121'  (handy for building kernels below)

:class:`Device` is a thread-local singleton: on the same thread, ``Device(0)``
always returns the same object, so repeated calls are cheap. Rich device
attributes live under :attr:`Device.properties` (for example
``dev.properties.multiprocessor_count``).


Compiling a kernel
------------------

CUDA C++ source is compiled at runtime with :class:`Program`. You describe the
compile with :class:`ProgramOptions` (here: the C++ standard and the target
architecture, taken from ``dev.arch``), call :meth:`Program.compile` to get an
:class:`ObjectCode`, then pull out a callable :class:`Kernel` by name.

.. code-block:: python

   code = r"""
   extern "C" __global__
   void scale(float* data, float factor, size_t n) {
       size_t i = threadIdx.x + blockIdx.x * blockDim.x;
       if (i < n)
           data[i] *= factor;
   }
   """

   opts = ProgramOptions(std="c++17", arch=f"sm_{dev.arch}")
   prog = Program(code, code_type="c++", options=opts)
   mod = prog.compile("cubin")          # "cubin" | "ptx" | "ltoir"
   scale = mod.get_kernel("scale")

We wrap the kernel in ``extern "C"`` so its name is exactly ``"scale"``, which is
the name we pass to :meth:`~ObjectCode.get_kernel`.

When something goes wrong, ``cuda.core`` raises a regular Python exception rather
than returning an error code. A compilation failure is especially friendly: the
exception carries the compiler's log, so you can see exactly what went wrong
(other errors raise an exception with the CUDA error name and description).

.. code-block:: python

   bad = r'extern "C" __global__ void k() { not_a_real_symbol; }'
   try:
       Program(bad, code_type="c++", options=opts).compile("cubin")
   except Exception as e:
       # the exception carries the compiler log
       assert 'identifier "not_a_real_symbol" is undefined' in str(e)


Streams
-------

A :class:`Stream` is an ordered queue of GPU work. Operations on the same stream
run sequentially; operations on separate streams may run concurrently. Most
``cuda.core`` operations that touch the GPU take a stream so you stay in control
of ordering. Create one from the device:

.. code-block:: python

   stream = dev.create_stream()

We create it first because the steps that follow, allocating, copying, and
launching, are all issued on a stream. We will pass this ``stream`` to each of
them in the next two sections.


Allocating memory
-----------------

Memory comes from a :class:`MemoryResource`. Two you'll use early:

- :meth:`Device.allocate` hands you a **device** :class:`Buffer` from the
  device's default memory resource.
- :class:`PinnedMemoryResource` allocates **host** memory that is page-locked
  (and host-accessible), which is convenient for staging data to and from the GPU.

Like :meth:`Device.allocate`, :class:`PinnedMemoryResource` is stream-ordered, so
its :meth:`~PinnedMemoryResource.allocate` also takes a ``stream``.

From here on we use `NumPy <https://numpy.org/>`_ for host data and to check
results, but only as a convenience: ``cuda.core`` works directly on raw memory
buffers and typed pointers, and does not depend on NumPy or any other array
library.

.. code-block:: python

   import numpy as np

   from cuda.core import PinnedMemoryResource

   n = 1024
   nbytes = n * np.dtype(np.float32).itemsize

   pinned = PinnedMemoryResource()
   host = pinned.allocate(nbytes, stream=stream)   # host-accessible buffer
   dbuf = dev.allocate(nbytes, stream=stream)      # device buffer

   print(host.size, host.is_host_accessible)    # 4096 True
   print(dbuf.is_device_accessible)             # True

A :class:`Buffer` is an owning handle to an allocation. Because every
:class:`Buffer` implements ``__dlpack__``, we can get a NumPy view of the
*host* buffer with :func:`numpy.from_dlpack` (no copy, no raw pointers) and
fill it in place. (``from_dlpack`` hands back raw bytes, so we ``.view`` it as
``float32``.)

.. code-block:: python

   host_np = np.from_dlpack(host).view(np.float32)   # writable view, zero-copy
   host_np[:] = np.arange(n, dtype=np.float32)

The same trick works on the *device* side with CuPy: ``cp.from_dlpack(dbuf)``
gives a CuPy array backed by the device buffer.


Copying and launching
---------------------

Copy host to device with :meth:`Buffer.copy_from`, run the kernel with
:func:`launch`, then copy device to host with :meth:`Buffer.copy_to`. These are
all stream-ordered and asynchronous; :meth:`Stream.sync` blocks until the queued
work finishes.

:class:`LaunchConfig` describes the grid and block; :func:`launch` takes the
stream, the config, the kernel, and then the kernel arguments. A :class:`Buffer`
can be passed straight through as a pointer argument. Scalars, however, must
carry an explicit C type, so we use NumPy scalars (``np.float32``,
``np.uint64``) to match the kernel signature.

.. code-block:: python

   dbuf.copy_from(host, stream=stream)          # H2D

   block = 256
   grid = (n + block - 1) // block
   config = LaunchConfig(grid=grid, block=block)
   launch(stream, config, scale, dbuf, np.float32(3.0), np.uint64(n))

   dbuf.copy_to(host, stream=stream)            # D2H
   stream.sync()                                # wait for all of the above

   assert np.array_equal(host_np, np.arange(n, dtype=np.float32) * 3.0)

That is the core ``cuda.core`` workflow: **select a device, compile, allocate,
copy, launch, sync.** Everything below builds on it.


Timing with events
------------------

An :class:`Event` marks a point in a stream. Create timing-enabled events,
:meth:`record <Stream.record>` them around some work, synchronize, and subtract
to get the elapsed time between them in milliseconds.

.. code-block:: python

   start = dev.create_event({"timing_enabled": True})
   end = dev.create_event({"timing_enabled": True})

   stream.record(start)
   for _ in range(100):
       launch(stream, config, scale, dbuf, np.float32(1.0), np.uint64(n))
   stream.record(end)
   end.sync()

   print(f"100 launches took {end - start:.4f} ms")

Events are also how you build cross-stream dependencies without stalling the
host: :meth:`Stream.wait` makes one stream wait on an event (or another stream).


Working with multiple streams
-----------------------------

A single stream is enough for ordered work, but multiple streams let independent
work proceed in parallel. The two launches below touch different buffers, so the
GPU is free to run them at the same time. When a later step *does* depend on
another stream's result, :meth:`Stream.wait` joins them: it makes one stream wait
for another's work to finish before continuing.

.. code-block:: python

   stream_a = dev.create_stream()
   stream_b = dev.create_stream()

   buf_a = dev.allocate(nbytes, stream=stream_a)
   buf_b = dev.allocate(nbytes, stream=stream_b)

   host_np[:] = np.arange(n, dtype=np.float32)   # known input
   buf_a.copy_from(host, stream=stream_a)
   buf_b.copy_from(host, stream=stream_b)

   # Independent work on each stream: free to run concurrently.
   launch(stream_a, config, scale, buf_a, np.float32(2.0), np.uint64(n))
   launch(stream_b, config, scale, buf_b, np.float32(5.0), np.uint64(n))

   # stream_b now needs stream_a's result, so it waits before reading it.
   stream_b.wait(stream_a)
   buf_b.copy_from(buf_a, stream=stream_b)   # safe: buf_a is ready

   buf_b.copy_to(host, stream=stream_b)
   stream_b.sync()
   assert np.array_equal(host_np, np.arange(n, dtype=np.float32) * 2.0)

Without the :meth:`Stream.wait`, the copy on ``stream_b`` could race ahead of the
kernel on ``stream_a``. Whether the two independent launches actually overlap is
up to the GPU scheduler, which can run them together only when each leaves the
device underutilized; using separate streams expresses that the work is
independent and lets the runtime overlap it when it can. The
:doc:`examples <examples>` show this scaled up across multiple GPUs.


Capturing work in a CUDA graph
------------------------------

Repeating the same GPU work many times means paying the CPU cost of submitting
each operation on every repeat. A CUDA graph lets you record that work once and
then launch the whole graph repeatedly, instead of re-issuing each operation
every time. Use the stream's graph builder: begin building, issue your
operations into the *builder* instead of the stream, then complete the graph.

.. code-block:: python

   gb = stream.create_graph_builder()
   gb.begin_building()

   launch(gb, config, scale, dbuf, np.float32(1.0), np.uint64(n))
   launch(gb, config, scale, dbuf, np.float32(1.0), np.uint64(n))

   graph = gb.end_building().complete()

   graph.upload(stream)
   graph.launch(stream)     # replay the whole graph in one shot
   stream.sync()

See :sample-file:`cudaGraphs.py <cudaGraphs/cudaGraphs.py>`
for a complete capture-and-replay example with a measured speedup.

Beyond the stream capture shown here, ``cuda.core`` also provides an explicit
graph interface (the :mod:`cuda.core.graph` module) for building, inspecting, and
modifying graphs directly, including graphs first produced by capture. The
project's `graph tests
<https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/tests/graph>`_ have
many usage examples.


Working with CuPy and PyTorch
-----------------------------

``cuda.core`` is designed to interoperate with the rest of the Python GPU
ecosystem, so in real workflows you often skip manual host buffers entirely and
operate directly on CuPy or PyTorch arrays.

**Current device/context.** Because :meth:`Device.set_current` sets a normal
CUDA context (the standard primary context), other CUDA-runtime libraries pick
it up automatically: if CuPy or PyTorch has already selected a device,
``Device()`` shares it, and vice versa.

**Passing array data to a kernel.** Both CuPy and PyTorch expose their device
pointer, which you pass to :func:`launch` like any other buffer pointer. We reuse
the ``scale`` kernel and ``config`` from above, applying a factor of ``2.0`` to an
array of ones:

.. code-block:: python

   # CuPy exposes its device pointer as .data.ptr
   import cupy as cp

   a = cp.ones(n, dtype=cp.float32)
   dev.sync()  # CuPy fills "a" on its own stream; sync before our stream reads it
   launch(stream, config, scale, a.data.ptr, np.float32(2.0), np.uint64(a.size))
   stream.sync()
   assert bool((a == 2).all())

.. code-block:: python

   # PyTorch streams implement the __cuda_stream__ protocol natively
   import torch

   t = torch.ones(n, dtype=torch.float32, device="cuda")
   ts = dev.create_stream(torch.cuda.current_stream())
   launch(ts, config, scale, t.data_ptr(), np.float32(2.0), np.uint64(t.numel()))
   ts.sync()
   assert bool((t == 2).all())

The ``__cuda_stream__`` protocol is how an object advertises that it represents a
CUDA stream, so :meth:`Device.create_stream` can accept it and drive ``cuda.core``
work on another library's stream. PyTorch implements it natively as of version
2.10. See the :doc:`interoperability guide <interoperability>` for the protocol
details and how to support it in your own types.

**Array-library-agnostic views.** To accept *any* CuPy/PyTorch/NumPy-like array
that supports DLPack or the CUDA Array Interface, decorate a function with
:func:`~cuda.core.utils.args_viewable_as_strided_memory`. The chosen arguments
become :class:`~cuda.core.utils.StridedMemoryView` objects exposing ``ptr``,
``shape``, ``dtype``, and ``is_device_accessible``:

.. code-block:: python

   from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory

   @args_viewable_as_strided_memory((0,))
   def scale_array(arr, work_stream, kern, factor):
       view = arr.view(work_stream.handle)
       assert isinstance(view, StridedMemoryView)
       assert view.is_device_accessible
       size = view.shape[0]
       cfg = LaunchConfig(grid=(size + 255) // 256, block=256)
       launch(work_stream, cfg, kern, view.ptr, np.float32(factor), np.uint64(size))
       work_stream.sync()

   # Works on a CuPy array, a PyTorch tensor, or anything else with DLPack/CAI:
   buf = cp.ones(n, dtype=cp.float32)
   dev.sync()
   scale_array(buf, stream, scale, 2.0)
   assert bool((buf == 2).all())

``cuda.core`` buffers also implement ``__dlpack__``, so a device :class:`Buffer`
can be handed to any DLPack importer for zero-copy exchange.


Cleaning up
-----------

Buffers, streams, events, graphs, and graph builders hold CUDA resources. They
are released when garbage-collected, but you can release them explicitly with
``close()``, which the :cuda-core-examples:`cuda.core examples </>` do in a
``finally`` block.

.. code-block:: python

   graph.close()
   gb.close()
   dbuf.close(stream=stream)
   host.close(stream=stream)
   stream.close()


Where to go next
----------------

You now know the essential ``cuda.core`` workflow and how to plug it into the
Python GPU ecosystem. From here:

- :doc:`Examples <examples>`: runnable programs for templated kernels
  (``name_expressions``), multi-GPU, graphs, JIT link-time optimization, TMA,
  and interop.
- :doc:`Interoperability <interoperability>`: the ``__cuda_stream__`` protocol,
  DLPack/CAI, and ``StridedMemoryView`` in depth.
- :doc:`API reference <api>`: every public class and function, including
  :class:`Linker` (runtime linking/LTO), the memory-resource family
  (:class:`DeviceMemoryResource`, :class:`ManagedMemoryResource`,
  :class:`VirtualMemoryResource`, ...), and the :mod:`graph <cuda.core.graph>`
  node types.
- :doc:`Environment variables <environment_variables>`: runtime knobs such as
  the per-thread default stream.
- Prefer writing kernels in Python instead of CUDA C++? `Numba CUDA
  <https://nvidia.github.io/numba-cuda/>`_ compiles a subset of Python into CUDA
  kernels, and `numba-cuda-mlir <https://nvidia.github.io/numba-cuda-mlir/>`_ is
  its next-generation evolution.
