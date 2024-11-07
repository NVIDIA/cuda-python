:orphan:

.. This page is to generate documentation for private classes exposed to users,
   i.e., users cannot instantiate it by themselves but may use it's properties
   or methods via returned values from public APIs. These classes must be referred
   in public APIs returning their instances.

.. currentmodule:: cuda.core.experimental

CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   _memory.Buffer
   _stream.Stream
   _event.Event


CUDA compilation toolchain
--------------------------

.. autosummary::
   :toctree: generated/

   _module.Kernel
   _module.ObjectCode
