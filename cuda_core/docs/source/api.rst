.. SPDX-License-Identifier: Apache-2.0

.. module:: cuda.core.experimental

``cuda.core.experimental`` API Reference
========================================

All of the APIs listed (or cross-referenced from) below are considered *experimental*
and subject to future changes without deprecation notice. Once stablized they will be
moved out of the ``experimental`` namespace.


CUDA runtime
------------

.. autosummary::
   :toctree: generated/

   Device
   launch

   :template: dataclass.rst

   EventOptions
   StreamOptions
   LaunchConfig

   Buffer
   MemoryResource
   DeviceMemoryResource
   LegacyPinnedMemoryResource


CUDA compilation toolchain
--------------------------

.. autosummary::
   :toctree: generated/

   Program
   Linker
   ObjectCode

   :template: dataclass.rst

   ProgramOptions
   LinkerOptions


CUDA system information
-----------------------

.. autoproperty:: cuda.core.experimental._system.System.driver_version
.. autoproperty:: cuda.core.experimental._system.System.num_devices
.. autoproperty:: cuda.core.experimental._system.System.devices


.. module:: cuda.core.experimental.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: dataclass.rst

   StridedMemoryView
