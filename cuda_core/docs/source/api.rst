.. currentmodule:: cuda.core.experimental

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


CUDA compilation toolchain
--------------------------

.. autosummary::
   :toctree: generated/

   Program


.. module:: cuda.core.experimental.utils

Utility functions
-----------------

.. autosummary::
   :toctree: generated/

   args_viewable_as_strided_memory

   :template: dataclass.rst

   StridedMemoryView
