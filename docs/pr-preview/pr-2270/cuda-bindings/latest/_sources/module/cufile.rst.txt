.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

.. default-role:: cpp:any
.. module:: cuda.bindings.cufile

cufile
======

The ``cuda.bindings.cufile`` Python module wraps the
`cuFile C APIs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html>`_.
Supported on Linux only.

Currently using this module requires NumPy to be present. Any recent NumPy 1.x or 2.x should work.


Functions
---------

.. autosummary::
   :toctree: generated/

   handle_register
   handle_deregister
   buf_register
   buf_deregister
   read
   write
   driver_open
   use_count
   driver_get_properties
   driver_set_poll_mode
   driver_set_max_direct_io_size
   driver_set_max_cache_size
   driver_set_max_pinned_mem_size
   batch_io_set_up
   batch_io_submit
   batch_io_get_status
   batch_io_cancel
   batch_io_destroy
   read_async
   write_async
   stream_register
   stream_deregister
   get_version
   get_parameter_size_t
   get_parameter_bool
   get_parameter_string
   set_parameter_size_t
   set_parameter_bool
   set_parameter_string
   op_status_error
   driver_close


Types
-----

.. autosummary::
   :toctree: generated/

   IOEvents
   Descr
   IOParams
   OpError
   DriverStatusFlags
   DriverControlFlags
   FeatureFlags
   FileHandleType
   Opcode
   Status
   BatchMode
   SizeTConfigParameter
   BoolConfigParameter
   StringConfigParameter
   cuFileError
