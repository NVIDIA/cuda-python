.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

.. default-role:: cpp:any

cudla
=====

Note
----

The cuDLA bindings require a Jetson platform with DLA hardware (Xavier or Orin).
cuDLA is not available on desktop GPUs.

Functions
---------

cuDLA defines the following functions for DLA device management and inference.

.. autofunction:: cuda.bindings.cudla.get_version
.. autofunction:: cuda.bindings.cudla.device_get_count
.. autofunction:: cuda.bindings.cudla.create_device
.. autofunction:: cuda.bindings.cudla.destroy_device
.. autofunction:: cuda.bindings.cudla.mem_register
.. autofunction:: cuda.bindings.cudla.mem_unregister
.. autofunction:: cuda.bindings.cudla.module_load_from_memory
.. autofunction:: cuda.bindings.cudla.module_get_attributes
.. autofunction:: cuda.bindings.cudla.module_unload
.. autofunction:: cuda.bindings.cudla.submit_task
.. autofunction:: cuda.bindings.cudla.device_get_attribute
.. autofunction:: cuda.bindings.cudla.get_last_error
.. autofunction:: cuda.bindings.cudla.set_task_timeout_in_ms

Types
-----

.. autoclass:: cuda.bindings.cudla.ExternalMemoryHandleDesc
.. autoclass:: cuda.bindings.cudla.ExternalSemaphoreHandleDesc
.. autoclass:: cuda.bindings.cudla.ModuleTensorDescriptor
.. autoclass:: cuda.bindings.cudla.Fence
.. autoclass:: cuda.bindings.cudla.DevAttribute
.. autoclass:: cuda.bindings.cudla.ModuleAttribute
.. autoclass:: cuda.bindings.cudla.WaitEvents
.. autoclass:: cuda.bindings.cudla.SignalEvents
.. autoclass:: cuda.bindings.cudla.Task

Enums
-----

.. autoclass:: cuda.bindings.cudla.Status

   .. autoattribute:: cuda.bindings.cudla.Status.Success

.. autoclass:: cuda.bindings.cudla.Mode

   .. autoattribute:: cuda.bindings.cudla.Mode.CUDA_DLA
   .. autoattribute:: cuda.bindings.cudla.Mode.STANDALONE

.. autoclass:: cuda.bindings.cudla.ModuleAttributeType
.. autoclass:: cuda.bindings.cudla.FenceType
.. autoclass:: cuda.bindings.cudla.ModuleLoadFlags
.. autoclass:: cuda.bindings.cudla.SubmissionFlags
.. autoclass:: cuda.bindings.cudla.AccessPermissionFlags
.. autoclass:: cuda.bindings.cudla.DevAttributeType
