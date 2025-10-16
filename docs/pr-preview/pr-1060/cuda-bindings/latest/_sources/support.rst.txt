.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

``cuda.bindings`` Support Policy
================================

The ``cuda.bindings`` module has the following support policy:

1. The module shares the same ``major.minor`` version with the CUDA Toolkit. The patch version (the
   third number in the version string), however, is reserved to reflect Python-only changes and
   is out of sync with the Toolkit patch version.
2. The module is actively maintained to support the latest CUDA major version and its prior major
   version. For example, as of writing the bindings for CUDA 12 & 13 are maintained. Any fix in the
   latest bindings would be backported to the prior major version.
3. The module supports `CUDA minor version compatibility`_, meaning that ``cuda.bindings`` 12.x
   supports any Toolkit 12.y. (Whether or not a binding API would actually correctly function
   depends on the underlying driver and the Toolkit versions, as described in the compatibility
   documentation.)
4. The module supports all Python versions following the `CPython EOL schedule`_. As of writing
   Python 3.9 - 3.13 are supported.
5. The module exposes a Cython layer from which types and functions could be ``cimport``'d. While
   we strive to keep this layer stable, due to Cython limitations a new *minor* release of this
   module could require Cython layer users to rebuild their projects and update their pinning to
   this module.

The NVIDIA CUDA Python team reserves rights to amend the above support policy. Any major changes,
however, will be announced to the users in advance.


.. _CUDA minor version compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility
.. _CPython EOL schedule: https://devguide.python.org/versions/
