.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _cuda-core-support:

``cuda.core`` Support Policy
============================

Versioning Scheme
-----------------

``cuda.core`` follows `Semantic Versioning (SemVer) <https://semver.org/>`_ with the version
format ``major.minor.patch``:

- **Major**: Bumped when a new CUDA major release is out and support for the oldest CUDA major
  version is dropped. Breaking API changes only happen at major-version boundaries.
- **Minor**: Bumped when new, backward-compatible features are added, or when a new Python feature
  release is out and the oldest supported Python version reaches EOL.
- **Patch**: Bumped for bug fixes and backward-compatible maintenance updates.

Unlike ``cuda.bindings``, the ``cuda.core`` version is *not* aligned with the CUDA Toolkit version.
Consult the table below or the :doc:`release notes <release>` to determine which CUDA versions are
supported by a given ``cuda.core`` release.

Project Lifecycle & Release Cadence
***********************************

- ``cuda.core`` follows its own release cadence, independent of CUDA Toolkit releases, as long as
  SemVer guarantees are maintained.

   - We currently aim for bimonthly releases, though this is subject to change.

- Major version releases are aligned to CUDA major version releases.
- New features may be delivered in minor releases at any time — not gated by the CUDA Toolkit
  release schedule.
- Patch releases can be made on an as-needed basis, subject to urgency and the team's bandwidth.
- We currently do not plan to maintain multiple releases, nor have any backport policy for new features or bug fixes.
- Deprecation notices will be issued at least for one (1) minor release, before the actual removal
  happens.

CUDA Version Support
--------------------

``cuda.core`` is actively maintained to support the two (2) most recent CUDA major versions. For
example, ``cuda.core`` 1.x supports CUDA 12 and 13.

In particular, what this entails is that all CUDA minor versions within the two major releases
(12.x, 13.x) are supported by the same ``cuda-core`` package.

When a new CUDA major version is released and support for the oldest major version is dropped,
``cuda.core`` will release a new major version (e.g., 1.x → 2.0.0).

.. list-table:: CUDA Version Support Matrix
   :header-rows: 1

   * - ``cuda.core`` version
     - Supported CUDA versions
   * - 1.x
     - 12, 13

As with any CUDA library, certain features may impose additional requirements on the minimum
``cuda-bindings``, CUDA library, or CUDA driver versions. Refer to the individual module
documentation for details.

Python Version Support
----------------------

``cuda.core`` supports all Python versions following the `CPython EOL schedule
<https://devguide.python.org/versions/>`_. As of writing, Python 3.10 – 3.14 are supported.

When a new Python feature version is released and the oldest supported version reaches EOL,
``cuda.core`` will bump its minor version accordingly.

Free-threading Build Support
----------------------------

Starting ``cuda-core`` 0.4.0, packages for the `free-threaded interpreter
<https://docs.python.org/3/howto/free-threading-python.html>`_ are shipped to PyPI and conda-forge.
This support is currently *experimental*.

For now, you are responsible for making sure that calls into the underlying CUDA libraries
are thread-safe. This is subject to change.

----

The NVIDIA CUDA Python team reserves the right to amend the above support policy. Any major changes,
however, will be announced to users in advance.
