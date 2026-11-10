.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _cuda-core-support:

``cuda.core`` Support Policy
============================

``cuda.core`` has its own version numbers and release schedule, independent of the CUDA
Toolkit. For example, ``cuda.core`` 1.x supports both CUDA 12 and CUDA 13.

Common Cases
------------

**Installing a wheel.** For CUDA 13, ``cuda-core`` |release| requires ``cuda-bindings``
|cuda-bindings-floor-cu13| or newer within CUDA 13. Installing ``cuda-core[cu13]`` selects
compatible bindings automatically. The :ref:`requirements table <cuda-core-bindings-floor>`
also gives the minimum for CUDA 12.

**Keeping core while upgrading bindings.** Suppose your installed core wheel works with
``cuda-bindings`` version 13.5.0. You can keep that wheel when upgrading to a newer bindings
release such as 13.6.0.
Newer bindings within the same CUDA major are supported; publishing them does not change an
existing core wheel's minimum requirement.

**Keeping an older driver.** Bindings generated from CUDA 13.5 headers can be used with a
driver supporting CUDA 13.0. The bindings minimum does not require a newer driver or newer
CUDA Toolkit libraries. Individual core features still need a driver and libraries that
provide them.

**Building from source.** Bindings generated from CUDA 13.6 headers require CUDA 13.6 headers
to build core; CUDA 13.5 headers will not work. That custom core build then needs bindings
generated from CUDA 13.6 or newer headers within CUDA 13 at run time. This matching-header
requirement applies when compiling core, not when installing a published wheel.

CUDA Version Support
--------------------

``cuda.core`` supports the two most recent CUDA major versions and all their minor versions
at run time:

.. list-table:: CUDA Version Support Matrix
   :header-rows: 1

   * - ``cuda.core`` version
     - Supported CUDA versions
   * - 1.x
     - 12, 13

Within a supported CUDA major, the wheel works with any driver and CUDA Toolkit library
versions. Individual features can require newer drivers or libraries; consult their module
documentation. When an older driver lacks a feature, core may raise an error or emulate the
feature, rather than crash or return an incorrect result.

``cuda-bindings`` has a separate minimum version, described below. That minimum does not
raise the CUDA driver or Toolkit library requirements.

.. _cuda-core-bindings-floor:

``cuda-bindings`` Version Requirements
--------------------------------------

Each core release declares a minimum bindings version for each supported CUDA major, called
its *floor*. This is a deliberate support baseline: core can rely on that release's APIs and
fixes without maintaining compatibility with older bindings.

.. list-table:: ``cuda-bindings`` floors
   :header-rows: 1

   * - ``cuda-core`` version
     - CUDA 12
     - CUDA 13
   * - |release|
     - ``cuda-bindings`` >= |cuda-bindings-floor-cu12|
     - ``cuda-bindings`` >= |cuda-bindings-floor-cu13|

The floor is the newest bindings release of that CUDA major at the time of the core release,
including its patch version. Published core wheels are built against that floor's headers.
A released wheel keeps its floor when newer bindings become available. A new core release
selects new floors, including when its changes need a newer bindings API; the
:doc:`release notes <release>` list changes under "Breaking Changes".

For published wheels, installing ``cuda-core[cu12]`` or ``cuda-core[cu13]`` selects compatible
bindings. The ``cu12`` and ``cu13`` extras in ``pyproject.toml`` declare the floors; the build,
import check, documentation and CI derive them from there.

At Run Time
***********

``import cuda.core`` requires bindings that satisfy all three conditions:

1. They belong to the same **CUDA major** as the selected core build: CUDA 12 or CUDA 13.
2. Their release version is at least that build's declared floor.
3. They were generated from headers at least as new, by major.minor, as the headers used to
   compile that core build.

The floor alone satisfies the published wheels. A source build using newer headers can
require newer bindings, as described below. Newer bindings of the same CUDA major are
supported. If the installed bindings do not meet the requirements, import fails with a
message naming the required version and how to install it.

When Building from Source
*************************

A source build requires bindings at or above the release's floor. The ``cuda.h`` found
through ``CUDA_PATH`` or ``CUDA_HOME`` must have the same major.minor as the header from which
those bindings were generated. Building against a Toolkit older than the floor's minor is
unsupported. An unsupported configuration fails the build with a message naming what was
found and what is required.

The built artifact records the headers actually used. At run time, it requires bindings
generated from headers at least that new within the same CUDA major, even when the declared
release floor is older.
See :doc:`installation instructions <install>` for source-build commands.

Python Version Support
----------------------

``cuda.core`` supports all Python versions following the `CPython EOL schedule
<https://devguide.python.org/versions/>`_. Currently, Python 3.11 through 3.15 are supported.

When a new Python feature version is released and the oldest supported Python version
reaches EOL, ``cuda.core`` bumps its minor version accordingly.

Free-threading Build Support
----------------------------

Since ``cuda-core`` 0.4.0, packages for the `free-threaded interpreter
<https://docs.python.org/3/howto/free-threading-python.html>`_ are shipped to PyPI and
conda-forge. This support is currently *experimental*.

You are responsible for making sure calls into the underlying CUDA libraries are thread-safe.
This is subject to change.

Versioning Scheme
-----------------

``cuda.core`` follows `Semantic Versioning (SemVer) <https://semver.org/>`_ with the version
format ``major.minor.patch``:

- **Major**: Bumped when a new CUDA major release is out and support for the oldest CUDA major
  is dropped. For example, dropping CUDA 12 would require core 2.0.0. Breaking API changes
  happen only at major-version boundaries.
- **Minor**: Bumped for backward-compatible features, or when a new Python feature release
  coincides with the oldest supported Python version reaching EOL.
- **Patch**: Bumped for bug fixes and backward-compatible maintenance updates.

Project Lifecycle & Release Cadence
***********************************

- Releases follow core's own schedule, independent of CUDA Toolkit releases, while preserving
  the SemVer guarantees. We currently aim for bimonthly releases, subject to change.
- Major releases accompany CUDA major-version transitions, as described above.
- New features may arrive in minor releases at any time. Patch releases are made as needed,
  subject to urgency and the team's bandwidth.
- We currently do not plan to maintain multiple releases or provide a backport policy for new
  features or bug fixes.
- Deprecation notices are issued at least one minor release before removal.

----

The NVIDIA CUDA Python team reserves the right to amend this support policy. Major changes
will be announced to users in advance.
