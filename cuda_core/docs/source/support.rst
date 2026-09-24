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
(12.x, 13.x) are supported by the same ``cuda-core`` package, at run time: any CUDA driver and any
CUDA Toolkit libraries of a supported major work with the same ``cuda-core`` wheel. The one input
this does not extend to is ``cuda-bindings``, which has a per-release minimum (see
:ref:`cuda-core-bindings-floor` below).

When a new CUDA major version is released and support for the oldest major version is dropped,
``cuda.core`` will release a new major version (e.g., 1.x → 2.0.0).

.. list-table:: CUDA Version Support Matrix
   :header-rows: 1

   * - ``cuda.core`` version
     - Supported CUDA versions
   * - 1.x
     - 12, 13

As with any CUDA library, certain features may impose additional requirements on the minimum
CUDA library or CUDA driver versions. Refer to the individual module documentation for details.

.. _cuda-core-bindings-floor:

``cuda-bindings`` Version Requirements
**************************************

Each ``cuda-core`` release declares, for each supported CUDA major version, a minimum
``cuda-bindings`` version, its *floor*: the newest ``cuda-bindings`` release of that major at the
time of the ``cuda-core`` release, which is the version the published wheels are built against.
The floors of the current release are declared by the ``cu12``/``cu13`` extras of ``cuda-core``
(in ``pyproject.toml``); the build, the import-time check, this page and CI all read them from
there.

.. list-table:: ``cuda-bindings`` floors
   :header-rows: 1

   * - ``cuda-core`` version
     - CUDA 12
     - CUDA 13
   * - |release|
     - ``cuda-bindings`` >= |cuda-bindings-floor-cu12|
     - ``cuda-bindings`` >= |cuda-bindings-floor-cu13|

- **At run time**, ``import cuda.core`` requires an installed ``cuda-bindings`` of the same major
  as the ``cuda-core`` build in use, at least as new as that build's floor, and generated from a
  ``cuda.h`` at least as new (by major.minor) as the one the build compiled against; the published
  wheels are built against the floor's header, so the floor alone satisfies them. An older
  ``cuda-bindings`` fails at import with a message that names the version found, the version
  required, and the ``pip`` command that fixes it. A newer ``cuda-bindings`` of the same major is
  supported.
- **At build time**, a source build requires ``cuda-bindings`` at or above the floor and a
  ``cuda.h`` (``CUDA_PATH`` or ``CUDA_HOME``) of the same major.minor as the header that
  ``cuda-bindings`` was generated from. Any other configuration fails the build with a message
  that names what was found and what is required. Building against an older CUDA Toolkit than
  the floor's minor is not supported.
- **The CUDA driver** is unaffected by the floor. Feature availability is decided by the driver alone: a
  feature the installed driver lacks raises when it is used.

A floor moves with each ``cuda-core`` release, to the newest ``cuda-bindings`` of each major at
that time, and in any release whose changes need a newer ``cuda-bindings`` API. Every move is
listed under "Breaking Changes" in the :doc:`release notes <release>`.

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
