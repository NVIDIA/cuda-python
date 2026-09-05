.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.core

Error Handling
==============

``cuda.core`` reports failures with Python exceptions. This page describes what
an exception from ``cuda.core`` guarantees about the state it leaves behind,
what happens when a failure occurs where no exception can be raised, and the
few situations in which ``cuda.core`` cannot fully undo a failed operation.

Exceptions
----------

A CUDA driver, runtime, NVRTC, NVVM or nvJitLink call that fails raises an
exception (``CUDAError`` for driver and runtime failures) whose message contains
the CUDA error name and its description. Invalid arguments and misuse raise the
usual Python exception types (``TypeError``, ``ValueError``, ``RuntimeError``).

When a ``cuda.core`` call raises, the following hold:

- A call that creates a resource creates nothing. If a later step of the call
  fails after the resource was created, the resource is destroyed before the
  exception propagates.
- The calling thread's current CUDA context is the one that was current when
  the call began. The only method that changes the current context on purpose
  is :meth:`Device.set_current`; every other method that must run in a
  different context restores the caller's context before returning, whether it
  succeeds or fails. See `Context restoration failures`_ for the one case in
  which the driver refuses to restore it.
- Objects that were modified by a call that failed midway remain usable and
  consistent, but some operations do not have an all-or-nothing outcome. Their
  documentation says so where it applies (for example the graph mutation
  methods that add several driver edges).

``cuda.core`` does not swallow driver errors. When a second failure occurs
while an exception is being raised, for example the caller's context cannot be
restored after a failed call, or the rollback of a partially built graph node
fails, the second failure is attached to the exception as a note
(:meth:`BaseException.add_note`), which appears in the traceback and in
``__notes__``. Python 3.10 has no exception notes; there the information is
appended to the message when ``cuda.core`` constructs the exception, and
reported as described in the next section otherwise.

Failures that cannot be raised
------------------------------

Some ``cuda.core`` code runs where no Python exception can propagate:

- resources released by the garbage collector or by the deferred cleanup of
  CUDA graphs, and the CUDA driver calls those releases make, including the
  context switch and restoration around such a release;
- callbacks invoked by CUDA.

A CUDA error in one of these places is reported as a :class:`CUDAWarning`. The
message names the failed driver call and the CUDA error. The warning means the
affected resource may have leaked; ``cuda.core`` never leaves a resource in use
by CUDA with its memory released (it prefers a leak to a dangling pointer).

:class:`CUDAWarning` derives from :class:`RuntimeWarning`, so it is shown by
default and can be filtered like any other warning. To make these failures
loud in a test suite::

   import warnings
   import cuda.core

   warnings.filterwarnings("error", category=cuda.core.CUDAWarning)

Because the report comes from a destructor or callback, an escalated warning is
delivered through :func:`sys.unraisablehook` rather than raised into user code.
pytest reports it as ``PytestUnraisableExceptionWarning``, which its
``-W error`` option turns into a test failure.

``CUDA_ERROR_DEINITIALIZED`` is not reported. It means the CUDA driver is
shutting down, which happens during process exit; cleanup failures at that
point are expected and there is nothing left to clean up.

Context restoration failures
----------------------------

Methods that run in a context other than the current one, such as
:meth:`Device.create_stream` when another device is current, switch the current
context, perform the driver call, and switch back. Restoring the caller's
context can fail only when the driver is shutting down
(``CUDA_ERROR_DEINITIALIZED``), when the caller's context was destroyed in the
meantime (``CUDA_ERROR_INVALID_CONTEXT``), or when the driver is reporting an
earlier, unrecoverable kernel fault (see `Sticky errors`_). None of these can
be fixed by retrying, so ``cuda.core`` does not retry.

When restoration fails in an ordinary call, the resource created by the call is
destroyed and a ``CUDAError`` is raised for the failed ``cuCtxSetCurrent``,
with a note stating that the caller's context could not be restored and which
context is now current. If the call itself failed as well, its own error is
raised and the restoration failure is the note. Call
:meth:`Device.set_current` before issuing further CUDA work on that thread.

When restoration fails inside a destructor or callback, a :class:`CUDAWarning`
is issued and the thread keeps the context that the cleanup used.

Sticky errors
-------------

Some CUDA errors mark the process as unusable for further CUDA work, for
example ``CUDA_ERROR_ILLEGAL_ADDRESS`` or ``CUDA_ERROR_LAUNCH_FAILED`` after a
kernel fault. The CUDA documentation calls for the process to be terminated and
relaunched after such an error, and every later CUDA call returns the same
error. Because these faults are detected asynchronously, the call that first
raises the error is often unrelated to the kernel that caused it.

``cuda.core`` raises these errors like any other and does not attempt to
recover from them. It does not terminate the process for you: the exception
carries the Python traceback of the call that observed the fault, and your
application decides how to shut down.

Interpreter shutdown
--------------------

Once the interpreter starts finalizing, ``cuda.core`` no longer touches Python
objects from CUDA callbacks or destructors. Resources whose release would
require Python at that point are intentionally leaked; the operating system and
the driver reclaim them when the process exits. Release all ``cuda.core``
objects explicitly (with ``close()`` or a ``with`` block) if their deterministic
release matters.

Process termination
-------------------

``cuda.core`` does not abort the process in response to a CUDA error, including
errors that cannot be raised, and including failures to restore the caller's
context. Aborting is reserved for an internal invariant violation where
continuing could corrupt memory.
