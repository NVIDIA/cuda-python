.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.core

Concurrency and Thread Safety
=============================

``cuda.core`` allows concurrent reads of its objects from multiple host
threads, but concurrent *mutation* of the same object is **not supported** --
for example, adding nodes to the same graph, or closing a resource while
another thread is using it. Whenever an object is shared across threads and at
least one of them may mutate it, the application is responsible for providing
external synchronization.

The library does protect the integrity of its own internal state (such as
cached attributes and reference counting), so that concurrent reads of the same
object, or any kind of concurrent use of *distinct* objects, cannot corrupt the
interpreter. This is an integrity guarantee only: the ordering and outcome of
concurrent operations on a shared object are otherwise undefined.

Additional limitations apply because ``cuda.core`` inherits the concurrency
constraints of the underlying CUDA driver. Distinct ``cuda.core`` objects can
share driver or context state, so operating on separate objects is not always
safe. For example, modifying peer device access from one thread while another
thread accesses device memory affected by that change is unsafe, even though the
two threads use different objects. The application is responsible for
synchronizing operations that concurrently read and modify shared driver state.
