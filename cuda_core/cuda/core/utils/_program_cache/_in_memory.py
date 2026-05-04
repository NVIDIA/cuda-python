# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory bytes-in / bytes-out program cache."""

from __future__ import annotations

import collections
import threading

from cuda.core._module import ObjectCode

from ._abc import ProgramCacheResource, _as_key_bytes, _extract_bytes


class InMemoryProgramCache(ProgramCacheResource):
    """In-memory program cache with LRU eviction.

    Suitable for single-process workflows that want to avoid disk I/O --
    a typical application compiles its kernels once per process and
    looks them up many times. Entries live only for the lifetime of
    the process; use :class:`FileStreamProgramCache` when the cache
    should persist across runs.

    Like :class:`FileStreamProgramCache`, this backend is bytes-in /
    bytes-out: ``__setitem__`` accepts ``bytes``, ``bytearray``,
    ``memoryview``, or any :class:`~cuda.core.ObjectCode` (path-backed
    too -- the file is read at write time so the cached entry holds the
    binary content, not a path). ``__getitem__`` returns ``bytes``.

    Parameters
    ----------
    max_size_bytes:
        Optional cap on the sum of stored payload sizes. When exceeded,
        LRU eviction runs until the total fits. ``None`` means
        unbounded. The size-only bound mirrors
        :class:`FileStreamProgramCache`.

    Notes
    -----
    Recency is updated on :meth:`__getitem__`; ``get`` is the
    recommended lookup since the cache deliberately omits
    ``__contains__`` (the ``if key in cache: ...`` idiom is racy
    across processes; see :class:`ProgramCacheResource`).

    Thread safety: a :class:`threading.RLock` serialises every method,
    so the cache can be shared across threads without external
    locking.
    """

    def __init__(
        self,
        *,
        max_size_bytes: int | None = None,
    ) -> None:
        if max_size_bytes is not None and max_size_bytes < 0:
            raise ValueError("max_size_bytes must be non-negative or None")
        self._max_size_bytes = max_size_bytes
        # Key insertion order encodes LRU order: oldest first, newest last.
        # Each value is ``(payload_bytes, payload_size)``; caching the size
        # avoids recomputing ``len(data)`` on every eviction pass.
        self._entries: collections.OrderedDict[bytes, tuple[bytes, int]] = collections.OrderedDict()
        self._total_bytes = 0
        # Reentrant so helper methods that also take the lock can nest
        # without deadlocking.
        self._lock = threading.RLock()

    def __getitem__(self, key: object) -> bytes:
        k = _as_key_bytes(key)
        with self._lock:
            try:
                data, _size = self._entries[k]
            except KeyError:
                raise KeyError(key) from None
            # Touch LRU: a real read promotes the entry to "most recent"
            # so eviction prefers genuinely cold entries.
            self._entries.move_to_end(k)
            return data

    def __setitem__(self, key: object, value: bytes | bytearray | memoryview | ObjectCode) -> None:
        data = _extract_bytes(value)
        size = len(data)
        k = _as_key_bytes(key)
        with self._lock:
            existing = self._entries.pop(k, None)
            if existing is not None:
                self._total_bytes -= existing[1]
            self._entries[k] = (data, size)
            self._total_bytes += size
            self._evict_to_caps()

    def __delitem__(self, key: object) -> None:
        k = _as_key_bytes(key)
        with self._lock:
            try:
                _data, size = self._entries.pop(k)
            except KeyError:
                raise KeyError(key) from None
            self._total_bytes -= size

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0

    # -- eviction ------------------------------------------------------------

    def _evict_to_caps(self) -> None:
        """Evict oldest entries until the size cap is satisfied.

        Called from ``__setitem__`` after an insert/update. Pops from
        the front of the OrderedDict (oldest first). If the
        just-inserted entry on its own exceeds ``max_size_bytes``, the
        loop will evict it too -- mirroring
        :class:`FileStreamProgramCache` (a write that cannot fit does
        not survive its own size-cap pass).
        """
        if self._max_size_bytes is None:
            return
        while self._entries and self._total_bytes > self._max_size_bytes:
            _k, (_data, size) = self._entries.popitem(last=False)
            self._total_bytes -= size
