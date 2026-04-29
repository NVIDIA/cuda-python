# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class and value-coercion helpers shared by every cache backend."""

from __future__ import annotations

import abc
import collections.abc
from pathlib import Path

from cuda.core._module import ObjectCode


def _extract_bytes(value: object) -> bytes:
    """Return the raw binary bytes to store on disk.

    Accepts ``bytes``, ``bytearray``, ``memoryview``, or any
    :class:`ObjectCode`. Path-backed ``ObjectCode`` (created via
    ``ObjectCode.from_cubin('/path')`` etc.) is read from the filesystem
    at write time so the cached entry is the binary content itself, not
    a path that could later be moved or modified.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, ObjectCode):
        code = value.code
        if isinstance(code, str):
            return Path(code).read_bytes()
        return bytes(code)
    raise TypeError(f"cache values must be bytes-like or ObjectCode, got {type(value).__name__}")


def _as_key_bytes(key: object) -> bytes:
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    if isinstance(key, str):
        return key.encode("utf-8")
    raise TypeError(f"cache keys must be bytes or str, got {type(key).__name__}")


class ProgramCacheResource(abc.ABC):
    """Abstract base class for compiled-program caches.

    Concrete implementations store and retrieve **raw binary bytes** keyed
    by ``bytes`` or ``str``. A ``str`` key is encoded as UTF-8 before
    being used, so ``"k"`` and ``b"k"`` refer to the same entry. A typical
    key is produced by :func:`make_program_cache_key`, which returns
    ``bytes``.

    The values written are the compiled program bytes themselves --
    cubin, PTX, LTO-IR, etc. Reads return raw bytes so cache files
    remain consumable by external NVIDIA tools (``cuobjdump``,
    ``nvdisasm``, ``cuda-gdb``, ...).

    Most callers don't interact with this object directly. The
    recommended usage is :meth:`cuda.core.Program.compile`'s ``cache=``
    keyword, which derives the key, returns a fresh
    :class:`~cuda.core.ObjectCode` on hit, and stores the compile
    result on miss::

        with FileStreamProgramCache() as cache:
            obj = program.compile("cubin", cache=cache)

    The escape hatch -- only needed when the compile inputs require an
    ``extra_digest`` (header / PCH content fingerprints, NVVM
    libdevice) -- is to call :func:`make_program_cache_key` yourself
    and use the cache as a plain ``bytes`` mapping::

        from cuda.core._module import ObjectCode

        key = make_program_cache_key(
            code=source,
            code_type="c++",
            options=options,
            target_type="cubin",
            extra_digest=header_fingerprint(),
        )
        data = cache.get(key)
        if data is None:
            obj = program.compile("cubin")
            cache[key] = obj  # extracts bytes(obj.code)
        else:
            obj = ObjectCode._init(data, "cubin")

    The cache layer does no payload validation; bytes go in and come
    back out unchanged. Symbol-mapping metadata that
    :class:`~cuda.core.ObjectCode` carries when produced with NVRTC
    name expressions is **not** preserved across a cache round-trip --
    the binary alone is stored. Callers that need ``symbol_mapping``
    for ``get_kernel(name_expression)`` should compile fresh, or look
    the mangled symbol up by hand.

    .. note:: **Concurrent-access idiom.**

        Use :meth:`get` (or ``data = cache[key]`` inside a ``try /
        except KeyError``) for lookups. There is intentionally no
        ``__contains__``: the obvious ``if key in cache: data =
        cache[key]`` idiom is racy across processes (another writer
        can ``os.replace`` over the entry, or eviction can unlink
        it, between the check and the read), and exposing
        ``__contains__`` invites that pattern. ``get`` answers
        both questions in one filesystem-level operation, so a
        successful return always carries the bytes.
    """

    @abc.abstractmethod
    def __getitem__(self, key: bytes | str) -> bytes:
        """Retrieve the cached binary bytes.

        Raises
        ------
        KeyError
            If ``key`` is not in the cache.
        """

    @abc.abstractmethod
    def __setitem__(self, key: bytes | str, value: bytes | bytearray | memoryview | ObjectCode) -> None:
        """Store ``value`` under ``key``.

        Path-backed :class:`~cuda.core.ObjectCode` is read from disk at
        write time so the cached entry holds the bytes, not a path.
        """

    @abc.abstractmethod
    def __delitem__(self, key: bytes | str) -> None:
        """Remove the entry associated with ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not in the cache.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of entries currently in the cache.

        Implementations that store entries on disk by hashed key may
        count orphaned files (entries from a previous
        ``_KEY_SCHEMA_VERSION`` that are still on disk but no longer
        reachable by post-bump lookups) until eviction reaps them.
        Callers that need an exact count of live entries should not
        rely on ``__len__`` across schema bumps.
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove every entry from the cache."""

    def get(self, key: bytes | str, default: bytes | None = None) -> bytes | None:
        """Return ``self[key]`` or ``default`` if absent."""
        try:
            return self[key]
        except KeyError:
            return default

    def update(
        self,
        items: (
            collections.abc.Mapping[bytes | str, bytes | bytearray | memoryview | ObjectCode]
            | collections.abc.Iterable[tuple[bytes | str, bytes | bytearray | memoryview | ObjectCode]]
        ),
        /,
    ) -> None:
        """Bulk ``__setitem__``.

        Accepts a mapping or an iterable of ``(key, value)`` pairs. Each
        write goes through ``__setitem__`` so backend-specific value
        coercion (e.g. extracting bytes from an :class:`~cuda.core.ObjectCode`)
        and size-cap enforcement run on every entry. Not transactional --
        a failure mid-iteration leaves earlier writes committed.
        """
        if isinstance(items, collections.abc.Mapping):
            items = items.items()
        for key, value in items:
            self[key] = value

    def close(self) -> None:  # noqa: B027
        """Release backend resources. No-op by default."""

    def __enter__(self) -> ProgramCacheResource:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
