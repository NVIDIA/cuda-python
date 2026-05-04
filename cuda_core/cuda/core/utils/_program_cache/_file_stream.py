# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""On-disk bytes-in / bytes-out program cache.

Atomic writes via :func:`os.replace`. Concurrent readers see either the
old entry or the new one, never a partial file. Each entry is the raw
compiled binary so files are directly consumable by external NVIDIA
tools (``cuobjdump``, ``nvdisasm``, ``cuda-gdb``).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

from cuda.core._module import ObjectCode

from ._abc import ProgramCacheResource, _as_key_bytes, _extract_bytes

_ENTRIES_SUBDIR = "entries"
_TMP_SUBDIR = "tmp"
# Temp files older than this are assumed to belong to a crashed writer and
# are eligible for cleanup. Picked large enough that no real ``os.replace``
# write should still be in flight (writes are bounded by mkstemp + write +
# fsync + replace, all fast on healthy disks).
_TMP_STALE_AGE_SECONDS = 3600


_SHARING_VIOLATION_WINERRORS = (5, 32, 33)  # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
_REPLACE_RETRY_DELAYS = (0.0, 0.005, 0.010, 0.020, 0.050, 0.100)  # ~185ms budget


# Exposed as a module-level flag so tests can toggle it without monkeypatching
# ``os.name`` itself (pathlib reads ``os.name`` at instantiation time).
_IS_WINDOWS = os.name == "nt"


def _default_cache_dir() -> Path:
    """OS-conventional default location for the file-stream cache.

    Resolves to the user-cache root for the calling user, with a
    ``program-cache`` leaf so future tooling can place sibling caches
    under the same ``cuda-python`` vendor directory:

    * Linux: ``$XDG_CACHE_HOME/cuda-python/program-cache``
      (default ``~/.cache/cuda-python/program-cache`` per the XDG Base
      Directory spec).
    * Windows: ``%LOCALAPPDATA%\\cuda-python\\program-cache``
      (Windows uses local AppData -- caches don't roam; falls back to
      ``~/AppData/Local`` if the env var is unset).

    CUDA does not support macOS, so no macOS branch is provided.
    """
    if _IS_WINDOWS:
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "cuda-python" / "program-cache"


def _replace_with_sharing_retry(tmp_path: Path, target: Path) -> bool:
    """Atomic rename with Windows-specific retry on sharing/lock violations.

    Returns True on success. Returns False only after the retry budget is
    exhausted on Windows with a genuine sharing violation -- the caller then
    treats the cache write as dropped. Any other ``PermissionError`` (ACLs,
    read-only dir, unexpected winerror, or any POSIX failure) propagates.

    ``ERROR_ACCESS_DENIED`` (winerror 5) is treated as a sharing violation
    because Windows surfaces it when a file is held open without
    ``FILE_SHARE_WRITE`` (Python's default for ``open(p, "wb")``) or while
    a previous unlink is in ``PENDING_DELETE`` -- both are transient.
    """
    for i, delay in enumerate(_REPLACE_RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            os.replace(tmp_path, target)
            return True
        except PermissionError as exc:
            if not _IS_WINDOWS or getattr(exc, "winerror", None) not in _SHARING_VIOLATION_WINERRORS:
                raise
            # Windows sharing violation; loop and try again unless this was the
            # last attempt, in which case fall through and return False.
            if i == len(_REPLACE_RETRY_DELAYS) - 1:
                return False
    return False


def _stat_and_read_with_sharing_retry(path: Path) -> tuple[os.stat_result, bytes]:
    """Snapshot stat and read bytes, retrying briefly on Windows transient
    sharing-violation ``PermissionError``.

    Reads race the rewriter's ``os.replace``: on Windows, the destination
    can be momentarily inaccessible (winerror 5/32/33) while the rename
    completes. Mirroring ``_replace_with_sharing_retry``'s budget keeps
    transient contention from being mistaken for a real read failure.

    Raises ``FileNotFoundError`` on miss or after exhausting the Windows
    sharing-retry budget. Non-Windows ``PermissionError`` propagates.

    On Windows, EACCES (errno 13) is treated as transient too: ``io.open``
    sometimes surfaces a pending-delete or share-mode mismatch as bare
    EACCES with no ``winerror`` attribute, indistinguishable here from
    a true sharing violation. Real ACL problems on a path the cache owns
    would surface consistently; the bounded retry budget keeps the cost
    of treating them as transient negligible.
    """
    last_exc: BaseException | None = None
    for delay in _REPLACE_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            return path.stat(), path.read_bytes()
        except FileNotFoundError:
            raise
        except PermissionError as exc:
            if not _is_windows_sharing_violation(exc):
                raise
            last_exc = exc
    raise FileNotFoundError(path) from last_exc


_UTIME_SUPPORTS_FD = os.utime in os.supports_fd


def _touch_atime(path: Path, st_before: os.stat_result) -> None:
    """Bump ``path``'s atime to "now", preserving its mtime, iff the
    file's stat still matches ``st_before``.

    Eviction sorts by ``st_atime`` so reads must reliably refresh atime
    regardless of OS or filesystem default behavior:

    * Linux ``relatime`` (default) only updates atime when the existing
      atime is older than mtime, which would skew LRU once an entry has
      been read once.
    * NTFS on Windows Vista+ disables atime updates by default
      (``NtfsDisableLastAccessUpdate``) and most modern installations
      keep that off, so a bare read never bumps atime.
    * ``noatime``-mounted filesystems disable updates entirely.

    Calling ``os.utime`` with explicit times bypasses all of the above
    and writes atime directly. The stat-guard is critical: if another
    process ``os.replace``-d a fresh entry into ``path`` between the
    read and this touch, blindly applying ``st_before.st_mtime_ns``
    would roll the new entry's mtime back to the old value and confuse
    the eviction stat-guard (which checks ``(ino, size, mtime_ns)``)
    into deleting a freshly-committed file.

    Where ``os.utime`` supports file descriptors (Linux, macOS), the
    fstat-then-utime pair runs against the same open fd: even if another
    writer replaces the path between our ``os.open`` and the ``fstat``,
    the fd still refers to the file we opened, so the comparison and the
    utime both target the same inode. This closes the residual TOCTOU
    window that a path-based stat + path-based utime would have.

    On Windows, ``os.utime`` is path-only; the fallback re-stats the
    path and accepts a small TOCTOU window between the second stat and
    the utime. That window is microseconds and the worst-case outcome
    is the racing writer's mtime being rolled back by a few hundred
    nanoseconds -- the eviction stat-guard would then refuse to evict
    the slightly-stale entry, costing one cache miss (recompile) but
    not a corrupt eviction.

    Best-effort: any ``OSError`` (read-only mount, restrictive ACLs,
    ...) is swallowed -- size enforcement still bounds the cache, but
    eviction degrades toward FIFO.
    """
    new_atime_ns = time.time_ns()
    if _UTIME_SUPPORTS_FD:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            try:
                st_now = os.fstat(fd)
            except OSError:
                return
            if (st_now.st_ino, st_now.st_size, st_now.st_mtime_ns) != (
                st_before.st_ino,
                st_before.st_size,
                st_before.st_mtime_ns,
            ):
                return
            with contextlib.suppress(OSError):
                os.utime(fd, ns=(new_atime_ns, st_before.st_mtime_ns))
        finally:
            os.close(fd)
        return

    # Path-based fallback (Windows). Best-effort -- residual TOCTOU window
    # documented above.
    try:
        st_now = path.stat()
    except OSError:
        return
    if (st_now.st_ino, st_now.st_size, st_now.st_mtime_ns) != (
        st_before.st_ino,
        st_before.st_size,
        st_before.st_mtime_ns,
    ):
        return
    with contextlib.suppress(OSError):
        os.utime(path, ns=(new_atime_ns, st_before.st_mtime_ns))


def _is_windows_sharing_violation(exc: BaseException) -> bool:
    """Return True if ``exc`` is a Windows sharing/lock violation that
    :func:`_unlink_with_sharing_retry` would have retried.

    Used by best-effort callers to filter out the exhausted-retry case
    while letting other ``PermissionError`` instances (POSIX ACL
    issues, Windows non-sharing winerrors) propagate -- those are real
    configuration problems, not transient contention.

    The ``EACCES`` fallback only fires when ``winerror`` is absent: a
    bare ``EACCES`` (no winerror attached) is the way ``io.open``
    surfaces a pending-delete or share-mode mismatch on Windows. When
    ``winerror`` IS set but is NOT in the sharing set, the OS told us
    exactly what failed and it isn't a sharing violation -- treating it
    as transient would silently swallow real errors like a corrupt
    ACL.
    """
    if not _IS_WINDOWS:
        return False
    if not isinstance(exc, PermissionError):
        return False
    winerror = getattr(exc, "winerror", None)
    if winerror in _SHARING_VIOLATION_WINERRORS:
        return True
    return winerror is None and exc.errno == errno.EACCES


def _unlink_with_sharing_retry(path: Path) -> None:
    """Unlink with Windows-specific retry on sharing/lock violations.

    On Windows, ``Path.unlink`` raises ``PermissionError`` (winerror 5,
    32, or 33; sometimes bare ``EACCES``) when another process holds
    the file open without ``FILE_SHARE_DELETE``. Python's default
    ``open(p, "rb")`` does not pass that flag, so a reader from another
    process briefly blocks our unlink while it reads. Retry with the
    same backoff budget as :func:`_replace_with_sharing_retry` so
    transient contention is not turned into a propagated error.

    Raises ``FileNotFoundError`` if the file is absent; the last
    ``PermissionError`` if the Windows retry budget is exhausted; and
    propagates any non-sharing ``PermissionError`` (or any non-Windows
    ``PermissionError``) immediately. Best-effort callers should use
    :func:`_is_windows_sharing_violation` to filter the exhausted-retry
    case and re-raise any other ``PermissionError``.
    """
    last_exc: PermissionError | None = None
    for delay in _REPLACE_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            path.unlink()
            return
        except FileNotFoundError:
            raise
        except PermissionError as exc:
            if not _is_windows_sharing_violation(exc):
                raise
            last_exc = exc
    if last_exc is not None:
        raise last_exc


def _prune_if_stat_unchanged(path: Path, st_before: os.stat_result) -> None:
    """Unlink ``path`` iff its stat still matches ``st_before``.

    Guards against a cross-process race: a reader that sees a corrupt
    record can have it atomically replaced (via ``os.replace``) by a
    writer before the reader decides to prune. Comparing
    ``(ino, size, mtime_ns)`` before and after rules out that case --
    any mismatch means someone else wrote a new file and we must not
    delete their work. The residual TOCTOU window between stat and
    unlink is narrow; worst case, a very-recently-written entry is
    removed and the next read recompiles.

    Best-effort: a Windows sharing violation that survives the retry
    budget leaves the file in place. The caller is in an eviction or
    cleanup pass, so re-trying on the next pass is the right outcome.
    """
    try:
        st_now = path.stat()
    except FileNotFoundError:
        return
    key_before = (st_before.st_ino, st_before.st_size, st_before.st_mtime_ns)
    key_now = (st_now.st_ino, st_now.st_size, st_now.st_mtime_ns)
    if key_before != key_now:
        return
    try:
        _unlink_with_sharing_retry(path)
    except FileNotFoundError:
        pass
    except PermissionError as exc:
        # Swallow only the exhausted-Windows-sharing case. POSIX ACL
        # errors and Windows non-sharing winerrors are real configuration
        # problems and must surface, not be silently lost during a prune.
        if not _is_windows_sharing_violation(exc):
            raise


class FileStreamProgramCache(ProgramCacheResource):
    """Persistent program cache backed by a directory of atomic files.

    Designed for multi-process use: writes stage a temporary file and then
    :func:`os.replace` it into place, so concurrent readers never observe a
    partially-written entry. Each entry on disk is the raw compiled binary
    -- cubin / PTX / LTO-IR -- with no header, framing, or pickle wrapper,
    so the files are directly consumable by external NVIDIA tools
    (``cuobjdump``, ``nvdisasm``, ``cuda-gdb``).

    Eviction is by least-recently-*read* time: every successful read bumps
    the entry's ``atime``, and the size enforcer evicts oldest atime
    first.

    .. note:: **Best-effort writes.**

        On Windows, ``os.replace`` raises ``PermissionError`` (winerror
        32 / 33) when another process holds the target file open. This
        backend retries with bounded backoff (~185 ms) and, if still
        failing, drops the cache write silently and returns success-shaped
        control flow. The next call will see no entry and recompile. POSIX
        and other ``PermissionError`` codes propagate.

    .. note:: **Atomic for readers, not crash-durable.**

        Each entry's temp file is ``fsync``-ed before ``os.replace``, but
        the containing directory is **not** ``fsync``-ed. A host crash
        between write and the next directory commit may lose recently
        added entries; surviving entries remain consistent.

    .. note:: **Cross-version sharing.**

        The cache is safe to share across ``cuda.core`` patch releases:
        every key produced by :func:`make_program_cache_key` encodes the
        relevant backend/compiler/runtime fingerprints for its
        compilation path (NVRTC entries pin the NVRTC version, NVVM
        entries pin the libNVVM library and IR versions, PTX/linker
        entries pin the chosen linker backend and its version -- and,
        when the cuLink/driver backend is selected, the driver version
        too; nvJitLink-backed PTX entries are deliberately
        driver-version independent). Bumping ``_KEY_SCHEMA_VERSION``
        (mixed into the digest by ``make_program_cache_key``) produces
        new keys that don't collide with old entries: post-bump
        lookups miss the old on-disk paths, and the orphaned files
        are reaped on the next size-cap eviction pass. Entries are
        stored verbatim as the compiled binary, so cross-patch sharing
        only requires that the compiler-pinning surface above stays
        stable -- there is no Python-pickle compatibility involved.

    Parameters
    ----------
    path:
        Directory that owns the cache. Created if missing. If omitted,
        the OS-conventional user cache directory is used:
        ``$XDG_CACHE_HOME/cuda-python/program-cache`` (Linux, defaulting
        to ``~/.cache/cuda-python/program-cache``) or
        ``%LOCALAPPDATA%\\cuda-python\\program-cache`` (Windows).
    max_size_bytes:
        Optional soft cap on total on-disk size. Enforced opportunistically
        on writes; concurrent writers may briefly exceed it. Eviction is by
        least-recently-read time (oldest ``st_atime`` first).
    """

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        max_size_bytes: int | None = None,
    ) -> None:
        if max_size_bytes is not None and max_size_bytes < 0:
            raise ValueError("max_size_bytes must be non-negative or None")
        self._root = Path(path) if path is not None else _default_cache_dir()
        self._entries = self._root / _ENTRIES_SUBDIR
        self._tmp = self._root / _TMP_SUBDIR
        self._max_size_bytes = max_size_bytes
        self._root.mkdir(parents=True, exist_ok=True)
        self._entries.mkdir(exist_ok=True)
        self._tmp.mkdir(exist_ok=True)
        # Opportunistic startup sweep of orphaned temp files left by any
        # crashed writers. Age-based so concurrent in-flight writes from
        # other processes are preserved.
        self._sweep_stale_tmp_files()

    # -- key-to-path helpers -------------------------------------------------

    def _path_for_key(self, key: object) -> Path:
        k = _as_key_bytes(key)
        # Hash the key to a fixed-length identifier so arbitrary-length user
        # keys never exceed per-component filename limits (typically 255 on
        # ext4 / NTFS). With a 256-bit blake2b digest, the cache relies on
        # cryptographic collision resistance for key uniqueness -- two
        # distinct keys hashing to the same path is astronomically unlikely
        # (~2^-128 with the 32-byte digest in use here).
        digest = hashlib.blake2b(k, digest_size=32).hexdigest() if k else "empty"
        if len(digest) < 3:
            digest = digest.rjust(3, "0")
        return self._entries / digest[:2] / digest[2:]

    # -- mapping API ---------------------------------------------------------

    def __getitem__(self, key: object) -> bytes:
        path = self._path_for_key(key)
        try:
            # The helper retries on Windows transient sharing-violation
            # PermissionErrors so a racing rewriter doesn't turn a hit
            # into a spurious propagated error.
            st, data = _stat_and_read_with_sharing_retry(path)
        except FileNotFoundError:
            raise KeyError(key) from None
        # Bump atime to "now" so eviction (which sorts by st_atime) treats
        # this read as the entry's most recent use. Best-effort: filesystems
        # mounted ``noatime`` or with restrictive ACLs may refuse, in which
        # case the cap still bounds size but eviction degrades toward FIFO
        # rather than true LRU.
        _touch_atime(path, st)
        return data

    def __setitem__(self, key: object, value: bytes | bytearray | memoryview | ObjectCode) -> None:
        data = _extract_bytes(value)
        target = self._path_for_key(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-create ``tmp/`` if something deleted it after ``__init__``
        # (operators clearing the cache by hand, ``rm -rf cache_dir/tmp``,
        # another process's overzealous wipe). Cheap and idempotent;
        # without it, every subsequent write would crash with
        # FileNotFoundError even though we could trivially recover.
        self._tmp.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(prefix="entry-", dir=self._tmp)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            # Retry os.replace under Windows sharing/lock violations; only
            # give up (and drop the cache write) after a bounded backoff, so
            # transient contention is not turned into a silent miss.
            # Non-sharing PermissionErrors and all POSIX PermissionErrors
            # propagate immediately (real config problem).
            if not _replace_with_sharing_retry(tmp_path, target):
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
                return
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise
        self._enforce_size_cap()

    def __delitem__(self, key: object) -> None:
        path = self._path_for_key(key)
        try:
            _unlink_with_sharing_retry(path)
        except FileNotFoundError:
            raise KeyError(key) from None

    def __len__(self) -> int:
        """Return the number of files currently in ``entries/``.

        This is a count of on-disk files, not of keys reachable through
        ``make_program_cache_key``. After a ``_KEY_SCHEMA_VERSION`` bump
        old entries become unreachable by lookup but remain on disk
        until eviction reaps them; ``__len__`` keeps counting them
        until then. The same is true for entries written by callers
        using arbitrary user keys -- the backend has no way to tell a
        live entry from an orphan without knowing the caller's keying
        scheme.
        """
        count = 0
        for path in self._iter_entry_paths():
            if path.is_file():
                count += 1
        return count

    def clear(self) -> None:
        # Snapshot stat alongside path so we can refuse to unlink an entry
        # that was concurrently replaced by another process between the
        # snapshot scan and the unlink. Same stat-guard contract as
        # ``_prune_if_stat_unchanged`` and ``_enforce_size_cap``.
        snapshot = []
        for path in self._iter_entry_paths():
            try:
                snapshot.append((path, path.stat()))
            except FileNotFoundError:
                continue
        for path, st_before in snapshot:
            _prune_if_stat_unchanged(path, st_before)
        # Sweep ONLY stale temp files. Deleting a young temp would race with
        # another process between ``mkstemp`` and ``os.replace`` and turn its
        # write into ``FileNotFoundError`` instead of a successful commit.
        self._sweep_stale_tmp_files()
        # Remove empty subdirs (best-effort; concurrent writers may re-create).
        if self._entries.exists():
            for sub in sorted(self._entries.iterdir(), reverse=True):
                if sub.is_dir():
                    with contextlib.suppress(OSError):
                        sub.rmdir()

    # -- internals -----------------------------------------------------------

    def _iter_entry_paths(self) -> Iterable[Path]:
        if not self._entries.exists():
            return
        for sub in self._entries.iterdir():
            if not sub.is_dir():
                continue
            for entry in sub.iterdir():
                if entry.is_file():
                    yield entry

    def _sweep_stale_tmp_files(self) -> None:
        """Remove temp files left behind by crashed writers.

        Age threshold is conservative (``_TMP_STALE_AGE_SECONDS``) so an
        in-flight write from another process is not interrupted. Best
        effort: a missing file or a permission failure is ignored.
        """
        if not self._tmp.exists():
            return
        cutoff = time.time() - _TMP_STALE_AGE_SECONDS
        for tmp in self._tmp.iterdir():
            if not tmp.is_file():
                continue
            try:
                if tmp.stat().st_mtime < cutoff:
                    tmp.unlink()
            except (FileNotFoundError, PermissionError):
                continue

    def _enforce_size_cap(self) -> None:
        if self._max_size_bytes is None:
            return
        # Sweep stale temp files first so a long-dead writer's leftovers
        # don't drag the apparent size up and force needless eviction.
        self._sweep_stale_tmp_files()
        entries = []
        total = 0
        # Count both committed entries AND surviving temp files: temp files
        # occupy disk too, even if they're young. Without this the soft cap
        # silently undercounts in-flight writes.
        for path in self._iter_entry_paths():
            try:
                st = path.stat()
            except FileNotFoundError:
                continue
            # Carry the full stat so eviction can guard against a concurrent
            # os.replace that swapped a fresh entry into this path between
            # snapshot and unlink. Eviction below sorts by ``st_atime`` so
            # entries that callers actually read recently survive
            # write-only churn (true LRU instead of FIFO).
            entries.append((st.st_atime, st.st_size, path, st))
            total += st.st_size
        if self._tmp.exists():
            for tmp in self._tmp.iterdir():
                if not tmp.is_file():
                    continue
                try:
                    total += tmp.stat().st_size
                except FileNotFoundError:
                    continue
        if total <= self._max_size_bytes:
            return
        entries.sort(key=lambda e: e[0])  # oldest atime first
        for _atime, size, path, st_before in entries:
            if total <= self._max_size_bytes:
                return
            # _prune_if_stat_unchanged refuses if a writer replaced the file
            # between snapshot and now, so eviction can't silently delete a
            # freshly-committed entry from another process.
            try:
                stat_now = path.stat()
            except FileNotFoundError:
                total -= size
                continue
            if (stat_now.st_ino, stat_now.st_size, stat_now.st_mtime_ns) != (
                st_before.st_ino,
                st_before.st_size,
                st_before.st_mtime_ns,
            ):
                # File was replaced -- don't unlink, but update ``total`` to
                # reflect the replacement's actual size or the cap check
                # below could declare us done while still over the limit.
                total += stat_now.st_size - size
                continue
            # Tolerate Windows sharing violations during eviction: another
            # process may briefly hold the file open for a read. Skip this
            # entry; a later eviction pass will retry. Same outcome as if
            # the stat-guard above had triggered. Other PermissionErrors
            # (POSIX ACL, Windows non-sharing winerrors) are real config
            # problems -- surface them rather than silently exceed the cap.
            try:
                _unlink_with_sharing_retry(path)
                total -= size
            except FileNotFoundError:
                pass
            except PermissionError as exc:
                if not _is_windows_sharing_violation(exc):
                    raise
