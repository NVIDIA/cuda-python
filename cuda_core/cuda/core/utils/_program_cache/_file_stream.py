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
import threading
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


def _stat_key(st: os.stat_result) -> tuple:
    """Stat fingerprint used by every stat-guarded path.

    ``(st_ino, st_size, st_mtime_ns)`` is the smallest triple that
    distinguishes "same file" from "file replaced under us": ``st_ino``
    catches replacement, ``st_size`` and ``st_mtime_ns`` catch a write
    that happens to land on the same inode (e.g. truncate-and-write in
    place). Centralised so all four readers compare the same fields.
    """
    return (st.st_ino, st.st_size, st.st_mtime_ns)


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


def _with_sharing_retry(op, *args, on_exhausted=None, **kwargs):
    """Run ``op(*args, **kwargs)`` retrying transient Windows sharing
    violations under the bounded ``_REPLACE_RETRY_DELAYS`` budget.

    On Windows, ``os.replace``/``read_bytes``/``unlink`` can surface
    winerror 5/32/33 (or bare EACCES via ``_is_windows_sharing_violation``)
    while another process briefly holds the file open without share-delete
    rights. The retry hides that contention. Other ``PermissionError``s
    (real ACLs, unexpected winerror) propagate immediately.

    Successful returns and any non-``PermissionError`` exceptions
    (including ``FileNotFoundError``) bubble up unchanged. After the
    budget is exhausted, the helper either calls ``on_exhausted(last_exc)``
    if provided, or re-raises the last sharing-violation exception.
    """
    last_exc: PermissionError | None = None
    for delay in _REPLACE_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        try:
            return op(*args, **kwargs)
        except PermissionError as exc:
            if not _is_windows_sharing_violation(exc):
                raise
            last_exc = exc
    if on_exhausted is not None:
        return on_exhausted(last_exc)
    assert last_exc is not None  # at least one iteration ran and caught a PermissionError
    raise last_exc


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

    def _do_replace() -> bool:
        os.replace(tmp_path, target)
        return True

    return _with_sharing_retry(_do_replace, on_exhausted=lambda _exc: False)


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

    def _do_stat_and_read() -> tuple[os.stat_result, bytes]:
        return path.stat(), path.read_bytes()

    def _exhausted(last_exc):
        raise FileNotFoundError(path) from last_exc

    return _with_sharing_retry(_do_stat_and_read, on_exhausted=_exhausted)


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
            if _stat_key(st_now) != _stat_key(st_before):
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
    if _stat_key(st_now) != _stat_key(st_before):
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
    _with_sharing_retry(path.unlink)


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
    if _stat_key(st_before) != _stat_key(st_now):
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
        if max_size_bytes is not None and max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive or None (0 would evict every write)")
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
        # Incremental size tracker. Without it every ``__setitem__`` would
        # walk ``entries/`` + ``tmp/`` to compute the total -- O(n) per
        # write. With it: writes update the tracker by the net delta in O(1)
        # and only walk on eviction (which already needs the scan to sort
        # entries by atime). The tracker is seeded by one full scan at open
        # time and refreshed on every eviction pass; cross-process drift
        # (other writers/deleters) self-corrects the next time eviction
        # fires. The lock guards mutations so multi-threaded writers in
        # the same process don't interleave the read-modify-write on the
        # int. Skipped entirely when ``max_size_bytes is None`` -- without
        # a cap the tracker is dead weight.
        self._size_lock = threading.Lock()
        self._tracked_size_bytes = self._compute_total_size() if max_size_bytes is not None else 0

    # -- key-to-path helpers -------------------------------------------------

    def _path_for_key(self, key: object) -> Path:
        k = _as_key_bytes(key)
        # Hash the key to a fixed-length identifier so arbitrary-length user
        # keys never exceed per-component filename limits (typically 255 on
        # ext4 / NTFS). With a 256-bit blake2b digest, the cache relies on
        # cryptographic collision resistance for key uniqueness -- two
        # distinct keys hashing to the same path is astronomically unlikely
        # (~2^-128 with the 32-byte digest in use here).
        digest = hashlib.blake2b(k, digest_size=32).hexdigest()
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

        # Stat the existing entry (if any) BEFORE the replace so we can
        # update the tracker by the net delta. A racing writer that lands
        # an ``os.replace`` between this stat and our own makes ``old_size``
        # slightly off; the next ``_enforce_size_cap`` reconciles by
        # re-scanning. Skipped when ``max_size_bytes is None`` (no tracker).
        old_size = 0
        if self._max_size_bytes is not None:
            try:
                old_size = target.stat().st_size
            except FileNotFoundError:
                old_size = 0

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

        if self._max_size_bytes is None:
            return

        # O(1) tracker update. Only run the scan-heavy ``_enforce_size_cap``
        # when this write actually pushes the running total above the cap.
        new_size = len(data)
        with self._size_lock:
            self._tracked_size_bytes += new_size - old_size
            over_cap = self._tracked_size_bytes > self._max_size_bytes
        if over_cap:
            self._enforce_size_cap()

    def __delitem__(self, key: object) -> None:
        path = self._path_for_key(key)
        # Stat before unlink so we can decrement the tracker by the actual
        # on-disk size. Best-effort: if the file vanishes between stat and
        # unlink (concurrent eviction), we treat the delete as a miss --
        # matching the behaviour callers expect (KeyError) and leaving the
        # tracker untouched (the racing eviction already accounted for it).
        size = 0
        if self._max_size_bytes is not None:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                raise KeyError(key) from None
        try:
            _unlink_with_sharing_retry(path)
        except FileNotFoundError:
            raise KeyError(key) from None
        if self._max_size_bytes is not None:
            with self._size_lock:
                # Clamp at zero. A racing ``_enforce_size_cap`` can re-seed the
                # tracker between our stat and our subtract; if its scan ran
                # AFTER we unlinked, its reseed value didn't include ``size``,
                # so subtracting ``size`` again here would undercount reality
                # by ``size``. Repeated under contention, an unclamped subtract
                # walks the tracker negative -- and once negative, the
                # ``tracker > cap`` check that gates ``_enforce_size_cap``
                # never fires, so eviction dies silently and there is no
                # self-healing path (the only reseed point is the function
                # that no longer runs). Clamping leaves us at worst
                # undercounting (the next reseed corrects it) instead of
                # entering the permanently-broken negative state.
                self._tracked_size_bytes = max(0, self._tracked_size_bytes - size)

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
        # ``_iter_entry_paths`` already filters with ``entry.is_file()``,
        # so don't stat each path a second time here.
        return sum(1 for _ in self._iter_entry_paths())

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
        # The directory is now (almost) empty -- but a concurrent writer may
        # have landed a fresh entry between the snapshot and the unlink, and
        # young temp files were intentionally preserved. Re-derive the
        # tracker from the post-clear state instead of zeroing blindly.
        if self._max_size_bytes is not None:
            actual = self._compute_total_size()
            with self._size_lock:
                self._tracked_size_bytes = actual

    # -- internals -----------------------------------------------------------

    def _iter_entry_paths(self) -> Iterable[Path]:
        # ``os.scandir`` returns ``DirEntry`` objects whose ``is_dir`` /
        # ``is_file`` methods consult the cached dirent type from the
        # ``readdir`` result on filesystems that report it (ext4, NTFS, ...),
        # avoiding a per-entry ``stat`` syscall. ``Path.iterdir`` also wraps
        # ``scandir`` but discards the cached type, forcing a separate
        # ``stat`` for every ``Path.is_dir`` / ``Path.is_file``. The ``with``
        # blocks release the underlying directory handle deterministically
        # when the consumer stops early -- otherwise a leaked handle blocks
        # deletes/renames on Windows until GC.
        try:
            with os.scandir(self._entries) as outer:
                for sub in outer:
                    if not sub.is_dir(follow_symlinks=False):
                        continue
                    try:
                        with os.scandir(sub.path) as inner:
                            yield from (Path(entry.path) for entry in inner if entry.is_file(follow_symlinks=False))
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            return

    def _compute_total_size(self) -> int:
        """Walk ``entries/`` + ``tmp/`` and return the on-disk byte total.

        Used to seed the tracker at open time and to refresh it after every
        eviction pass. Best-effort: files that vanish under us during the
        walk (concurrent eviction by this or another process) are skipped.
        Tracked total may briefly differ from this scan's result under
        cross-process contention; the next eviction will reconcile.
        """
        total = 0
        for path in self._iter_entry_paths():
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total + self._sum_tmp_sizes()

    def _iter_tmp_entries(self) -> Iterable[os.DirEntry]:
        # Mirror ``_iter_entry_paths``: scandir + cached d_type for the
        # file/dir filter + deterministic handle close on early exit.
        # Yields ``DirEntry`` (not Path) so callers can use ``entry.stat``
        # / ``entry.path`` directly without an extra wrap.
        try:
            with os.scandir(self._tmp) as it:
                yield from (entry for entry in it if entry.is_file(follow_symlinks=False))
        except FileNotFoundError:
            return

    def _sum_tmp_sizes(self) -> int:
        """Sum sizes of every file in ``tmp/``, skipping vanished entries.

        Both ``_compute_total_size`` (open-time seed) and
        ``_enforce_size_cap`` (eviction reconciliation) need this --
        temp files occupy disk too, so undercounting them would let
        bursts of in-flight writes silently exceed ``max_size_bytes``.
        """
        total = 0
        for entry in self._iter_tmp_entries():
            try:
                total += entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
        return total

    def _sweep_stale_tmp_files(self) -> None:
        """Remove temp files left behind by crashed writers.

        Age threshold is conservative (``_TMP_STALE_AGE_SECONDS``) so an
        in-flight write from another process is not interrupted. Best
        effort: a missing file or a permission failure is ignored.
        """
        cutoff = time.time() - _TMP_STALE_AGE_SECONDS
        for entry in self._iter_tmp_entries():
            try:
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    os.unlink(entry.path)
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
        #
        # Trade-off under burst concurrency: many young temp files (each
        # below the stale-sweep threshold) can push ``total`` above
        # ``max_size_bytes`` with only committed entries left to evict.
        # That can over-evict committed entries during the burst; once
        # the burst subsides and the temps land via ``os.replace`` (or
        # are reaped by a later sweep), the cap re-stabilises. This is
        # consistent with the documented soft-cap contract -- callers
        # that need a hard bound should leave the cap None and prune
        # externally.
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
        total += self._sum_tmp_sizes()
        if total <= self._max_size_bytes:
            # Re-seed the tracker from the scan: catches drift from
            # cross-process writers/deleters that the per-write delta
            # accounting wouldn't have observed. Reaching here means the
            # tracker was over-cap but the disk truth is under-cap, so
            # this assignment is the cheapest reconciliation point we get.
            with self._size_lock:
                self._tracked_size_bytes = total
            return
        entries.sort(key=lambda e: e[0])  # oldest atime first
        for _atime, size, path, st_before in entries:
            if total <= self._max_size_bytes:
                break
            # _prune_if_stat_unchanged refuses if a writer replaced the file
            # between snapshot and now, so eviction can't silently delete a
            # freshly-committed entry from another process.
            try:
                stat_now = path.stat()
            except FileNotFoundError:
                total -= size
                continue
            if _stat_key(stat_now) != _stat_key(st_before):
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
        # Reconcile: after the eviction pass, ``total`` reflects what we
        # believe the disk now holds. Re-seed the tracker so the next write
        # accumulates from a fresh baseline.
        with self._size_lock:
            self._tracked_size_bytes = total
