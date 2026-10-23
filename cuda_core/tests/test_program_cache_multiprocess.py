# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess stress tests for FileStreamProgramCache.

These run without a GPU. They exercise the atomic-rename write path from
multiple processes launched via ``multiprocessing.get_context("spawn")``.
"""

from __future__ import annotations

import multiprocessing as _mp


def _worker_write(root: str, key: bytes, payload: bytes) -> None:
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(root) as cache:
        cache[key] = payload


def _worker_write_many(root: str, base: int, n: int) -> None:
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(root) as cache:
        for i in range(n):
            key = f"proc-{base}-key-{i}".encode()
            cache[key] = f"payload-{base}-{i}".encode()


def _worker_reader(root: str, key: bytes, expected: bytes, rounds: int, result_queue) -> None:
    """Hit ``cache.get(key)`` ``rounds`` times and report (hits, mismatches).

    A mismatch means a read returned non-``None`` but the bytes differed
    from ``expected`` -- that's the torn-file case the cache must never
    expose.
    """
    from cuda.core.utils import FileStreamProgramCache

    hits = 0
    mismatches = 0
    for _ in range(rounds):
        with FileStreamProgramCache(root) as cache:
            got = cache.get(key)
            if got is None:
                continue
            hits += 1
            if got != expected:
                mismatches += 1
    result_queue.put((hits, mismatches))


def test_concurrent_writers_same_key_no_corruption(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    root = str(tmp_path / "fc")
    ctx = _mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_worker_write,
            args=(root, b"shared", f"v{i}".encode() * 64),
        )
        for i in range(6)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    with FileStreamProgramCache(root) as cache:
        # At least one writer must have succeeded; on Windows some writes
        # may silently fail due to PermissionError on os.replace.
        got = cache.get(b"shared")
        assert got is not None, "no writer succeeded"
        assert got.startswith(b"v")


def test_concurrent_writers_distinct_keys_all_survive(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    root = str(tmp_path / "fc")
    n_procs = 4
    per_proc = 25
    ctx = _mp.get_context("spawn")
    procs = [ctx.Process(target=_worker_write_many, args=(root, base, per_proc)) for base in range(n_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    with FileStreamProgramCache(root) as cache:
        for base in range(n_procs):
            for i in range(per_proc):
                key = f"proc-{base}-key-{i}".encode()
                assert cache.get(key) is not None


def test_concurrent_reader_never_sees_torn_file(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    root = str(tmp_path / "fc")
    # Seed 'k' so the reader can hit; the writer writes unrelated keys so 'k'
    # is never overwritten while the reader is active.
    seed = b"seed" * 256
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = seed

    ctx = _mp.get_context("spawn")
    queue = ctx.Queue()
    writer = ctx.Process(target=_worker_write_many, args=(root, 99, 50))
    reader = ctx.Process(target=_worker_reader, args=(root, b"k", seed, 200, queue))
    reader.start()
    writer.start()
    writer.join(timeout=60)
    reader.join(timeout=60)
    assert writer.exitcode == 0
    assert reader.exitcode == 0
    hits, mismatches = queue.get(timeout=5)
    # 'k' was never overwritten, so every read must hit AND every hit must
    # carry the seeded bytes verbatim -- a half-written file would land in
    # mismatches.
    assert hits == 200
    assert mismatches == 0
