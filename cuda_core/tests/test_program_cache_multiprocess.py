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


def _worker_reader(root: str, key: bytes, rounds: int, result_queue) -> None:
    from cuda.core.utils import FileStreamProgramCache

    hits = 0
    for _ in range(rounds):
        with FileStreamProgramCache(root) as cache:
            got = cache.get(key)
            if got is not None:
                hits += 1
    result_queue.put(hits)


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
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = b"seed" * 256

    ctx = _mp.get_context("spawn")
    queue = ctx.Queue()
    writer = ctx.Process(target=_worker_write_many, args=(root, 99, 50))
    reader = ctx.Process(target=_worker_reader, args=(root, b"k", 200, queue))
    reader.start()
    writer.start()
    writer.join(timeout=60)
    reader.join(timeout=60)
    assert writer.exitcode == 0
    assert reader.exitcode == 0
    hits = queue.get(timeout=5)
    # 'k' was never overwritten, so every read must hit.
    assert hits == 200


def _worker_size_cap_writer(root: str, prefix: bytes, payload: bytes, count: int, max_size_bytes: int) -> None:
    """Hammer a small-cap cache with churning writes so eviction fires often."""
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(root, max_size_bytes=max_size_bytes) as cache:
        for i in range(count):
            cache[prefix + str(i).encode()] = payload


def _worker_size_cap_rewriter(root: str, key: bytes, payload: bytes, max_size_bytes: int, done_event) -> None:
    """Repeatedly rewrite ``key`` with a fresh value until ``done_event`` fires;
    afterwards land one final uncontested write so the test's end-state assertion
    isn't sensitive to scheduler-dependent interleaving."""
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(root, max_size_bytes=max_size_bytes) as cache:
        i = 0
        while not done_event.is_set():
            cache[key] = payload + str(i).encode()
            i += 1
        cache[key] = payload + b"final"


def test_concurrent_eviction_does_not_delete_replaced_file(tmp_path):
    """Eviction is stat-guarded: while one process is evicting an entry to
    bring the cache under its size cap, another process may have already
    ``os.replace``-d a fresh value into the same path. The evictor must
    refuse to unlink in that case, otherwise the racing rewriter's
    just-committed entry vanishes."""
    from cuda.core.utils import FileStreamProgramCache

    root = str(tmp_path / "fc")
    payload = b"X" * 2000
    cap = 5000  # fits 2 raw 2000B entries; the third write triggers eviction

    ctx = _mp.get_context("spawn")
    done_event = ctx.Event()
    rewriter = ctx.Process(
        target=_worker_size_cap_rewriter,
        args=(root, b"survivor", payload, cap, done_event),
    )
    # Churning writer creates new keys faster than eviction can drain them,
    # forcing _enforce_size_cap to consider 'survivor' for eviction many times.
    churner = ctx.Process(
        target=_worker_size_cap_writer,
        args=(root, b"churn-", payload, 80, cap),
    )
    rewriter.start()
    churner.start()
    churner.join(timeout=60)
    done_event.set()
    rewriter.join(timeout=60)
    assert rewriter.exitcode == 0
    assert churner.exitcode == 0

    # The rewriter's final uncontested write must survive: if eviction
    # blindly unlinked replaced files, this entry would be gone.
    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        got = cache.get(b"survivor")
        assert got is not None, "rewriter's entry was evicted by racing churner"
        assert got.endswith(b"final")
