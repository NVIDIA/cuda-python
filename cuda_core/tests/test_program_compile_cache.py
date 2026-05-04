# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``Program.compile(cache=...)`` convenience integration."""

from __future__ import annotations

import pytest

from cuda.core import Program, ProgramOptions
from cuda.core import _program as _program_module
from cuda.core._module import ObjectCode
from cuda.core.utils import (
    FileStreamProgramCache,
    make_program_cache_key,
)


class _RecordingCache:
    """Minimal recording stub for the bytes-in / bytes-out cache protocol.

    Mirrors :class:`FileStreamProgramCache`'s contract: ``__setitem__``
    accepts bytes-like or :class:`ObjectCode` (extracts bytes), and
    ``get`` returns the stored bytes (or ``None``).

    Intentionally does NOT subclass ``ProgramCacheResource`` -- the wrapper
    should be duck-typed, so we test the duck-typed surface directly.
    """

    def __init__(self, preseed=None):
        self._store: dict[bytes, bytes] = {}
        for k, v in (preseed or {}).items():
            self._store[k] = self._extract(v)
        self.get_calls: list[bytes] = []
        self.set_calls: list[tuple[bytes, bytes]] = []
        self.get_side_effect: BaseException | None = None
        self.set_side_effect: BaseException | None = None

    @staticmethod
    def _extract(value) -> bytes:
        if isinstance(value, ObjectCode):
            return bytes(value.code)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        raise TypeError(f"unexpected value type: {type(value).__name__}")

    def get(self, key, default=None):
        self.get_calls.append(key)
        if self.get_side_effect is not None:
            raise self.get_side_effect
        return self._store.get(key, default)

    def __setitem__(self, key, value):
        data = self._extract(value)
        self.set_calls.append((key, data))
        if self.set_side_effect is not None:
            raise self.set_side_effect
        self._store[key] = data


_KERNEL = 'extern "C" __global__ void k() {}'
_SENTINEL_BYTES = b"sentinel-cubin-bytes"


def _make_sentinel_object_code():
    """Construct a cache-safe ``ObjectCode`` that doesn't require compilation."""
    return ObjectCode._init(_SENTINEL_BYTES, "cubin", name="sentinel")


def test_cache_miss_runs_compile_then_stores(monkeypatch):
    """On cache miss: get(key) once, _program_compile_uncached once, __setitem__ once."""
    sentinel = _make_sentinel_object_code()
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))

    def _return_sentinel(_program, *_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _return_sentinel)
    cache = _RecordingCache()

    result = program.compile("cubin", cache=cache)

    # On miss the wrapper returns the freshly-compiled ObjectCode unchanged.
    assert result is sentinel
    assert len(cache.get_calls) == 1
    assert len(cache.set_calls) == 1
    # The cache stored the binary bytes extracted from the ObjectCode.
    assert cache.set_calls[0][1] == _SENTINEL_BYTES


def test_cache_hit_returns_object_code_reconstructed_from_bytes(monkeypatch):
    """On hit: get(key) returns bytes, the wrapper rebuilds an ObjectCode with
    the same code_type and ProgramOptions.name. _program_compile_uncached is
    NOT called and there is no __setitem__."""
    options = ProgramOptions(arch="sm_80", name="my_program")
    program = Program(_KERNEL, "c++", options)
    key = make_program_cache_key(
        code=_KERNEL,
        code_type="c++",
        options=options,
        target_type="cubin",
    )

    def _explode(_program, *_args, **_kwargs):
        raise AssertionError("_program_compile_uncached must not be called on cache hit")

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _explode)
    cache = _RecordingCache(preseed={key: _SENTINEL_BYTES})

    result = program.compile("cubin", cache=cache)

    assert isinstance(result, ObjectCode)
    assert bytes(result.code) == _SENTINEL_BYTES
    assert result.code_type == "cubin"
    assert result.name == "my_program"
    assert cache.get_calls == [key]
    assert cache.set_calls == []


def test_cache_hit_emits_ptx_loadability_warning_when_driver_too_old(monkeypatch):
    """The uncached NVRTC path warns when the active driver can't load
    freshly-generated PTX. That loadability is a property of the driver,
    not of how the bytes were produced, so a cache hit on the same
    (NVRTC + ptx target_type) compile must emit the same warning. Without
    this mirror, ``compile("ptx", cache=cache)`` would silently hand back
    PTX that won't actually load on the active driver."""
    options = ProgramOptions(arch="sm_80", name="warn_program")
    program = Program(_KERNEL, "c++", options)
    key = make_program_cache_key(code=_KERNEL, code_type="c++", options=options, target_type="ptx")

    monkeypatch.setattr(_program_module, "_can_load_generated_ptx", lambda: False)

    def _explode(_program, *_args, **_kwargs):
        raise AssertionError("_program_compile_uncached must not be called on cache hit")

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _explode)
    cache = _RecordingCache(preseed={key: _SENTINEL_BYTES})

    with pytest.warns(RuntimeWarning, match="driver"):
        result = program.compile("ptx", cache=cache)

    assert bytes(result.code) == _SENTINEL_BYTES


def test_cache_hit_no_ptx_warning_when_driver_supports_it(monkeypatch):
    """Inverse of the warning test: when the driver can load the PTX, no
    warning is emitted on cache hit (the wrapper must not be over-eager)."""
    import warnings

    options = ProgramOptions(arch="sm_80", name="quiet_program")
    program = Program(_KERNEL, "c++", options)
    key = make_program_cache_key(code=_KERNEL, code_type="c++", options=options, target_type="ptx")

    monkeypatch.setattr(_program_module, "_can_load_generated_ptx", lambda: True)

    def _explode(_program, *_args, **_kwargs):
        raise AssertionError("_program_compile_uncached must not be called on cache hit")

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _explode)
    cache = _RecordingCache(preseed={key: _SENTINEL_BYTES})

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        program.compile("ptx", cache=cache)


def test_cache_rejects_name_expressions():
    """``name_expressions`` is incompatible with ``cache=``: the cache stores
    raw binary bytes, but ``ObjectCode.symbol_mapping`` (populated by
    NVRTC name-expression mangling) is not preserved across a cache
    round-trip. Without an explicit rejection, the first call (miss)
    would return an ObjectCode with mappings while every subsequent call
    (hit) would return one without -- silently breaking later
    ``get_kernel(name_expression)`` lookups."""
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))
    cache = _RecordingCache()

    with pytest.raises(ValueError, match="name_expressions"):
        program.compile("cubin", name_expressions=("foo",), cache=cache)

    # Wrapper rejects BEFORE touching the cache.
    assert cache.get_calls == []
    assert cache.set_calls == []


def test_cache_accepts_empty_name_expressions(monkeypatch):
    """An empty ``name_expressions`` (the default) must NOT be rejected --
    it's the no-op case, fully supported by the cache."""
    sentinel = _make_sentinel_object_code()
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))
    monkeypatch.setattr(_program_module, "_program_compile_uncached", lambda _self, *_args, **_kwargs: sentinel)
    cache = _RecordingCache()

    # Default empty tuple, explicit empty tuple, and explicit empty list
    # all go through.
    program.compile("cubin", cache=cache)
    program.compile("cubin", name_expressions=(), cache=cache)
    program.compile("cubin", name_expressions=[], cache=cache)


def test_cache_raises_for_extra_digest_required_option():
    """Options that require an ``extra_digest`` propagate a ValueError."""
    program = Program(
        _KERNEL,
        "c++",
        ProgramOptions(arch="sm_80", include_path=["/some/dir"]),
    )
    cache = _RecordingCache()

    with pytest.raises(ValueError, match="extra_digest"):
        program.compile("cubin", cache=cache)

    assert cache.get_calls == []
    assert cache.set_calls == []


def test_cache_raises_for_nvrtc_name_with_dir_component():
    """NVRTC ``options.name`` with a directory component must propagate a
    ValueError: NVRTC resolves quoted ``#include`` directives relative to
    that directory, so neighbour-header changes wouldn't invalidate the
    cache without an extra_digest."""
    program = Program(
        _KERNEL,
        "c++",
        ProgramOptions(arch="sm_80", name="kernels/foo.cu"),
    )
    cache = _RecordingCache()

    with pytest.raises(ValueError, match="directory component"):
        program.compile("cubin", cache=cache)

    assert cache.get_calls == []
    assert cache.set_calls == []


def test_cache_raises_for_side_effect_option(tmp_path):
    """Options with compile-time side effects can't be cached."""
    program = Program(
        _KERNEL,
        "c++",
        ProgramOptions(arch="sm_80", create_pch=str(tmp_path / "k.pch")),
    )
    cache = _RecordingCache()

    with pytest.raises(ValueError):
        program.compile("cubin", cache=cache)

    assert cache.get_calls == []
    assert cache.set_calls == []


def test_cache_miss_compile_failure_does_not_store(monkeypatch):
    """If _program_compile_uncached raises after a miss, the cache is not written."""
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))

    def _boom(_program, *_args, **_kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _boom)
    cache = _RecordingCache()

    with pytest.raises(RuntimeError, match="compile failed"):
        program.compile("cubin", cache=cache)

    assert len(cache.get_calls) == 1
    assert cache.set_calls == []


def test_cache_read_exception_propagates(monkeypatch):
    """Exceptions from cache.get propagate and compile is not invoked."""
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))

    def _explode(_program, *_args, **_kwargs):
        raise AssertionError("_program_compile_uncached must not be called when get raises")

    monkeypatch.setattr(_program_module, "_program_compile_uncached", _explode)
    cache = _RecordingCache()
    cache.get_side_effect = RuntimeError("broken")

    with pytest.raises(RuntimeError, match="broken"):
        program.compile("cubin", cache=cache)


def test_cache_write_exception_propagates(monkeypatch):
    """Exceptions from cache.__setitem__ propagate after compile runs."""
    sentinel = _make_sentinel_object_code()
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))
    monkeypatch.setattr(_program_module, "_program_compile_uncached", lambda _self, *_args, **_kwargs: sentinel)
    cache = _RecordingCache()
    cache.set_side_effect = RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        program.compile("cubin", cache=cache)

    assert len(cache.get_calls) == 1
    assert len(cache.set_calls) == 1


def test_no_cache_kwarg_does_not_derive_key(monkeypatch):
    """Without cache=, no cache-module functions run; compile goes straight through."""
    sentinel = _make_sentinel_object_code()
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))
    monkeypatch.setattr(_program_module, "_program_compile_uncached", lambda _self, *_args, **_kwargs: sentinel)

    # If the implementation accidentally derived a key, it would call
    # make_program_cache_key. Replace it with a raising stub to catch that.
    from cuda.core.utils import _program_cache as _pc

    def _cache_path_must_not_run(*_args, **_kwargs):
        raise AssertionError("cache path must not run when cache= is omitted")

    monkeypatch.setattr(_pc, "make_program_cache_key", _cache_path_must_not_run)

    result = program.compile("cubin")

    assert result is sentinel


def test_filestream_hit_returns_byte_equal_object_code(init_cuda, tmp_path):
    """End-to-end: real compile, FileStreamProgramCache roundtrip, second
    compile returns an ObjectCode whose bytes match the first compile."""
    program = Program(_KERNEL, "c++", ProgramOptions(arch="sm_80"))
    cache_dir = tmp_path / "fc"

    with FileStreamProgramCache(cache_dir) as cache:
        first = program.compile("cubin", cache=cache)

    with FileStreamProgramCache(cache_dir) as cache:
        second = program.compile("cubin", cache=cache)

    assert bytes(second.code) == bytes(first.code)
    assert second.code_type == first.code_type


def test_cache_kwarg_roundtrip_across_reopen(init_cuda, tmp_path, monkeypatch):
    """Compile with cache= in one 'session', reopen and fetch via cache= again."""
    init_args = (_KERNEL, "c++", ProgramOptions(arch="sm_80", name="cached_kernel"))
    cache_path = tmp_path / "fc"

    with FileStreamProgramCache(cache_path) as cache:
        program = Program(*init_args)
        first = program.compile("cubin", cache=cache)

    # Fresh process / fresh Program and cache-handle -- same cache path.
    with FileStreamProgramCache(cache_path) as cache:
        program = Program(*init_args)

        # If the reopened cache misses, the wrapper would fall back to
        # _program_compile_uncached -- replace it with a raising stub so
        # the test can only succeed via a hit.
        def _must_not_recompile(_program, *_args, **_kwargs):
            raise AssertionError("cache miss: reopened cache didn't serve entry")

        monkeypatch.setattr(_program_module, "_program_compile_uncached", _must_not_recompile)
        second = program.compile("cubin", cache=cache)

    assert bytes(second.code) == bytes(first.code)
    assert second.name == "cached_kernel"
