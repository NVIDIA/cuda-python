# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import time

import pytest


def test_program_cache_resource_is_abstract():
    from cuda.core.utils import ProgramCacheResource

    assert issubclass(ProgramCacheResource, abc.ABC)
    with pytest.raises(TypeError, match="abstract"):
        ProgramCacheResource()


def test_program_cache_resource_requires_core_methods():
    from cuda.core.utils import ProgramCacheResource

    required = {
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__len__",
        "clear",
    }
    assert required <= ProgramCacheResource.__abstractmethods__
    # ``__contains__`` is intentionally NOT abstract: the racy
    # ``key in cache; data = cache[key]`` idiom should be discouraged
    # in favour of ``data = cache.get(key)``.
    assert "__contains__" not in ProgramCacheResource.__abstractmethods__


def _build_empty_subclass():
    from cuda.core.utils import ProgramCacheResource

    class _Empty(ProgramCacheResource):
        def __getitem__(self, key):
            raise KeyError(key)

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            raise KeyError(key)

        def __len__(self):
            return 0

        def clear(self):
            pass

    return _Empty


def test_program_cache_resource_default_get_returns_default_on_miss():
    sentinel = object()
    cache = _build_empty_subclass()()
    assert cache.get(b"missing", default=sentinel) is sentinel


def test_program_cache_resource_default_get_returns_none_without_default():
    cache = _build_empty_subclass()()
    assert cache.get(b"missing") is None


def test_program_cache_resource_close_is_noop_by_default():
    cache = _build_empty_subclass()()
    cache.close()  # does not raise


def test_program_cache_resource_context_manager_closes():
    from cuda.core.utils import ProgramCacheResource

    closed = []

    class _Tracked(ProgramCacheResource):
        def __getitem__(self, key):
            raise KeyError(key)

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            raise KeyError(key)

        def __len__(self):
            return 0

        def clear(self):
            pass

        def close(self):
            closed.append(True)

    with _Tracked():
        pass
    assert closed == [True]


# ---------------------------------------------------------------------------
# make_program_cache_key
# ---------------------------------------------------------------------------


def _opts(**kw):
    from cuda.core import ProgramOptions

    kw.setdefault("arch", "sm_80")
    return ProgramOptions(**kw)


def _make_key(**overrides):
    """Call ``make_program_cache_key`` with a sensible default baseline.

    Tests only need to state the field(s) they care about; everything
    unspecified defaults to a valid cubin-from-c++ compile over "a"."""
    from cuda.core.utils import make_program_cache_key

    base = dict(code="a", code_type="c++", options=_opts(), target_type="cubin")
    return make_program_cache_key(**{**base, **overrides})


def test_make_program_cache_key_returns_bytes():
    key = _make_key()
    assert isinstance(key, bytes)
    assert len(key) == 32


def test_make_program_cache_key_propagates_as_bytes_typeerror(monkeypatch):
    """A ``TypeError`` out of ``ProgramOptions.as_bytes`` must propagate --
    regressing this to a silent retry/fallback would mint cache keys for
    inputs the real compile path rejects."""
    options = _opts()

    def _boom(*args, **kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(options, "as_bytes", _boom)
    with pytest.raises(TypeError, match="boom"):
        _make_key(options=options)


@pytest.mark.parametrize("code_type, code", [("c++", "void k(){}"), ("ptx", ".version 7.0")])
def test_make_program_cache_key_is_deterministic(code_type, code):
    assert _make_key(code=code, code_type=code_type) == _make_key(code=code, code_type=code_type)


def test_make_program_cache_key_accepts_bytes_code():
    # NVVM IR is bytes; accept both str and bytes equivalently (str is UTF-8).
    k_str = _make_key(code="abc", code_type="nvvm", target_type="ptx")
    k_bytes = _make_key(code=b"abc", code_type="nvvm", target_type="ptx")
    assert k_str == k_bytes


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param({"code": "a"}, {"code": "b"}, id="code"),
        pytest.param({"target_type": "ptx"}, {"target_type": "cubin"}, id="target_type"),
        pytest.param({"options": _opts(arch="sm_80")}, {"options": _opts(arch="sm_90")}, id="arch"),
        pytest.param(
            {"options": _opts(use_fast_math=True)},
            {"options": _opts(use_fast_math=False)},
            id="option",
        ),
        pytest.param(
            {"options": _opts(name="kernel-a")},
            {"options": _opts(name="kernel-b")},
            id="options.name",
        ),
        # no extra_digest vs some digest -- adding a digest must perturb the key.
        pytest.param({}, {"extra_digest": b"\x01" * 32}, id="extra_digest_added"),
        pytest.param(
            {"extra_digest": b"\x01" * 32},
            {"extra_digest": b"\x02" * 32},
            id="extra_digest_value",
        ),
    ],
)
def test_make_program_cache_key_differs_on(a, b):
    """Every invalidation axis: code, target, arch, option flag, options.name,
    extra_digest presence and value."""
    assert _make_key(**a) != _make_key(**b)


@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param(("driver", "13200"), ("nvJitLink", "12030"), id="backend_flip"),
        pytest.param(("nvJitLink", "12030"), ("nvJitLink", "12040"), id="version_bump"),
    ],
)
def test_make_program_cache_key_ptx_linker_probe_changes(first, second, monkeypatch):
    """PTX keys must reflect both the linker backend choice (nvJitLink vs
    driver) and its version."""
    from cuda.core.utils import _program_cache

    monkeypatch.setattr(_program_cache._keys, "_linker_backend_and_version", lambda: first)
    k1 = _make_key(code=".version 7.0", code_type="ptx")
    monkeypatch.setattr(_program_cache._keys, "_linker_backend_and_version", lambda: second)
    k2 = _make_key(code=".version 7.0", code_type="ptx")
    assert k1 != k2


def test_make_program_cache_key_name_expressions_order_insensitive():
    assert _make_key(name_expressions=("f", "g")) == _make_key(name_expressions=("g", "f"))


@pytest.mark.parametrize("bad", [123, 1.5, object(), None])
def test_make_program_cache_key_rejects_invalid_name_expressions_element(bad):
    """For NVRTC, Program.compile only forwards str/bytes name_expressions;
    persisting a key for an invalid input is just a foot-gun. Reject up front."""
    with pytest.raises(TypeError, match="name_expressions"):
        _make_key(name_expressions=("ok", bad))


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("ptx", ".version 7.0", "cubin", id="ptx"),
        pytest.param("nvvm", "abc", "ptx", id="nvvm"),
    ],
)
def test_make_program_cache_key_ignores_invalid_name_expressions_for_non_nvrtc(code_type, code, target_type):
    """Program.compile silently ignores name_expressions on PTX/NVVM, so
    the cache helper must not reject invalid elements there either --
    otherwise legitimate non-NVRTC compiles fail the cache layer."""
    # Should not raise even though 123 isn't a valid NVRTC name.
    _make_key(code=code, code_type=code_type, target_type=target_type, name_expressions=(123, object()))


@pytest.mark.parametrize("code_type", ["PTX", "C++", "NVVM", "Ptx", "c++"])
def test_make_program_cache_key_normalises_code_type_case(code_type):
    """Program() normalises code_type to lower; the cache helper must do
    the same so callers using ``Program(code, "PTX")`` get the same routing
    and the same key as the lowercase form."""
    # Pick a target valid for any of the lowered code types.
    if code_type.lower() == "nvvm":
        target = "ptx"
        code = "abc"
    elif code_type.lower() == "ptx":
        target = "cubin"
        code = ".version 7.0"
    else:
        target = "cubin"
        code = "void k(){}"
    upper_key = _make_key(code=code, code_type=code_type, target_type=target)
    lower_key = _make_key(code=code, code_type=code_type.lower(), target_type=target)
    assert upper_key == lower_key


def test_make_program_cache_key_name_expressions_str_bytes_distinct():
    """``Program.compile`` records the *original* Python object as the key in
    ``ObjectCode.symbol_mapping``. Returning a cached ObjectCode whose
    mapping-key type differs from the caller's later ``get_kernel`` lookup
    would silently miss, so ``"foo"`` and ``b"foo"`` must produce distinct
    cache keys."""
    assert _make_key(name_expressions=("foo",)) != _make_key(name_expressions=(b"foo",))


def test_make_program_cache_key_rejects_bytearray_in_name_expressions():
    """``bytearray`` is unhashable, and ``Program.compile`` stores each
    element of ``name_expressions`` as a dict key
    (``symbol_mapping[n] = ...`` in ``_program.pyx``). Accepting it in the
    cache helper would mean hits served for inputs the uncached compile
    path crashes on -- so reject up front."""
    with pytest.raises(TypeError, match="bytearray"):
        _make_key(name_expressions=("ok", bytearray(b"bad")))


@pytest.mark.parametrize(
    "code_type, target_type",
    [
        pytest.param("c++", "cubin", id="nvrtc"),
        pytest.param("ptx", "cubin", id="ptx"),
    ],
)
def test_make_program_cache_key_rejects_bytes_code_outside_nvvm(code_type, target_type):
    """``Program()`` only accepts bytes-like code for NVVM; c++ and PTX
    require str. The cache helper must mirror that rejection."""
    with pytest.raises(TypeError, match="code must be str for code_type"):
        _make_key(code=b"abc", code_type=code_type, target_type=target_type)


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("c++", "void k(){}", "cubin", id="nvrtc"),
        pytest.param("ptx", ".version 7.0", "cubin", id="ptx"),
    ],
)
def test_make_program_cache_key_rejects_extra_sources_outside_nvvm(code_type, code, target_type):
    """``Program(code, code_type)`` rejects ``extra_sources`` for non-NVVM
    backends. The cache key path should mirror that and not silently
    fingerprint a configuration the real compile would refuse."""
    with pytest.raises(ValueError, match="extra_sources"):
        _make_key(
            code=code,
            code_type=code_type,
            target_type=target_type,
            options=_opts(extra_sources=[("foo.cu", "int x = 0;")]),
        )


@pytest.mark.parametrize(
    "kwargs, exc_type, match",
    [
        pytest.param({"code_type": "fortran"}, ValueError, "code_type", id="unknown_code_type"),
        pytest.param({"target_type": "exe"}, ValueError, "target_type", id="unknown_target_type"),
        pytest.param({"code": 12345}, TypeError, "code", id="non_str_bytes_code"),
        # Backend-specific target matrix -- Program.compile rejects these
        # combinations, so caching a key for them would be a lie.
        pytest.param(
            {"code_type": "ptx", "target_type": "ltoir"},
            ValueError,
            "not valid for code_type",
            id="ptx_cannot_ltoir",
        ),
        pytest.param(
            {"code_type": "nvvm", "target_type": "cubin"},
            ValueError,
            "not valid for code_type",
            id="nvvm_cannot_cubin",
        ),
    ],
)
def test_make_program_cache_key_rejects(kwargs, exc_type, match):
    with pytest.raises(exc_type, match=match):
        _make_key(**kwargs)


def test_make_program_cache_key_supported_targets_matches_program_compile():
    """``_SUPPORTED_TARGETS_BY_CODE_TYPE`` duplicates the backend target
    matrix in ``_program.pyx``. Guard against drift: parse the pyx source
    with :mod:`tokenize` (which skips string literals and comments) to
    extract ``SUPPORTED_TARGETS`` and assert the two views agree."""
    import ast
    import io
    import tokenize
    from pathlib import Path

    from cuda.core.utils._program_cache._keys import _SUPPORTED_TARGETS_BY_CODE_TYPE

    backend_to_code_type = {"NVRTC": "c++", "NVVM": "nvvm"}
    linker_backends = ("nvJitLink", "driver")

    pyx = Path(__file__).parent.parent / "cuda" / "core" / "_program.pyx"
    text = pyx.read_text()
    marker_idx = text.index("cdef dict SUPPORTED_TARGETS")
    tokens = tokenize.generate_tokens(io.StringIO(text[marker_idx:]).readline)

    depth = 0
    start_offset = None
    end_offset = None
    lines = text[marker_idx:].splitlines(keepends=True)
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line))

    def _offset(row, col):
        return line_starts[row - 1] + col

    for tok in tokens:
        if tok.type != tokenize.OP:
            continue
        if tok.string == "{":
            if depth == 0:
                start_offset = _offset(tok.start[0], tok.start[1])
            depth += 1
        elif tok.string == "}":
            depth -= 1
            if depth == 0:
                end_offset = _offset(tok.end[0], tok.end[1])
                break
    assert start_offset is not None and end_offset is not None, "could not locate SUPPORTED_TARGETS literal"
    pyx_targets = ast.literal_eval(text[marker_idx + start_offset : marker_idx + end_offset])

    for backend, code_type in backend_to_code_type.items():
        assert frozenset(pyx_targets[backend]) == _SUPPORTED_TARGETS_BY_CODE_TYPE[code_type], (
            backend,
            code_type,
        )
    linker_sets = [frozenset(pyx_targets[b]) for b in linker_backends]
    assert all(s == linker_sets[0] for s in linker_sets)
    assert linker_sets[0] == _SUPPORTED_TARGETS_BY_CODE_TYPE["ptx"]


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("nvvm", "abc", "ptx", id="nvvm"),
        pytest.param("ptx", ".version 7.0", "cubin", id="ptx"),
    ],
)
def test_make_program_cache_key_ignores_name_expressions_for_non_nvrtc(code_type, code, target_type):
    """Program.compile only forwards ``name_expressions`` on the NVRTC path
    (_program.pyx). Folding them into the key for NVVM/PTX compiles would
    cause identical compiles to miss the cache for no behavioural reason."""
    k_none = _make_key(code=code, code_type=code_type, target_type=target_type)
    k_with = _make_key(code=code, code_type=code_type, target_type=target_type, name_expressions=("foo", "bar"))
    assert k_none == k_with


@pytest.mark.parametrize(
    "a, b",
    [
        # ``debug`` / ``lineinfo`` / ``link_time_optimization`` are truthy-only
        # gates in the linker; False and None produce identical output.
        pytest.param({"debug": False}, {"debug": None}, id="debug_false_eq_none"),
        pytest.param({"lineinfo": False}, {"lineinfo": None}, id="lineinfo_false_eq_none"),
        pytest.param(
            {"link_time_optimization": False},
            {"link_time_optimization": None},
            id="lto_false_eq_none",
        ),
        # ``time`` is a presence gate: the linker emits ``-time`` for any
        # non-None value, so True / "path" produce the same flag.
        pytest.param({"time": True}, {"time": "timing.csv"}, id="time_true_eq_path"),
        # ``no_cache`` has an ``is True`` gate; False and None equivalent.
        pytest.param({"no_cache": False}, {"no_cache": None}, id="no_cache_false_eq_none"),
    ],
)
def test_make_program_cache_key_ptx_linker_equivalent_options_hash_same(a, b, monkeypatch):
    """The linker folds several PTX-relevant fields through simple gates:
    truthy-only (``debug``, ``lineinfo``, ``link_time_optimization``),
    presence-only (``time``), ``is True`` (``no_cache``). Semantically
    equivalent inputs under those gates must hash to the same key."""
    # Pin the linker probe so the only variable is the options gate.
    from cuda.core.utils import _program_cache

    monkeypatch.setattr(_program_cache._keys, "_linker_backend_and_version", lambda: ("nvJitLink", "12030"))
    k_a = _make_key(code=".version 7.0", code_type="ptx", options=_opts(**a))
    k_b = _make_key(code=".version 7.0", code_type="ptx", options=_opts(**b))
    assert k_a == k_b


@pytest.mark.parametrize(
    "field, a, b",
    [
        pytest.param("ftz", True, False, id="ftz"),
        pytest.param("prec_div", True, False, id="prec_div"),
        pytest.param("prec_sqrt", True, False, id="prec_sqrt"),
        pytest.param("fma", True, False, id="fma"),
    ],
)
def test_make_program_cache_key_ptx_driver_ignored_fields_collapse(field, a, b, monkeypatch):
    """The driver (cuLink) linker silently ignores ftz/prec_div/prec_sqrt/fma
    (only emits a DeprecationWarning). Under the driver backend, those
    fields must not perturb the PTX cache key -- two otherwise-equivalent
    compiles differing only in these flags produce identical ObjectCode."""
    from cuda.core import _linker

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: True)  # driver
    k_a = _make_key(code=".version 7.0", code_type="ptx", options=_opts(**{field: a}))
    k_b = _make_key(code=".version 7.0", code_type="ptx", options=_opts(**{field: b}))
    assert k_a == k_b


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param("-v", ["-v"], id="str_vs_list"),
        pytest.param("-v", ("-v",), id="str_vs_tuple"),
        pytest.param(["-v"], ("-v",), id="list_vs_tuple"),
        # Empty sequence emits no -Xptxas flags; must match None.
        pytest.param(None, [], id="none_vs_empty_list"),
        pytest.param(None, (), id="none_vs_empty_tuple"),
        pytest.param([], (), id="empty_list_vs_empty_tuple"),
    ],
)
def test_make_program_cache_key_ptx_ptxas_options_canonicalized(a, b, monkeypatch):
    """_prepare_nvjitlink_options emits the same -Xptxas= flags for str,
    list, and tuple shapes of ptxas_options. The cache key must treat them
    as equivalent so equivalent compiles don't miss the cache."""
    from cuda.core import _linker

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: False)  # nvJitLink
    k_a = _make_key(code=".version 7.0", code_type="ptx", options=_opts(ptxas_options=a))
    k_b = _make_key(code=".version 7.0", code_type="ptx", options=_opts(ptxas_options=b))
    assert k_a == k_b


def test_make_program_cache_key_ptx_driver_ignored_fields_still_matter_under_nvjitlink(monkeypatch):
    """nvJitLink does honour those fields; they must still differentiate keys there."""
    from cuda.core import _linker

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: False)  # nvJitLink
    k_a = _make_key(code=".version 7.0", code_type="ptx", options=_opts(ftz=True))
    k_b = _make_key(code=".version 7.0", code_type="ptx", options=_opts(ftz=False))
    assert k_a != k_b


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("c++", "void k(){}", "cubin", id="nvrtc"),
        pytest.param("ptx", ".version 7.0", "cubin", id="ptx"),
    ],
)
def test_make_program_cache_key_use_libdevice_ignored_for_non_nvvm(code_type, code, target_type):
    """``use_libdevice`` is only consumed on the NVVM path; NVRTC and PTX
    ignore it, so toggling it must not perturb the cache key elsewhere."""
    k_off = _make_key(code=code, code_type=code_type, target_type=target_type, options=_opts(use_libdevice=False))
    k_on = _make_key(code=code, code_type=code_type, target_type=target_type, options=_opts(use_libdevice=True))
    k_none = _make_key(code=code, code_type=code_type, target_type=target_type, options=_opts(use_libdevice=None))
    assert k_off == k_on == k_none


def test_make_program_cache_key_nvvm_use_libdevice_requires_extra_digest():
    """NVVM with ``use_libdevice=True`` links an external libdevice bitcode
    file whose contents the cache can't observe; require an extra_digest
    or the cached ObjectCode can silently drift under a toolkit upgrade."""
    from cuda.core.utils import make_program_cache_key

    with pytest.raises(ValueError, match="libdevice"):
        make_program_cache_key(
            code="abc",
            code_type="nvvm",
            options=_opts(use_libdevice=True),
            target_type="ptx",
        )
    # With an extra_digest, it's accepted; different digests produce
    # different keys so a caller can represent a libdevice change.
    k_a = make_program_cache_key(
        code="abc",
        code_type="nvvm",
        options=_opts(use_libdevice=True),
        target_type="ptx",
        extra_digest=b"libdev-a" * 4,
    )
    k_b = make_program_cache_key(
        code="abc",
        code_type="nvvm",
        options=_opts(use_libdevice=True),
        target_type="ptx",
        extra_digest=b"libdev-b" * 4,
    )
    assert k_a != k_b


def test_make_program_cache_key_nvvm_use_libdevice_false_equals_none():
    """Program_init gates ``use_libdevice`` on truthiness, so False and None
    compile identically and must hash the same way. (True without an
    extra_digest is rejected; see test_...requires_extra_digest.)"""
    k_none = _make_key(code="abc", code_type="nvvm", target_type="ptx", options=_opts(use_libdevice=None))
    k_false = _make_key(code="abc", code_type="nvvm", target_type="ptx", options=_opts(use_libdevice=False))
    assert k_none == k_false
    # With an explicit extra_digest, True produces a different key.
    k_true = _make_key(
        code="abc",
        code_type="nvvm",
        target_type="ptx",
        options=_opts(use_libdevice=True),
        extra_digest=b"libdev" * 8,
    )
    assert k_true != k_none


def test_make_program_cache_key_nvvm_library_version_changes_key(monkeypatch):
    """Updating libNVVM (different ``module.version()``) must invalidate
    NVVM cache entries even when the IR version stays constant; a patch
    upgrade can change codegen without bumping the IR pair."""

    class _FakeNVVM:
        def __init__(self, lib_version):
            self._lib_version = lib_version

        def version(self):
            return self._lib_version

        def ir_version(self):
            return (1, 8, 3, 0)  # constant -- only the lib version varies

    fake_old = _FakeNVVM((12, 3))
    fake_new = _FakeNVVM((12, 4))
    from cuda.core import _program

    monkeypatch.setattr(_program, "_get_nvvm_module", lambda: fake_old)
    k_old = _make_key(code="abc", code_type="nvvm", target_type="ptx")
    monkeypatch.setattr(_program, "_get_nvvm_module", lambda: fake_new)
    k_new = _make_key(code="abc", code_type="nvvm", target_type="ptx")
    assert k_old != k_new


def test_make_program_cache_key_nvvm_fingerprint_uses_get_nvvm_module(monkeypatch):
    """The fingerprint must call _get_nvvm_module() rather than importing
    cuda.bindings.nvvm directly -- otherwise it bypasses the availability
    /cuda-bindings-version gate and could disagree with the actual NVVM
    compile path."""
    sentinel_called = {"n": 0}

    class _SentinelNVVM:
        def version(self):
            sentinel_called["n"] += 1
            return (12, 9)

        def ir_version(self):
            return (1, 8, 3, 0)

    from cuda.core import _program

    monkeypatch.setattr(_program, "_get_nvvm_module", lambda: _SentinelNVVM())
    _make_key(code="abc", code_type="nvvm", target_type="ptx")
    assert sentinel_called["n"] == 1


def test_make_program_cache_key_nvvm_probe_changes_key(monkeypatch):
    """NVVM keys must reflect the NVVM toolchain identity (IR version)
    so an upgraded libNVVM does not silently reuse pre-upgrade entries."""
    from cuda.core.utils import _program_cache

    monkeypatch.setattr(_program_cache._keys, "_nvvm_fingerprint", lambda: "ir=1.8.3.0")
    k1 = _make_key(code="abc", code_type="nvvm", target_type="ptx")
    monkeypatch.setattr(_program_cache._keys, "_nvvm_fingerprint", lambda: "ir=2.0.3.0")
    k2 = _make_key(code="abc", code_type="nvvm", target_type="ptx")
    assert k1 != k2


@pytest.mark.parametrize(
    "option_kw",
    [
        pytest.param({"time": True}, id="time_true"),
        # ``_prepare_driver_options`` checks ``is not None``, so even the
        # "falsy-but-set" cases must still be rejected at key time.
        pytest.param({"time": False}, id="time_false"),
        pytest.param({"ptxas_options": "-v"}, id="ptxas_options_str"),
        pytest.param({"ptxas_options": ["-v", "-O2"]}, id="ptxas_options_list"),
        pytest.param({"ptxas_options": []}, id="ptxas_options_empty_list"),
        # ProgramOptions.ptxas_options also accepts tuples (and frozenset ()
        # literal is falsy). Lock in parity for all accepted shapes.
        pytest.param({"ptxas_options": ("-v",)}, id="ptxas_options_tuple"),
        pytest.param({"ptxas_options": ()}, id="ptxas_options_empty_tuple"),
        pytest.param({"split_compile": 0}, id="split_compile_zero"),
        pytest.param({"split_compile": 4}, id="split_compile_nonzero"),
        # split_compile_extended is a LinkerOptions-only field; ProgramOptions
        # does not expose it, so it cannot reach the driver linker via
        # Program.compile and is not part of the cache-time guard.
    ],
)
def test_make_program_cache_key_ptx_rejects_driver_linker_unsupported(option_kw, monkeypatch):
    """When the driver (cuLink) linker backend is selected, options that
    ``_prepare_driver_options`` rejects must also be rejected at key time
    so we never cache a compilation that would fail. Uses ``is not None``
    to exactly mirror the driver-linker's own gate."""
    from cuda.core import _linker

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: True)  # driver
    with pytest.raises(ValueError, match="driver linker"):
        _make_key(code=".version 7.0", code_type="ptx", options=_opts(**option_kw))


def test_make_program_cache_key_ptx_accepts_driver_linker_unsupported_with_nvjitlink(monkeypatch):
    """Under nvJitLink those same options are valid and must not be
    rejected at key time."""
    from cuda.core import _linker

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: False)  # nvJitLink
    # Should not raise.
    _make_key(code=".version 7.0", code_type="ptx", options=_opts(time=True))


def test_filestream_cache_replace_retries_on_sharing_violation(tmp_path, monkeypatch):
    """Under Windows sharing/lock violations, os.replace is retried with a
    bounded backoff; a transient violation that clears within the budget
    must still produce a successful cache write."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    real_replace = _os.replace
    calls = {"n": 0}

    def _flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            exc = PermissionError("sharing violation")
            exc.winerror = 32
            raise exc
        return real_replace(src, dst)

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        monkeypatch.setattr(_os, "replace", _flaky_replace)
        cache[b"k"] = _fake_object_code(b"v")  # succeeds on third attempt
        assert calls["n"] == 3
        assert cache[b"k"] == b"v"


def test_filestream_cache_unlink_retries_on_sharing_violation(tmp_path, monkeypatch):
    """``Path.unlink`` retries on Windows sharing violations (winerror 5/32/33
    or bare EACCES). Without this, an evictor unlinking an entry that another
    process briefly holds open for a read crashes the writer that triggered
    eviction -- which is what the win-64 multiprocess CI was hitting."""
    from pathlib import Path

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    real_unlink = Path.unlink
    calls = {"n": 0}

    def _flaky_unlink(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            exc = PermissionError("sharing violation")
            exc.winerror = 32
            raise exc
        return real_unlink(self, *args, **kwargs)

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = _fake_object_code(b"v")
        monkeypatch.setattr(Path, "unlink", _flaky_unlink)
        del cache[b"k"]  # succeeds on third attempt
        assert calls["n"] == 3
        assert cache.get(b"k") is None


def test_filestream_cache_unlink_propagates_non_windows_permission_error(tmp_path, monkeypatch):
    """On non-Windows, PermissionError on unlink is a real config error and
    must surface, never silently retried."""
    from pathlib import Path

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", False)

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = _fake_object_code(b"v")

        def _denied(self, *args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "unlink", _denied)
        with pytest.raises(PermissionError, match="denied"):
            del cache[b"k"]


def test_filestream_cache_unlink_skips_locked_entries_in_eviction(tmp_path, monkeypatch):
    """Eviction tolerates a Windows sharing violation: a locked entry is
    skipped (a later eviction pass will retry), and the eviction does not
    propagate the error to the writer that triggered it."""
    from pathlib import Path

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    # Tiny cap so every write triggers eviction.
    with FileStreamProgramCache(tmp_path / "fc", max_size_bytes=200) as cache:
        cache[b"a"] = b"a" * 100
        cache[b"b"] = b"b" * 100  # both still fit

        # Now make ALL unlink attempts fail with a sharing violation. The
        # next write triggers eviction; eviction must swallow the error
        # rather than crash the writer.
        def _always_locked(self, *args, **kwargs):
            exc = PermissionError("sharing violation")
            exc.winerror = 32
            raise exc

        monkeypatch.setattr(Path, "unlink", _always_locked)
        # Should not raise.
        cache[b"c"] = b"c" * 100


def test_filestream_cache_eviction_propagates_non_sharing_permission_error(tmp_path, monkeypatch):
    """Eviction must NOT silently swallow PermissionErrors that aren't
    Windows sharing violations -- those (POSIX ACL, Windows non-sharing
    winerrors) are real config issues and would otherwise let the cache
    sit over its cap with no signal to the caller."""
    from pathlib import Path

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", False)

    with FileStreamProgramCache(tmp_path / "fc", max_size_bytes=200) as cache:
        cache[b"a"] = b"a" * 100
        cache[b"b"] = b"b" * 100

        def _denied(self, *args, **kwargs):
            raise PermissionError("acl denied")

        monkeypatch.setattr(Path, "unlink", _denied)
        with pytest.raises(PermissionError, match="acl denied"):
            cache[b"c"] = b"c" * 100


def test_is_windows_sharing_violation_predicate(monkeypatch):
    """``_is_windows_sharing_violation`` must:

    * recognise winerror 5/32/33 as transient sharing violations,
    * recognise a bare ``EACCES`` (no winerror) as transient too,
    * NOT swallow a Windows ``PermissionError`` whose winerror is
      something else (e.g. winerror 1224 = ``ERROR_USER_MAPPED_FILE``)
      even when ``errno`` happens to be ``EACCES``,
    * never return True off-Windows.
    """
    import errno as _errno

    from cuda.core.utils import _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    def _make(winerror, errno_value):
        exc = PermissionError("test")
        if winerror is not None:
            exc.winerror = winerror
        exc.errno = errno_value
        return exc

    # Sharing winerrors: True regardless of errno.
    for w in _program_cache._file_stream._SHARING_VIOLATION_WINERRORS:
        assert _program_cache._file_stream._is_windows_sharing_violation(_make(w, _errno.EACCES))
        assert _program_cache._file_stream._is_windows_sharing_violation(_make(w, _errno.EPERM))

    # Bare EACCES (no winerror): True.
    assert _program_cache._file_stream._is_windows_sharing_violation(_make(None, _errno.EACCES))

    # No winerror, non-EACCES errno: False.
    assert not _program_cache._file_stream._is_windows_sharing_violation(_make(None, _errno.EPERM))

    # Windows non-sharing winerror plus EACCES: False (the regression
    # this guard is here for; a non-sharing winerror tells the OS
    # exactly what failed and we must not silently swallow it).
    assert not _program_cache._file_stream._is_windows_sharing_violation(_make(1224, _errno.EACCES))
    assert not _program_cache._file_stream._is_windows_sharing_violation(_make(1224, _errno.EPERM))

    # Off-Windows: always False.
    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", False)
    assert not _program_cache._file_stream._is_windows_sharing_violation(_make(32, _errno.EACCES))
    assert not _program_cache._file_stream._is_windows_sharing_violation(_make(None, _errno.EACCES))


def test_filestream_cache_eviction_propagates_windows_non_sharing_eacces(tmp_path, monkeypatch):
    """Even on Windows, a ``PermissionError`` carrying a non-sharing
    ``winerror`` plus ``errno.EACCES`` must propagate -- the OS named the
    failure mode, and silently swallowing it would mask real config
    problems."""
    import errno as _errno
    from pathlib import Path

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    with FileStreamProgramCache(tmp_path / "fc", max_size_bytes=200) as cache:
        cache[b"a"] = b"a" * 100
        cache[b"b"] = b"b" * 100

        def _named_failure(self, *args, **kwargs):
            exc = PermissionError("user mapped file")
            exc.winerror = 1224  # ERROR_USER_MAPPED_FILE -- not a sharing violation
            exc.errno = _errno.EACCES
            raise exc

        monkeypatch.setattr(Path, "unlink", _named_failure)
        with pytest.raises(PermissionError, match="user mapped file"):
            cache[b"c"] = b"c" * 100


@pytest.mark.parametrize(
    "option_kw",
    [
        # Populated path-like options
        pytest.param({"include_path": "/usr/local/include"}, id="include_path"),
        pytest.param({"pre_include": "stdint.h"}, id="pre_include"),
        pytest.param({"pch": True}, id="pch"),
        pytest.param({"pch_dir": "pch-cache"}, id="pch_dir"),
        # Non-list/tuple Sequence: the compiler iterates it via ``is_sequence``
        # (``isinstance(v, Sequence)``), so the guard must too.
        pytest.param({"include_path": range(1)}, id="include_path_nonempty_range"),
        # Empty-string path-like options -- NVRTC still emits a flag
        # (``--use-pch=``, ``--pch-dir=``, ``--pre-include=``) so the guard
        # must fire for them too.
        pytest.param({"use_pch": ""}, id="use_pch_empty_string"),
        pytest.param({"pch_dir": ""}, id="pch_dir_empty_string"),
        pytest.param({"pre_include": ""}, id="pre_include_empty_string"),
        # For path-shaped fields (``use_pch``, ``pch_dir``), NVRTC's gate is
        # ``is not None``, so even False emits a real flag and must be caught.
        pytest.param({"use_pch": False}, id="use_pch_false"),
        pytest.param({"pch_dir": False}, id="pch_dir_false"),
        # ``include_path`` / ``pre_include`` are NOT in that group: the
        # compiler only emits them for str or non-empty sequences, so
        # ``False`` is silently ignored at compile time -- test the accept
        # path below, not the reject path.
    ],
)
def test_make_program_cache_key_rejects_external_content_without_extra_digest(option_kw):
    """Options that pull in external file content must force an extra_digest:
    the cache cannot observe header/PCH bytes, so silently omitting them
    would yield stale cache hits after header edits."""
    with pytest.raises(ValueError, match="extra_digest"):
        _make_key(options=_opts(**option_kw))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("kernels/foo.cu", id="forward_slash"),
        pytest.param("kernels\\foo.cu", id="backslash"),
        pytest.param("/abs/foo.cu", id="absolute_unix"),
        pytest.param("../parent/foo.cu", id="parent_relative"),
    ],
)
def test_make_program_cache_key_rejects_nvrtc_name_with_dir_component(name):
    """NVRTC uses ``options.name`` as the source filename and resolves
    quoted ``#include "x.h"`` directives relative to its directory. Without
    an ``extra_digest`` the cache cannot observe edits to those neighbour
    headers, so a stale cached binary could be served. Reject the input so
    callers either pass an extra_digest or strip the directory component."""
    with pytest.raises(ValueError, match="directory component"):
        _make_key(options=_opts(name=name))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("foo.cu", id="bare_filename"),
        pytest.param("default_program", id="default"),
        pytest.param("", id="empty"),
    ],
)
def test_make_program_cache_key_accepts_nvrtc_name_without_dir_component(name):
    """Names without a directory component are fine: NVRTC's relative-include
    resolution doesn't reach outside the in-memory program."""
    _make_key(options=_opts(name=name))  # Should not raise.


def test_make_program_cache_key_accepts_nvrtc_name_with_dir_when_extra_digest_supplied():
    """``extra_digest`` is the escape hatch: the caller has fingerprinted
    whatever the directory contributes, so the guard stands down."""
    _make_key(
        options=_opts(name="kernels/foo.cu"),
        extra_digest=b"caller-fingerprint",
    )


@pytest.mark.parametrize(
    "option_kw",
    [
        pytest.param({"include_path": []}, id="include_path_empty_list"),
        pytest.param({"include_path": ()}, id="include_path_empty_tuple"),
        pytest.param({"pre_include": []}, id="pre_include_empty_list"),
        # ``_prepare_nvrtc_options_impl`` only emits include_path / pre_include
        # for str or non-empty sequence, so False (or any non-str non-sequence)
        # is silently ignored at compile time and must not trip the guard.
        pytest.param({"include_path": False}, id="include_path_false"),
        pytest.param({"pre_include": False}, id="pre_include_false"),
        # Empty non-list/tuple Sequence: ``_prepare_nvrtc_options_impl`` uses
        # ``is_sequence`` (i.e. ``isinstance(v, Sequence)``); a zero-length
        # sequence produces no emission regardless of type.
        pytest.param({"include_path": range(0)}, id="include_path_empty_range"),
    ],
)
def test_make_program_cache_key_accepts_empty_external_content(option_kw):
    """Truly empty sequences mean 'no external inputs' -- they must not
    force an extra_digest. (Empty *strings* are rejected separately because
    NVRTC still emits a flag for them.)"""
    _make_key(options=_opts(**option_kw))  # Should not raise.


def test_make_program_cache_key_ptx_ignores_nvrtc_only_options():
    """PTX compiles go through ``_translate_program_options`` which drops
    NVRTC-only fields (include_path, pch_*, frandom_seed, ...). Those
    fields must not perturb the PTX cache key; otherwise a shared
    ProgramOptions that happens to set them causes spurious misses."""
    base = _make_key(code=".version 7.0", code_type="ptx", options=_opts())
    # Each of these only affects NVRTC, never Linker.
    for kw in (
        {"define_macro": "FOO"},
        {"frandom_seed": "1234"},
        {"ofast_compile": "min"},
        {"std": "c++17"},
        {"disable_warnings": True},
    ):
        assert _make_key(code=".version 7.0", code_type="ptx", options=_opts(**kw)) == base, kw


@pytest.mark.parametrize(
    "option_kw",
    [
        pytest.param({"include_path": "/usr/local/include"}, id="include_path"),
        pytest.param({"pre_include": "stdint.h"}, id="pre_include"),
        pytest.param({"pch": True}, id="pch"),
        pytest.param({"use_pch": "pch.file"}, id="use_pch"),
        pytest.param({"pch_dir": "pch-cache"}, id="pch_dir"),
    ],
)
def test_make_program_cache_key_accepts_external_content_options_for_ptx(option_kw):
    """The external-content guard is NVRTC-only: ``Program.compile`` for PTX
    inputs translates options via ``_translate_program_options``, which
    drops include_path/pre_include/PCH fields entirely. A PTX compile must
    not be blocked just because a reused ProgramOptions object carries
    irrelevant header settings."""
    _make_key(code=".version 7.0", code_type="ptx", options=_opts(**option_kw))  # no raise


def test_make_program_cache_key_accepts_external_content_with_extra_digest():
    """With an extra_digest, external-content options are accepted and
    different digests produce different keys so callers can represent
    header edits."""
    opts = _opts(include_path="/usr/local/include")
    k_a = _make_key(options=opts, extra_digest=b"header-a" * 4)
    k_b = _make_key(options=opts, extra_digest=b"header-b" * 4)
    assert k_a != k_b


@pytest.mark.parametrize(
    "option_kw, extra_digest",
    [
        pytest.param({"create_pch": "out.pch"}, None, id="create_pch"),
        # Even with extra_digest, create_pch is rejected: a cache hit skips
        # compilation, so the side effect (writing the PCH) would not run.
        pytest.param({"create_pch": "out.pch"}, b"x" * 32, id="create_pch_with_extra_digest"),
        pytest.param({"create_pch": ""}, None, id="create_pch_empty_string"),
        # NVRTC emits ``--create-pch=False`` for any non-None value, so False
        # still triggers the side effect and must be rejected.
        pytest.param({"create_pch": False}, None, id="create_pch_false"),
        pytest.param({"time": "timing.csv"}, None, id="time"),
        pytest.param({"time": False}, None, id="time_false"),
        pytest.param({"fdevice_time_trace": "trace.json"}, None, id="fdevice_time_trace"),
        pytest.param({"fdevice_time_trace": False}, None, id="fdevice_time_trace_false"),
    ],
)
def test_make_program_cache_key_rejects_side_effect_options_nvrtc(option_kw, extra_digest):
    """Options that write files as a compile-time side effect must refuse
    key generation when the target backend is NVRTC; a cache hit would skip
    compilation and the artifact would never be produced."""
    with pytest.raises(ValueError, match="side effect"):
        _make_key(options=_opts(**option_kw), extra_digest=extra_digest)


@pytest.mark.parametrize(
    "option_kw",
    [
        # ``time`` goes through Linker's ``-time`` flag which only logs to the
        # info log -- no filesystem side effect -- so PTX compiles with
        # ``time=True`` must cache normally.
        pytest.param({"time": True}, id="time_true"),
        pytest.param({"time": "whatever.csv"}, id="time_path"),
    ],
)
def test_make_program_cache_key_accepts_side_effect_options_for_ptx(option_kw):
    """The side-effect guard is NVRTC-specific: PTX (linker) and NVVM must
    not be blocked by options whose side effects only apply under NVRTC."""
    _make_key(code=".version 7.0", code_type="ptx", options=_opts(**option_kw))  # no raise


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("c++", "a", "cubin", id="nvrtc"),
        pytest.param("ptx", ".version 7.0", "cubin", id="linker"),
        pytest.param("nvvm", "abc", "ptx", id="nvvm"),
    ],
)
def test_make_program_cache_key_survives_cuda_core_version_change(code_type, code, target_type, monkeypatch):
    """The docstring promises cross-patch sharing within a schema version, so
    cuda.core's own ``__version__`` must NOT be mixed into the digest."""
    import cuda.core._version as _version_mod

    monkeypatch.setattr(_version_mod, "__version__", "0.0.0")
    k_a = _make_key(code=code, code_type=code_type, target_type=target_type)
    monkeypatch.setattr(_version_mod, "__version__", "999.999.999")
    k_b = _make_key(code=code, code_type=code_type, target_type=target_type)
    assert k_a == k_b


def test_make_program_cache_key_driver_version_does_not_perturb_ptx_under_nvjitlink(monkeypatch):
    """nvJitLink does NOT route PTX compilation through cuLink, so a
    changing driver version must not invalidate PTX cache keys when
    nvJitLink is the active linker backend."""
    from cuda.core.utils import _program_cache

    monkeypatch.setattr(_program_cache._keys, "_linker_backend_and_version", lambda: ("nvJitLink", "12030"))
    monkeypatch.setattr(_program_cache._keys, "_driver_version", lambda: 13200)
    k_a = _make_key(code=".version 7.0", code_type="ptx")
    monkeypatch.setattr(_program_cache._keys, "_driver_version", lambda: 13300)
    k_b = _make_key(code=".version 7.0", code_type="ptx")
    assert k_a == k_b


@pytest.mark.parametrize(
    "code_type, code, target_type",
    [
        pytest.param("c++", "a", "cubin", id="nvrtc"),
        pytest.param("nvvm", "abc", "ptx", id="nvvm"),
    ],
)
def test_make_program_cache_key_driver_probe_failure_does_not_perturb_non_linker(
    code_type, code, target_type, monkeypatch
):
    """The driver version is only consumed on the linker (PTX) path because
    cuLink runs through the driver. NVRTC and NVVM produce identical bytes
    regardless of the driver version, so a failed driver probe must NOT
    perturb their cache keys -- otherwise driver upgrades would invalidate
    perfectly good caches."""
    from cuda.core.utils import _program_cache

    def _broken():
        raise RuntimeError("driver probe failed")

    k_ok = _make_key(code=code, code_type=code_type, target_type=target_type)
    monkeypatch.setattr(_program_cache._keys, "_driver_version", _broken)
    k_broken = _make_key(code=code, code_type=code_type, target_type=target_type)
    assert k_ok == k_broken


@pytest.mark.parametrize(
    "probe_name, code_type, code",
    [
        pytest.param("_nvrtc_version", "c++", "a", id="nvrtc"),
        pytest.param("_linker_backend_and_version", "ptx", ".ptx", id="linker"),
    ],
)
def test_make_program_cache_key_fails_closed_on_probe_failure(probe_name, code_type, code, monkeypatch):
    """A failed probe (a) must produce a key that differs from a working
    probe (so environments never silently share cache entries), and (b)
    must produce a *stable* key across calls -- otherwise the persistent
    cache could not be reused in broken environments. ``_driver_version``
    is exercised separately because it's only invoked transitively from
    ``_linker_backend_and_version`` on the cuLink driver path."""
    from cuda.core.utils import _program_cache

    def _broken():
        raise RuntimeError("probe failed")

    k_ok = _make_key(code=code, code_type=code_type)
    monkeypatch.setattr(_program_cache._keys, probe_name, _broken)
    k_broken1 = _make_key(code=code, code_type=code_type)
    k_broken2 = _make_key(code=code, code_type=code_type)
    assert k_ok != k_broken1
    assert k_broken1 == k_broken2  # stable: same failure -> same key


def test_make_program_cache_key_driver_probe_failure_taints_ptx_under_cuLink(monkeypatch):
    """When the driver linker is active, _linker_backend_and_version
    invokes _driver_version internally; a failing driver probe must (a)
    perturb the PTX key away from the success key, AND (b) be stable
    across repeated calls so the persistent cache stays usable in the
    failed environment."""
    from cuda.core import _linker
    from cuda.core.utils import _program_cache

    def _broken():
        raise RuntimeError("driver probe failed")

    monkeypatch.setattr(_linker, "_decide_nvjitlink_or_driver", lambda: True)
    k_ok = _make_key(code=".ptx", code_type="ptx")
    monkeypatch.setattr(_program_cache._keys, "_driver_version", _broken)
    k_broken1 = _make_key(code=".ptx", code_type="ptx")
    k_broken2 = _make_key(code=".ptx", code_type="ptx")
    assert k_ok != k_broken1
    assert k_broken1 == k_broken2  # stable: same failure -> same key


def _fake_object_code(payload: bytes = b"fake-cubin", name: str = "unit"):
    """Build an ObjectCode without touching the driver."""
    from cuda.core._module import ObjectCode

    return ObjectCode._init(payload, "cubin", name=name)


# ---------------------------------------------------------------------------
# FileStreamProgramCache -- single-process CRUD
# ---------------------------------------------------------------------------


def test_filestream_cache_empty_on_create(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        assert len(cache) == 0
        assert cache.get(b"nope") is None
        with pytest.raises(KeyError):
            cache[b"nope"]


def test_filestream_cache_roundtrip(tmp_path):
    """Cache returns the exact bytes that were written. ObjectCode metadata
    (name, code_type, symbol_mapping) is NOT preserved -- the cache stores
    just the binary."""
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k1"] = _fake_object_code(b"v1", name="x")
        assert cache.get(b"k1") is not None
        assert cache[b"k1"] == b"v1"


def test_filestream_cache_delete(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = _fake_object_code()
        del cache[b"k"]
        assert cache.get(b"k") is None
        with pytest.raises(KeyError):
            del cache[b"k"]


def test_filestream_cache_len_counts_all(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"a"] = _fake_object_code(b"1")
        cache[b"b"] = _fake_object_code(b"2")
        cache[b"c"] = _fake_object_code(b"3")
        assert len(cache) == 3


def test_filestream_cache_clear(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"a"] = _fake_object_code()
        cache.clear()
        assert len(cache) == 0


def test_filestream_cache_persists_across_reopen(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = _fake_object_code(b"persisted")
    with FileStreamProgramCache(root) as cache:
        assert cache[b"k"] == b"persisted"


def test_filestream_cache_permission_error_propagates_on_posix(tmp_path, monkeypatch):
    """On non-Windows, PermissionError from os.replace is a real config error
    and must not be silently swallowed."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", False)

    with FileStreamProgramCache(tmp_path / "fc") as cache:

        def _denied(src, dst):
            raise PermissionError("denied")

        monkeypatch.setattr(_os, "replace", _denied)
        with pytest.raises(PermissionError, match="denied"):
            cache[b"k"] = _fake_object_code(b"v")


def test_filestream_cache_write_phase_permission_error_propagates_on_windows(tmp_path, monkeypatch):
    """Even on Windows, a PermissionError from the write phase (mkstemp /
    fdopen / fsync) is a real config problem -- the Windows carve-out is
    only for the os.replace race. A write-phase error must propagate."""
    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    def _denied(*args, **kwargs):
        raise PermissionError("mkstemp denied")

    monkeypatch.setattr(_program_cache._file_stream.tempfile, "mkstemp", _denied)

    with FileStreamProgramCache(tmp_path / "fc") as cache, pytest.raises(PermissionError, match="mkstemp"):
        cache[b"k"] = _fake_object_code(b"v")


@pytest.mark.parametrize(
    "winerror, should_raise",
    [
        pytest.param(5, False, id="access_denied_swallowed"),
        pytest.param(32, False, id="sharing_violation_swallowed"),
        pytest.param(33, False, id="lock_violation_swallowed"),
        pytest.param(1, True, id="other_winerror_propagates"),
        pytest.param(None, True, id="no_winerror_propagates"),
    ],
)
def test_filestream_cache_permission_error_windows_is_narrowed(tmp_path, monkeypatch, winerror, should_raise):
    """On Windows, ERROR_ACCESS_DENIED (5), ERROR_SHARING_VIOLATION (32) and
    ERROR_LOCK_VIOLATION (33) are all transient "target held open by another
    process / pending delete" cases worth swallowing after the bounded retry.
    Any other PermissionError -- unrelated winerrors, missing winerror
    attribute, etc. -- is a real problem and must propagate."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)

    def _denied(src, dst):
        exc = PermissionError("simulated")
        exc.winerror = winerror
        raise exc

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        monkeypatch.setattr(_os, "replace", _denied)
        if should_raise:
            with pytest.raises(PermissionError, match="simulated"):
                cache[b"k"] = _fake_object_code(b"v")
        else:
            cache[b"k"] = _fake_object_code(b"v")  # swallowed
            assert cache.get(b"k") is None


def test_filestream_cache_atomic_no_half_written_file(tmp_path, monkeypatch):
    # Simulate a crash during write: patch os.replace to raise.
    import os as _os

    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:

        def _boom(src, dst):
            raise RuntimeError("crash during replace")

        monkeypatch.setattr(_os, "replace", _boom)
        with pytest.raises(RuntimeError, match="crash"):
            cache[b"k"] = _fake_object_code(b"v")
        monkeypatch.undo()
        assert cache.get(b"k") is None


def test_filestream_cache_prune_only_if_stat_unchanged(tmp_path):
    """The reader-unlink-vs-writer-replace race: if a concurrent writer
    atomically replaced a file between the reader's read and the reader's
    prune, the pruner must NOT delete the replacement."""
    from cuda.core.utils import FileStreamProgramCache
    from cuda.core.utils._program_cache._file_stream import _prune_if_stat_unchanged

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = _fake_object_code(b"v1")
        path = cache._path_for_key(b"k")
        stale_stat = path.stat()
        # Simulate a concurrent writer replacing the file.
        time.sleep(0.02)
        cache[b"k"] = _fake_object_code(b"v2")

    # Reader decides to prune using the stale stat; the guard refuses.
    _prune_if_stat_unchanged(path, stale_stat)
    assert path.exists()

    # With a fresh stat matching the current file, pruning proceeds.
    _prune_if_stat_unchanged(path, path.stat())
    assert not path.exists()


def test_filestream_cache_touch_atime_only_if_stat_unchanged(tmp_path):
    """The atime-touch is also stat-guarded so a racing rewriter's freshly
    replaced file does NOT get its mtime rolled back to the previous
    entry's value. Without the guard, the eviction stat-check (which keys
    on (ino, size, mtime_ns)) would mistake the replacement for the old
    entry and delete a just-committed file."""
    from cuda.core.utils import FileStreamProgramCache
    from cuda.core.utils._program_cache._file_stream import _touch_atime

    same_size_bytes = b"v" * 64
    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = same_size_bytes
        path = cache._path_for_key(b"k")
        stale_stat = path.stat()
        # Concurrent writer replaces with same-size payload (same st_size,
        # different ino/mtime) -- this is the dangerous case: ino and
        # mtime differ, only the stat-guard saves us.
        time.sleep(0.02)
        cache[b"k"] = same_size_bytes
        new_mtime_ns = path.stat().st_mtime_ns

    _touch_atime(path, stale_stat)
    # The new file's mtime must be untouched.
    assert path.stat().st_mtime_ns == new_mtime_ns

    # With a stat that matches the current file, atime is updated and
    # mtime is preserved.
    fresh_stat = path.stat()
    _touch_atime(path, fresh_stat)
    after = path.stat()
    assert after.st_mtime_ns == fresh_stat.st_mtime_ns
    assert after.st_atime_ns >= fresh_stat.st_atime_ns


def test_filestream_cache_returns_bytes_verbatim(tmp_path):
    """The cache stores raw binary and does no payload validation: whatever
    bytes were written are returned exactly. External tools (cuobjdump,
    nvdisasm) can read the entry file directly."""
    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    payload = b"\x7fELF\x02\x01\x01\x00" + b"\xab" * 256  # plausible cubin header
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = payload
        assert cache[b"k"] == payload
        path = cache._path_for_key(b"k")
        # On-disk content equals the input bytes verbatim -- no header,
        # no pickle frame, no length prefix.
        assert path.read_bytes() == payload


def test_filestream_cache_accepts_bytes_directly(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = b"raw"
        assert cache[b"k"] == b"raw"


def test_filestream_cache_accepts_bytearray_and_memoryview(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"a"] = bytearray(b"ba")
        cache[b"b"] = memoryview(b"mv")
        assert cache[b"a"] == b"ba"
        assert cache[b"b"] == b"mv"


def test_filestream_cache_rejects_non_bytes_non_object_code(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache, pytest.raises(TypeError, match="bytes-like or ObjectCode"):
        cache[b"k"] = "a string"


def test_filestream_cache_accepts_path_backed_object_code(tmp_path):
    """Path-backed ObjectCode is now read at write time so the cache stores
    the binary content (not the path), keeping cache files self-contained
    even if the source path is later moved or deleted."""
    from cuda.core._module import ObjectCode
    from cuda.core.utils import FileStreamProgramCache

    src = tmp_path / "src.cubin"
    src.write_bytes(b"hello-cubin-bytes")
    path_backed = ObjectCode.from_cubin(str(src), name="x")

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[b"k"] = path_backed
        assert cache[b"k"] == b"hello-cubin-bytes"

    # Mutating / removing the source must not affect the cached entry.
    src.unlink()
    with FileStreamProgramCache(tmp_path / "fc") as cache:
        assert cache[b"k"] == b"hello-cubin-bytes"


def test_program_cache_resource_update_accepts_mapping_and_pairs(tmp_path):
    """``update`` is a default ABC method; it must accept either a Mapping
    or an iterable of (key, value) pairs and dispatch each item through
    ``__setitem__`` so backend coercion (bytes extraction, size-cap
    enforcement) still runs."""
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc-mapping") as cache:
        cache.update({b"a": b"v-a", b"b": b"v-b"})
        assert cache[b"a"] == b"v-a"
        assert cache[b"b"] == b"v-b"

    with FileStreamProgramCache(tmp_path / "fc-pairs") as cache:
        cache.update([(b"x", b"v-x"), (b"y", b"v-y")])
        assert cache[b"x"] == b"v-x"
        assert cache[b"y"] == b"v-y"


def test_filestream_cache_input_forms_are_byte_equivalent(tmp_path):
    """Whether the caller writes raw bytes, a bytearray, a memoryview, a
    bytes-backed ObjectCode, or a path-backed ObjectCode pointing at a file
    with the same bytes, the cache content is byte-identical and the on-disk
    file has those exact bytes. Demonstrates the transparency contract:
    callers don't have to normalise their input shape themselves."""
    from cuda.core._module import ObjectCode
    from cuda.core.utils import FileStreamProgramCache

    payload = b"\x7fELF\x02\x01\x01\x00fake-cubin-bytes"
    src = tmp_path / "src.cubin"
    src.write_bytes(payload)

    inputs = {
        b"raw-bytes": payload,
        b"bytearray": bytearray(payload),
        b"memoryview": memoryview(payload),
        b"obj-bytes-backed": ObjectCode._init(payload, "cubin", name="x"),
        b"obj-path-backed": ObjectCode.from_cubin(str(src), name="y"),
    }

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache.update(inputs)
        for k in inputs:
            assert cache[k] == payload, f"value for {k!r} round-tripped to a different byte string"
            on_disk = cache._path_for_key(k).read_bytes()
            assert on_disk == payload, f"on-disk file for {k!r} is not the raw payload"


def test_filestream_cache_rejects_negative_size_cap(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with pytest.raises(ValueError, match="non-negative"):
        FileStreamProgramCache(tmp_path / "fc", max_size_bytes=-1)


def test_default_cache_dir_lives_under_user_cache_root(monkeypatch, tmp_path):
    """The cache root is platform-specific:

    * Linux: ``$XDG_CACHE_HOME`` or ``~/.cache``.
    * Windows: ``%LOCALAPPDATA%`` or ``~/AppData/Local``.

    Both branches must end in ``cuda-python/program-cache``; that suffix
    is what guarantees a stable on-disk layout across releases.
    """
    from pathlib import Path

    from cuda.core.utils import _program_cache
    from cuda.core.utils._program_cache._file_stream import _default_cache_dir

    # Path must end with cuda-python/program-cache regardless of platform.
    assert _default_cache_dir().parts[-2:] == ("cuda-python", "program-cache")

    # Linux branch: XDG_CACHE_HOME wins when set.
    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert _default_cache_dir() == tmp_path / "xdg" / "cuda-python" / "program-cache"

    # Linux branch: falls back to ``~/.cache`` when XDG_CACHE_HOME is unset.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    assert _default_cache_dir() == tmp_path / "home" / ".cache" / "cuda-python" / "program-cache"

    # Windows branch: LOCALAPPDATA wins when set.
    monkeypatch.setattr(_program_cache._file_stream, "_IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    assert _default_cache_dir() == tmp_path / "appdata" / "cuda-python" / "program-cache"

    # Windows branch: falls back to ``~/AppData/Local`` when LOCALAPPDATA is unset.
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert _default_cache_dir() == tmp_path / "home" / "AppData" / "Local" / "cuda-python" / "program-cache"


def test_filestream_cache_uses_default_dir_when_path_omitted(tmp_path, monkeypatch):
    from cuda.core.utils import FileStreamProgramCache, _program_cache

    monkeypatch.setattr(_program_cache._file_stream, "_default_cache_dir", lambda: tmp_path / "default-fc")

    with FileStreamProgramCache() as cache:
        cache[b"k"] = b"hello"
        assert cache[b"k"] == b"hello"
    assert (tmp_path / "default-fc" / "entries").is_dir()


def test_filestream_cache_recreates_tmp_dir_if_missing(tmp_path):
    """If ``tmp/`` is deleted out from under a live cache (operator clearing
    by hand, another process's wipe, etc.), the next write must recreate
    it rather than crash with ``FileNotFoundError``. The cache already
    recreates the per-shard ``entries/<2-char>/`` directory; ``tmp/``
    deserves the same treatment since ``mkstemp`` writes there."""
    import shutil

    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = b"first"
        # Nuke tmp/ between writes; the second write must still succeed.
        shutil.rmtree(root / "tmp")
        cache[b"k2"] = b"second"
        assert cache[b"k"] == b"first"
        assert cache[b"k2"] == b"second"
        assert (root / "tmp").is_dir()


def test_filestream_cache_sweeps_stale_tmp_files_on_open(tmp_path):
    """A crashed writer can leave files in ``tmp/``; the next ``open`` must
    sweep ones older than the staleness threshold so disk usage doesn't
    grow without bound."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    root = tmp_path / "fc"
    # Create the cache directory layout, then plant two temp files: one
    # young (must be preserved as it could be an in-flight write) and one
    # ancient (must be swept).
    with FileStreamProgramCache(root):
        pass
    young = root / "tmp" / "entry-young"
    young.write_bytes(b"in-flight")
    ancient = root / "tmp" / "entry-ancient"
    ancient.write_bytes(b"crashed-writer-leftover")
    ancient_mtime = time.time() - _program_cache._file_stream._TMP_STALE_AGE_SECONDS - 60
    _os.utime(ancient, (ancient_mtime, ancient_mtime))

    with FileStreamProgramCache(root):
        # Reopen triggers _sweep_stale_tmp_files.
        assert young.exists(), "young temp file must not be swept"
        assert not ancient.exists(), "ancient temp file should have been swept"


def test_filestream_cache_clear_preserves_young_tmp_files(tmp_path):
    """clear() must not delete young temp files: another process could be
    mid-write between ``mkstemp`` and ``os.replace``, and unlinking under
    it turns the writer's harmless rename into ``FileNotFoundError``.
    Stale temps (older than the threshold) are still swept."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache, _program_cache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"k"] = _fake_object_code(b"v")
    young_tmp = root / "tmp" / "entry-young"
    young_tmp.write_bytes(b"in-flight")
    ancient_tmp = root / "tmp" / "entry-ancient"
    ancient_tmp.write_bytes(b"crashed")
    ancient_mtime = time.time() - _program_cache._file_stream._TMP_STALE_AGE_SECONDS - 60
    _os.utime(ancient_tmp, (ancient_mtime, ancient_mtime))

    with FileStreamProgramCache(root) as cache:
        cache.clear()
    # Committed entry is gone, ancient orphan is gone, young temp survives.
    # Filenames are hash-like (no extension), so use a file filter rather
    # than a "*.*" glob.
    remaining_entries = [p for p in (root / "entries").rglob("*") if p.is_file()]
    assert not remaining_entries
    assert young_tmp.exists()
    assert not ancient_tmp.exists()


def test_filestream_cache_clear_does_not_unlink_replaced_file(tmp_path):
    """``clear()``'s scan-then-unlink loop must use the stat-guard so a
    concurrent writer's ``os.replace`` between snapshot and unlink doesn't
    delete the fresh entry. Race injection: subclass the cache and have
    ``_iter_entry_paths``'s post-yield cleanup os.replace path_a, then call
    ``clear()`` and verify the fresh contents survive."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"a"] = _fake_object_code(b"A" * 200, name="a")
        cache[b"b"] = _fake_object_code(b"B" * 200, name="b")
        path_a = cache._path_for_key(b"a")

    class _RaceCache(FileStreamProgramCache):
        race_armed = True

        def _iter_entry_paths(self):
            yield from super()._iter_entry_paths()
            # Generator cleanup runs at StopIteration, between clear()'s
            # scan and its unlink loop.
            if _RaceCache.race_armed and path_a.exists():
                _RaceCache.race_armed = False
                tmp = path_a.parent / "_inflight"
                tmp.write_bytes(b"\x80\x05fresh-by-other-writer-" * 32)
                _os.replace(tmp, path_a)

    with _RaceCache(root) as cache:
        cache.clear()

    # The fresh file must survive: clear() saw a stat mismatch and skipped.
    assert path_a.exists(), "stat guard failed -- clear() unlinked a concurrently-replaced file"
    assert path_a.read_bytes().startswith(b"\x80\x05fresh-by-other-writer-")


def test_filestream_cache_clear_does_not_break_concurrent_writer(tmp_path):
    """Simulate a writer that has already produced a temp file but has not
    yet executed ``os.replace``; a concurrent ``clear()`` from another
    cache instance must NOT unlink that temp, so the writer's
    ``os.replace`` still succeeds."""
    import os as _os

    from cuda.core.utils import FileStreamProgramCache

    root = tmp_path / "fc"
    with FileStreamProgramCache(root) as cache:
        cache[b"seed"] = _fake_object_code(b"seed")

    # Stage a temp file that mimics an in-flight write.
    inflight_tmp = root / "tmp" / "entry-inflight"
    inflight_tmp.write_bytes(b"in-flight payload")  # contents do not matter

    # Concurrent clear() from another cache handle.
    with FileStreamProgramCache(root) as other:
        other.clear()

    # The writer can now finish: rename the staged file into entries/.
    target = root / "entries" / "ab" / "cdef"
    target.parent.mkdir(parents=True, exist_ok=True)
    _os.replace(inflight_tmp, target)
    assert target.exists()


def test_filestream_cache_size_cap_does_not_unlink_replaced_file(tmp_path):
    """The PRODUCTION ``_enforce_size_cap`` must compare the snapshot stat
    to the current stat before unlinking; if the file was replaced under
    us (a concurrent writer's ``os.replace``), the unlink is skipped.

    Race injection without reimplementing the method: subclass the cache
    and override only ``_iter_entry_paths`` so that the cleanup code
    *after* the generator's last yield runs an ``os.replace`` on path_a.
    Python's for-loop calls ``next()`` until ``StopIteration``; the
    generator code after its last yield runs at that ``StopIteration``,
    which is exactly between ``_enforce_size_cap``'s scan loop and its
    eviction loop. Eviction's per-entry re-stat then sees a different
    stat for path_a and the production code's stat-guard must skip it.
    """
    import os as _os

    from cuda.core.utils import FileStreamProgramCache

    # Cap fits two 2000-byte entries (raw payload only -- no per-entry
    # framing) but not three.
    cap = 5000
    root = tmp_path / "fc"
    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        cache[b"a"] = _fake_object_code(b"A" * 2000, name="a")
        time.sleep(0.02)
        cache[b"b"] = _fake_object_code(b"B" * 2000, name="b")
        path_a = cache._path_for_key(b"a")
    assert path_a.exists(), "cap too small -- 'a' was evicted before the test ran"

    class _RaceCache(FileStreamProgramCache):
        race_armed = True

        def _iter_entry_paths(self):
            yield from super()._iter_entry_paths()
            # Generator cleanup runs at StopIteration, between
            # _enforce_size_cap's scan and its eviction loop. Fire the race
            # here exactly once.
            if _RaceCache.race_armed and path_a.exists():
                _RaceCache.race_armed = False
                tmp = path_a.parent / "_inflight"
                tmp.write_bytes(b"\x80\x05fresh-by-other-writer-" * 32)
                _os.replace(tmp, path_a)

    with _RaceCache(root, max_size_bytes=cap) as cache:
        # Trigger eviction by adding 'c'; eviction's scan exhausts our
        # racing generator, the cleanup fires, then the eviction loop's
        # re-stat sees the new stat and the production stat-guard MUST
        # refuse to unlink path_a.
        time.sleep(0.02)
        cache[b"c"] = _fake_object_code(b"C" * 2000, name="c")

    # The race-injected fresh file must survive: production stat-guard worked.
    assert path_a.exists(), "stat guard failed -- evicted a concurrently-replaced file"
    assert path_a.read_bytes().startswith(b"\x80\x05fresh-by-other-writer-")


def test_filestream_cache_size_cap_counts_tmp_files(tmp_path):
    """Surviving temp files occupy disk too; the soft cap must include
    them, otherwise an attacker (or a flurry of crashed writers) could
    inflate disk usage well past max_size_bytes."""
    from cuda.core.utils import FileStreamProgramCache

    cap = 4000
    root = tmp_path / "fc"
    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        cache[b"a"] = _fake_object_code(b"A" * 1500, name="a")
        time.sleep(0.02)
        cache[b"b"] = _fake_object_code(b"B" * 1500, name="b")
    # Plant a young temp file that pushes total over the cap.
    young_tmp = root / "tmp" / "entry-leftover"
    young_tmp.write_bytes(b"X" * 2500)

    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        # New write triggers _enforce_size_cap; 'a' must be evicted because
        # the temp file's bytes count toward the cap now.
        time.sleep(0.02)
        cache[b"c"] = _fake_object_code(b"C" * 200, name="c")
        assert cache.get(b"a") is None
        assert cache.get(b"c") is not None


def test_filestream_cache_handles_long_keys(tmp_path):
    """Arbitrary-length keys must not overflow per-component filename limits.
    The filename is a fixed-length 256-bit blake2b digest; key uniqueness
    relies on the digest's collision resistance."""
    from cuda.core.utils import FileStreamProgramCache

    long_bytes_key = b"x" * 4096
    long_str_key = "y" * 4096
    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache[long_bytes_key] = _fake_object_code(b"b", name="nb")
        cache[long_str_key] = _fake_object_code(b"s", name="ns")
        assert cache.get(long_bytes_key) is not None
        assert cache.get(long_str_key) is not None
        assert cache[long_bytes_key] == b"b"
        assert cache[long_str_key] == b"s"


def test_filestream_cache_accepts_str_keys(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        cache["my-key"] = _fake_object_code(b"v")
        assert cache.get("my-key") is not None
        assert cache.get(b"my-key") is not None


def test_filestream_cache_size_cap_evicts_oldest(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    # Big payloads, small cap; after the third entry, the cap is exceeded and
    # the entry with the oldest atime (a) must be evicted.
    cap = 3000
    root = tmp_path / "fc"
    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        cache[b"a"] = b"A" * 2000
        time.sleep(0.02)
        cache[b"b"] = b"B" * 2000
        time.sleep(0.02)
        cache[b"c"] = b"C" * 2000

    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        assert cache.get(b"a") is None
        assert cache.get(b"c") is not None


def test_filestream_cache_atime_lru_promotes_recently_read(tmp_path):
    """Eviction sorts by ``st_atime``: an entry that was recently READ
    survives even if it was the first one WRITTEN. This is the practical
    win over mtime-based FIFO eviction."""
    from cuda.core.utils import FileStreamProgramCache

    # Cap fits two payloads but not three; the third write triggers exactly
    # one eviction, and we want it to evict 'b' (oldest atime), not 'a'.
    cap = 5000
    root = tmp_path / "fc"
    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        cache[b"a"] = b"A" * 2000  # oldest write
        time.sleep(0.02)
        cache[b"b"] = b"B" * 2000  # newer write
        time.sleep(0.02)
        # Bump 'a' to most-recently-used.
        _ = cache[b"a"]
        time.sleep(0.02)
        cache[b"c"] = b"C" * 2000  # 6000 total -> one eviction

    with FileStreamProgramCache(root, max_size_bytes=cap) as cache:
        # 'b' is now the oldest atime -- evicted.
        assert cache.get(b"b") is None
        # 'a' was read after 'b' so its atime is newer -- survives.
        assert cache.get(b"a") is not None
        assert cache.get(b"c") is not None


def test_filestream_cache_unbounded_by_default(tmp_path):
    from cuda.core.utils import FileStreamProgramCache

    with FileStreamProgramCache(tmp_path / "fc") as cache:
        for i in range(20):
            cache[f"k{i}".encode()] = _fake_object_code(b"X" * 1024, name=f"n{i}")
        assert len(cache) == 20


def test_make_program_cache_key_changes_with_key_schema_version(monkeypatch):
    """Bumping ``_KEY_SCHEMA_VERSION`` produces a different cache key for
    the same logical inputs. That's what makes a schema bump invalidate
    old entries that were stored under :func:`make_program_cache_key`
    output: post-bump, the new key hashes to a different on-disk path,
    so lookups miss and old entries become orphans (harmless, reaped on
    the next size-cap eviction). Raw user keys passed straight to
    ``cache[k]`` bypass this mixing -- the schema version only
    participates via :func:`make_program_cache_key`."""
    from cuda.core import ProgramOptions
    from cuda.core.utils import _program_cache, make_program_cache_key

    args = {
        "code": "void k(){}",
        "code_type": "c++",
        "options": ProgramOptions(arch="sm_80"),
        "target_type": "cubin",
    }
    key_before = make_program_cache_key(**args)
    monkeypatch.setattr(_program_cache._keys, "_KEY_SCHEMA_VERSION", _program_cache._keys._KEY_SCHEMA_VERSION + 1)
    key_after = make_program_cache_key(**args)

    assert key_before != key_after


# ---------------------------------------------------------------------------
# End-to-end: real NVRTC compilation through persistent cache
# ---------------------------------------------------------------------------


def test_cache_roundtrip_with_real_compilation(tmp_path, init_cuda):
    """Compile a real kernel, persist its bytes, reopen the cache, and
    reconstruct an ``ObjectCode`` from the cached bytes.

    Exercises the full user workflow: NVRTC compile → persistent store →
    fresh process (simulated by closing and reopening the cache handle)
    → driver-side module load from an ObjectCode rebuilt from the
    cached bytes.
    """
    from cuda.core import Program, ProgramOptions
    from cuda.core._module import Kernel, ObjectCode
    from cuda.core.utils import FileStreamProgramCache, make_program_cache_key

    code = 'extern "C" __global__ void my_kernel() {}'
    code_type = "c++"
    target_type = "cubin"
    options = ProgramOptions(name="cached_kernel")

    program = Program(code, code_type, options=options)
    try:
        compiled = program.compile(target_type)
    finally:
        program.close()

    key = make_program_cache_key(
        code=code,
        code_type=code_type,
        options=options,
        target_type=target_type,
    )

    # First "process": compile and store the binary.
    with FileStreamProgramCache(tmp_path / "fc") as cache:
        assert cache.get(key) is None
        cache[key] = compiled  # extracts bytes(compiled.code)

    # Second "process": reopen, retrieve bytes, rebuild ObjectCode.
    with FileStreamProgramCache(tmp_path / "fc") as cache:
        assert cache.get(key) is not None
        cached_bytes = cache[key]

    assert cached_bytes == bytes(compiled.code)
    rebuilt = ObjectCode._init(cached_bytes, target_type, name="cached_kernel")
    # The reconstructed ObjectCode must still be usable against the driver.
    assert isinstance(rebuilt.get_kernel("my_kernel"), Kernel)


# ---------------------------------------------------------------------------
# InMemoryProgramCache
# ---------------------------------------------------------------------------


def test_inmemory_cache_empty_on_create():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    assert len(cache) == 0
    assert cache.get(b"nope") is None
    with pytest.raises(KeyError):
        cache[b"nope"]


def test_inmemory_cache_roundtrip_object_code():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"k1"] = _fake_object_code(b"v1", name="x")
    assert cache.get(b"k1") is not None
    assert cache[b"k1"] == b"v1"


def test_inmemory_cache_accepts_bytes_directly():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"k"] = b"raw-payload"
    assert cache[b"k"] == b"raw-payload"


def test_inmemory_cache_accepts_bytearray_and_memoryview():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"a"] = bytearray(b"ba-payload")
    cache[b"b"] = memoryview(b"mv-payload")
    assert cache[b"a"] == b"ba-payload"
    assert cache[b"b"] == b"mv-payload"


def test_inmemory_cache_accepts_path_backed_object_code(tmp_path):
    """Path-backed ObjectCode should be read at write time so the cache
    holds the bytes, not a path -- mirrors FileStreamProgramCache."""
    from cuda.core._module import ObjectCode
    from cuda.core.utils import InMemoryProgramCache

    payload = b"\x7fELF\x02\x01fake-cubin-bytes"
    src = tmp_path / "src.cubin"
    src.write_bytes(payload)
    obj = ObjectCode.from_cubin(str(src), name="x")

    cache = InMemoryProgramCache()
    cache[b"k"] = obj
    # If we mutate the source file, the cached entry must be unchanged.
    src.write_bytes(b"changed")
    assert cache[b"k"] == payload


def test_inmemory_cache_str_and_bytes_keys_alias():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache["k"] = b"v"
    assert cache[b"k"] == b"v"
    assert cache.get(b"k") is not None
    assert cache.get("k") is not None


def test_inmemory_cache_rejects_non_str_non_bytes_key():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    with pytest.raises(TypeError):
        cache[123] = b"v"
    with pytest.raises(TypeError):
        cache[123]
    with pytest.raises(TypeError):
        cache.get(123)


def test_inmemory_cache_rejects_non_bytes_non_object_code_value():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    with pytest.raises(TypeError):
        cache[b"k"] = "a string"


def test_inmemory_cache_delete():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"k"] = _fake_object_code()
    del cache[b"k"]
    assert cache.get(b"k") is None
    with pytest.raises(KeyError):
        del cache[b"k"]


def test_inmemory_cache_len_counts_all():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"a"] = _fake_object_code(b"1")
    cache[b"b"] = _fake_object_code(b"2")
    cache[b"c"] = _fake_object_code(b"3")
    assert len(cache) == 3


def test_inmemory_cache_clear():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"a"] = _fake_object_code()
    cache[b"b"] = _fake_object_code()
    cache.clear()
    assert len(cache) == 0
    assert cache.get(b"a") is None


def test_inmemory_cache_get_returns_default_on_miss():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    assert cache.get(b"missing") is None
    assert cache.get(b"missing", b"fallback") == b"fallback"


def test_inmemory_cache_get_returns_value_on_hit():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache[b"k"] = b"v"
    assert cache.get(b"k") == b"v"


def test_inmemory_cache_update_accepts_mapping_and_pairs():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    cache.update({b"a": b"v-a", b"b": b"v-b"})
    assert cache[b"a"] == b"v-a"
    assert cache[b"b"] == b"v-b"

    cache2 = InMemoryProgramCache()
    cache2.update([(b"x", b"v-x"), (b"y", b"v-y")])
    assert cache2[b"x"] == b"v-x"
    assert cache2[b"y"] == b"v-y"


def test_inmemory_cache_overwrite_replaces_value_and_updates_size():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache(max_size_bytes=1000)
    cache[b"k"] = b"x" * 100
    assert cache[b"k"] == b"x" * 100
    cache[b"k"] = b"y" * 50
    assert cache[b"k"] == b"y" * 50
    assert len(cache) == 1
    # Internal accounting should track the replacement, not double-count.
    assert cache._total_bytes == 50


def test_inmemory_cache_rejects_negative_size_cap():
    from cuda.core.utils import InMemoryProgramCache

    with pytest.raises(ValueError):
        InMemoryProgramCache(max_size_bytes=-1)


def test_inmemory_cache_size_cap_evicts_oldest():
    from cuda.core.utils import InMemoryProgramCache

    # Cap fits two 100-byte entries; the third write evicts the LRU one.
    cache = InMemoryProgramCache(max_size_bytes=250)
    cache[b"a"] = b"a" * 100
    cache[b"b"] = b"b" * 100
    cache[b"c"] = b"c" * 100  # forces eviction of b"a"
    assert cache.get(b"a") is None
    assert cache.get(b"b") is not None
    assert cache.get(b"c") is not None


def test_inmemory_cache_read_promotes_lru():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache(max_size_bytes=250)
    cache[b"a"] = b"a" * 100
    cache[b"b"] = b"b" * 100
    # Read of a promotes it past b in LRU order.
    _ = cache[b"a"]
    cache[b"c"] = b"c" * 100  # b is now the oldest; b should evict
    assert cache.get(b"a") is not None
    assert cache.get(b"b") is None
    assert cache.get(b"c") is not None


def test_inmemory_cache_oversized_write_evicts_itself():
    """A single write larger than max_size_bytes does not survive its own
    size-cap pass -- mirrors FileStreamProgramCache."""
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache(max_size_bytes=10)
    cache[b"big"] = b"x" * 100
    assert cache.get(b"big") is None
    assert len(cache) == 0


def test_inmemory_cache_unbounded_when_max_size_none():
    from cuda.core.utils import InMemoryProgramCache

    cache = InMemoryProgramCache()
    for i in range(50):
        cache[f"k{i}".encode()] = b"x" * 1024
    assert len(cache) == 50
