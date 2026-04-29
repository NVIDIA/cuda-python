# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cache-key construction.

A backend-strategy hierarchy (:class:`_KeyBackend`) owns the per-code-type
guard / fingerprint / version-probe logic; :func:`make_program_cache_key`
dispatches to the right backend and assembles the digest.
"""

from __future__ import annotations

import abc
import collections.abc
import hashlib
from typing import Sequence

from cuda.core._program import ProgramOptions
from cuda.core._utils.cuda_utils import (
    driver as _driver,
)
from cuda.core._utils.cuda_utils import (
    handle_return as _handle_return,
)
from cuda.core._utils.cuda_utils import (
    nvrtc as _nvrtc,
)


# Bump when the key schema changes in a way that invalidates existing caches.
_KEY_SCHEMA_VERSION = 1

_VALID_CODE_TYPES = frozenset({"c++", "ptx", "nvvm"})
_VALID_TARGET_TYPES = frozenset({"ptx", "cubin", "ltoir"})

# code_type -> allowed target_type set, mirroring Program.compile's
# SUPPORTED_TARGETS matrix in _program.pyx.
_SUPPORTED_TARGETS_BY_CODE_TYPE = {
    "c++": frozenset({"ptx", "cubin", "ltoir"}),
    "ptx": frozenset({"cubin", "ptx"}),
    "nvvm": frozenset({"ptx", "ltoir"}),
}


# ProgramOptions fields that reach the Linker via _translate_program_options
# (see cuda_core/cuda/core/_program.pyx). All other fields on ProgramOptions
# are NVRTC-only and must NOT perturb a PTX cache key: a PTX compile with a
# shared ProgramOptions that happens to set include_path/pch/frandom_seed
# would otherwise miss the cache unnecessarily.
_LINKER_RELEVANT_FIELDS = (
    "name",
    "arch",
    "max_register_count",
    "time",
    "link_time_optimization",
    "debug",
    "lineinfo",
    "ftz",
    "prec_div",
    "prec_sqrt",
    "fma",
    "split_compile",
    "ptxas_options",
    "no_cache",
)


# Map each linker-relevant ProgramOptions field to the gate the Linker uses
# to turn it into a flag (see ``_prepare_nvjitlink_options`` and
# ``_prepare_driver_options`` in _linker.pyx). Collapsing inputs through
# these gates means semantically-equivalent configurations
# (``debug=False`` vs ``None``, ``time=True`` vs ``time="path"``) hash to
# the same cache key instead of forcing spurious misses.
def _gate_presence(v):
    return v is not None


def _gate_truthy(v):
    return bool(v)


def _gate_is_true(v):
    return v is True


def _gate_tristate_bool(v):
    return None if v is None else bool(v)


def _gate_identity(v):
    return v


def _gate_ptxas_options(v):
    # ``_prepare_nvjitlink_options`` emits one ``-Xptxas=<s>`` per element, and
    # treats ``str`` as a single-element sequence. Canonicalize to a tuple so
    # ``"-v"`` / ``["-v"]`` / ``("-v",)`` all hash the same. An empty sequence
    # emits no flags, so collapse it to ``None`` too.
    if v is None:
        return None
    if isinstance(v, str):
        return ("-Xptxas=" + v,)
    if isinstance(v, collections.abc.Sequence):
        if len(v) == 0:
            return None
        return tuple(f"-Xptxas={s}" for s in v)
    return v


_LINKER_FIELD_GATES = {
    "name": _gate_identity,
    "arch": _gate_identity,
    "max_register_count": _gate_identity,
    "time": _gate_presence,  # linker emits ``-time`` iff value is not None
    "link_time_optimization": _gate_truthy,
    "debug": _gate_truthy,
    "lineinfo": _gate_truthy,
    "ftz": _gate_tristate_bool,
    "prec_div": _gate_tristate_bool,
    "prec_sqrt": _gate_tristate_bool,
    "fma": _gate_tristate_bool,
    "split_compile": _gate_identity,
    "ptxas_options": _gate_ptxas_options,
    "no_cache": _gate_is_true,
}


# LinkerOptions fields the ``cuLink`` driver backend silently ignores
# (emits only a DeprecationWarning; no actual flag reaches the compiler).
# When the driver backend is active, collapse them to a single sentinel in
# the fingerprint so nvJitLink<->driver parity of ``ObjectCode`` doesn't
# cause cache misses from otherwise-equivalent configurations.
_DRIVER_IGNORED_LINKER_FIELDS = frozenset({"ftz", "prec_div", "prec_sqrt", "fma"})


def _linker_option_fingerprint(options: ProgramOptions, *, use_driver_linker: bool | None) -> list[bytes]:
    """Backend-aware fingerprint of ProgramOptions fields consumed by the Linker.

    Each field passes through the gate the Linker itself uses so equivalent
    inputs (e.g. ``debug=False`` / ``None``) hash to the same bytes. When
    the driver (cuLink) linker backend is in use, fields it silently
    ignores collapse to one sentinel so those options don't perturb the
    key on driver-backed hosts either. ``use_driver_linker=None`` means we
    couldn't probe the backend; we don't collapse driver-ignored fields in
    that case, to stay conservative.
    """
    parts = []
    driver_ignored = use_driver_linker is True
    for name in _LINKER_RELEVANT_FIELDS:
        if driver_ignored and name in _DRIVER_IGNORED_LINKER_FIELDS:
            parts.append(f"{name}=<driver-ignored>".encode())
            continue
        gated = _LINKER_FIELD_GATES[name](getattr(options, name, None))
        parts.append(f"{name}={gated!r}".encode())
    return parts


# ProgramOptions fields that map to LinkerOptions fields the cuLink (driver)
# backend rejects outright (see _prepare_driver_options in _linker.pyx).
# ``split_compile_extended`` exists on LinkerOptions but is not exposed via
# ProgramOptions / _translate_program_options, so it cannot reach the driver
# linker from the cache path and is omitted here.
_DRIVER_LINKER_UNSUPPORTED_FIELDS = ("time", "ptxas_options", "split_compile")


def _driver_version() -> int:
    return int(_handle_return(_driver.cuDriverGetVersion()))


def _nvrtc_version() -> tuple[int, int]:
    major, minor = _handle_return(_nvrtc.nvrtcVersion())
    return int(major), int(minor)


def _linker_backend_and_version() -> tuple[str, str]:
    """Return ``(backend, version)`` for the linker used on PTX inputs.

    Raises any underlying probe exception. ``make_program_cache_key`` catches
    and mixes the exception's class name into the digest, so the same probe
    failure produces the same key across processes -- the cache stays
    persistent in broken environments, while never sharing a key with a
    working probe (``_probe_failed`` label vs. ``driver``/``nvrtc``/...).

    nvJitLink version lookup goes through ``sys.modules`` first so we hit the
    same module ``_decide_nvjitlink_or_driver()`` already loaded. That keeps
    fingerprinting aligned with whichever ``cuda.bindings.nvjitlink`` import
    path the linker actually uses.
    """
    import sys

    from cuda.core._linker import _decide_nvjitlink_or_driver

    use_driver = _decide_nvjitlink_or_driver()
    if use_driver:
        return ("driver", str(_driver_version()))
    nvjitlink = sys.modules.get("cuda.bindings.nvjitlink")
    if nvjitlink is None:
        from cuda.bindings import nvjitlink

    return ("nvJitLink", str(nvjitlink.version()))


def _nvvm_fingerprint() -> str:
    """Stable identifier for the loaded NVVM toolchain.

    Combines the libNVVM library version (``module.version()``) with the IR
    version reported by ``module.ir_version()``. The library version is the
    primary invalidation lever: a libNVVM patch upgrade can change codegen
    while keeping the same IR major/minor, so keying only on the IR pair
    would silently reuse stale entries. Paired with cuda-core, the IR pair
    adds defence in depth without making the key any less stable.

    Both calls go through ``_get_nvvm_module()`` so this fingerprint follows
    the same availability / cuda-bindings-version gate that real NVVM
    compilation does -- if NVVM is unusable at compile time, the probe
    fails the same way and ``_probe`` mixes the failure label into the key.
    """
    from cuda.core._program import _get_nvvm_module

    module = _get_nvvm_module()
    lib_major, lib_minor = module.version()
    major, minor, debug_major, debug_minor = module.ir_version()
    return f"lib={lib_major}.{lib_minor};ir={major}.{minor}.{debug_major}.{debug_minor}"


# ProgramOptions fields that reference external files whose *contents* the
# cache key cannot observe without reading the filesystem. Callers that set
# any of these must supply an ``extra_digest`` covering the dependency surface
# (e.g. a hash over all reachable headers / PCH bytes).
_EXTERNAL_CONTENT_OPTIONS = (
    "include_path",
    "pre_include",
    "pch",
    "use_pch",
    "pch_dir",
)

# ProgramOptions fields whose compilation effect is not captured in the
# returned ``ObjectCode`` -- they produce a filesystem artifact as a side
# effect. A cache hit skips compilation, so that artifact would never be
# written. Reject these outright: the persistent cache is for pure ObjectCode
# reuse, not for replaying compile-time side effects.
#   * create_pch        -- writes a PCH file (NVRTC).
#   * time              -- writes NVRTC timing info to a file.
#   * fdevice_time_trace -- writes a device-compilation time trace file (NVRTC).
# These are all NVRTC-specific; the Linker's ``-time`` logs to the info log
# (not a file) and NVVM explicitly rejects all three at compile time. The
# side-effect guard is therefore gated on ``backend == "nvrtc"`` below.
_SIDE_EFFECT_OPTIONS = ("create_pch", "time", "fdevice_time_trace")


# ProgramOptions fields gated by plain truthiness in ``_program.pyx`` (the
# compiler writes the flag only when the value is truthy).
_BOOLEAN_OPTION_FIELDS = frozenset({"pch"})

# Fields whose compiler emission requires ``isinstance(value, str)`` or a
# non-empty sequence; anything else (``False``, ``int``, ``None``, ``[]``)
# is silently ignored at compile time.
_STR_OR_SEQUENCE_OPTION_FIELDS = frozenset({"include_path", "pre_include"})


def _option_is_set(options: ProgramOptions, name: str) -> bool:
    """Match how ``_program.pyx`` gates option emission, per field shape.

    - Boolean flags (``pch``): truthy only.
    - str-or-sequence fields (``include_path``, ``pre_include``): ``str``
      (including empty) or a non-empty ``collections.abc.Sequence`` (list,
      tuple, range, user subclass, ...); everything else (``False``, ``int``,
      empty sequence, ``None``) is ignored by the compiler and must not
      trigger a cache-time guard.
    - Path/string-shaped fields (``create_pch``, ``time``,
      ``fdevice_time_trace``, ``use_pch``, ``pch_dir``): ``is not None`` --
      the compiler emits ``--flag=<value>`` for any non-None value, so
      ``False`` / ``""`` / ``0`` must still count as set.
    """
    value = getattr(options, name, None)
    if value is None:
        return False
    if name in _BOOLEAN_OPTION_FIELDS:
        return bool(value)
    if name in _STR_OR_SEQUENCE_OPTION_FIELDS:
        # Mirror ``_prepare_nvrtc_options_impl``: it checks ``isinstance(v, str)``
        # first, then ``is_sequence(v)`` (which is ``isinstance(v, Sequence)``).
        # We therefore accept any ``collections.abc.Sequence`` (range, deque,
        # user subclass, etc.), not just list/tuple.
        if isinstance(value, str):
            return True
        if isinstance(value, collections.abc.Sequence):
            return len(value) > 0
        return False
    return True


def _hash_probe_failure(update, label: str, exc: BaseException) -> None:
    """Mix a probe failure into the digest under a stable, content-free label.

    Hashing only the exception's CLASS NAME (not its message) keeps the
    digest stable across repeated calls within one process (e.g. NVVM's
    loader reports different messages on first vs. cached-failure attempts)
    AND across processes that hit the same failure mode. The
    ``_probe_failed`` label differs from every backend's success label, so a
    broken environment never collides with a working one -- the cache
    "fails closed" between broken and working environments while staying
    persistent within either.
    """
    update(f"{label}_probe_failed", type(exc).__name__.encode())


class _KeyBackend(abc.ABC):
    """Strategy for deriving the cache key for one ``Program`` ``code_type``.

    Each subclass owns the backend-specific guard logic, code coercion,
    option fingerprinting, name-expression handling, version probing, and
    extra-payload hashing. The orchestrator :func:`make_program_cache_key`
    validates the code_type / target_type pair, dispatches to the right
    backend, and assembles the digest.
    """

    @abc.abstractmethod
    def validate(self, options: ProgramOptions, target_type: str, extra_digest: bytes | None) -> None:
        """Reject inputs the cache cannot key safely.

        Raises ``ValueError`` for options that have compile-time side
        effects, options that pull in external file content the cache
        can't observe, or any other backend-specific invariants.
        """

    def encode_code(self, code: object, code_type: str) -> bytes:
        """Coerce ``code`` to bytes. Default rejects bytes-like input
        (only NVVM accepts it; ``Program()`` does the same)."""
        if isinstance(code, str):
            return code.encode("utf-8")
        if isinstance(code, (bytes, bytearray)):
            raise TypeError(
                f"code must be str for code_type={code_type!r}; bytes/bytearray are only accepted for code_type='nvvm'."
            )
        raise TypeError(f"code must be str or bytes, got {type(code).__name__}")

    @abc.abstractmethod
    def option_fingerprint(self, options: ProgramOptions, target_type: str) -> list[bytes]:
        """Fingerprint of the ``ProgramOptions`` fields that reach the compiler."""

    def encode_name_expressions(self, name_expressions: Sequence) -> tuple[bytes, ...] | None:  # noqa: ARG002
        """Sorted, type-tagged name expressions, or ``None`` if the
        backend does not consume them.

        ``None`` means the orchestrator emits no ``names_count`` /
        ``name`` entries at all (a backend that ignores
        ``name_expressions`` should never have them perturb its key). An
        empty tuple means the backend supports them but the caller
        passed none -- the orchestrator still emits ``names_count=0`` so
        the schema is stable across "absent" and "empty".
        """
        return None

    @abc.abstractmethod
    def hash_version_probe(self, update) -> None:
        """Mix the runtime/compiler version probe into the digest via
        ``update(label, payload)``. On probe failure, mix
        ``_hash_probe_failure(update, "<label>", exc)`` instead so the
        digest is stable across processes hitting the same failure
        mode.
        """

    def hash_extra_payload(self, options: ProgramOptions, update) -> None:  # noqa: B027
        """Mix backend-specific extras (e.g. NVVM ``extra_sources`` /
        ``use_libdevice``). Default: nothing.
        """


class _NvrtcBackend(_KeyBackend):
    def validate(self, options, target_type, extra_digest):  # noqa: ARG002
        # Side-effect options are NVRTC-specific:
        # ``time``/``fdevice_time_trace`` write artifacts via NVRTC,
        # ``create_pch`` writes via NVRTC. The Linker's ``-time`` logs to
        # the info log (not a file), and NVVM explicitly rejects all three
        # at compile time, so the side-effect guard is meaningful only for
        # the NVRTC path.
        side_effects = [name for name in _SIDE_EFFECT_OPTIONS if _option_is_set(options, name)]
        if side_effects:
            raise ValueError(
                f"make_program_cache_key() refuses to build a key for options that "
                f"have compile-time side effects ({', '.join(side_effects)}); a "
                f"cache hit skips compilation, so the side effect would not occur. "
                f"Disable the option, or compile directly without the cache."
            )
        # ``extra_sources`` is NVVM-only -- ``Program`` raises for non-NVVM
        # backends (_program.pyx). Reject here so callers get the same
        # error from the cache-key path as from a real compile.
        if getattr(options, "extra_sources", None) is not None:
            raise ValueError(
                "extra_sources is only valid for code_type='nvvm'; Program() rejects it for code_type='c++'."
            )
        if extra_digest is None:
            # ``Program.compile`` for PTX inputs runs
            # ``_translate_program_options``, which drops these entirely;
            # NVVM rejects them. Only NVRTC reads the external content.
            external = [name for name in _EXTERNAL_CONTENT_OPTIONS if _option_is_set(options, name)]
            if external:
                raise ValueError(
                    f"make_program_cache_key() refuses to build a key for options that "
                    f"pull in external file content ({', '.join(external)}) without an "
                    f"extra_digest; compute a digest over the header/PCH bytes the "
                    f"compile will read and pass it as extra_digest=..."
                )
            # NVRTC uses ``options.name`` as the source filename and
            # resolves quoted ``#include "x.h"`` directives relative to
            # the directory component of that name. The directory's
            # contents are external to anything else the key observes,
            # so a name with a directory component requires the same
            # ``extra_digest`` treatment as ``include_path`` etc.
            options_name = getattr(options, "name", None)
            if isinstance(options_name, str) and ("/" in options_name or "\\" in options_name):
                raise ValueError(
                    f"make_program_cache_key() refuses to build a key for options.name="
                    f"{options_name!r} (NVRTC source-filename with a directory "
                    f"component) without an extra_digest; NVRTC resolves quoted "
                    f"#include directives relative to that directory, so a digest "
                    f"covering the headers it may pull in must be supplied."
                )

    def option_fingerprint(self, options, target_type):
        # ``ProgramOptions.as_bytes("nvrtc", ...)`` gives the real
        # compile-time flag surface for NVRTC.
        return options.as_bytes("nvrtc", target_type)

    def encode_name_expressions(self, name_expressions):
        # ``"foo"`` and ``b"foo"`` get distinct tags because
        # ``Program.compile`` records the original Python object as the
        # ``ObjectCode.symbol_mapping`` key, so a cached ObjectCode whose
        # mapping-key type differs from what the caller's later
        # ``get_kernel`` passes would silently miss. Reject ``bytearray``
        # because ``Program.compile`` also uses the raw element as a dict
        # key -- bytearray is unhashable, so a cache miss would compile
        # then crash in ``symbol_mapping[n] = ...``. Accepting it here
        # would let the cache serve hits for inputs the uncached path
        # can't handle.
        def _tag(n):
            if isinstance(n, bytes):
                return b"b:" + n
            if isinstance(n, str):
                return b"s:" + n.encode("utf-8")
            if isinstance(n, bytearray):
                raise TypeError(
                    "name_expressions elements must be str or bytes; "
                    "bytearray is not accepted because Program.compile uses "
                    "each element as a dict key and bytearray is unhashable."
                )
            raise TypeError(f"name_expressions elements must be str or bytes; got {type(n).__name__}")

        return tuple(sorted(_tag(n) for n in name_expressions))

    def hash_version_probe(self, update):
        try:
            major, minor = _nvrtc_version()
        except Exception as exc:
            _hash_probe_failure(update, "nvrtc", exc)
            return
        update("nvrtc", f"{major}.{minor}".encode("ascii"))


class _LinkerBackend(_KeyBackend):
    def _decide_driver(self) -> bool | None:
        """``True`` if the cuLink driver linker will be used, ``False`` if
        nvJitLink, ``None`` if the probe failed (in which case
        :meth:`hash_version_probe` mixes a ``_probe_failed`` taint into
        the digest instead of a backend label).
        """
        try:
            from cuda.core._linker import _decide_nvjitlink_or_driver

            return _decide_nvjitlink_or_driver()
        except Exception:
            return None

    def validate(self, options, target_type, extra_digest):  # noqa: ARG002
        if getattr(options, "extra_sources", None) is not None:
            raise ValueError(
                "extra_sources is only valid for code_type='nvvm'; Program() rejects it for code_type='ptx'."
            )
        # PTX compiles go through the Linker. When the driver (cuLink)
        # backend is selected (nvJitLink unavailable), ``Program.compile``
        # rejects a subset of options that nvJitLink would accept; reject
        # them here too so we never store a key for a compilation that
        # can't succeed in this environment. If the probe fails we can't
        # tell which backend will run, so skip -- the failed-probe taint
        # in ``hash_version_probe`` already poisons the key.
        if self._decide_driver() is True:
            # Mirror ``_prepare_driver_options``'s exact gate: ``is not
            # None`` for these fields, so ``time=False`` or
            # ``ptxas_options=[]`` is still a rejection. Do NOT use the
            # truthiness-based ``_option_is_set`` helper here.
            unsupported = [
                name for name in _DRIVER_LINKER_UNSUPPORTED_FIELDS if getattr(options, name, None) is not None
            ]
            if unsupported:
                raise ValueError(
                    f"the cuLink driver linker does not support these options: "
                    f"{', '.join(unsupported)}; Program.compile() would reject this "
                    f"configuration before producing an ObjectCode."
                )

    def option_fingerprint(self, options, target_type):  # noqa: ARG002
        # For PTX inputs the Linker reads only a subset of ProgramOptions
        # (see ``_translate_program_options`` in _program.pyx); fingerprint
        # just those fields so shared ProgramOptions carrying NVRTC-only
        # flags (``include_path``, ``pch_*``, ``frandom_seed``, ...) don't
        # force spurious cache misses on PTX.
        return _linker_option_fingerprint(options, use_driver_linker=self._decide_driver())

    def hash_version_probe(self, update):
        # Only cuLink (driver-backed linker) goes through the CUDA driver
        # for codegen. nvJitLink is a separate library, so a driver
        # upgrade under it does not change the compiled bytes -- skip the
        # driver version there. ``_linker_backend_and_version`` already
        # returns the driver version when the driver backend is active,
        # so the bytes are still in the digest via ``linker_version``.
        try:
            lb_name, lb_version = _linker_backend_and_version()
        except Exception as exc:
            _hash_probe_failure(update, "linker", exc)
            return
        update("linker_backend", lb_name.encode("ascii"))
        update("linker_version", lb_version.encode("ascii"))


class _NvvmBackend(_KeyBackend):
    def encode_code(self, code, code_type):  # noqa: ARG002
        # NVVM accepts both str and bytes (matching ``Program()``).
        if isinstance(code, str):
            return code.encode("utf-8")
        if isinstance(code, (bytes, bytearray)):
            return bytes(code)
        raise TypeError(f"code must be str or bytes, got {type(code).__name__}")

    def validate(self, options, target_type, extra_digest):  # noqa: ARG002
        # NVVM with ``use_libdevice=True`` reads external libdevice
        # bitcode at compile time (see Program_init in _program.pyx). The
        # file is resolved from the active toolkit, so a changed
        # CUDA_HOME / libdevice upgrade changes the linked output without
        # touching any key input the cache can observe. Require the
        # caller to supply an ``extra_digest`` that fingerprints the
        # libdevice bytes (or simply disable use_libdevice for
        # caching-sensitive workflows).
        if extra_digest is None and getattr(options, "use_libdevice", None):
            raise ValueError(
                "make_program_cache_key() refuses to build an NVVM key with "
                "use_libdevice=True and no extra_digest: the linked libdevice "
                "bitcode can change out from under a cached ObjectCode. Pass an "
                "extra_digest that fingerprints the libdevice file you intend "
                "to link against, or disable use_libdevice."
            )

    def option_fingerprint(self, options, target_type):
        return options.as_bytes("nvvm", target_type)

    def hash_version_probe(self, update):
        try:
            fp = _nvvm_fingerprint()
        except Exception as exc:
            _hash_probe_failure(update, "nvvm", exc)
            return
        update("nvvm", fp.encode("ascii"))

    def hash_extra_payload(self, options, update):
        extra_sources = getattr(options, "extra_sources", None)
        if extra_sources:
            # ``extra_sources`` is hashed in caller-provided order on purpose.
            # NVVM module linking is order-dependent in the general case
            # (overlapping symbols, weak definitions, definition order can
            # change which body wins), so canonicalising by sorting on the
            # source name would produce the same key for two compiles whose
            # outputs may legitimately differ. If a future test proves the
            # relevant input subset is order-insensitive, sorting can be
            # introduced under that proof; absent that proof, preserving
            # caller order is the safe default.
            update("extra_sources_count", str(len(extra_sources)).encode("ascii"))
            for item in extra_sources:
                # ``extra_sources`` is a sequence of (name, source) tuples.
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    name, src = item
                    update("extra_source_name", str(name).encode("utf-8"))
                    if isinstance(src, str):
                        update("extra_source_code", src.encode("utf-8"))
                    elif isinstance(src, (bytes, bytearray)):
                        update("extra_source_code", bytes(src))
                    else:
                        update("extra_source_code", str(src).encode("utf-8"))
                else:
                    # Fallback for unexpected format.
                    update("extra_source", str(item).encode("utf-8"))
        # ``use_libdevice`` is gated on truthiness to match Program_init's
        # gate -- ``False`` and ``None`` collapse to the same key.
        if getattr(options, "use_libdevice", None):
            update("use_libdevice", b"1")


# Stateless backends; one shared instance per code_type.
_BACKENDS_BY_CODE_TYPE: dict[str, _KeyBackend] = {
    "c++": _NvrtcBackend(),
    "ptx": _LinkerBackend(),
    "nvvm": _NvvmBackend(),
}


def make_program_cache_key(
    *,
    code: str | bytes,
    code_type: str,
    options: ProgramOptions,
    target_type: str,
    name_expressions: Sequence[str | bytes | bytearray] = (),
    extra_digest: bytes | None = None,
) -> bytes:
    """Build a stable cache key from compile inputs.

    Parameters
    ----------
    code:
        Source text. ``str`` is encoded as UTF-8.
    code_type:
        One of ``"c++"``, ``"ptx"``, ``"nvvm"``.
    options:
        A :class:`cuda.core.ProgramOptions`. Its ``arch`` must be set (the
        default ``ProgramOptions.__post_init__`` populates it from the current
        device).
    target_type:
        One of ``"ptx"``, ``"cubin"``, ``"ltoir"``.
    name_expressions:
        Optional iterable of mangled-name lookups. Order is not significant.
        Elements may be ``str`` or ``bytes``; ``"foo"`` and ``b"foo"`` produce
        distinct keys because ``Program.compile`` records the original Python
        object as the ``ObjectCode.symbol_mapping`` key, and ``get_kernel``
        lookups must use the same type the cache key recorded. ``bytearray``
        is rejected because ``Program.compile`` stores each element as a
        dict key and ``bytearray`` is unhashable.
    extra_digest:
        Caller-supplied bytes mixed into the key. Required whenever
        :class:`cuda.core.ProgramOptions` sets any option that pulls in
        external file content (``include_path``, ``pre_include``, ``pch``,
        ``use_pch``, ``pch_dir``) -- the cache cannot read those files on
        the caller's behalf, so the caller must fingerprint the header /
        PCH surface and pass it here. Callers may pass this for other
        inputs too (embedded kernels, generated sources, etc.).

    Returns
    -------
    bytes
        A 32-byte blake2b digest suitable for use as a cache key.

    Raises
    ------
    ValueError
        If ``options`` sets an option with compile-time side effects (such
        as ``create_pch``) -- a cache hit skips compilation, so the side
        effect would not occur.
    ValueError
        If ``extra_digest`` is ``None`` while ``options`` sets any option
        whose compilation effect depends on external file content that the
        key cannot otherwise observe.

    Examples
    --------
    For most workflows you should not call ``make_program_cache_key``
    yourself -- pass ``cache=`` to :meth:`cuda.core.Program.compile`,
    which derives the key, returns the cached
    :class:`~cuda.core.ObjectCode` on hit, and stores the compile
    result on miss::

        from cuda.core import Program, ProgramOptions
        from cuda.core.utils import FileStreamProgramCache

        source = 'extern "C" __global__ void k(int *a){ *a = 1; }'
        options = ProgramOptions(arch="sm_80")

        with FileStreamProgramCache() as cache:
            obj = Program(source, "c++", options=options).compile("cubin", cache=cache)

    Call ``make_program_cache_key`` directly when the compile inputs
    require an ``extra_digest`` (the cache cannot read external file
    content on the caller's behalf) -- ``Program.compile(cache=...)``
    refuses those inputs with a ``ValueError`` pointing here::

        from cuda.core._module import ObjectCode
        from cuda.core.utils import FileStreamProgramCache, make_program_cache_key

        with FileStreamProgramCache() as cache:
            key = make_program_cache_key(
                code=source,
                code_type="c++",
                options=options,
                target_type="cubin",
                extra_digest=fingerprint_headers(options.include_path),
            )
            data = cache.get(key)
            if data is None:
                obj = Program(source, "c++", options=options).compile("cubin")
                cache[key] = obj  # extracts bytes(obj.code)
            else:
                obj = ObjectCode._init(data, "cubin")

    The cache stores raw binary bytes -- cubin / PTX / LTO-IR with no
    pickle, JSON, or framing -- so entry files are directly consumable
    by external NVIDIA tools (``cuobjdump``, ``nvdisasm``, ...). Note
    that an :class:`~cuda.core.ObjectCode` round-tripped through the
    cache loses ``symbol_mapping``: callers that compile with
    ``name_expressions`` and rely on ``get_kernel(name_expression)``
    after a cache hit must either compile fresh or look up the mangled
    symbol explicitly.

    Options that read external files (``include_path``, ``pre_include``,
    ``pch``, ``use_pch``, ``pch_dir``; ``use_libdevice=True`` on the NVVM
    path; and on NVRTC, an ``options.name`` with a directory component,
    which NVRTC uses for relative-include resolution) require
    ``extra_digest`` -- fingerprint the bytes the compiler will pull in
    and pass that digest so changes to those files force a cache miss.
    Options that have compile-time side effects (``create_pch``,
    ``time``, ``fdevice_time_trace``) cannot be cached and raise
    ``ValueError``; compile directly, or disable the flag, for those
    cases.
    """
    # Mirror Program.compile (_program.pyx Program_init lowercases code_type
    # before dispatch); a caller that passes "PTX" or "C++" must get the
    # same routing and the same cache key as the lowercase form.
    code_type = code_type.lower() if isinstance(code_type, str) else code_type
    if code_type not in _VALID_CODE_TYPES:
        raise ValueError(f"code_type={code_type!r} is not supported (must be one of {sorted(_VALID_CODE_TYPES)})")
    if target_type not in _VALID_TARGET_TYPES:
        raise ValueError(f"target_type={target_type!r} is not supported (must be one of {sorted(_VALID_TARGET_TYPES)})")
    supported_for_code = _SUPPORTED_TARGETS_BY_CODE_TYPE[code_type]
    if target_type not in supported_for_code:
        raise ValueError(
            f"target_type={target_type!r} is not valid for code_type={code_type!r}"
            f" (supported: {sorted(supported_for_code)}). Program.compile() rejects"
            f" this combination, so caching a key for it is meaningless."
        )

    backend = _BACKENDS_BY_CODE_TYPE[code_type]
    backend.validate(options, target_type, extra_digest)

    code_bytes = backend.encode_code(code, code_type)
    option_bytes = backend.option_fingerprint(options, target_type)
    name_tags = backend.encode_name_expressions(name_expressions)

    hasher = hashlib.blake2b(digest_size=32)

    def _update(label: str, payload: bytes) -> None:
        hasher.update(label.encode("ascii"))
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)

    _update("schema", str(_KEY_SCHEMA_VERSION).encode("ascii"))
    backend.hash_version_probe(_update)
    _update("code_type", code_type.encode("ascii"))
    _update("target_type", target_type.encode("ascii"))
    _update("code", code_bytes)
    _update("option_count", str(len(option_bytes)).encode("ascii"))
    for opt in option_bytes:
        _update("option", bytes(opt))
    if name_tags is not None:
        # ``encode_name_expressions`` returns ``None`` from backends that
        # ignore name_expressions and a (possibly-empty) tuple from those
        # that consume them. Hashing ``names_count=0`` for the latter
        # keeps the schema stable across "absent" and "empty" inputs.
        _update("names_count", str(len(name_tags)).encode("ascii"))
        for n in name_tags:
            _update("name", n)
    backend.hash_extra_payload(options, _update)

    # ``Program.compile()`` propagates ``options.name`` onto the returned
    # ObjectCode, so two compiles identical in everything but name produce
    # ObjectCodes that differ in their public ``name`` attribute. The key
    # must reflect that or a cache hit could hand back an entry with the
    # wrong name. Universal across backends.
    options_name = getattr(options, "name", None)
    if options_name is not None:
        _update("options_name", str(options_name).encode("utf-8"))

    if extra_digest is not None:
        _update("extra_digest", bytes(extra_digest))

    return hasher.digest()
