#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark FIPS-available hashlib candidates for cuda.core program-cache use.

This mirrors the two relevant call sites:

* ``FileStreamProgramCache._path_for_key()``: hash a cache key to a stable
  filename component via ``hexdigest()``.
* ``make_program_cache_key()``: incrementally build the digest from labeled
  payload chunks and return ``digest()``.

This is a review/support tool, not a production dependency. The benchmark is
intentionally stdlib-only so reviewers can run it directly.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

_DEFAULT_ALGORITHMS = (
    "sha1",
    "sha224",
    "sha256",
    "sha384",
    "sha512",
    "sha512_224",
    "sha512_256",
    "sha3_224",
    "sha3_256",
    "sha3_384",
    "sha3_512",
    "shake_128",
    "shake_256",
)

_SHAKE_DIGEST_BYTES = 32


@dataclass(frozen=True)
class HashCase:
    name: str
    runner: Callable[[Callable[..., object]], None]


def _supports_usedforsecurity(constructor: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return False
    return "usedforsecurity" in signature.parameters


def _make_constructor(name: str) -> Callable[..., object]:
    constructor = getattr(hashlib, name, None)
    if constructor is not None:
        if _supports_usedforsecurity(constructor):
            return lambda data=b"": constructor(data, usedforsecurity=False)
        return constructor

    def _constructor(data=b""):
        try:
            return hashlib.new(name, data, usedforsecurity=False)
        except TypeError:
            return hashlib.new(name, data)

    return _constructor


def _file_stream_case(name: str, key: bytes) -> HashCase:
    def _runner(constructor: Callable[..., object]) -> None:
        _hex_digest(constructor(key))

    return HashCase(name, _runner)


def _program_cache_case(name: str, payloads: tuple[tuple[str, bytes], ...]) -> HashCase:
    def _runner(constructor: Callable[..., object]) -> None:
        hasher = constructor()
        for label, payload in payloads:
            hasher.update(label.encode("ascii"))
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
        _digest_bytes(hasher)

    return HashCase(name, _runner)


def _end_to_end_case(name: str, payloads: tuple[tuple[str, bytes], ...]) -> HashCase:
    def _runner(constructor: Callable[..., object]) -> None:
        hasher = constructor()
        for label, payload in payloads:
            hasher.update(label.encode("ascii"))
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
        key = _digest_bytes(hasher)
        _hex_digest(constructor(key))

    return HashCase(name, _runner)


def _digest_bytes(hasher: object) -> bytes:
    try:
        return hasher.digest()
    except TypeError:
        return hasher.digest(_SHAKE_DIGEST_BYTES)


def _hex_digest(hasher: object) -> str:
    try:
        return hasher.hexdigest()
    except TypeError:
        return hasher.hexdigest(_SHAKE_DIGEST_BYTES)


def _sample_cases() -> tuple[HashCase, ...]:
    file_stream_key = bytes.fromhex("ab" * 32)
    long_file_stream_key = (b"cuda-core-cache-key-" * 128)[:4096]

    source = b"""
extern "C" __global__ void saxpy(float a, const float* x, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = a * x[i] + y[i];
}
""".strip()
    ptx = b"""
.version 8.0
.target sm_90
.address_size 64
.visible .entry saxpy() { ret; }
""".strip()
    option_bytes = (
        b"name='saxpy'",
        b"arch='sm_90'",
        b"max_register_count=None",
        b"time=False",
        b"link_time_optimization=False",
        b"debug=False",
        b"lineinfo=False",
        b"ftz=None",
        b"prec_div=None",
        b"prec_sqrt=None",
        b"fma=None",
        b"split_compile=None",
        b"ptxas_options=None",
        b"no_cache=False",
    )
    names = (b"saxpy", b"_Z5saxpyv")
    extra_digest = bytes.fromhex("cd" * 32)

    cpp_payloads = (
        ("schema", b"2"),
        ("nvrtc", b"13.2"),
        ("code_type", b"c++"),
        ("target_type", b"cubin"),
        ("code", source),
        ("option_count", str(len(option_bytes)).encode("ascii")),
        *tuple(("option", item) for item in option_bytes),
        ("names_count", str(len(names)).encode("ascii")),
        *tuple(("name", item) for item in names),
        ("options_name", b"saxpy"),
        ("extra_digest", extra_digest),
    )
    ptx_payloads = (
        ("schema", b"2"),
        ("linker", b"nvJitLink-13.2"),
        ("code_type", b"ptx"),
        ("target_type", b"cubin"),
        ("code", ptx),
        ("option_count", str(len(option_bytes)).encode("ascii")),
        *tuple(("option", item) for item in option_bytes),
        ("names_count", b"0"),
        ("extra_digest", extra_digest),
    )

    return (
        _file_stream_case("file_stream_key_32b", file_stream_key),
        _file_stream_case("file_stream_key_4k", long_file_stream_key),
        _program_cache_case("program_cache_cpp", cpp_payloads),
        _program_cache_case("program_cache_ptx", ptx_payloads),
        _end_to_end_case("end_to_end_cpp", cpp_payloads),
        _end_to_end_case("end_to_end_ptx", ptx_payloads),
    )


def _benchmark_case(
    case: HashCase,
    constructor: Callable[..., object],
    *,
    loops: int,
    repeat: int,
) -> tuple[float, float]:
    samples_ns: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        for _ in range(loops):
            case.runner(constructor)
        elapsed = time.perf_counter_ns() - start
        samples_ns.append(elapsed / loops)
    return statistics.mean(samples_ns), min(samples_ns)


def _format_ns(value: float) -> str:
    return f"{value:,.1f}"


def _write_line(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loops",
        type=int,
        default=200_000,
        help="Iterations per repeat for each algorithm/case pair.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="Independent timing repeats for each algorithm/case pair.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(_DEFAULT_ALGORITHMS),
        help="hashlib algorithm names to benchmark.",
    )
    args = parser.parse_args()

    cases = _sample_cases()
    widths = {
        "algorithm": max(len("Algorithm"), max(len(name) for name in args.algorithms)),
        "case": max(len(case.name) for case in cases),
    }

    _write_line(
        f"{'Algorithm':<{widths['algorithm']}}  "
        f"{'Case':<{widths['case']}}  {'mean ns/op':>12}  {'best ns/op':>12}"
    )
    _write_line("-" * (widths["algorithm"] + widths["case"] + 28))

    for algorithm in args.algorithms:
        constructor = _make_constructor(algorithm)
        for case in cases:
            mean_ns, best_ns = _benchmark_case(case, constructor, loops=args.loops, repeat=args.repeat)
            _write_line(
                f"{algorithm:<{widths['algorithm']}}  "
                f"{case.name:<{widths['case']}}  "
                f"{_format_ns(mean_ns):>12}  {_format_ns(best_ns):>12}"
            )


if __name__ == "__main__":
    main()
