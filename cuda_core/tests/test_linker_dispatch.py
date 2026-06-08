# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Linker backend dispatch logic.

These cover the full decision matrix of :func:`cuda.core._linker._choose_backend`
with mocked version and availability inputs, so they run without a GPU, without
a specific nvJitLink version installed, and without a CUDA driver.
"""

import pytest

from cuda.core import _linker


class TestChooseBackend:
    """Parametrized unit tests for :func:`cuda.core._linker._choose_backend`.

    The decision matrix axes are:

    * ``driver_major``: CUDA driver major version.
    * ``nvjitlink_version``: ``(major, minor)`` tuple or ``None`` if nvJitLink is
      unavailable / too old.
    * ``inputs_have_ltoir``: any input ``ObjectCode`` has ``code_type="ltoir"``.
    * ``lto_requested``: ``LinkerOptions.link_time_optimization`` or ``ptx``.
    """

    @pytest.mark.parametrize(
        ("driver_major", "nvjitlink_version", "has_ltoir", "lto_requested", "expected"),
        [
            # No nvJitLink available + no LTO needed -> driver.
            (12, None, False, False, "driver"),
            (13, None, False, False, "driver"),
            # Matching driver/nvJitLink majors -> always nvJitLink.
            (12, (12, 3), False, False, "nvjitlink"),
            (12, (12, 9), True, True, "nvjitlink"),
            (12, (12, 9), True, False, "nvjitlink"),
            (12, (12, 9), False, True, "nvjitlink"),
            (13, (13, 0), False, False, "nvjitlink"),
            (13, (13, 0), True, True, "nvjitlink"),
            # Cross-major, no LTO requirement -> driver fallback.
            (13, (12, 9), False, False, "driver"),
            (12, (13, 0), False, False, "driver"),
            # Unknown driver (e.g., build containers) optimistically picks nvJitLink when available.
            (None, (12, 9), False, False, "nvjitlink"),
            (None, (12, 9), True, False, "nvjitlink"),
            (None, (12, 9), False, True, "nvjitlink"),
            (None, (13, 0), True, True, "nvjitlink"),
            # Unknown driver + no nvJitLink + no LTO -> driver (will fail at use-time, not dispatch).
            (None, None, False, False, "driver"),
        ],
    )
    def test_returns_expected_backend(self, driver_major, nvjitlink_version, has_ltoir, lto_requested, expected):
        assert _linker._choose_backend(driver_major, nvjitlink_version, has_ltoir, lto_requested) == expected

    @pytest.mark.parametrize(
        ("driver_major", "nvjitlink_version", "has_ltoir", "lto_requested", "match"),
        [
            # No nvJitLink + LTO IR input -> cannot satisfy.
            (12, None, True, False, "nvJitLink is not available"),
            (13, None, True, True, "nvJitLink is not available"),
            # No nvJitLink + link_time_optimization requested.
            (12, None, False, True, "nvJitLink is not available"),
            # Cross-major + LTO IR input.
            (13, (12, 9), True, False, "matching major versions"),
            (12, (13, 0), True, True, "matching major versions"),
            # Cross-major + link_time_optimization requested (no ltoir input).
            (13, (12, 9), False, True, "matching major versions"),
            (12, (13, 0), False, True, "matching major versions"),
            # Unknown driver + no nvJitLink + LTO needs cannot be satisfied.
            (None, None, True, False, "nvJitLink is not available"),
            (None, None, False, True, "nvJitLink is not available"),
        ],
    )
    def test_raises_when_unsatisfiable(self, driver_major, nvjitlink_version, has_ltoir, lto_requested, match):
        with pytest.raises(RuntimeError, match=match):
            _linker._choose_backend(driver_major, nvjitlink_version, has_ltoir, lto_requested)
