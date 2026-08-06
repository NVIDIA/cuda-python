# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from compatibility_guard_rails_test_utils import (
    STRICTNESS,
    _assert_real_ctk_backed_path,
    _default_process_wide_guard_rails_mode,  # noqa: F401
    clear_real_host_probe_caches,  # noqa: F401
)
from local_helpers import (
    have_distribution,
    require_real_cuda_toolkit_version_from_cuda_h,
    require_real_driver_cuda_version,
)

from cuda.pathfinder import (
    BitcodeLibNotFoundError,
    CompatibilityCheckError,
    CompatibilityGuardRails,
    CompatibilityInsufficientMetadataError,
    DynamicLibNotFoundError,
    StaticLibNotFoundError,
)


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_driver(info_summary_append):
    real_driver = require_real_driver_cuda_version()
    info_summary_append(
        f"real driver CUDA version={real_driver.major}.{real_driver.minor} (encoded={real_driver.encoded})"
    )


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_ctk(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    info_summary_append(
        f"real cuda.h CTK version={real_ctk.version.major}.{real_ctk.version.minor} "
        f"via {real_ctk.found_via} at {real_ctk.cuda_h_path!r}"
    )


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_wheel_ctk_items_are_compatible(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    real_driver = require_real_driver_cuda_version()
    guard_rails = CompatibilityGuardRails(
        ctk_version=f"=={real_ctk.version.major}.{real_ctk.version.minor}",
        driver_cuda_version=real_driver,
    )

    try:
        loaded = guard_rails.load_nvidia_dynamic_lib("nvrtc")
        header_dir = guard_rails.find_nvidia_header_directory("nvrtc")
        static_lib = guard_rails.find_static_lib("cudadevrt")
        bitcode_lib = guard_rails.find_bitcode_lib("device")
        nvcc = guard_rails.find_nvidia_binary_utility("nvcc")
    except (
        CompatibilityCheckError,
        CompatibilityInsufficientMetadataError,
        DynamicLibNotFoundError,
        StaticLibNotFoundError,
        BitcodeLibNotFoundError,
    ) as exc:
        if STRICTNESS == "all_must_work":
            raise
        pytest.skip(f"real CTK check unavailable: {exc.__class__.__name__}: {exc}")

    assert isinstance(loaded.abs_path, str)
    assert header_dir is not None
    for path in (loaded.abs_path, header_dir, static_lib, bitcode_lib):
        _assert_real_ctk_backed_path(path)
    if have_distribution(r"^nvidia-cuda-nvcc-cu12$"):
        # For CUDA 12, NVIDIA publishes a PyPI package named nvidia-cuda-nvcc-cu12,
        # but the wheels only contain nvcc-adjacent compiler components such as
        # ptxas, CRT headers, libnvvm, and libdevice; the nvcc executable itself
        # is not included.
        if nvcc is not None:
            # nvcc found elsewhere, e.g. /usr/local or Conda.
            _assert_real_ctk_backed_path(nvcc)
    else:
        if nvcc is None:
            if STRICTNESS == "all_must_work":
                raise AssertionError("Expected CTK-backed nvcc to be discoverable.")
            info_summary_append("real CTK-backed nvcc executable not found; continuing without asserting nvcc")
        else:
            _assert_real_ctk_backed_path(nvcc)


@pytest.mark.usefixtures("clear_real_host_probe_caches")
def test_real_wheel_component_version_does_not_override_ctk_line(info_summary_append):
    real_ctk = require_real_cuda_toolkit_version_from_cuda_h()
    real_driver = require_real_driver_cuda_version()
    guard_rails = CompatibilityGuardRails(
        ctk_version=f"=={real_ctk.version.major}.{real_ctk.version.minor}",
        driver_cuda_version=real_driver,
    )

    try:
        header_dir = guard_rails.find_nvidia_header_directory("cufft")
    except (CompatibilityCheckError, CompatibilityInsufficientMetadataError) as exc:
        if STRICTNESS == "all_must_work":
            raise
        pytest.skip(f"real cufft CTK check unavailable: {exc.__class__.__name__}: {exc}")

    if header_dir is None:
        if STRICTNESS == "all_must_work":
            raise AssertionError("Expected CTK-backed cufft headers to be discoverable.")
        pytest.skip("real cufft CTK check unavailable: cufft headers not found")

    _assert_real_ctk_backed_path(header_dir)
