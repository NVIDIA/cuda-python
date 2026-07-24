# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ci.tools.fetch_ctk_redistrib import get_preview_packages


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_get_preview_packages_linux_x86_64():
    packages, skipped = get_preview_packages(
        host_platform="linux-64",
        cuda_version="13.4.0",
        components="cuda_cudart,cuda_nvrtc,cuda_cccl,libnvjitlink,libcudla",
    )

    assert packages == [
        "cuda-cudart-dev-13-4",
        "cuda-nvrtc-dev-13-4",
        "cccl-13-4",
        "libnvjitlink-dev-13-4",
    ]
    assert skipped == ["libcudla"]


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_get_preview_packages_linux_aarch64():
    packages, skipped = get_preview_packages(
        host_platform="linux-aarch64",
        cuda_version="13.4.0",
        components="cuda_cudart,libcudla,libcufile",
    )

    assert packages == [
        "cuda-cudart-dev-13-4",
        "libcudla-dev-13-4",
        "libcufile-dev-13-4",
    ]
    assert skipped == []


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_get_preview_packages_rejects_windows():
    with pytest.raises(ValueError, match="not supported"):
        get_preview_packages(
            host_platform="win-64",
            cuda_version="13.4.0",
            components="cuda_cudart",
        )


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize(
    ("cuda_version", "components", "message"),
    [
        ("13.4", "cuda_cudart", "invalid cuda-version"),
        ("13.4.0", "unknown", "unsupported CUDA prerelease component"),
    ],
)
def test_get_preview_packages_rejects_invalid_input(cuda_version, components, message):
    with pytest.raises(ValueError, match=message):
        get_preview_packages(
            host_platform="linux-64",
            cuda_version=cuda_version,
            components=components,
        )
