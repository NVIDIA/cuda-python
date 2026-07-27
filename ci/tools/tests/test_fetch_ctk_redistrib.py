# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from ci.tools.fetch_ctk_redistrib import get_preview_installer, get_preview_packages, host_platform_to_subdir


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
    ("host_platform", "architecture", "sha256"),
    [
        ("win-64", "x86_64", "b743a3323116bf33404953ef58a9b9a3319368241f6352e933e9461409e9a759"),
        ("win-arm64", "arm64", "a1f68c81160b16d519c4087788b9c07de41306c3f1b872471ceee0996621374d"),
    ],
)
def test_get_preview_installer_windows(host_platform, architecture, sha256):
    installer = get_preview_installer(host_platform=host_platform, cuda_version="13.4.0")

    assert installer.url == (
        f"https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_{architecture}.exe"
    )
    assert installer.sha256 == sha256


@pytest.mark.agent_authored(model="gpt-5.6-sol")
@pytest.mark.parametrize("host_platform", ["linux-64", "linux-aarch64", "win-unknown"])
def test_get_preview_installer_rejects_unsupported_platform(host_platform):
    with pytest.raises(ValueError, match="not supported"):
        get_preview_installer(host_platform=host_platform, cuda_version="13.4.0")


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_windows_arm64_redistrib_subdir():
    assert host_platform_to_subdir("win-arm64") == "windows-arm64"


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
