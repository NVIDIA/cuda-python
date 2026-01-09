# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for artifact search patterns across platforms and environments."""

from dataclasses import dataclass

from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


@dataclass(frozen=True)
class ArtifactSearchConfig:
    """Configuration for searching a specific artifact (platform-resolved at init).

    Attributes:
        canonical_name: Platform-agnostic identifier for the artifact.
        filenames: Filenames to search for on the current platform.
        conda_dirs: Directory paths relative to CONDA_PREFIX.
        cuda_home_dirs: Directory paths relative to CUDA_HOME.
        search_targets_subdirs: Whether to search targets/*/{lib,lib64} subdirs
            for cross-compilation setups (Linux only).
    """

    canonical_name: str
    filenames: tuple[str, ...]
    conda_dirs: tuple[str, ...]
    cuda_home_dirs: tuple[str, ...]
    search_targets_subdirs: bool = False


def _create_config(
    canonical_name: str,
    linux_filenames: tuple[str, ...],
    windows_filenames: tuple[str, ...],
    conda_linux_dirs: tuple[str, ...],
    conda_windows_dirs: tuple[str, ...],
    cuda_home_linux_dirs: tuple[str, ...],
    cuda_home_windows_dirs: tuple[str, ...],
    search_targets_subdirs: bool = False,
) -> ArtifactSearchConfig:
    """Create a platform-specific config by selecting appropriate values at init time."""
    return ArtifactSearchConfig(
        canonical_name=canonical_name,
        filenames=windows_filenames if IS_WINDOWS else linux_filenames,
        conda_dirs=conda_windows_dirs if IS_WINDOWS else conda_linux_dirs,
        cuda_home_dirs=cuda_home_windows_dirs if IS_WINDOWS else cuda_home_linux_dirs,
        search_targets_subdirs=search_targets_subdirs,
    )


# Registry of all supported artifacts with their search configurations
# Platform selection happens once at module import time via _create_config
ARTIFACT_CONFIGS = {
    "libcudadevrt.a": _create_config(
        canonical_name="cudadevrt",
        linux_filenames=("libcudadevrt.a",),
        windows_filenames=("cudadevrt.lib",),
        conda_linux_dirs=("lib",),
        conda_windows_dirs=("Library/lib", "Library/lib/x64"),
        cuda_home_linux_dirs=("lib64", "lib"),
        cuda_home_windows_dirs=("lib/x64", "lib"),
        search_targets_subdirs=True,
    ),
    "libdevice.10.bc": _create_config(
        canonical_name="libdevice",
        linux_filenames=("libdevice.10.bc",),
        windows_filenames=("libdevice.10.bc",),
        conda_linux_dirs=("nvvm/libdevice",),
        conda_windows_dirs=("Library/nvvm/libdevice",),
        cuda_home_linux_dirs=("nvvm/libdevice",),
        cuda_home_windows_dirs=("nvvm/libdevice",),
        search_targets_subdirs=False,
    ),
}
