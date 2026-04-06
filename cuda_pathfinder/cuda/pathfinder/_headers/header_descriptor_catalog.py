# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical authored descriptor catalog for header directories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HeaderPackagedWith = Literal["ctk", "other"]


@dataclass(frozen=True, slots=True)
class HeaderDescriptorSpec:
    name: str
    packaged_with: HeaderPackagedWith
    header_basename: str
    site_packages_dirs: tuple[str, ...] = ()
    available_on_linux: bool = True
    available_on_windows: bool = True
    # Relative path(s) from anchor point to the include directory.
    anchor_include_rel_dirs: tuple[str, ...] = ("include",)
    # Subdirectories within the include dir to check before the include dir itself.
    include_subdirs: tuple[str, ...] = ()
    # Windows-only additional subdirectories within the include dir.
    include_subdirs_windows: tuple[str, ...] = ()
    # System install directories (glob patterns).
    system_install_dirs: tuple[str, ...] = ()
    # Whether to use targets/<arch>/include layout for conda on Linux.
    conda_targets_layout: bool = True
    # Whether to attempt CTK-root canary probing (spawns a subprocess).
    use_ctk_root_canary: bool = True


HEADER_DESCRIPTOR_CATALOG: tuple[HeaderDescriptorSpec, ...] = (
    # -----------------------------------------------------------------------
    # CTK (CUDA Toolkit) headers
    # -----------------------------------------------------------------------
    HeaderDescriptorSpec(
        name="cccl",
        packaged_with="ctk",
        header_basename="cuda/std/version",
        site_packages_dirs=(
            "nvidia/cu13/include/cccl",  # cuda-toolkit[cccl]==13.*
            "nvidia/cuda_cccl/include",  # cuda-toolkit[cccl]==12.*
        ),
        include_subdirs=("cccl",),
        include_subdirs_windows=("targets/x64/cccl", "targets/x64"),
    ),
    HeaderDescriptorSpec(
        name="cublas",
        packaged_with="ctk",
        header_basename="cublas.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cublas/include"),
    ),
    HeaderDescriptorSpec(
        name="cudart",
        packaged_with="ctk",
        header_basename="cuda_runtime.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cuda_runtime/include"),
    ),
    HeaderDescriptorSpec(
        name="cufft",
        packaged_with="ctk",
        header_basename="cufft.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cufft/include"),
    ),
    HeaderDescriptorSpec(
        name="cufile",
        packaged_with="ctk",
        header_basename="cufile.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cufile/include"),
        available_on_windows=False,
    ),
    HeaderDescriptorSpec(
        name="curand",
        packaged_with="ctk",
        header_basename="curand.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/curand/include"),
    ),
    HeaderDescriptorSpec(
        name="cusolver",
        packaged_with="ctk",
        header_basename="cusolverDn.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cusolver/include"),
    ),
    HeaderDescriptorSpec(
        name="cusparse",
        packaged_with="ctk",
        header_basename="cusparse.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cusparse/include"),
    ),
    HeaderDescriptorSpec(
        name="npp",
        packaged_with="ctk",
        header_basename="npp.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/npp/include"),
    ),
    HeaderDescriptorSpec(
        name="profiler",
        packaged_with="ctk",
        header_basename="cuda_profiler_api.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cuda_profiler_api/include"),
    ),
    HeaderDescriptorSpec(
        name="nvcc",
        packaged_with="ctk",
        header_basename="fatbinary_section.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cuda_nvcc/include"),
    ),
    HeaderDescriptorSpec(
        name="nvfatbin",
        packaged_with="ctk",
        header_basename="nvFatbin.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/nvfatbin/include"),
    ),
    HeaderDescriptorSpec(
        name="nvjitlink",
        packaged_with="ctk",
        header_basename="nvJitLink.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/nvjitlink/include"),
    ),
    HeaderDescriptorSpec(
        name="nvjpeg",
        packaged_with="ctk",
        header_basename="nvjpeg.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/nvjpeg/include"),
    ),
    HeaderDescriptorSpec(
        name="nvrtc",
        packaged_with="ctk",
        header_basename="nvrtc.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cuda_nvrtc/include"),
    ),
    HeaderDescriptorSpec(
        name="nvvm",
        packaged_with="ctk",
        header_basename="nvvm.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cuda_nvcc/nvvm/include"),
        anchor_include_rel_dirs=("nvvm/include",),
    ),
    HeaderDescriptorSpec(
        name="cudla",
        packaged_with="ctk",
        header_basename="cudla.h",
        site_packages_dirs=("nvidia/cu13/include",),
        available_on_windows=False,
    ),
    # -----------------------------------------------------------------------
    # Third-party / separately packaged headers
    # -----------------------------------------------------------------------
    HeaderDescriptorSpec(
        name="cusolverMp",
        packaged_with="other",
        header_basename="cusolverMp.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cu12/include"),
        available_on_windows=False,
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="cusparseLt",
        packaged_with="other",
        header_basename="cusparseLt.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cusparselt/include"),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="cute",
        packaged_with="other",
        header_basename="cute/tensor.hpp",
        site_packages_dirs=("cutlass_library/source/include",),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="cutensor",
        packaged_with="other",
        header_basename="cutensor.h",
        site_packages_dirs=("cutensor/include",),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="cutlass",
        packaged_with="other",
        header_basename="cutlass/cutlass.h",
        site_packages_dirs=("cutlass_library/source/include",),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="mathdx",
        packaged_with="other",
        header_basename="libmathdx.h",
        site_packages_dirs=("nvidia/cu13/include", "nvidia/cu12/include"),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
    HeaderDescriptorSpec(
        name="nvshmem",
        packaged_with="other",
        header_basename="nvshmem.h",
        site_packages_dirs=("nvidia/nvshmem/include",),
        available_on_windows=False,
        system_install_dirs=("/usr/include/nvshmem_*",),
        conda_targets_layout=False,
        use_ctk_root_canary=False,
    ),
)
