# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical authored descriptor catalog for dynamic libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PackagedWith = Literal["ctk", "other", "driver"]


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    name: str
    packaged_with: PackagedWith
    linux_sonames: tuple[str, ...] = ()
    windows_dlls: tuple[str, ...] = ()
    site_packages_linux: tuple[str, ...] = ()
    site_packages_windows: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    anchor_rel_dirs_linux: tuple[str, ...] = ("lib64", "lib")
    anchor_rel_dirs_windows: tuple[str, ...] = ("bin/x64", "bin")
    ctk_root_canary_anchor_libnames: tuple[str, ...] = ()
    requires_add_dll_directory: bool = False
    requires_rtld_deepbind: bool = False


DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
    # -----------------------------------------------------------------------
    # CTK (CUDA Toolkit) libraries
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cudart",
        packaged_with="ctk",
        linux_sonames=("libcudart.so.12", "libcudart.so.13"),
        windows_dlls=("cudart64_12.dll", "cudart64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_runtime/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cuda_runtime/bin"),
    ),
    DescriptorSpec(
        name="nvfatbin",
        packaged_with="ctk",
        linux_sonames=("libnvfatbin.so.12", "libnvfatbin.so.13"),
        windows_dlls=("nvfatbin_120_0.dll", "nvfatbin_130_0.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvfatbin/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/nvfatbin/bin"),
    ),
    DescriptorSpec(
        name="nvJitLink",
        packaged_with="ctk",
        linux_sonames=("libnvJitLink.so.12", "libnvJitLink.so.13"),
        windows_dlls=("nvJitLink_120_0.dll", "nvJitLink_130_0.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvjitlink/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/nvjitlink/bin"),
    ),
    DescriptorSpec(
        name="nvrtc",
        packaged_with="ctk",
        linux_sonames=("libnvrtc.so.12", "libnvrtc.so.13"),
        windows_dlls=("nvrtc64_120_0.dll", "nvrtc64_130_0.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cuda_nvrtc/bin"),
        requires_add_dll_directory=True,
    ),
    DescriptorSpec(
        name="nvvm",
        packaged_with="ctk",
        linux_sonames=("libnvvm.so.4",),
        windows_dlls=("nvvm64.dll", "nvvm64_40_0.dll", "nvvm70.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_nvcc/nvvm/lib64"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cuda_nvcc/nvvm/bin"),
        anchor_rel_dirs_linux=("nvvm/lib64",),
        anchor_rel_dirs_windows=("nvvm/bin/*", "nvvm/bin"),
        ctk_root_canary_anchor_libnames=("cudart",),
    ),
    DescriptorSpec(
        name="cublas",
        packaged_with="ctk",
        linux_sonames=("libcublas.so.12", "libcublas.so.13"),
        windows_dlls=("cublas64_12.dll", "cublas64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cublas/bin"),
        dependencies=("cublasLt",),
    ),
    DescriptorSpec(
        name="cublasLt",
        packaged_with="ctk",
        linux_sonames=("libcublasLt.so.12", "libcublasLt.so.13"),
        windows_dlls=("cublasLt64_12.dll", "cublasLt64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cublas/bin"),
    ),
    DescriptorSpec(
        name="cufft",
        packaged_with="ctk",
        linux_sonames=("libcufft.so.11", "libcufft.so.12"),
        windows_dlls=("cufft64_11.dll", "cufft64_12.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufft/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cufft/bin"),
        requires_add_dll_directory=True,
    ),
    DescriptorSpec(
        name="cufftw",
        packaged_with="ctk",
        linux_sonames=("libcufftw.so.11", "libcufftw.so.12"),
        windows_dlls=("cufftw64_11.dll", "cufftw64_12.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufft/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cufft/bin"),
        dependencies=("cufft",),
    ),
    DescriptorSpec(
        name="curand",
        packaged_with="ctk",
        linux_sonames=("libcurand.so.10",),
        windows_dlls=("curand64_10.dll",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/curand/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/curand/bin"),
    ),
    DescriptorSpec(
        name="cusolver",
        packaged_with="ctk",
        linux_sonames=("libcusolver.so.11", "libcusolver.so.12"),
        windows_dlls=("cusolver64_11.dll", "cusolver64_12.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusolver/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cusolver/bin"),
        dependencies=("nvJitLink", "cusparse", "cublasLt", "cublas"),
    ),
    DescriptorSpec(
        name="cusolverMg",
        packaged_with="ctk",
        linux_sonames=("libcusolverMg.so.11", "libcusolverMg.so.12"),
        windows_dlls=("cusolverMg64_11.dll", "cusolverMg64_12.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusolver/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cusolver/bin"),
        dependencies=("nvJitLink", "cublasLt", "cublas"),
    ),
    DescriptorSpec(
        name="cusparse",
        packaged_with="ctk",
        linux_sonames=("libcusparse.so.12",),
        windows_dlls=("cusparse64_12.dll",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusparse/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cusparse/bin"),
        dependencies=("nvJitLink",),
    ),
    DescriptorSpec(
        name="nppc",
        packaged_with="ctk",
        linux_sonames=("libnppc.so.12", "libnppc.so.13"),
        windows_dlls=("nppc64_12.dll", "nppc64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
    ),
    DescriptorSpec(
        name="nppial",
        packaged_with="ctk",
        linux_sonames=("libnppial.so.12", "libnppial.so.13"),
        windows_dlls=("nppial64_12.dll", "nppial64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppicc",
        packaged_with="ctk",
        linux_sonames=("libnppicc.so.12", "libnppicc.so.13"),
        windows_dlls=("nppicc64_12.dll", "nppicc64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppidei",
        packaged_with="ctk",
        linux_sonames=("libnppidei.so.12", "libnppidei.so.13"),
        windows_dlls=("nppidei64_12.dll", "nppidei64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppif",
        packaged_with="ctk",
        linux_sonames=("libnppif.so.12", "libnppif.so.13"),
        windows_dlls=("nppif64_12.dll", "nppif64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppig",
        packaged_with="ctk",
        linux_sonames=("libnppig.so.12", "libnppig.so.13"),
        windows_dlls=("nppig64_12.dll", "nppig64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppim",
        packaged_with="ctk",
        linux_sonames=("libnppim.so.12", "libnppim.so.13"),
        windows_dlls=("nppim64_12.dll", "nppim64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppist",
        packaged_with="ctk",
        linux_sonames=("libnppist.so.12", "libnppist.so.13"),
        windows_dlls=("nppist64_12.dll", "nppist64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppisu",
        packaged_with="ctk",
        linux_sonames=("libnppisu.so.12", "libnppisu.so.13"),
        windows_dlls=("nppisu64_12.dll", "nppisu64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppitc",
        packaged_with="ctk",
        linux_sonames=("libnppitc.so.12", "libnppitc.so.13"),
        windows_dlls=("nppitc64_12.dll", "nppitc64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="npps",
        packaged_with="ctk",
        linux_sonames=("libnpps.so.12", "libnpps.so.13"),
        windows_dlls=("npps64_12.dll", "npps64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nvblas",
        packaged_with="ctk",
        linux_sonames=("libnvblas.so.12", "libnvblas.so.13"),
        windows_dlls=("nvblas64_12.dll", "nvblas64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cublas/bin"),
        dependencies=("cublas", "cublasLt"),
    ),
    DescriptorSpec(
        name="nvjpeg",
        packaged_with="ctk",
        linux_sonames=("libnvjpeg.so.12", "libnvjpeg.so.13"),
        windows_dlls=("nvjpeg64_12.dll", "nvjpeg64_13.dll"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvjpeg/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/nvjpeg/bin"),
    ),
    DescriptorSpec(
        name="cufile",
        packaged_with="ctk",
        linux_sonames=("libcufile.so.0",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufile/lib"),
    ),
    DescriptorSpec(
        name="cupti",
        packaged_with="ctk",
        linux_sonames=("libcupti.so.12", "libcupti.so.13"),
        windows_dlls=(
            "cupti64_2026.1.0.dll",
            "cupti64_2025.4.1.dll",
            "cupti64_2025.3.1.dll",
            "cupti64_2025.2.1.dll",
            "cupti64_2025.1.1.dll",
            "cupti64_2024.3.2.dll",
            "cupti64_2024.2.1.dll",
            "cupti64_2024.1.1.dll",
            "cupti64_2023.3.1.dll",
            "cupti64_2023.2.2.dll",
            "cupti64_2023.1.1.dll",
            "cupti64_2022.4.1.dll",
        ),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_cupti/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cuda_cupti/bin"),
        anchor_rel_dirs_linux=("extras/CUPTI/lib64", "lib"),
        anchor_rel_dirs_windows=("extras/CUPTI/lib64", "bin"),
        ctk_root_canary_anchor_libnames=("cudart",),
    ),
    # -----------------------------------------------------------------------
    # Third-party / separately packaged libraries
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cublasmp",
        packaged_with="other",
        linux_sonames=("libcublasmp.so.0",),
        site_packages_linux=("nvidia/cublasmp/cu13/lib", "nvidia/cublasmp/cu12/lib"),
        dependencies=("cublas", "cublasLt", "nvshmem_host"),
    ),
    DescriptorSpec(
        name="cufftMp",
        packaged_with="other",
        linux_sonames=("libcufftMp.so.12", "libcufftMp.so.11"),
        site_packages_linux=("nvidia/cufftmp/cu13/lib", "nvidia/cufftmp/cu12/lib"),
        dependencies=("nvshmem_host",),
        requires_rtld_deepbind=True,
    ),
    DescriptorSpec(
        name="mathdx",
        packaged_with="other",
        linux_sonames=("libmathdx.so.0",),
        windows_dlls=("mathdx64_0.dll",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cu12/lib"),
        site_packages_windows=("nvidia/cu13/bin/x86_64", "nvidia/cu12/bin"),
        dependencies=("nvrtc",),
    ),
    DescriptorSpec(
        name="cudss",
        packaged_with="other",
        linux_sonames=("libcudss.so.0",),
        windows_dlls=("cudss64_0.dll",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cu12/lib"),
        site_packages_windows=("nvidia/cu13/bin", "nvidia/cu12/bin"),
        dependencies=("cublas", "cublasLt"),
    ),
    DescriptorSpec(
        name="cusparseLt",
        packaged_with="other",
        linux_sonames=("libcusparseLt.so.0",),
        windows_dlls=("cusparseLt.dll",),
        site_packages_linux=("nvidia/cusparselt/lib",),
        site_packages_windows=("nvidia/cusparselt/bin",),
    ),
    DescriptorSpec(
        name="cutensor",
        packaged_with="other",
        linux_sonames=("libcutensor.so.2",),
        windows_dlls=("cutensor.dll",),
        site_packages_linux=("cutensor/lib",),
        site_packages_windows=("cutensor/bin",),
        dependencies=("cublasLt",),
    ),
    DescriptorSpec(
        name="cutensorMg",
        packaged_with="other",
        linux_sonames=("libcutensorMg.so.2",),
        windows_dlls=("cutensorMg.dll",),
        site_packages_linux=("cutensor/lib",),
        site_packages_windows=("cutensor/bin",),
        dependencies=("cutensor", "cublasLt"),
    ),
    DescriptorSpec(
        name="nccl",
        packaged_with="other",
        linux_sonames=("libnccl.so.2",),
        site_packages_linux=("nvidia/nccl/lib",),
    ),
    DescriptorSpec(
        name="nvpl_fftw",
        packaged_with="other",
        linux_sonames=("libnvpl_fftw.so.0",),
        site_packages_linux=("nvpl/lib",),
    ),
    DescriptorSpec(
        name="nvshmem_host",
        packaged_with="other",
        linux_sonames=("libnvshmem_host.so.3",),
        site_packages_linux=("nvidia/nvshmem/lib",),
    ),
    # -----------------------------------------------------------------------
    # Driver libraries (system-search only, no CTK cascade)
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cuda",
        packaged_with="driver",
        linux_sonames=("libcuda.so.1",),
        windows_dlls=("nvcuda.dll",),
    ),
    DescriptorSpec(
        name="nvml",
        packaged_with="driver",
        linux_sonames=("libnvidia-ml.so.1",),
        windows_dlls=("nvml.dll",),
    ),
)
