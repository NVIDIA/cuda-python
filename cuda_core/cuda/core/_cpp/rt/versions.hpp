// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The one place the C++ under _cpp/ consults CUDA_VERSION.
//
// cuda.core supports one build configuration per CUDA major series. The
// cuda.h it compiles against has the same major.minor as the cuda-bindings
// present at build time, and that cuda-bindings is at or above the series'
// floor (see cuda/core/_bindings_floor.py and
// https://github.com/NVIDIA/cuda-python/issues/2783). build_hooks.py enforces
// both before it compiles and passes the decision down as two macros:
//
//   CUDA_CORE_BUILD_MAJOR        the CUDA major series of the build, 12 or 13.
//                                The only version the C++ may branch on, as
//                                `#if CUDA_CORE_BUILD_MAJOR >= 13`, and always
//                                for a difference between major series.
//   CUDA_CORE_MIN_CUDA_VERSION   the floor's major.minor as a CUDA_VERSION
//                                value, for example 13040.
//
// This file checks cuda.h against both macros again, so that a build that
// bypasses build_hooks.py still cannot compile against an unsupported header.
// No other file may use a minor-version fence such as
// `#if CUDA_VERSION >= 130x0`. Such fences compiled features out of source
// builds against an older header, and the run-time checks, which looked at
// cuda-bindings and the driver, never noticed. tests/test_rt_layout.py
// enforces that this is the only file that names CUDA_VERSION.
//
// Downstream Cython code that cimports _rt includes this header without the
// macros. It then learns only the major from cuda.h and skips the floor check.

#include <cuda.h>

#ifndef CUDA_CORE_BUILD_MAJOR
#define CUDA_CORE_BUILD_MAJOR (CUDA_VERSION / 1000)
#endif

#if (CUDA_VERSION / 1000) != CUDA_CORE_BUILD_MAJOR
#error "cuda.h does not belong to the CUDA major series this cuda.core build targets (CUDA_CORE_BUILD_MAJOR)"
#endif

#ifdef CUDA_CORE_MIN_CUDA_VERSION
#if CUDA_VERSION < CUDA_CORE_MIN_CUDA_VERSION
#error "cuda.h is older than the floor of this cuda.core release for its CUDA major series (see the cuda.core support policy)"
#endif
#endif
