# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time

from runner.runtime import alloc_persistent, ensure_context

from cuda.bindings import driver as cuda

ensure_context()

PTR = alloc_persistent(1 << 20)

cuuint32_t = cuda.cuuint32_t
cuuint64_t = cuda.cuuint64_t

# Tiled: rank-2 float32, 128x128, 64x64 tile.
TILED_DTYPE = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32
TILED_RANK = 2
TILED_GLOBAL_DIM = (cuuint64_t(128), cuuint64_t(128))
TILED_GLOBAL_STRIDES = (cuuint64_t(128 * 4),)
TILED_BOX_DIM = (cuuint32_t(64), cuuint32_t(64))
TILED_ELEMENT_STRIDES = (cuuint32_t(1), cuuint32_t(1))
TILED_INTERLEAVE = cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
TILED_SWIZZLE = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
TILED_L2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
TILED_OOB = cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

# Im2col: rank-3 float16, 32x64x64.
IM2COL_DTYPE = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16
IM2COL_RANK = 3
IM2COL_GLOBAL_DIM = (cuuint64_t(32), cuuint64_t(64), cuuint64_t(64))
IM2COL_GLOBAL_STRIDES = (cuuint64_t(32 * 2), cuuint64_t(32 * 64 * 2))
IM2COL_PIXEL_BOX_LOWER = (0,)
IM2COL_PIXEL_BOX_UPPER = (0,)
IM2COL_CHANNELS = 32
IM2COL_PIXELS = 32
IM2COL_ELEMENT_STRIDES = (cuuint32_t(1), cuuint32_t(1), cuuint32_t(1))
IM2COL_INTERLEAVE = cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE
IM2COL_SWIZZLE = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
IM2COL_L2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE
IM2COL_OOB = cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE

_SUCCESS = cuda.CUresult.CUDA_SUCCESS


def _probe_tiled() -> bool:
    try:
        err, _ = cuda.cuTensorMapEncodeTiled(
            TILED_DTYPE,
            TILED_RANK,
            PTR,
            TILED_GLOBAL_DIM,
            TILED_GLOBAL_STRIDES,
            TILED_BOX_DIM,
            TILED_ELEMENT_STRIDES,
            TILED_INTERLEAVE,
            TILED_SWIZZLE,
            TILED_L2,
            TILED_OOB,
        )
    except Exception:
        return False
    return err == _SUCCESS


def _probe_im2col() -> bool:
    try:
        err, _ = cuda.cuTensorMapEncodeIm2col(
            IM2COL_DTYPE,
            IM2COL_RANK,
            PTR,
            IM2COL_GLOBAL_DIM,
            IM2COL_GLOBAL_STRIDES,
            IM2COL_PIXEL_BOX_LOWER,
            IM2COL_PIXEL_BOX_UPPER,
            IM2COL_CHANNELS,
            IM2COL_PIXELS,
            IM2COL_ELEMENT_STRIDES,
            IM2COL_INTERLEAVE,
            IM2COL_SWIZZLE,
            IM2COL_L2,
            IM2COL_OOB,
        )
    except Exception:
        return False
    return err == _SUCCESS


_ENCODE_IM2COL_WIDE = getattr(cuda, "cuTensorMapEncodeIm2colWide", None)
_IM2COL_WIDE_MODE_CLS = getattr(cuda, "CUtensorMapIm2ColWideMode", None)


def _probe_im2col_wide() -> bool:
    if _ENCODE_IM2COL_WIDE is None or _IM2COL_WIDE_MODE_CLS is None:
        return False
    try:
        mode = _IM2COL_WIDE_MODE_CLS.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W
        err, _ = _ENCODE_IM2COL_WIDE(
            IM2COL_DTYPE,
            IM2COL_RANK,
            PTR,
            IM2COL_GLOBAL_DIM,
            IM2COL_GLOBAL_STRIDES,
            0,
            0,
            IM2COL_CHANNELS,
            IM2COL_PIXELS,
            IM2COL_ELEMENT_STRIDES,
            IM2COL_INTERLEAVE,
            mode,
            cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            IM2COL_L2,
            IM2COL_OOB,
        )
    except Exception:
        return False
    return err == _SUCCESS


_TILED_OK = _probe_tiled()
_IM2COL_OK = _probe_im2col()
_IM2COL_WIDE_OK = _probe_im2col_wide()

if _IM2COL_WIDE_OK:
    _IM2COL_WIDE_MODE_W = _IM2COL_WIDE_MODE_CLS.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W
    _IM2COL_WIDE_SWIZZLE = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B

SKIPPED_BENCHMARKS: set[str] = set()
if not _TILED_OK:
    SKIPPED_BENCHMARKS.add("bench_tensor_map_encode_tiled")
if not _IM2COL_OK:
    SKIPPED_BENCHMARKS.add("bench_tensor_map_encode_im2col")
if not _IM2COL_WIDE_OK:
    SKIPPED_BENCHMARKS.add("bench_tensor_map_encode_im2col_wide")


def bench_tensor_map_encode_tiled(loops: int) -> float:
    _fn = cuda.cuTensorMapEncodeTiled
    _dt = TILED_DTYPE
    _rank = TILED_RANK
    _addr = PTR
    _gdim = TILED_GLOBAL_DIM
    _gstr = TILED_GLOBAL_STRIDES
    _bdim = TILED_BOX_DIM
    _estr = TILED_ELEMENT_STRIDES
    _inter = TILED_INTERLEAVE
    _swz = TILED_SWIZZLE
    _l2 = TILED_L2
    _oob = TILED_OOB

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_dt, _rank, _addr, _gdim, _gstr, _bdim, _estr, _inter, _swz, _l2, _oob)
    return time.perf_counter() - t0


def bench_tensor_map_encode_im2col(loops: int) -> float:
    _fn = cuda.cuTensorMapEncodeIm2col
    _dt = IM2COL_DTYPE
    _rank = IM2COL_RANK
    _addr = PTR
    _gdim = IM2COL_GLOBAL_DIM
    _gstr = IM2COL_GLOBAL_STRIDES
    _lower = IM2COL_PIXEL_BOX_LOWER
    _upper = IM2COL_PIXEL_BOX_UPPER
    _ch = IM2COL_CHANNELS
    _px = IM2COL_PIXELS
    _estr = IM2COL_ELEMENT_STRIDES
    _inter = IM2COL_INTERLEAVE
    _swz = IM2COL_SWIZZLE
    _l2 = IM2COL_L2
    _oob = IM2COL_OOB

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_dt, _rank, _addr, _gdim, _gstr, _lower, _upper, _ch, _px, _estr, _inter, _swz, _l2, _oob)
    return time.perf_counter() - t0


def bench_tensor_map_encode_im2col_wide(loops: int) -> float:
    _fn = _ENCODE_IM2COL_WIDE
    _dt = IM2COL_DTYPE
    _rank = IM2COL_RANK
    _addr = PTR
    _gdim = IM2COL_GLOBAL_DIM
    _gstr = IM2COL_GLOBAL_STRIDES
    _lower_w = 0
    _upper_w = 0
    _ch = IM2COL_CHANNELS
    _px = IM2COL_PIXELS
    _estr = IM2COL_ELEMENT_STRIDES
    _inter = IM2COL_INTERLEAVE
    _mode = _IM2COL_WIDE_MODE_W
    _swz = _IM2COL_WIDE_SWIZZLE
    _l2 = IM2COL_L2
    _oob = IM2COL_OOB

    t0 = time.perf_counter()
    for _ in range(loops):
        _fn(_dt, _rank, _addr, _gdim, _gstr, _lower_w, _upper_w, _ch, _px, _estr, _inter, _mode, _swz, _l2, _oob)
    return time.perf_counter() - t0
