// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * hags_status
 *
 * Return codes:
 *   -1 : Not available on this platform (not compiled with MSVC on Windows)
 *    0 : Failure obtaining HwSchSupported/HwSchEnabled
 *    1 : HwSchSupported == 0 or HwSchEnabled == 0 (HAGS not fully enabled)
 *    2 : HwSchSupported == 1 and HwSchEnabled == 1 (HAGS fully enabled)
 */
int hags_status(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif
