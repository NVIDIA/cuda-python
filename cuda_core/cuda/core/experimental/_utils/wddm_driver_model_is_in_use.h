// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wddm_driver_model_is_in_use
 *
 * Return codes:
 *   -2 : Failed to get device count from NVML
 *   -1 : Not available on this platform (not compiled with MSVC on Windows) or NVML initialization failed
 *    0 : No WDDM driver model found (all devices use TCC or other driver models)
 *    1 : WDDM driver model is in use (at least one device uses WDDM)
 */
int wddm_driver_model_is_in_use(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif
