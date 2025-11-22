// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Query NVML for the Windows WDDM driver model, looping over all GPUs.
//
// On non-Windows platforms this always returns -1 and performs no NVML calls.
//
// Example compilation command (Windows/MSVC):
//     cl /nologo /c wddm_driver_model_is_in_use.c /I"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\include"
// Needed for linking:
//     /link /LIBPATH:"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\lib\\x64" nvml.lib
//
#include "wddm_driver_model_is_in_use.h"

#ifdef _MSC_VER

#include "nvml.h"  // from NVIDIA GPU Computing Toolkit

static int wddm_driver_model_is_in_use_impl(void)
{
    unsigned deviceCount = 0;
    nvmlReturn_t result = nvmlDeviceGetCount_v2(&deviceCount);
    if (result != NVML_SUCCESS) {
        return -2;
    }
    for (unsigned i_dev = 0; i_dev < deviceCount; ++i_dev) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex_v2(i_dev, &device);
        if (result == NVML_SUCCESS) {
            nvmlDriverModel_t currentModel = 0;
            nvmlDriverModel_t pendingModel = 0;
            result = nvmlDeviceGetDriverModel(device, &currentModel, &pendingModel);
            if (result == NVML_SUCCESS) {
                if (currentModel == NVML_DRIVER_WDDM || pendingModel == NVML_DRIVER_WDDM) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

int wddm_driver_model_is_in_use(void)
{
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        return -1;
    }
    int return_code = wddm_driver_model_is_in_use_impl();
    nvmlShutdown();
    return return_code;
}

#else  // !_MSC_VER

int wddm_driver_model_is_in_use(void)
{
    // WDDM is a Windows-only concept; on non-Windows platforms we report -1
    // to indicate that the driver model could not be determined.
    return -1;
}

#endif  // _MSC_VER
