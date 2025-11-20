// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// drvmodel.c
// Query NVML for the Windows driver model (WDDM / WDM(TCC) / MCDM) of each GPU.
//
// Build example (MSVC, adjust paths as needed):
//   cl /nologo /W3 drvmodel.c /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64" nvml.lib
//
// On success, prints something like:
//   GPU 0: NVIDIA RTX A6000
//     Current driver model: WDDM
//     Pending driver model: WDDM

#include <stdio.h>
#include <stdlib.h>

#include "nvml.h"  // from NVIDIA NVML package / CUDA toolkit

static const char *driverModelToString(nvmlDriverModel_t m)
{
    switch (m) {
    case NVML_DRIVER_WDDM:
        return "WDDM (display device)";
    case NVML_DRIVER_WDM:
        return "WDM (TCC, compute device)";
#ifdef NVML_DRIVER_MCDM
    case NVML_DRIVER_MCDM:
        return "MCDM (Microsoft compute device)";
#endif
    default:
        return "Unknown";
    }
}

int main(void)
{
    nvmlReturn_t result;
    unsigned int deviceCount = 0;
    unsigned int i;

    result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "nvmlInit_v2() failed: %s\n", nvmlErrorString(result));
        return EXIT_FAILURE;
    }

    result = nvmlDeviceGetCount_v2(&deviceCount);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "nvmlDeviceGetCount_v2() failed: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No NVIDIA GPUs found.\n");
        nvmlShutdown();
        return EXIT_SUCCESS;
    }

    for (i = 0; i < deviceCount; ++i) {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
        nvmlDriverModel_t currentModel = 0;
        nvmlDriverModel_t pendingModel = 0;

        result = nvmlDeviceGetHandleByIndex_v2(i, &device);
        if (result != NVML_SUCCESS) {
            fprintf(stderr,
                    "nvmlDeviceGetHandleByIndex_v2(%u) failed: %s\n",
                    i, nvmlErrorString(result));
            continue;
        }

        result = nvmlDeviceGetName(device, name, sizeof(name));
        if (result != NVML_SUCCESS) {
            snprintf(name, sizeof(name), "<unknown>");
        }

        result = nvmlDeviceGetDriverModel(device, &currentModel, &pendingModel);
        if (result == NVML_ERROR_NOT_SUPPORTED) {
            printf("GPU %u: %s\n", i, name);
            printf("  Driver model query not supported (non-Windows or unsupported device).\n");
            continue;
        } else if (result != NVML_SUCCESS) {
            fprintf(stderr,
                    "nvmlDeviceGetDriverModel(%u) failed: %s\n",
                    i, nvmlErrorString(result));
            continue;
        }

        printf("GPU %u: %s\n", i, name);
        printf("  Current driver model: %s\n", driverModelToString(currentModel));
        printf("  Pending driver model: %s\n", driverModelToString(pendingModel));
    }

    nvmlShutdown();
    return EXIT_SUCCESS;
}
