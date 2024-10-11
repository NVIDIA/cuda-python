// Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "loader.h"

#define DXCORE_MAX_PATH 260

#if defined(_WIN32)
#include "windows.h"
#define _getAddr GetProcAddress
#define _Handle HMODULE
static const size_t sysrootName64_length = (sizeof("System32") - 1);
static const char* sysrootName64 = "System32";
static const size_t libcudaName64_length = (sizeof("\\nvcuda64.dll") - 1);
static const char* libcudaName64 = "\\nvcuda64.dll";
static const size_t sysrootNameX86_length = (sizeof("SysWOW64") - 1);
static const char* sysrootNameX86 = "SysWOW64";
static const size_t libcudaNameX86_length = (sizeof("\\nvcuda32.dll") - 1);
static const char* libcudaNameX86 = "\\nvcuda32.dll";
static size_t sysrootName_length = NULL;
static const char* sysrootName = NULL;

#else
#include <dlfcn.h>
#include <unistd.h>
#define _getAddr dlsym
#define _Handle void*
static const size_t libcudaNameLinux_length = (sizeof("/libcuda.so.1.1") - 1);
static const char* libcudaNameLinux = "/libcuda.so.1.1";
#endif
static size_t libcudaName_length = 0;
static const char* libcudaName = NULL;

struct dxcore_enumAdapters2;
struct dxcore_queryAdapterInfo;

typedef int (*pfnDxcoreEnumAdapters2)(const dxcore_enumAdapters2 *pParams);
typedef int (*pfnDxcoreQueryAdapterInfo)(const dxcore_queryAdapterInfo *pParams);

struct dxcore_lib {
    _Handle hDxcoreLib;
    pfnDxcoreEnumAdapters2 pDxcoreEnumAdapters2;
    pfnDxcoreQueryAdapterInfo pDxcoreQueryAdapterInfo;
};

struct dxcore_luid
{
    unsigned int lowPart;
    int highPart;
};

struct dxcore_adapterInfo
{
    unsigned int              hAdapter;
    struct dxcore_luid        AdapterLuid;
    unsigned int              NumOfSources;
    unsigned int              bPresentMoveRegionsPreferred;
};

struct dxcore_enumAdapters2
{
    unsigned int                   NumAdapters;
    struct dxcore_adapterInfo     *pAdapters;
};

enum dxcore_kmtqueryAdapterInfoType
{
    DXCORE_QUERYDRIVERVERSION = 13,
    DXCORE_QUERYREGISTRY = 48,
};

enum dxcore_queryregistry_type {
    DXCORE_QUERYREGISTRY_DRIVERSTOREPATH = 2,
};

enum dxcore_queryregistry_status {
    DXCORE_QUERYREGISTRY_STATUS_SUCCESS = 0,
    DXCORE_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW = 1,
    DXCORE_QUERYREGISTRY_STATUS_FAIL = 2,
};

struct dxcore_queryregistry_info {
    enum dxcore_queryregistry_type        QueryType;
    unsigned int                          QueryFlags;
    wchar_t                               ValueName[DXCORE_MAX_PATH];
    unsigned int                          ValueType;
    unsigned int                          PhysicalAdapterIndex;
    unsigned int                          OutputValueSize;
    enum dxcore_queryregistry_status      Status;
    union {
        unsigned long long                    OutputQword;
        wchar_t                               Output;
    };
};

struct dxcore_queryAdapterInfo
{
    unsigned int                           hAdapter;
    enum dxcore_kmtqueryAdapterInfoType    Type;
    void                                   *pPrivateDriverData;
    unsigned int                           PrivateDriverDataSize;
};

static int dxcore_query_adapter_info_helper(struct dxcore_lib* pLib,
                                            unsigned int hAdapter,
                                            enum dxcore_kmtqueryAdapterInfoType type,
                                            void* pPrivateDriverDate,
                                            unsigned int privateDriverDataSize)
{
    struct dxcore_queryAdapterInfo queryAdapterInfo = {};

    queryAdapterInfo.hAdapter = hAdapter;
    queryAdapterInfo.Type = type;
    queryAdapterInfo.pPrivateDriverData = pPrivateDriverDate;
    queryAdapterInfo.PrivateDriverDataSize = privateDriverDataSize;

    return pLib->pDxcoreQueryAdapterInfo(&queryAdapterInfo);
}

static int dxcore_query_adapter_wddm_version(struct dxcore_lib* pLib, unsigned int hAdapter, unsigned int* version)
{
        return dxcore_query_adapter_info_helper(pLib,
                                                hAdapter,
                                                DXCORE_QUERYDRIVERVERSION,
                                                (void*)version,
                                                (unsigned int)sizeof(*version));
}

static int dxcore_query_adapter_driverstore_path(struct dxcore_lib* pLib, unsigned int hAdapter, char** ppDriverStorePath)
{
    struct dxcore_queryregistry_info params = {};
    struct dxcore_queryregistry_info* pValue = NULL;
    wchar_t* pOutput;
    size_t outputSizeInBytes;
    size_t outputSize;

    // 1. Fetch output size
    params.QueryType = DXCORE_QUERYREGISTRY_DRIVERSTOREPATH;

    if (dxcore_query_adapter_info_helper(pLib,
                                         hAdapter,
                                         DXCORE_QUERYREGISTRY,
                                         (void*)&params,
                                         (unsigned int)sizeof(struct dxcore_queryregistry_info)))
    {
        return (-1);
    }

    if (params.OutputValueSize > DXCORE_MAX_PATH * sizeof(wchar_t)) {
        return (-1);
    }

    outputSizeInBytes = (size_t)params.OutputValueSize;
    outputSize = outputSizeInBytes / sizeof(wchar_t);

    // 2. Retrieve output
    pValue = (struct dxcore_queryregistry_info*)calloc(sizeof(struct dxcore_queryregistry_info) + outputSizeInBytes + sizeof(wchar_t), 1);
    if (!pValue) {
        return (-1);
    }

    pValue->QueryType = DXCORE_QUERYREGISTRY_DRIVERSTOREPATH;
    pValue->OutputValueSize = (unsigned int)outputSizeInBytes;

    if (dxcore_query_adapter_info_helper(pLib,
                                         hAdapter,
                                         DXCORE_QUERYREGISTRY,
                                         (void*)pValue,
                                         (unsigned int)(sizeof(struct dxcore_queryregistry_info) + outputSizeInBytes)))
    {
        free(pValue);
        return (-1);
    }
    pOutput = (wchar_t*)(&pValue->Output);

    // Make sure no matter what happened the wchar_t string is null terminated
    pOutput[outputSize] = L'\0';

    // Convert the output into a regular c string
    *ppDriverStorePath = (char*)calloc(outputSize + 1, sizeof(char));
    if (!*ppDriverStorePath) {
        free(pValue);
        return (-1);
    }
    wcstombs(*ppDriverStorePath, pOutput, outputSize);

    free(pValue);

    return 0;
}

static char* replaceSystemPath(char* path)
{
    char *replacedPath = (char*)calloc(DXCORE_MAX_PATH + 1, sizeof(char));

#if defined(_WIN32)
    wchar_t *systemPath = (wchar_t*)calloc(DXCORE_MAX_PATH + 1, sizeof(wchar_t));
    // Get system root path
    if (GetSystemDirectoryW(systemPath, DXCORE_MAX_PATH) == 0) {
        free(replacedPath);
        free(systemPath);
        return NULL;
    }
    wcstombs(replacedPath, systemPath, DXCORE_MAX_PATH);
    free(systemPath);

    // Replace the /SystemRoot/ part of the registry-obtained path with
    // the actual system root path from above
    char* sysrootPath = strstr(path, sysrootName);
    strncat(replacedPath, sysrootPath + sysrootName_length, DXCORE_MAX_PATH - strlen(replacedPath));
#else
    strncat(replacedPath, path, DXCORE_MAX_PATH);
#endif

    // Append nvcuda dll
    if (libcudaName_length < DXCORE_MAX_PATH - strlen(replacedPath)) {
        strncat(replacedPath, libcudaName, libcudaName_length);
    }
    else {
        strncat(replacedPath, libcudaName, DXCORE_MAX_PATH - strlen(replacedPath));
    }

    return replacedPath;
}

static int dxcore_check_adapter(struct dxcore_lib *pLib, char *libPath, struct dxcore_adapterInfo *pAdapterInfo)
{
    unsigned int wddmVersion = 0;
    char* driverStorePath = NULL;

    if (dxcore_query_adapter_wddm_version(pLib, pAdapterInfo->hAdapter, &wddmVersion)) {
        return 1;
    }

    if (wddmVersion < 2500) {
        return 1;
    }

    if (dxcore_query_adapter_driverstore_path(pLib, pAdapterInfo->hAdapter, &driverStorePath)) {
        return 1;
    }

    // Replace with valid path
    char* replacedPath = replaceSystemPath(driverStorePath);
    if (!replacedPath) {
        free(driverStorePath);
        free(replacedPath);
        return 1;
    }

    // Does file exist?
#if defined(_WIN32)
    if (GetFileAttributes(replacedPath) == INVALID_FILE_ATTRIBUTES) {
        free(driverStorePath);
        free(replacedPath);
        return 1;
    }
#else
    if (access(replacedPath, F_OK) < 0) {
        free(driverStorePath);
        free(replacedPath);
        return 1;
    }
#endif

    memcpy(libPath, replacedPath, DXCORE_MAX_PATH);
    free(driverStorePath);
    free(replacedPath);

    return 0;
}

static int dxcore_enum_adapters(struct dxcore_lib *pLib, char *libPath)
{
    struct dxcore_enumAdapters2 params = {0};
    unsigned int adapterIndex = 0;

    if (pLib->pDxcoreEnumAdapters2(&params)) {
        return 1;
    }
    params.pAdapters = (dxcore_adapterInfo*)calloc(params.NumAdapters, sizeof(struct dxcore_adapterInfo));
    if (pLib->pDxcoreEnumAdapters2(&params)) {
        free(params.pAdapters);
        return 1;
    }

    for (adapterIndex = 0; adapterIndex < params.NumAdapters; adapterIndex++) {
        if (!dxcore_check_adapter(pLib, libPath, &params.pAdapters[adapterIndex])) {
            free(params.pAdapters);
            return 0;
        }
    }

    free(params.pAdapters);
    return 1;
}

int getCUDALibraryPath(char *libPath, bool isBit64)
{
    struct dxcore_lib lib = {0};

    if (!libPath) {
        return 1;
    }

    // Configure paths based on app's bit configuration
#if defined(_WIN32)
    if (isBit64) {
        sysrootName_length = sysrootName64_length;
        sysrootName = sysrootName64;
        libcudaName_length = libcudaName64_length;
        libcudaName = libcudaName64;
    }
    else {
        sysrootName_length = sysrootNameX86_length;
        sysrootName = sysrootNameX86;
        libcudaName_length = libcudaNameX86_length;
        libcudaName = libcudaNameX86;
    }
#else
    libcudaName_length = libcudaNameLinux_length;
    libcudaName = libcudaNameLinux;
#endif

#if defined(_WIN32)
    lib.hDxcoreLib = LoadLibraryExW(L"gdi32.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    lib.hDxcoreLib = dlopen("libdxcore.so", RTLD_LAZY);
#endif
    if (!lib.hDxcoreLib) {
        return 1;
    }

    lib.pDxcoreEnumAdapters2 = (pfnDxcoreEnumAdapters2)_getAddr(lib.hDxcoreLib, "D3DKMTEnumAdapters2");
    if (!lib.pDxcoreEnumAdapters2) {
        return 1;
    }
    lib.pDxcoreQueryAdapterInfo = (pfnDxcoreQueryAdapterInfo)_getAddr(lib.hDxcoreLib, "D3DKMTQueryAdapterInfo");
    if (!lib.pDxcoreQueryAdapterInfo) {
        return 1;
    }

    if (dxcore_enum_adapters(&lib, libPath)) {
        return 1;
    }
    return 0;
}
