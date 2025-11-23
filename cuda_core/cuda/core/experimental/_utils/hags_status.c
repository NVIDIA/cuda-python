// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Note, this may or may not exist, but is NOT the ground truth:
// reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode
// The HwSchMode registry value is only a user override (force on/off).
// If absent, Windows uses the driver's WDDM caps defaults.
// Actual HAGS state comes from D3DKMT_WDDM_2_7_CAPS, not the registry.

// Possibly useful for experimentation:
// reg delete "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v HwSchMode /f

#ifdef _MSC_VER
#include <windows.h>
#include <d3dkmthk.h>
#include <d3dkmdt.h>
#endif

int hags_status(void)
{
#ifdef _MSC_VER
    DISPLAY_DEVICEW dd;
    HDC hdc;
    int i;
    BOOL foundPrimary = FALSE;
    NTSTATUS status;

    D3DKMT_OPENADAPTERFROMHDC openData;
    D3DKMT_QUERYADAPTERINFO   query;
    D3DKMT_WDDM_2_7_CAPS      caps;
    D3DKMT_CLOSEADAPTER       closeData;

    // Find the primary display device
    ZeroMemory(&dd, sizeof(dd));
    dd.cb = sizeof(dd);

    for (i = 0; EnumDisplayDevicesW(NULL, i, &dd, 0); ++i) {
        if (dd.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE) {
            foundPrimary = TRUE;
            break;
        }
    }

    if (!foundPrimary)
        return 0;

    hdc = CreateDCW(NULL, dd.DeviceName, NULL, NULL);
    if (!hdc)
        return 0;

    ZeroMemory(&openData, sizeof(openData));
    openData.hDc = hdc;
    status = D3DKMTOpenAdapterFromHdc(&openData);

    DeleteDC(hdc);

    if (status != 0)
        return 0;

    ZeroMemory(&caps, sizeof(caps));
    ZeroMemory(&query, sizeof(query));

    query.hAdapter             = openData.hAdapter;
    query.Type                 = KMTQAITYPE_WDDM_2_7_CAPS;
    query.pPrivateDriverData   = &caps;
    query.PrivateDriverDataSize = sizeof(caps);

    status = D3DKMTQueryAdapterInfo(&query);

    ZeroMemory(&closeData, sizeof(closeData));
    closeData.hAdapter = openData.hAdapter;
    D3DKMTCloseAdapter(&closeData);

    if (status != 0)
        return 0;

    if (!caps.HwSchSupported || !caps.HwSchEnabled)
        return 1;

    return 2;
#else
    return -1;
#endif
}
