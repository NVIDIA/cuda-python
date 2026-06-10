# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# configure_driver_mode.ps1 -- set the NVIDIA driver mode on a Windows CI
# runner and cycle the display devices so the new mode takes effect
# without rebooting. Always runs (whether or not install_gpu_driver.ps1
# just ran). When install_gpu_driver.ps1 has run, this single device
# cycle also activates the freshly-installed driver.
#
# Inputs (env):
#   DRIVER_MODE  One of WDDM, TCC, MCDM.

function Set-DriverMode {

    # Map matrix DRIVER_MODE to nvidia-smi -fdm code.
    # This assumes we have the prior knowledge on which GPU can use which mode.
    $driver_mode = $env:DRIVER_MODE
    if ($driver_mode -eq "WDDM") {
        Write-Output "Setting driver mode to WDDM..."
        nvidia-smi -fdm 0
    } elseif ($driver_mode -eq "TCC") {
        Write-Output "Setting driver mode to TCC..."
        nvidia-smi -fdm 1
    } elseif ($driver_mode -eq "MCDM") {
        Write-Output "Setting driver mode to MCDM..."
        nvidia-smi -fdm 2
    } else {
        Write-Output "Unknown driver mode: $driver_mode"
        exit 1
    }

    # Only restart NVIDIA display adapters, not other display devices (e.g. QEMU VGA)
    $nvidia_devices = Get-PnpDevice -Class Display -FriendlyName "NVIDIA*"
    foreach ($device in $nvidia_devices) {
        Write-Output "Restarting device: $($device.FriendlyName) ($($device.InstanceId))"
        pnputil /disable-device "$($device.InstanceId)"
        pnputil /enable-device "$($device.InstanceId)"
    }

    # Poll nvidia-smi until NVML can initialize, or give up after ~60s.
    # A fixed sleep is not enough on slower-coming-back-up multi-GPU rows
    # (e.g. 2x H100 MCDM) where pnputil enable returns before NVML is
    # ready. Pattern borrowed from the runner-team `nvgha-driver.ps1`.
    Write-Output "Waiting for nvidia-smi/NVML to come back up after device cycle..."
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Seconds 2
        & nvidia-smi.exe 2>&1 | Out-Null
    } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nvidia-smi did not return cleanly within 60s of the device cycle"
        exit 1
    }
}

# Run the functions
Set-DriverMode
