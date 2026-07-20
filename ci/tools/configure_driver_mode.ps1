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

    # Only restart NVIDIA display adapters, not other display devices (e.g. QEMU VGA).
    $nvidia_devices = @(Get-PnpDevice -Class Display -FriendlyName "NVIDIA*")
    $gpu_count = $nvidia_devices.Count
    foreach ($device in $nvidia_devices) {
        Write-Output "Restarting device: $($device.FriendlyName) ($($device.InstanceId))"
        pnputil /disable-device "$($device.InstanceId)"
        pnputil /enable-device "$($device.InstanceId)"
    }

    # Initial settle after the device cycle.
    Start-Sleep -Seconds 5

    # Poll nvidia-smi for N consecutive successes (N == cycled GPUs)
    # so a mid-init "ok" flap doesn't fool the loop; bail after ~60s.
    Write-Output "Waiting for nvidia-smi/NVML to come back up after device cycle..."
    $deadline = (Get-Date).AddSeconds(60)
    $consecutive_ok = 0
    do {
        Start-Sleep -Seconds 3
        & nvidia-smi.exe 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $consecutive_ok++ } else { $consecutive_ok = 0 }
    } while ($consecutive_ok -lt $gpu_count -and (Get-Date) -lt $deadline)
    if ($consecutive_ok -lt $gpu_count) {
        Write-Error "nvidia-smi did not return cleanly $gpu_count times in a row within 60s of the device cycle"
        exit 1
    }
}

# Run the functions
Set-DriverMode
