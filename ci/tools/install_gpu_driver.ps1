# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# install_gpu_driver.ps1 -- install a specific NVIDIA driver version on a
# Windows CI runner. Driver-mode selection and the post-install device
# power-cycle are the responsibility of configure_driver_mode.ps1, which
# the workflow runs immediately after this script (or by itself when
# DRIVER is 'latest'/'earliest' and the runner already brings up the
# right driver).
#
# Inputs (env):
#   DRIVER    Driver version, e.g. "610.47". Must NOT be 'latest' or
#             'earliest' -- those are runner-pre-installed and the
#             workflow is expected to skip this script for them.
#   GPU_TYPE  Lower-case GPU label from the matrix (e.g. "l4", "rtx4090").
#             Selects the data-center vs desktop installer variant.

# Install the driver
function Install-Driver {

    # Driver version is plumbed from the matrix via the DRIVER env var.
    $version = $env:DRIVER
    if (-not $version -or $version -eq 'latest' -or $version -eq 'earliest') {
        Write-Error "DRIVER env var must be a specific version string (e.g. '610.47'); got '$version'."
        exit 1
    }

    # Get GPU type from environment variable
    $gpu_type = $env:GPU_TYPE

    $data_center_gpus = @('a100', 'h100', 'l4', 't4', 'v100', 'rtxa6000', 'rtx6000ada')
    $desktop_gpus = @('rtx2080', 'rtx4090', 'rtxpro6000')

    if ($data_center_gpus -contains $gpu_type) {
        Write-Output "Data center GPU detected: $gpu_type"
        $filename="$version-data-center-tesla-desktop-winserver-2022-2025-dch-international.exe"
        $server_path="tesla/$version"
    } elseif ($desktop_gpus -contains $gpu_type) {
        Write-Output "Desktop GPU detected: $gpu_type"
        $filename="$version-desktop-win10-win11-64bit-international-dch-whql.exe"
        $server_path="Windows/$version"
    } else {
        Write-Output "Unknown GPU type: $gpu_type"
        exit 1
    }

    $url="https://us.download.nvidia.com/$server_path/$filename"
    $filepath="C:\NVIDIA-Driver\$filename"

    Write-Output "Installing NVIDIA driver version $version for GPU type $gpu_type"
    Write-Output "Download URL: $url"

    # Silent install arguments
    $install_args = '/s /noeula /noreboot';

    # Create the folder for the driver download
    if (!(Test-Path -Path 'C:\NVIDIA-Driver')) {
        New-Item -Path 'C:\' -Name 'NVIDIA-Driver' -ItemType 'directory' | Out-Null
    }

    # Download the file to a specified directory
    # Disabling progress bar due to https://github.com/GoogleCloudPlatform/compute-gpu-installation/issues/29
    $ProgressPreference_tmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Write-Output 'Downloading the driver installer...'
    Invoke-WebRequest $url -OutFile $filepath
    $ProgressPreference = $ProgressPreference_tmp
    Write-Output 'Download complete!'

    # Install the file with the specified path from earlier
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $filepath -ArgumentList $install_args -Wait
    Write-Output 'Install complete; driver mode + device cycle handled by configure_driver_mode.ps1.'
}

# Run the functions
Install-Driver
