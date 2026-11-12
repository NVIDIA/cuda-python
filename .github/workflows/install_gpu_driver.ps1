# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Install the driver
function Install-Driver {

    # Set the correct URL, filename, and arguments to the installer
    # This driver is picked to support Windows 11 & CUDA 12.8
    $url = 'https://us.download.nvidia.com/tesla/572.13/572.13-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe';
    $file_dir = 'C:\NVIDIA-Driver\572.13-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe';
    $install_args = '/s /noeula /noreboot';
    # sha256 of the installer above. Whoever changes $url owns this value:
    # compute it on a machine with an up-to-date trust store (for example
    # `curl -fsSL <url> | sha256sum`) and treat a CI mismatch as possible
    # tampering. Never copy the "got" value from a job log into this file.
    $expected_sha256 = 'e6663c8ca3685da149d357b03b89e0946d862a41788492a4c06f34744a2433a5';

    # Create the folder for the driver download
    if (!(Test-Path -Path 'C:\NVIDIA-Driver')) {
        New-Item -Path 'C:\' -Name 'NVIDIA-Driver' -ItemType 'directory' | Out-Null
    }

    # Download the file to a specified directory
    # Disabling progress bar due to https://github.com/GoogleCloudPlatform/compute-gpu-installation/issues/29
    $ProgressPreference_tmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Write-Output 'Downloading the driver installer...'
    Invoke-WebRequest $url -OutFile $file_dir
    $ProgressPreference = $ProgressPreference_tmp
    Write-Output 'Download complete!'

    # Verify the installer against the pinned sha256 before running it as admin.
    $actual_sha256 = (Get-FileHash -Algorithm SHA256 -Path $file_dir).Hash.ToLower()
    if ($actual_sha256 -ne $expected_sha256) {
        Write-Error "Driver installer sha256 mismatch: expected $expected_sha256, got $actual_sha256"
        exit 1
    }
    Write-Output "Driver installer sha256 verified: $actual_sha256"

    # Install the file with the specified path from earlier
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $file_dir -ArgumentList $install_args -Wait
    Write-Output 'Done!'
}

# Run the functions
Install-Driver
