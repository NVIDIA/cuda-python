# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Install the driver
function Install-Driver {

    # Set the correct URL, filename, and arguments to the installer
    # This driver is picked to support Windows 11 & CUDA 12.8
    $url = 'https://us.download.nvidia.com/tesla/572.13/572.13-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe';
    $file_dir = 'C:\NVIDIA-Driver\572.13-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe';
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
    Invoke-WebRequest $url -OutFile $file_dir
    $ProgressPreference = $ProgressPreference_tmp
    Write-Output 'Download complete!'

    # Install the file with the specified path from earlier
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $file_dir -ArgumentList $install_args -Wait
    Write-Output 'Done!'
}

# Run the functions
Install-Driver
