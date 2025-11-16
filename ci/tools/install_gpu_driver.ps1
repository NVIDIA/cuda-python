# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Detect GPU type from JOB_RUNNER environment variable
function Get-GPUType {
    param(
        [string]$JobRunner = $env:JOB_RUNNER
    )
    
    if ([string]::IsNullOrEmpty($JobRunner)) {
        Write-Output "Warning: JOB_RUNNER environment variable not set. Using default GPU detection."
        return "unknown"
    }
    
    # Extract GPU type from runner label (e.g., "windows-amd64-gpu-l4-latest-1")
    if ($JobRunner -match "gpu-([^-]+)") {
        $gpuType = $matches[1].ToLower()
        Write-Output "Detected GPU type: $gpuType"
        return $gpuType
    }
    
    Write-Output "Warning: Could not parse GPU type from JOB_RUNNER: $JobRunner"
    return "unknown"
}

# Determine if GPU is a data center GPU
function Is-DataCenterGPU {
    param(
        [string]$GpuType
    )
    
    $dataCenterGPUs = @("l4", "a100", "t4", "h100", "a10", "a30", "a40")
    return $dataCenterGPUs -contains $GpuType
}

# Get driver URL and filename based on GPU type
function Get-DriverInfo {
    param(
        [string]$GpuType,
        [string]$DriverMode = $env:DRIVER_MODE
    )
    
    $isDataCenter = Is-DataCenterGPU -GpuType $GpuType
    
    # Default driver version that supports Windows 11 & CUDA 13.0
    $driverVersion = "580.88"
    
    if ($isDataCenter) {
        # Data center GPU - use Tesla driver
        $filename = "$driverVersion-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe"
        $url = "https://us.download.nvidia.com/tesla/$driverVersion/$filename"
    } else {
        # Desktop GPU - use GeForce/Quadro driver
        $filename = "$driverVersion-desktop-win10-win11-64bit-international-dch-whql.exe"
        $url = "https://us.download.nvidia.com/Windows/$driverVersion/$filename"
    }
    
    return @{
        Url = $url
        Filename = $filename
        IsDataCenter = $isDataCenter
    }
}

# Set driver mode using nvidia-smi
function Set-DriverMode {
    param(
        [string]$DriverMode,
        [bool]$IsDataCenter
    )
    
    if ([string]::IsNullOrEmpty($DriverMode)) {
        Write-Output "No driver mode specified, skipping mode configuration"
        return
    }
    
    $DriverMode = $DriverMode.ToUpper()
    Write-Output "Configuring driver mode: $DriverMode"
    
    if (-not $IsDataCenter) {
        if ($DriverMode -ne "WDDM") {
            Write-Output "Warning: Desktop GPUs only support WDDM mode. Requested mode '$DriverMode' will be ignored."
        }
        # Desktop GPUs are always in WDDM mode, no configuration needed
        return
    }
    
    # Data center GPUs support TCC and MCDM (not WDDM)
    if ($DriverMode -eq "WDDM") {
        Write-Output "Warning: Data center GPUs do not support WDDM mode. Skipping mode configuration."
        return
    }
    
    try {
        # Check current mode
        $currentMode = & nvidia-smi -q | Select-String -Pattern "Driver Mode" | Out-String
        Write-Output "Current driver mode: $currentMode"
        
        if ($DriverMode -eq "TCC") {
            # Set TCC mode (nvidia-smi -fdm 0 sets TCC mode)
            Write-Output "Setting TCC mode..."
            & nvidia-smi -fdm 0
            
            # Verify mode was set
            Write-Output "Resetting display device..."
            # Reset display devices to apply the change
            $devcon = "C:\Windows\System32\pnputil.exe"
            if (Test-Path $devcon) {
                & $devcon /restart-device "PCI\VEN_10DE*"
            }
        } elseif ($DriverMode -eq "MCDM") {
            # Set MCDM mode (nvidia-smi -fdm 2 sets MCDM mode)
            Write-Output "Setting MCDM mode..."
            & nvidia-smi -fdm 2
            
            # Verify mode was set
            Write-Output "Resetting display device..."
            # Reset display devices to apply the change
            $devcon = "C:\Windows\System32\pnputil.exe"
            if (Test-Path $devcon) {
                & $devcon /restart-device "PCI\VEN_10DE*"
            }
        }
        
        # Wait for device reset
        Start-Sleep -Seconds 5
        
        # Verify new mode
        $newMode = & nvidia-smi -q | Select-String -Pattern "Driver Mode" | Out-String
        Write-Output "New driver mode: $newMode"
    } catch {
        Write-Output "Warning: Failed to set driver mode: $_"
    }
}

# Install the driver
function Install-Driver {
    param(
        [string]$GpuType = (Get-GPUType),
        [string]$DriverMode = $env:DRIVER_MODE
    )
    
    Write-Output "Installing GPU driver for GPU type: $GpuType"
    
    # Get driver information
    $driverInfo = Get-DriverInfo -GpuType $GpuType -DriverMode $DriverMode
    $url = $driverInfo.Url
    $filename = $driverInfo.Filename
    $isDataCenter = $driverInfo.IsDataCenter
    
    Write-Output "Driver URL: $url"
    Write-Output "Is Data Center GPU: $isDataCenter"
    
    $file_dir = "C:\NVIDIA-Driver\$filename"
    $install_args = '/s /noeula /noreboot'

    # Create the folder for the driver download
    if (!(Test-Path -Path 'C:\NVIDIA-Driver')) {
        New-Item -Path 'C:\' -Name 'NVIDIA-Driver' -ItemType 'directory' | Out-Null
    }

    # Download the file to a specified directory
    # Disabling progress bar due to https://github.com/GoogleCloudPlatform/compute-gpu-installation/issues/29
    $ProgressPreference_tmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Write-Output 'Downloading the driver installer...'
    try {
        Invoke-WebRequest $url -OutFile $file_dir
        $ProgressPreference = $ProgressPreference_tmp
        Write-Output 'Download complete!'
    } catch {
        $ProgressPreference = $ProgressPreference_tmp
        Write-Output "Error downloading driver: $_"
        Write-Output "Falling back to default driver..."
        # Fall back to the original hardcoded driver if download fails
        $url = 'https://us.download.nvidia.com/tesla/580.88/580.88-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe'
        $file_dir = 'C:\NVIDIA-Driver\580.88-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe'
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest $url -OutFile $file_dir
        $ProgressPreference = $ProgressPreference_tmp
        Write-Output 'Fallback download complete!'
    }

    # Install the file with the specified path from earlier
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $file_dir -ArgumentList $install_args -Wait
    Write-Output 'Driver installation complete!'
    
    # Set driver mode if specified
    Set-DriverMode -DriverMode $DriverMode -IsDataCenter $isDataCenter
}

# Run the functions
Install-Driver
