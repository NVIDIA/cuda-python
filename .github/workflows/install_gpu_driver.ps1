#Requires -RunAsAdministrator

# Install the driver
function Install-Driver {

    # Set the correct URL, filename, and arguments to the installer
    $url = 'https://us.download.nvidia.com/tesla/539.19/539.19-data-center-tesla-desktop-winserver-2019-2022-dch-international.exe';
    $file_dir = 'C:\NVIDIA-Driver\539.19-data-center-tesla-desktop-winserver-2019-2022-dch-international.exe';
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

    # Install the file with the specified path from earlier as well as the RunAs admin option
    Write-Output 'Running the driver installer...'
    Start-Process -FilePath $file_dir -ArgumentList $install_args -Wait
    Write-Output 'Done!'
}

# Run the functions
Install-Driver
