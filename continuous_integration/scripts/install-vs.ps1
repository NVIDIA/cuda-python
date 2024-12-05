# from https://github.com/rapidsai/devcontainers/blob/branch-25.02/windows/image/installers/install-vs.ps1

Param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('15', '16', '17')]
    [string]
    $msvcVersion,
    [Parameter(Mandatory=$false)]
    [string]
    $clVersion
)

$msvcPath = "C:\msbuild\$msvcVersion"

$vsComponentsMap = @{
    "15"     = "Microsoft.VisualStudio.Component.Windows10SDK.17763"
    "16"     = "Microsoft.VisualStudio.Component.Windows10SDK.20348"
    "17"     = "Microsoft.VisualStudio.Component.Windows10SDK.20348"
    "14.14"  = "Microsoft.VisualStudio.Component.VC.Tools.14.14"
    "14.15"  = "Microsoft.VisualStudio.Component.VC.Tools.14.15"
    "14.16"  = "Microsoft.VisualStudio.Component.VC.Tools.14.16"
    "14.27"  = "Microsoft.VisualStudio.Component.VC.14.27.x86.x64"
    "14.28"  = "Microsoft.VisualStudio.Component.VC.14.28.x86.x64"
    "14.29"  = "Microsoft.VisualStudio.Component.VC.14.29.x86.x64"
    "14.34"  = "Microsoft.VisualStudio.Component.VC.14.34.17.4.x86.x64"
    "14.35"  = "Microsoft.VisualStudio.Component.VC.14.35.17.5.x86.x64"
    "14.36"  = "Microsoft.VisualStudio.Component.VC.14.36.17.6.x86.x64"
    "14.37"  = "Microsoft.VisualStudio.Component.VC.14.37.17.7.x86.x64"
    "14.38"  = "Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64"
    "14.39"  = "Microsoft.VisualStudio.Component.VC.14.39.17.9.x86.x64"
    "latest" = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
}

# Always install/update core VC tools
$vsComponent = $vsComponentsMap[$msvcVersion]
$vsComponentString = "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add $vsComponent"

if ($clVersion) {
    $clComponent = $vsComponentsMap[$clVersion]
    $vsComponentString = "$vsComponentString --add $clComponent"
}

# 14.16 alone does not install build tools
if ($clVersion -eq "14.16") {
    $clComponent = $vsComponentsMap["14.15"]
    $vsComponentString = "$vsComponentString --add $clComponent"
}

Invoke-WebRequest -Uri "https://aka.ms/vs/$msvcVersion/release/vs_buildtools.exe" -UseBasicParsing -OutFile .\vs_buildtools.exe
Write-Output "Installing components: $vsComponentString"
Start-Process -NoNewWindow -PassThru -Wait -FilePath .\vs_buildtools.exe -ArgumentList "install --installWhileDownloading --installPath $msvcPath --wait --norestart --nocache --quiet $vsComponentString"

Remove-Item .\vs_buildtools.exe
