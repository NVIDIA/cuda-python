# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [UInt64]$ReserveBytes
)

$ErrorActionPreference = "Stop"

if (-not $env:RUNNER_TEMP) {
    throw "RUNNER_TEMP is required"
}
$path = [IO.Path]::GetFullPath($Executable)
$installRoot = [IO.Path]::GetFullPath((Join-Path $env:RUNNER_TEMP "setup-hardened-python/interpreters"))
if (-not $path.StartsWith($installRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to modify Python outside $installRoot`: $path"
}

$image = [IO.File]::ReadAllBytes($path)
if ($image.Length -lt 64 -or $image[0] -ne 0x4d -or $image[1] -ne 0x5a) {
    throw "Not a PE executable: $path"
}
$peOffset = [BitConverter]::ToInt32($image, 0x3c)
if ($peOffset -lt 64 -or $peOffset -gt $image.Length - 24 -or
    $image[$peOffset] -ne 0x50 -or $image[$peOffset + 1] -ne 0x45 -or
    $image[$peOffset + 2] -ne 0 -or $image[$peOffset + 3] -ne 0) {
    throw "Invalid PE header: $path"
}
$optionalOffset = $peOffset + 24
$optionalSize = [BitConverter]::ToUInt16($image, $peOffset + 20)
if ($optionalSize -lt 2 -or $optionalOffset + $optionalSize -gt $image.Length) {
    throw "Truncated PE optional header: $path"
}
$magic = [BitConverter]::ToUInt16($image, $optionalOffset)
if ($magic -eq 0x20b) {
    $width = 8
} elseif ($magic -eq 0x10b) {
    $width = 4
    if ($ReserveBytes -gt [UInt32]::MaxValue) {
        throw "Stack reserve is too large for a 32-bit PE executable"
    }
} else {
    throw "Unsupported PE optional header: $magic"
}
if ($optionalSize -lt 72 + $width) {
    throw "PE optional header does not contain a stack reserve: $path"
}

$reserveOffset = $optionalOffset + 72
$current = if ($width -eq 8) {
    [BitConverter]::ToUInt64($image, $reserveOffset)
} else {
    [BitConverter]::ToUInt32($image, $reserveOffset)
}
if ($current -lt $ReserveBytes) {
    $replacement = if ($width -eq 8) {
        [BitConverter]::GetBytes($ReserveBytes)
    } else {
        [BitConverter]::GetBytes([UInt32]$ReserveBytes)
    }
    $stream = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $stream.Position = $reserveOffset
        $stream.Write($replacement, 0, $width)
        $stream.Flush()
    } finally {
        $stream.Dispose()
    }
}

$updated = [IO.File]::ReadAllBytes($path)
$actual = if ($width -eq 8) {
    [BitConverter]::ToUInt64($updated, $reserveOffset)
} else {
    [BitConverter]::ToUInt32($updated, $reserveOffset)
}
if ($actual -lt $ReserveBytes) {
    throw "Stack reserve verification failed: $actual < $ReserveBytes"
}
Write-Output "[stack-reserve] $path`: $current -> $actual bytes"
