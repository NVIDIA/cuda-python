# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.ps1 -- probe the Windows certificate store on a CI runner
# for the compromised eMudhra intermediates and their Sectigo cross-signs.

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\..\.github\actions\upgrade-ca-certificates\emudhra_intermediates.ps1')

$EmSignG3Hex = '86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B'
# One host per path: via emSign ECC Root CA - G3, and via the Sectigo
# cross-sign to AAA Certificate Services.
$TestEndpoints = @(
    'https://dv-g3b-emudhra.we-have-a-problem.com/',
    'https://dv-g3b-aaa-emudhra.we-have-a-problem.com/'
)
$ProbeLabel = 'Windows OS trust store'
$SummaryFile = $env:GITHUB_STEP_SUMMARY

function Append-Summary([string]$Line) {
    if ($SummaryFile) {
        Add-Content -Path $SummaryFile -Value $Line -Encoding utf8
    }
}

function Record-Check([string]$Name, [string]$Status, [string]$Details) {
    Write-Output "[$Status] ${Name}: $Details"
    Append-Summary "| $Name | **$Status** | $Details |"
}

$GatingFailures = 0

Append-Summary "### $ProbeLabel"
Append-Summary ""
Append-Summary "| Check | Status | Details |"
Append-Summary "| --- | --- | --- |"

# 1. Root anchor state (informational). Microsoft's fix blocks the
# intermediates and leaves the root in place, so this is not a pass criterion.
$rootCerts = @(Get-ChildItem -Path 'Cert:\LocalMachine\Root' -ErrorAction Stop)
$inRoot = [bool]($rootCerts | Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })
$inDisallowed = [bool](Get-ChildItem -Path 'Cert:\LocalMachine\Disallowed' -ErrorAction SilentlyContinue |
    Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })
Record-Check 'emSign ECC Root CA - G3 store state' 'INFO' "LocalMachine\Root=$inRoot ($($rootCerts.Count) certs), LocalMachine\Disallowed=$inDisallowed"

# 2. All compromised intermediates and cross-signs explicitly distrusted (gating)
$certs = @(Get-EmudhraIntermediates)
try {
    $notDistrusted = @($certs | Where-Object { -not (Test-ExplicitlyDistrusted $_) } | ForEach-Object { $_.EmudhraLabel })
} finally {
    $certs | ForEach-Object { $_.Dispose() }
}
if ($notDistrusted.Count -eq 0) {
    Record-Check 'Compromised intermediates distrusted' 'PASS' "All $($certs.Count) explicitly distrusted"
} else {
    Record-Check 'Compromised intermediates distrusted' 'FAIL' "$($notDistrusted.Count) of $($certs.Count) not explicitly distrusted: $($notDistrusted -join '; ')"
    $GatingFailures++
}

# 3. OS-store client (Invoke-WebRequest) rejects each test endpoint (only a successful connection fails the job)
$progTmp = $ProgressPreference
$ProgressPreference = 'SilentlyContinue'
try {
    foreach ($endpoint in $TestEndpoints) {
        $checkName = "Invoke-WebRequest rejects ``$endpoint``"
        try {
            $res = Invoke-WebRequest -Uri $endpoint -UseBasicParsing -SkipHttpErrorCheck -TimeoutSec 15
            Record-Check $checkName 'FAIL' "Connection succeeded with HTTP $($res.StatusCode) (certificate was trusted)"
            $GatingFailures++
        } catch {
            $certRejected = $false
            $hostnameMismatchOnly = $false
            $ex = $_.Exception
            while ($null -ne $ex) {
                if (
                    ($ex -is [System.Security.Authentication.AuthenticationException] -and $ex.Message -match 'certificate chain|UntrustedRoot|RemoteCertificateChainErrors') -or
                    ($ex -is [System.Net.WebException] -and $ex.Status -eq [System.Net.WebExceptionStatus]::TrustFailure)
                ) {
                    $certRejected = $true
                    break
                }
                if ($ex -is [System.Security.Authentication.AuthenticationException] -and $ex.Message -match 'RemoteCertificateNameMismatch') {
                    $hostnameMismatchOnly = $true
                }
                $ex = $ex.InnerException
            }
            $cleanMsg = ($_.Exception.Message -replace '[\r\n]+', ' ' -replace '\|', '\|')
            if ($certRejected) {
                Record-Check $checkName 'PASS' "Rejected with certificate validation error ($cleanMsg)"
            } elseif ($hostnameMismatchOnly) {
                Record-Check $checkName 'FAIL' "Certificate chain was trusted; failed only hostname check ($cleanMsg)"
                $GatingFailures++
            } else {
                Record-Check $checkName 'INCONCLUSIVE' "Non-certificate or network error ($cleanMsg)"
            }
        }
    }
} finally {
    $ProgressPreference = $progTmp
}

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
