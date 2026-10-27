# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.ps1 -- probe the Windows certificate store on a CI runner
# for the compromised eMudhra/emSign root anchor (emSign ECC Root CA - G3).
#
# Optional environment overrides for local testing:
#   TRUST_PROBE_TEST_ENDPOINT   HTTPS endpoint to test (default: https://dv-g3b-emudhra.we-have-a-problem.com/)
#   TRUST_PROBE_LABEL           Heading label for GITHUB_STEP_SUMMARY

$ErrorActionPreference = 'Stop'

$EmSignG3Sha256 = '86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B'
$EmSignG3Hex = $EmSignG3Sha256.Replace(':', '')
$TestEndpoint = if ($null -ne $env:TRUST_PROBE_TEST_ENDPOINT) { $env:TRUST_PROBE_TEST_ENDPOINT } else { 'https://dv-g3b-emudhra.we-have-a-problem.com/' }
$ProbeLabel = if ($env:TRUST_PROBE_LABEL) { $env:TRUST_PROBE_LABEL } else { 'Windows OS trust store' }
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

# 1. Root anchor not trusted in Cert:\LocalMachine\Root (or present in Disallowed) (gating)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $rootCerts = @(Get-ChildItem -Path 'Cert:\LocalMachine\Root' -ErrorAction Stop)
    $inRoot = $false
    foreach ($cert in $rootCerts) {
        $hashBytes = $sha256.ComputeHash($cert.RawData)
        $hex = [System.BitConverter]::ToString($hashBytes).Replace('-', '').ToUpperInvariant()
        if ($hex -eq $EmSignG3Hex) {
            $inRoot = $true
            break
        }
    }

    $inDisallowed = $false
    if (Test-Path -Path 'Cert:\LocalMachine\Disallowed') {
        $disallowedCerts = @(Get-ChildItem -Path 'Cert:\LocalMachine\Disallowed' -ErrorAction SilentlyContinue)
        foreach ($cert in $disallowedCerts) {
            $hashBytes = $sha256.ComputeHash($cert.RawData)
            $hex = [System.BitConverter]::ToString($hashBytes).Replace('-', '').ToUpperInvariant()
            if ($hex -eq $EmSignG3Hex) {
                $inDisallowed = $true
                break
            }
        }
    }

    $storeSummary = "LocalMachine\Root=$inRoot ($($rootCerts.Count) certs), LocalMachine\Disallowed=$inDisallowed"
    if ((-not $inRoot) -or $inDisallowed) {
        Record-Check 'Root anchor not trusted' 'PASS' $storeSummary
    } else {
        Record-Check 'Root anchor not trusted' 'FAIL' "Present in LocalMachine\Root and not in Disallowed ($storeSummary)"
        $GatingFailures++
    }
} finally {
    $sha256.Dispose()
}

# 2. OS-store client (Invoke-WebRequest) rejects the test endpoint (only a successful connection fails the job)
if ([string]::IsNullOrEmpty($TestEndpoint)) {
    Record-Check 'Invoke-WebRequest rejects test endpoint' 'SKIP' 'Skipped via empty `TRUST_PROBE_TEST_ENDPOINT`'
} else {
    $progTmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        $null = Invoke-WebRequest -Uri $TestEndpoint -UseBasicParsing -TimeoutSec 15
        Record-Check 'Invoke-WebRequest rejects test endpoint' 'FAIL' "Connection to ``$TestEndpoint`` succeeded (certificate was trusted)"
        $GatingFailures++
    } catch {
        $certRejected = $false
        $ex = $_.Exception
        while ($null -ne $ex) {
            if (
                ($ex -is [System.Security.Authentication.AuthenticationException] -and $ex.Message -match 'remote certificate|certificate chain|UntrustedRoot|RemoteCertificateChainErrors') -or
                ($ex -is [System.Net.WebException] -and $ex.Status -eq [System.Net.WebExceptionStatus]::TrustFailure)
            ) {
                $certRejected = $true
                break
            }
            $ex = $ex.InnerException
        }
        $cleanMsg = ($_.Exception.Message -replace '[\r\n]+', ' ' -replace '\|', '\|')
        if ($certRejected) {
            Record-Check 'Invoke-WebRequest rejects test endpoint' 'PASS' "Rejected with certificate validation error ($cleanMsg)"
        } else {
            Record-Check 'Invoke-WebRequest rejects test endpoint' 'INCONCLUSIVE' "Non-certificate or network error ($cleanMsg)"
        }
    } finally {
        $ProgressPreference = $progTmp
    }
}

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
