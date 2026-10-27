# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.ps1 -- probe the Windows certificate store on a CI runner
# for the compromised eMudhra/emSign root anchor (emSign ECC Root CA - G3).

$ErrorActionPreference = 'Stop'

$EmSignG3Hex = '86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B'
$TestEndpoint = 'https://dv-g3b-emudhra.we-have-a-problem.com/'
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

# 1. Root anchor not trusted in Cert:\LocalMachine\Root (or present in Disallowed) (gating)
$rootCerts = @(Get-ChildItem -Path 'Cert:\LocalMachine\Root' -ErrorAction Stop)
$inRoot = [bool]($rootCerts | Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })
$inDisallowed = [bool](Get-ChildItem -Path 'Cert:\LocalMachine\Disallowed' -ErrorAction SilentlyContinue |
    Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })

$storeSummary = "LocalMachine\Root=$inRoot ($($rootCerts.Count) certs), LocalMachine\Disallowed=$inDisallowed"
if ((-not $inRoot) -or $inDisallowed) {
    Record-Check 'Root anchor not trusted' 'PASS' $storeSummary
} else {
    Record-Check 'Root anchor not trusted' 'FAIL' "Present in LocalMachine\Root and not in Disallowed ($storeSummary)"
    $GatingFailures++
}

# 2. OS-store client (Invoke-WebRequest) rejects the test endpoint (only a successful connection fails the job)
$progTmp = $ProgressPreference
$ProgressPreference = 'SilentlyContinue'
try {
    $null = Invoke-WebRequest -Uri $TestEndpoint -UseBasicParsing -TimeoutSec 15
    Record-Check 'Invoke-WebRequest rejects test endpoint' 'FAIL' "Connection to ``$TestEndpoint`` succeeded (certificate was trusted)"
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
        Record-Check 'Invoke-WebRequest rejects test endpoint' 'PASS' "Rejected with certificate validation error ($cleanMsg)"
    } elseif ($hostnameMismatchOnly) {
        Record-Check 'Invoke-WebRequest rejects test endpoint' 'FAIL' "Certificate chain was trusted; failed only hostname check ($cleanMsg)"
        $GatingFailures++
    } else {
        Record-Check 'Invoke-WebRequest rejects test endpoint' 'INCONCLUSIVE' "Non-certificate or network error ($cleanMsg)"
    }
} finally {
    $ProgressPreference = $progTmp
}

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
