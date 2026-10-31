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
# Disallowed CTL sequence number of Microsoft's 2026-10-20 out-of-band update
# that blocks the compromised eMudhra keys (Microsoft Trusted Root Program).
$EmudhraDisallowedCtlSequence = [UInt64]::Parse('01dd5f6ca45c9d86', [System.Globalization.NumberStyles]::HexNumber)
$CtlPolicyPath = 'HKLM:\SOFTWARE\Policies\Microsoft\SystemCertificates\AuthRoot'
$CtlAutoUpdatePath = 'HKLM:\SOFTWARE\Microsoft\SystemCertificates\AuthRoot\AutoUpdate'

function Append-Summary([string]$Line) {
    if ($SummaryFile) {
        Add-Content -Path $SummaryFile -Value $Line -Encoding utf8
    }
}

function Format-Details([string]$Text) {
    return ($Text -replace '[\r\n]+', ' ' -replace '\|', '\|')
}

function Record-Check([string]$Name, [string]$Status, [string]$Details) {
    Write-Output "[$Status] ${Name}: $Details"
    Append-Summary "| $Name | **$Status** | $(Format-Details $Details) |"
}

function Get-RegistryValueText([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 'key absent'
    }
    $value = (Get-Item -LiteralPath $Path).GetValue($Name)
    if ($null -eq $value) {
        return 'unset'
    }
    if ($value -is [byte[]]) {
        if ($value.Length -eq 8) {
            return [DateTime]::FromFileTimeUtc([BitConverter]::ToInt64($value, 0)).ToString('u')
        }
        return "$($value.Length) bytes"
    }
    return "$value"
}

# Runs a best-effort diagnostic. An exception becomes an INCONCLUSIVE row
# instead of ending the script before the gating checks and their exit code.
function Invoke-Diagnostic([string]$Name, [scriptblock]$Body) {
    try {
        & $Body
    } catch {
        Record-Check $Name 'INCONCLUSIVE' "Diagnostic failed: $($_.Exception.Message)"
    }
}

# Runs certutil -verifyCTL, logs its full output in a collapsed group, and
# records only the SequenceNumber line as an informational check. The sequence
# number is reported even when certutil fails: Windows versions without ML-DSA
# support exit non-zero (NTE_BAD_ALGID) on CTLs that carry post-quantum pilot
# roots.
function Record-CertutilCtl([string]$Name, [string[]]$CertutilArgs) {
    $output = @(& certutil.exe @CertutilArgs 2>&1 | ForEach-Object { "$_" })
    $exitCode = $LASTEXITCODE
    Write-Host "::group::certutil $($CertutilArgs -join ' ') (exit $exitCode)"
    $output | ForEach-Object { Write-Host $_ }
    Write-Host '::endgroup::'
    $sequenceLine = $output | Where-Object { $_ -match 'SequenceNumber' } | Select-Object -First 1
    $details = if ($sequenceLine) { $sequenceLine.Trim() } else { 'no SequenceNumber line in output' }
    if ($sequenceLine -match '\b([0-9a-fA-F]{16})\b') {
        $sequence = [UInt64]::Parse($Matches[1], [System.Globalization.NumberStyles]::HexNumber)
        $relation = if ($sequence -ge $EmudhraDisallowedCtlSequence) { 'includes' } else { 'predates' }
        $details += " ($relation the eMudhra update 01dd5f6ca45c9d86)"
    }
    if ($exitCode -ne 0) {
        $reason = "certutil $($CertutilArgs -join ' ') exited with $exitCode"
        if (($output -join "`n") -match 'NTE_BAD_ALGID|0x80090008') {
            $reason += ' (NTE_BAD_ALGID: this Windows version likely cannot process the post-quantum pilot roots in the CTL)'
        }
        Record-Check $Name 'INCONCLUSIVE' "$reason; $details"
        return
    }
    Record-Check $Name 'INFO' $details
}

$GatingFailures = 0

Append-Summary "### $ProbeLabel"
Append-Summary ""
Append-Summary "| Check | Status | Details |"
Append-Summary "| --- | --- | --- |"

# 1. Root anchor state (informational). Microsoft's October 26 out-of-band
# update removes this root, with the other eMudhra G3 roots, from the Windows
# trust list through the AuthRoot CTL. It is therefore expected to be absent
# once that update arrives, and is not a pass criterion; the explicit block on
# the intermediates (check 3) is.
$rootCerts = @(Get-ChildItem -Path 'Cert:\LocalMachine\Root' -ErrorAction Stop)
$inRoot = [bool]($rootCerts | Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })
$inDisallowed = [bool](Get-ChildItem -Path 'Cert:\LocalMachine\Disallowed' -ErrorAction SilentlyContinue |
    Where-Object { $_.GetCertHashString('SHA256') -eq $EmSignG3Hex })
Record-Check 'emSign ECC Root CA - G3 store state' 'INFO' "LocalMachine\Root=$inRoot ($($rootCerts.Count) certs), LocalMachine\Disallowed=$inDisallowed"

# 2. CTL update state (informational). Microsoft delivers its eMudhra block
# through the Disallowed CTL, which reaches a machine only through automatic
# CTL updates or an explicit import.
Invoke-Diagnostic 'CTL auto-update policy' {
    Record-Check 'CTL auto-update policy' 'INFO' ("DisableRootAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'DisableRootAutoUpdate'), " +
        "EnableDisallowedCertAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'EnableDisallowedCertAutoUpdate')")
}
Invoke-Diagnostic 'CTL auto-update registry state' {
    Record-Check 'CTL auto-update registry state' 'INFO' ("LastSyncTime=$(Get-RegistryValueText $CtlAutoUpdatePath 'LastSyncTime'), " +
        "DisallowedCertLastSyncTime=$(Get-RegistryValueText $CtlAutoUpdatePath 'DisallowedCertLastSyncTime'), " +
        "DisallowedCertEncodedCtl=$(Get-RegistryValueText $CtlAutoUpdatePath 'DisallowedCertEncodedCtl')")
}
Invoke-Diagnostic 'Local Disallowed CTL' { Record-CertutilCtl 'Local Disallowed CTL' @('-verifyCTL', 'Disallowed') }

# 3. All compromised intermediates and cross-signs explicitly distrusted (gating)
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

# 4. OS-store client (Invoke-WebRequest) rejects each test endpoint (only a successful connection fails the job)
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
            $cleanMsg = ($_.Exception.Message -replace '[\r\n]+', ' ')
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

# 5. Disallowed CTL currently published on Windows Update (informational). -f
# forces a download; this runs after the gating checks so it cannot change
# what they observed.
Invoke-Diagnostic 'Windows Update Disallowed CTL' { Record-CertutilCtl 'Windows Update Disallowed CTL' @('-f', '-verifyCTL', 'DisallowedWU') }

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
