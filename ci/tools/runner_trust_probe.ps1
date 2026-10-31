# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.ps1 -- probe the Windows certificate store on a CI runner
# for the compromised eMudhra intermediates and their Sectigo cross-signs.

$ErrorActionPreference = 'Stop'

$EmudhraHelperPath = Join-Path $PSScriptRoot '..\..\.github\actions\upgrade-ca-certificates\emudhra_intermediates.ps1'
. $EmudhraHelperPath

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

# Asks certutil whether the local Disallowed CTL lists each compromised
# certificate's key. certutil prints "Certificate match found" and exits 0 on a
# match, and exits 0x80094005 when the CTL has no entry for the certificate.
function Record-CtlMatch([string]$Name) {
    $noMatchExitCode = [int]0x80094005
    $certs = @(Get-EmudhraIntermediates)
    $workDir = Join-Path ([System.IO.Path]::GetTempPath()) "emudhra-ctl-match-$PID"
    New-Item -ItemType Directory -Path $workDir -Force | Out-Null
    try {
        $matched = @(); $notMatched = @(); $unclear = @()
        for ($i = 0; $i -lt $certs.Count; $i++) {
            $certFile = Join-Path $workDir "cert$i.crt"
            [System.IO.File]::WriteAllBytes($certFile, $certs[$i].RawData)
            $output = @(& certutil.exe -verifyCTL Disallowed $workDir $certFile 2>&1 | ForEach-Object { "$_" })
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0 -and ($output -match 'match found')) {
                $matched += $certs[$i].EmudhraLabel
            } elseif ($exitCode -eq $noMatchExitCode) {
                $notMatched += $certs[$i].EmudhraLabel
            } else {
                $unclear += "$($certs[$i].EmudhraLabel) (exit $exitCode)"
                if ($unclear.Count -eq 1) {
                    Write-Host "::group::certutil -verifyCTL Disallowed for $($certs[$i].EmudhraLabel) (exit $exitCode)"
                    $output | ForEach-Object { Write-Host $_ }
                    Write-Host '::endgroup::'
                }
            }
        }
        $details = "$($matched.Count) of $($certs.Count) matched"
        if ($notMatched.Count -gt 0) { $details += "; no match: $($notMatched -join '; ')" }
        if ($unclear.Count -gt 0) { $details += "; unexpected result: $($unclear -join '; ')" }
        Record-Check $Name 'INFO' $details
    } finally {
        $certs | ForEach-Object { $_.Dispose() }
        Remove-Item -LiteralPath $workDir -Recurse -Force
    }
}

# Counts the compromised certificates that Windows reports as explicitly
# distrusted, in a fresh pwsh process so no chain state cached by this
# process carries over.
function Get-ExplicitlyDistrustedCountFresh {
    $command = ". '$EmudhraHelperPath'; " +
        '$certs = @(Get-EmudhraIntermediates); ' +
        'try { @($certs | Where-Object { Test-ExplicitlyDistrusted $_ }).Count } ' +
        'finally { $certs | ForEach-Object { $_.Dispose() } }'
    $output = @(& (Get-Process -Id $PID).Path -NoProfile -NonInteractive -Command $command 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0 -or $output.Count -ne 1 -or $output[0] -notmatch '^\d+$') {
        throw "Fresh distrust check failed (exit $LASTEXITCODE): $($output -join ' ')"
    }
    return [int]$output[0]
}

# Applies one CTL policy change, restarts the CryptSvc service that maintains
# the CTLs, and records the explicit-distrust count and certutil matches
# observed afterwards.
function Invoke-PolicyExperimentStep([string]$Label, [scriptblock]$Change) {
    & $Change
    Restart-Service -Name CryptSvc -Force
    $policy = "DisableRootAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'DisableRootAutoUpdate'), " +
        "EnableDisallowedCertAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'EnableDisallowedCertAutoUpdate')"
    $count = Get-ExplicitlyDistrustedCountFresh
    # Out-Host keeps the recorded rows out of this function's return value.
    Record-Check "Experiment: $Label" 'INFO' "$count of $EmudhraIntermediatesCount explicitly distrusted after CryptSvc restart ($policy)" | Out-Host
    Record-CtlMatch "Experiment: $Label, CTL matches" | Out-Host
    foreach ($endpoint in $TestEndpoints) {
        $outcome = Get-TlsOutcomeFresh $endpoint
        Record-Check "Experiment: $Label, Invoke-WebRequest ``$endpoint``" 'INFO' "$($outcome.Kind): $($outcome.Detail)" | Out-Host
    }
    return $count
}

# Connects to an endpoint with Invoke-WebRequest and classifies the result as
# Trusted, Rejected (certificate validation error), NameMismatchOnly (chain
# trusted, hostname check failed) or Other (non-certificate or network error).
function Get-TlsOutcome {
    param([string]$Endpoint)
    $progTmp = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        $res = Invoke-WebRequest -Uri $Endpoint -UseBasicParsing -SkipHttpErrorCheck -TimeoutSec 15
        return [pscustomobject]@{ Kind = 'Trusted'; Detail = "HTTP $($res.StatusCode)" }
    } catch {
        $kind = 'Other'
        $ex = $_.Exception
        while ($null -ne $ex) {
            if (
                ($ex -is [System.Security.Authentication.AuthenticationException] -and $ex.Message -match 'certificate chain|UntrustedRoot|RemoteCertificateChainErrors') -or
                ($ex -is [System.Net.WebException] -and $ex.Status -eq [System.Net.WebExceptionStatus]::TrustFailure)
            ) {
                $kind = 'Rejected'
                break
            }
            if ($ex -is [System.Security.Authentication.AuthenticationException] -and $ex.Message -match 'RemoteCertificateNameMismatch') {
                $kind = 'NameMismatchOnly'
            }
            $ex = $ex.InnerException
        }
        return [pscustomobject]@{ Kind = $kind; Detail = ($_.Exception.Message -replace '[\r\n]+', ' ') }
    } finally {
        $ProgressPreference = $progTmp
    }
}

# Runs Get-TlsOutcome in a fresh pwsh process so no TLS or chain state cached
# by this process carries over.
function Get-TlsOutcomeFresh([string]$Endpoint) {
    $command = "function Get-TlsOutcome { ${function:Get-TlsOutcome} }; " +
        "`$o = Get-TlsOutcome '$Endpoint'; `"`$(`$o.Kind)``t`$(`$o.Detail)`""
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($command))
    $output = @(& (Get-Process -Id $PID).Path -NoProfile -NonInteractive -EncodedCommand $encoded 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0 -or $output.Count -ne 1 -or $output[0] -notmatch "^(\w+)`t(.*)$") {
        throw "Fresh TLS check failed (exit $LASTEXITCODE): $($output -join ' ')"
    }
    return [pscustomobject]@{ Kind = $Matches[1]; Detail = $Matches[2] }
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
Invoke-Diagnostic 'Windows version' {
    $cv = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
    Record-Check 'Windows version' 'INFO' "$($cv.ProductName) $($cv.DisplayVersion), build $($cv.CurrentBuild).$($cv.UBR), runner environment $env:RUNNER_ENVIRONMENT"
}
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
Invoke-Diagnostic 'Local Disallowed CTL matches' { Record-CtlMatch 'Local Disallowed CTL matches' }

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
foreach ($endpoint in $TestEndpoints) {
    $checkName = "Invoke-WebRequest rejects ``$endpoint``"
    $outcome = Get-TlsOutcome $endpoint
    switch ($outcome.Kind) {
        'Trusted' {
            Record-Check $checkName 'FAIL' "Connection succeeded with $($outcome.Detail) (certificate was trusted)"
            $GatingFailures++
        }
        'Rejected' {
            Record-Check $checkName 'PASS' "Rejected with certificate validation error ($($outcome.Detail))"
        }
        'NameMismatchOnly' {
            Record-Check $checkName 'FAIL' "Certificate chain was trusted; failed only hostname check ($($outcome.Detail))"
            $GatingFailures++
        }
        default {
            Record-Check $checkName 'INCONCLUSIVE' "Non-certificate or network error ($($outcome.Detail))"
        }
    }
}

# 5. Disallowed CTL currently published on Windows Update (informational). -f
# forces a download; this runs after the gating checks so it cannot change
# what they observed.
Invoke-Diagnostic 'Windows Update Disallowed CTL' { Record-CertutilCtl 'Windows Update Disallowed CTL' @('-f', '-verifyCTL', 'DisallowedWU') }

# 6. CTL policy experiment (informational). Tests whether the root auto-update
# policy is why a Disallowed CTL that lists the compromised keys is not applied.
# It changes machine policy and restarts CryptSvc, so it runs last and only on
# ephemeral GitHub-hosted runners where DisableRootAutoUpdate=1.
if ($env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    Record-Check 'Experiment: CTL policy' 'INFO' "Skipped: runner environment is '$env:RUNNER_ENVIRONMENT', not github-hosted"
} elseif ((Get-RegistryValueText $CtlPolicyPath 'DisableRootAutoUpdate') -ne '1') {
    Record-Check 'Experiment: CTL policy' 'INFO' 'Skipped: DisableRootAutoUpdate is not 1'
} else {
    Invoke-Diagnostic 'Experiment: CTL policy' {
        $count = Invoke-PolicyExperimentStep 'EnableDisallowedCertAutoUpdate=1' {
            Set-ItemProperty -LiteralPath $CtlPolicyPath -Name 'EnableDisallowedCertAutoUpdate' -Type DWord -Value 1
        }
        if ($count -lt $EmudhraIntermediatesCount) {
            [void](Invoke-PolicyExperimentStep 'DisableRootAutoUpdate removed' {
                Remove-ItemProperty -LiteralPath $CtlPolicyPath -Name 'DisableRootAutoUpdate'
            })
        }
    }
}

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
