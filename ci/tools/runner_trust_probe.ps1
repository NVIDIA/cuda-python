# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.ps1 -- probe the Windows certificate store on a CI runner
# for the compromised eMudhra intermediates and their Sectigo cross-signs.

$ErrorActionPreference = 'Stop'

$EmudhraHelperPath = Join-Path $PSScriptRoot '..\..\.github\actions\upgrade-ca-certificates\emudhra_intermediates.ps1'
. $EmudhraHelperPath
# SHA-256 (upper-case hex) -> label of each certificate in the impacted list.
$EmudhraListed = @{}
foreach ($entry in (Get-Content -LiteralPath $EmudhraIntermediatesPath -Raw | ConvertFrom-Json)) {
    $EmudhraListed[$entry.sha256.ToUpperInvariant()] = "$($entry.name) (issued by $($entry.issued_by))"
}

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

# Restarts the CryptSvc service that maintains the CTLs after an experiment
# change, then records the explicit-distrust count, certutil matches, TLS
# outcomes and TLS chains observed afterwards. Returns the count and whether
# any TLS connection still trusted its chain.
function Measure-ExperimentState([string]$Label) {
    Restart-Service -Name CryptSvc -Force
    $policy = "DisableRootAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'DisableRootAutoUpdate'), " +
        "EnableDisallowedCertAutoUpdate=$(Get-RegistryValueText $CtlPolicyPath 'EnableDisallowedCertAutoUpdate')"
    $count = Get-ExplicitlyDistrustedCountFresh
    # Out-Host keeps the recorded rows out of this function's return value.
    Record-Check "Experiment: $Label" 'INFO' "$count of $EmudhraIntermediatesCount explicitly distrusted after CryptSvc restart ($policy)" | Out-Host
    Record-CtlMatch "Experiment: $Label, CTL matches" | Out-Host
    $tlsTrusted = $false
    foreach ($endpoint in $TestEndpoints) {
        $outcome = Get-TlsOutcomeFresh $endpoint
        if ($outcome.Kind -in 'Trusted', 'NameMismatchOnly') {
            $tlsTrusted = $true
        }
        Record-Check "Experiment: $Label, Invoke-WebRequest ``$endpoint``" 'INFO' "$($outcome.Kind): $($outcome.Detail)" | Out-Host
        $chainCheck = "Experiment: $Label, TLS chain ``$endpoint``"
        Invoke-Diagnostic $chainCheck { Record-Check $chainCheck 'INFO' (Format-TlsChain (Get-TlsChainFresh $endpoint)) } | Out-Host
    }
    Record-NativeClients "Experiment: $Label, " | Out-Host
    return [pscustomobject]@{ Count = $count; TlsTrusted = $tlsTrusted }
}

# Runs a function defined in this script in a fresh pwsh process, so no TLS or
# chain state cached by this process carries over, and returns its result.
function Invoke-FunctionFresh([string]$Name, [string]$Argument) {
    $command = "function $Name { $((Get-Item -LiteralPath "function:$Name").ScriptBlock) }; " +
        "$Name '$Argument' | ConvertTo-Json -Compress -Depth 5"
    $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($command))
    $output = @(& (Get-Process -Id $PID).Path -NoProfile -NonInteractive -EncodedCommand $encoded 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0 -or $output.Count -ne 1) {
        throw "Fresh $Name failed (exit $LASTEXITCODE): $($output -join ' ')"
    }
    return ($output[0] | ConvertFrom-Json)
}

# Adds the compromised certificates missing from LocalMachine\Disallowed, as
# the upgrade-ca-certificates action does, and returns their thumbprints.
function Add-DisallowedStoreEntries {
    $certs = @(Get-EmudhraIntermediates)
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new('Disallowed', 'LocalMachine')
    try {
        $store.Open('ReadWrite')
        $added = @()
        foreach ($cert in $certs) {
            if ($store.Certificates.Find('FindByThumbprint', $cert.Thumbprint, $false).Count -eq 0) {
                $store.Add($cert)
                $added += $cert.Thumbprint
            }
        }
        return $added
    } finally {
        $store.Dispose()
        $certs | ForEach-Object { $_.Dispose() }
    }
}

function Remove-DisallowedStoreEntries([string[]]$Thumbprints) {
    $store = [System.Security.Cryptography.X509Certificates.X509Store]::new('Disallowed', 'LocalMachine')
    try {
        $store.Open('ReadWrite')
        foreach ($thumbprint in $Thumbprints) {
            foreach ($cert in $store.Certificates.Find('FindByThumbprint', $thumbprint, $false)) {
                $store.Remove($cert)
            }
        }
    } finally {
        $store.Dispose()
    }
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

function Get-TlsOutcomeFresh([string]$Endpoint) {
    return Invoke-FunctionFresh 'Get-TlsOutcome' $Endpoint
}

# Connects to an endpoint with SslStream and returns the chain .NET built for
# the server certificate (policy errors, chain status, and each element with
# its status flags and SHA-256) together with the certificates the server sent
# (the chain's ExtraStore). The validation callback accepts
# every certificate, so this observes the chain without gating on it.
function Get-TlsChain {
    param([string]$Endpoint)
    $uri = [Uri]$Endpoint
    $result = @{}
    $callback = {
        param($source, $certificate, $chain, $errors)
        $flagsOf = { param($statuses) $f = @($statuses | ForEach-Object { $_.Status }) -join ','; if ($f) { $f } else { 'NoError' } }
        $describe = {
            param($cert)
            $name = $cert.GetNameInfo('SimpleName', $false)
            if ($cert.Subject -ne $cert.Issuer) {
                $name += " (issued by $($cert.GetNameInfo('SimpleName', $true)))"
            }
            [pscustomobject]@{ Name = $name; Sha256 = $cert.GetCertHashString('SHA256'); Flags = '' }
        }
        $result.Errors = "$errors"
        $result.Status = & $flagsOf $chain.ChainStatus
        $result.Built = @(foreach ($element in $chain.ChainElements) {
                $d = & $describe $element.Certificate
                $d.Flags = & $flagsOf $element.ChainElementStatus
                $d
            })
        $result.Sent = @($chain.ChainPolicy.ExtraStore | ForEach-Object { & $describe $_ })
        return $true
    }.GetNewClosure()
    $tcp = [System.Net.Sockets.TcpClient]::new($uri.Host, $uri.Port)
    try {
        $ssl = [System.Net.Security.SslStream]::new($tcp.GetStream(), $false, $callback)
        try {
            $ssl.AuthenticateAsClient($uri.Host)
        } finally {
            $ssl.Dispose()
        }
    } finally {
        $tcp.Dispose()
    }
    return [pscustomobject]$result
}

function Get-TlsChainFresh([string]$Endpoint) {
    return Invoke-FunctionFresh 'Get-TlsChain' $Endpoint
}

function Format-CertRef($Cert) {
    $text = "$($Cert.Name) sha256 $($Cert.Sha256.Substring(0, 16))"
    if ($EmudhraListed.ContainsKey($Cert.Sha256)) {
        $text += ' (listed)'
    }
    return $text
}

function Format-TlsChain($Chain) {
    $built = @($Chain.Built | ForEach-Object { "$(Format-CertRef $_) [$($_.Flags)]" }) -join ' -> '
    $sent = @($Chain.Sent | ForEach-Object { Format-CertRef $_ }) -join ', '
    if (-not $sent) {
        $sent = 'none'
    }
    return "policy errors $($Chain.Errors); chain status $($Chain.Status); built: $built; server sent: $sent"
}

# Looks for certificates that carry one of the compromised keys or an eMudhra
# G3 intermediate name, in the certificate stores and in the CryptnetUrlCache
# where Windows keeps certificates downloaded through AIA. A copy that is not
# in the impacted list would let chain building route around the blocks.
function Record-EmudhraCopies([string]$Name) {
    $listedCerts = @(Get-EmudhraIntermediates)
    $keys = @{}
    foreach ($cert in $listedCerts) {
        $keys[[Convert]::ToBase64String($cert.GetPublicKey())] = $true
        $cert.Dispose()
    }
    $found = [System.Collections.Generic.List[string]]::new()
    $inspect = {
        param($cert, $where)
        $simple = $cert.GetNameInfo('SimpleName', $false)
        if ($keys.ContainsKey([Convert]::ToBase64String($cert.GetPublicKey())) -or $simple -cmatch '^EM .+ CA - G3') {
            $sha = $cert.GetCertHashString('SHA256')
            $tag = if ($EmudhraListed.ContainsKey($sha)) { 'listed' } else { 'NOT listed' }
            $found.Add("${where}: $simple (issued by $($cert.GetNameInfo('SimpleName', $true))), sha256 $($sha.Substring(0, 16)), " +
                "serial $($cert.SerialNumber), $($cert.NotBefore.ToString('yyyy-MM-dd'))..$($cert.NotAfter.ToString('yyyy-MM-dd')), $tag")
        }
    }

    $storeCount = 0
    foreach ($location in 'CurrentUser', 'LocalMachine') {
        foreach ($storeName in 'Root', 'AuthRoot', 'CA', 'TrustedPeople', 'TrustedPublisher', 'Disallowed') {
            $store = [System.Security.Cryptography.X509Certificates.X509Store]::new($storeName, $location)
            try {
                $store.Open([System.Security.Cryptography.X509Certificates.OpenFlags]'ReadOnly, OpenExistingOnly')
                foreach ($cert in $store.Certificates) {
                    $storeCount++
                    & $inspect $cert "$location\$storeName"
                    $cert.Dispose()
                }
            } catch [System.Security.Cryptography.CryptographicException] {
                # OpenExistingOnly: the store does not exist for this location.
            } finally {
                $store.Dispose()
            }
        }
    }

    $cacheDirs = @(
        (Join-Path $env:USERPROFILE 'AppData\LocalLow\Microsoft\CryptnetUrlCache\Content'),
        (Join-Path $env:SystemRoot 'System32\config\systemprofile\AppData\LocalLow\Microsoft\CryptnetUrlCache\Content'),
        (Join-Path $env:SystemRoot 'ServiceProfiles\LocalService\AppData\LocalLow\Microsoft\CryptnetUrlCache\Content'),
        (Join-Path $env:SystemRoot 'ServiceProfiles\NetworkService\AppData\LocalLow\Microsoft\CryptnetUrlCache\Content')
    )
    $cacheFiles = 0
    $cacheCerts = 0
    $unreadable = @()
    foreach ($dir in $cacheDirs) {
        if (-not (Test-Path -LiteralPath $dir)) {
            continue
        }
        try {
            $files = @(Get-ChildItem -LiteralPath $dir -File -Force)
        } catch {
            $unreadable += "$dir ($($_.Exception.Message))"
            continue
        }
        foreach ($file in $files) {
            $cacheFiles++
            $collection = [System.Security.Cryptography.X509Certificates.X509Certificate2Collection]::new()
            try {
                $collection.Import($file.FullName)
            } catch [System.Security.Cryptography.CryptographicException] {
                # CRLs, CTLs and OCSP responses share the cache; only certificates are of interest.
                continue
            }
            foreach ($cert in $collection) {
                $cacheCerts++
                & $inspect $cert "CryptnetUrlCache $($file.Name)"
                $cert.Dispose()
            }
        }
    }

    $details = "$($found.Count) found among $storeCount store certificates and $cacheCerts certificates in $cacheFiles cache files"
    if ($found.Count -gt 0) {
        $details += ": $($found -join '; ')"
    }
    if ($unreadable.Count -gt 0) {
        $details += "; unreadable cache directories: $($unreadable -join '; ')"
    }
    Record-Check $Name 'INFO' $details
}

# Connects with the inbox curl.exe, which uses Schannel and Windows chain
# building without .NET. --ssl-no-revoke keeps an online revocation check from
# masking the local trust decision.
function Get-CurlOutcome([string]$Endpoint) {
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    $output = @(& $curl --ssl-no-revoke --silent --show-error --output NUL --write-out 'HTTP %{http_code}' --max-time 15 $Endpoint 2>&1 | ForEach-Object { "$_" })
    $exitCode = $LASTEXITCODE
    $text = ($output -join ' ').Trim()
    if ($exitCode -eq 0) {
        return "Trusted: $text"
    }
    # 60 is CURLE_PEER_FAILED_VERIFICATION; Schannel also reports chain errors as 35 (CURLE_SSL_CONNECT_ERROR).
    if ($exitCode -eq 60 -or ($exitCode -eq 35 -and $text -match 'SEC_E_UNTRUSTED_ROOT|SEC_E_CERT|CERT_E_|CRYPT_E_|certificate')) {
        return "Rejected (exit $exitCode): $text"
    }
    return "Other (exit $exitCode): $text"
}

# Connects with gh, whose Go crypto/x509 uses Windows chain building on
# Windows. gh api refuses to run without a token, so it gets a placeholder and
# an empty config directory; no real credential is available to send to the
# test host. Go releases with the GO-2026-6809 fix reject the compromised
# intermediates on their own, so read this together with gh's Go version.
function Get-GhOutcome([string]$Endpoint) {
    $names = 'GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN', 'GH_HOST', 'GH_CONFIG_DIR'
    $saved = @{}
    foreach ($n in $names) {
        $saved[$n] = [Environment]::GetEnvironmentVariable($n)
        Remove-Item -LiteralPath "Env:$n" -ErrorAction SilentlyContinue
    }
    $configDir = Join-Path ([System.IO.Path]::GetTempPath()) "gh-probe-$PID"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    try {
        $env:GH_TOKEN = 'placeholder-not-a-token'
        $env:GH_CONFIG_DIR = $configDir
        $output = @(& gh api $Endpoint 2>&1 | ForEach-Object { "$_" })
        $exitCode = $LASTEXITCODE
    } finally {
        foreach ($n in $names) {
            if ($null -eq $saved[$n]) {
                Remove-Item -LiteralPath "Env:$n" -ErrorAction SilentlyContinue
            } else {
                Set-Item -LiteralPath "Env:$n" -Value $saved[$n]
            }
        }
        Remove-Item -LiteralPath $configDir -Recurse -Force
    }
    $certLine = $output | Where-Object { $_ -match 'x509:|certificate' } | Select-Object -First 1
    if ($certLine) {
        return "Rejected (exit $exitCode): $($certLine.Trim())"
    }
    $httpLine = $output | Where-Object { $_ -match '\bHTTP \d{3}\b' } | Select-Object -First 1
    if ($exitCode -eq 0) {
        return 'Trusted: response received'
    }
    if ($httpLine) {
        return "Trusted (exit $exitCode): $($httpLine.Trim())"
    }
    $first = ($output | Select-Object -First 1)
    return "Other (exit $exitCode): $first"
}

function Record-NativeClients([string]$Prefix) {
    $hasGh = [bool](Get-Command gh -ErrorAction SilentlyContinue)
    foreach ($endpoint in $TestEndpoints) {
        Invoke-Diagnostic "${Prefix}curl.exe ``$endpoint``" { Record-Check "${Prefix}curl.exe ``$endpoint``" 'INFO' (Get-CurlOutcome $endpoint) }
        if ($hasGh) {
            Invoke-Diagnostic "${Prefix}gh api ``$endpoint``" { Record-Check "${Prefix}gh api ``$endpoint``" 'INFO' (Get-GhOutcome $endpoint) }
        }
    }
}

function Get-ClientVersions {
    $parts = @(
        "pwsh $($PSVersionTable.PSVersion)",
        [System.Runtime.InteropServices.RuntimeInformation]::FrameworkDescription,
        "crypt32.dll $((Get-Item -LiteralPath (Join-Path $env:SystemRoot 'System32\crypt32.dll')).VersionInfo.FileVersion)",
        ((& (Join-Path $env:SystemRoot 'System32\curl.exe') -V | Select-Object -First 1) -replace '\s+Release-Date.*$', '')
    )
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh) {
        $parts += 'gh not found'
    } else {
        $parts += (& gh --version | Select-Object -First 1)
        if (Get-Command go -ErrorAction SilentlyContinue) {
            $parts += "gh built with $((& go version -m $gh.Source | Select-Object -First 1) -replace '^.*:\s*', '')"
        } else {
            $parts += "gh Go version unknown (go not found)"
        }
    }
    return ($parts -join '; ')
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
Invoke-Diagnostic 'Client versions' { Record-Check 'Client versions' 'INFO' (Get-ClientVersions) }
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
foreach ($endpoint in $TestEndpoints) {
    Invoke-Diagnostic "TLS chain ``$endpoint``" { Record-Check "TLS chain ``$endpoint``" 'INFO' (Format-TlsChain (Get-TlsChain $endpoint)) }
}
Record-NativeClients ''
Invoke-Diagnostic 'Certificate copies' { Record-EmudhraCopies 'Certificate copies' }

# 5. Disallowed CTL currently published on Windows Update (informational). -f
# forces a download; this runs after the gating checks so it cannot change
# what they observed.
Invoke-Diagnostic 'Windows Update Disallowed CTL' { Record-CertutilCtl 'Windows Update Disallowed CTL' @('-f', '-verifyCTL', 'DisallowedWU') }

# 6. Trust experiment (informational). First tests the upgrade-ca-certificates
# workaround (LocalMachine\Disallowed entries) under the image's policy, then
# removes those entries again. Then tests whether the root auto-update policy
# is why a Disallowed CTL that lists the compromised keys is not applied: set
# EnableDisallowedCertAutoUpdate=1, and if chain building or TLS still trusts
# the certificates, also remove DisableRootAutoUpdate. It changes machine
# trust state and restarts CryptSvc, so it runs last and only on ephemeral
# GitHub-hosted runners where DisableRootAutoUpdate=1.
if ($env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    Record-Check 'Experiment: CTL policy' 'INFO' "Skipped: runner environment is '$env:RUNNER_ENVIRONMENT', not github-hosted"
} elseif ((Get-RegistryValueText $CtlPolicyPath 'DisableRootAutoUpdate') -ne '1') {
    Record-Check 'Experiment: CTL policy' 'INFO' 'Skipped: DisableRootAutoUpdate is not 1'
} else {
    Invoke-Diagnostic 'Experiment: CTL policy' {
        $added = @(Add-DisallowedStoreEntries)
        [void](Measure-ExperimentState "LocalMachine\Disallowed import ($($added.Count) added)")
        Remove-DisallowedStoreEntries $added
        [void](Measure-ExperimentState 'LocalMachine\Disallowed import removed')

        Set-ItemProperty -LiteralPath $CtlPolicyPath -Name 'EnableDisallowedCertAutoUpdate' -Type DWord -Value 1
        $state = Measure-ExperimentState 'EnableDisallowedCertAutoUpdate=1'
        if ($state.Count -lt $EmudhraIntermediatesCount -or $state.TlsTrusted) {
            Remove-ItemProperty -LiteralPath $CtlPolicyPath -Name 'DisableRootAutoUpdate'
            [void](Measure-ExperimentState 'DisableRootAutoUpdate removed')
        }
        Record-EmudhraCopies 'Experiment: certificate copies after all steps'
    }
}

Append-Summary ""

if ($GatingFailures -gt 0) {
    Write-Error "${ProbeLabel}: $GatingFailures gating check(s) failed"
    exit 1
}
