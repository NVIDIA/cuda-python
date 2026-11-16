# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Helpers for the compromised eMudhra intermediates and their Sectigo
# cross-signs listed in impacted-intermediates.json. Dot-sourced by this
# action's Windows step and by ci/tools/runner_trust_probe.ps1.
#
# impacted-intermediates.json is the disclosure's file, unmodified (SHA-256
# f67b3abe9b4879f4e91874ed041beccd71257f1db41c588c724ccf12aa90c59e). The set
# is closed: new certificates for these keys would have to be issued by
# another CA, and all existing ones are revoked.

$EmudhraIntermediatesPath = Join-Path $PSScriptRoot 'impacted-intermediates.json'
$EmudhraIntermediatesCount = 16

# Uppercase hex SHA-256 of the certificate's DER encoding. Computed directly
# because X509Certificate.GetCertHashString(HashAlgorithmName) does not exist
# in Windows PowerShell 5.1 (.NET Framework).
function Get-CertificateSha256([System.Security.Cryptography.X509Certificates.X509Certificate2]$Cert) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [System.BitConverter]::ToString($sha256.ComputeHash($Cert.RawData)).Replace('-', '')
    } finally {
        $sha256.Dispose()
    }
}

# Returns the certificates as X509Certificate2 objects; callers dispose them.
# Works on both Windows PowerShell 5.1 and PowerShell 7+.
function Get-EmudhraIntermediates {
    # Windows PowerShell 5.1's ConvertFrom-Json emits a JSON array as a single
    # object; piping it through ForEach-Object enumerates it on both versions.
    $parsed = Get-Content -LiteralPath $EmudhraIntermediatesPath -Raw | ConvertFrom-Json
    $entries = @($parsed | ForEach-Object { $_ })
    if ($entries.Count -ne $EmudhraIntermediatesCount) {
        throw "Expected $EmudhraIntermediatesCount certificates in $EmudhraIntermediatesPath, found $($entries.Count)."
    }
    foreach ($entry in $entries) {
        # X509Certificate2.CreateFromPem is .NET 5+ only; decode the PEM body manually.
        $der = [System.Convert]::FromBase64String(($entry.pem -replace '-----[A-Z ]+-----', '' -replace '\s', ''))
        $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($der)
        if ((Get-CertificateSha256 $cert) -ne $entry.sha256.ToUpperInvariant()) {
            $cert.Dispose()
            throw "SHA-256 mismatch for $($entry.name) (issued by $($entry.issued_by))."
        }
        Add-Member -InputObject $cert -NotePropertyName EmudhraLabel -NotePropertyValue "$($entry.name) (issued by $($entry.issued_by))"
        $cert
    }
}

# True if Windows chain building reports the certificate as explicitly
# distrusted. This covers both a local LocalMachine\Disallowed entry and
# Microsoft's automatically updated disallowed list. Revocation checking and
# AIA downloads are off so the result reflects the local trust state only.
function Test-ExplicitlyDistrusted([System.Security.Cryptography.X509Certificates.X509Certificate2]$Cert) {
    $explicitDistrust = [System.Security.Cryptography.X509Certificates.X509ChainStatusFlags]::ExplicitDistrust
    $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
    try {
        $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
        # DisableCertificateDownloads does not exist on .NET Framework (Windows PowerShell 5.1).
        if ($chain.ChainPolicy | Get-Member -Name DisableCertificateDownloads) {
            $chain.ChainPolicy.DisableCertificateDownloads = $true
        }
        [void]$chain.Build($Cert)
        $statuses = @($chain.ChainStatus) + @($chain.ChainElements | ForEach-Object { $_.ChainElementStatus })
        return [bool]($statuses | Where-Object { $_.Status.HasFlag($explicitDistrust) })
    } finally {
        $chain.Dispose()
    }
}

# Classifies how Windows chain building treats the certificate, using only the
# local trust state (no revocation, no AIA downloads):
#   ExplicitDistrust : in LocalMachine\Disallowed or Microsoft's disallowed list
#   NoTrustAnchor    : the chain no longer ends in a trusted root, and
#                      UntrustedRoot / PartialChain are the only problems
#                      reported, e.g. because the root was removed
#   Trusted          : the chain builds cleanly, so the compromised key is usable
#   Indeterminate    : the build failed for some other reason; never a pass
# Verification time is pinned (the instant used by runner_trust_probe.sh) so an
# expired certificate cannot masquerade as untrusted.
# A PartialChain only shows that the root is untrusted if the issuers are
# available to the build, so callers pass the non-root issuers in
# $ExtraCertificates. Issuers that are roots need not be supplied: a missing
# root is exactly the condition this function reports as NoTrustAnchor.
function Get-ChainTrustState(
    [System.Security.Cryptography.X509Certificates.X509Certificate2]$Cert,
    [System.Security.Cryptography.X509Certificates.X509Certificate2[]]$ExtraCertificates = @()
) {
    $explicitDistrust = [System.Security.Cryptography.X509Certificates.X509ChainStatusFlags]::ExplicitDistrust
    $noAnchor = [int]([System.Security.Cryptography.X509Certificates.X509ChainStatusFlags]::UntrustedRoot -bor
        [System.Security.Cryptography.X509Certificates.X509ChainStatusFlags]::PartialChain)
    $chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
    try {
        foreach ($extra in $ExtraCertificates) {
            [void]$chain.ChainPolicy.ExtraStore.Add($extra)
        }
        $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::NoCheck
        $chain.ChainPolicy.VerificationTime = [DateTime]::new(2026, 10, 23, 0, 0, 0, [DateTimeKind]::Utc)
        # DisableCertificateDownloads does not exist on .NET Framework (Windows PowerShell 5.1).
        if ($chain.ChainPolicy | Get-Member -Name DisableCertificateDownloads) {
            $chain.ChainPolicy.DisableCertificateDownloads = $true
        }
        $built = $chain.Build($Cert)
        $statuses = @($chain.ChainStatus) + @($chain.ChainElements | ForEach-Object { $_.ChainElementStatus })
        $names = @($statuses | ForEach-Object { $_.Status.ToString() } | Where-Object { $_ -ne 'NoError' } | Sort-Object -Unique)
        if ($statuses | Where-Object { $_.Status.HasFlag($explicitDistrust) }) {
            $state = 'ExplicitDistrust'
        } elseif ($built) {
            $state = 'Trusted'
        } elseif (
            ($statuses | Where-Object { ([int]$_.Status -band $noAnchor) -ne 0 }) -and
            -not ($statuses | Where-Object { ([int]$_.Status -band (-bnot $noAnchor)) -ne 0 })
        ) {
            $state = 'NoTrustAnchor'
        } else {
            $state = 'Indeterminate'
        }
        return [pscustomobject]@{ State = $state; Statuses = ($names -join ',') }
    } finally {
        $chain.Dispose()
    }
}
