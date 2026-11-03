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

# Returns the certificates as X509Certificate2 objects; callers dispose them.
function Get-EmudhraIntermediates {
    $entries = @(Get-Content -LiteralPath $EmudhraIntermediatesPath -Raw | ConvertFrom-Json)
    if ($entries.Count -ne $EmudhraIntermediatesCount) {
        throw "Expected $EmudhraIntermediatesCount certificates in $EmudhraIntermediatesPath, found $($entries.Count)."
    }
    foreach ($entry in $entries) {
        $cert = [System.Security.Cryptography.X509Certificates.X509Certificate2]::CreateFromPem($entry.pem)
        if ($cert.GetCertHashString('SHA256') -ne $entry.sha256.ToUpperInvariant()) {
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
        $chain.ChainPolicy.DisableCertificateDownloads = $true
        [void]$chain.Build($Cert)
        $statuses = @($chain.ChainStatus) + @($chain.ChainElements | ForEach-Object { $_.ChainElementStatus })
        return [bool]($statuses | Where-Object { $_.Status.HasFlag($explicitDistrust) })
    } finally {
        $chain.Dispose()
    }
}
