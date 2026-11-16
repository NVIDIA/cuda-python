Issuer certificates for the Windows runner trust probe
======================================================

ci/tools/runner_trust_probe.ps1 passes these certificates to Windows chain
building, together with the chains in ../runner_trust_probe_chains, so a missing
parent certificate is not mistaken for a missing trust anchor.

  emsign-root-cs-ca-g3-cross.pem
      emSign Root CS CA - G3, issued by emSign ECC Root CA - G3 (the
      cross-certificate of the self-signed emSign Root CS CA - G3). It is the
      issuer of the two native code-signing intermediates (EM OV CS CA - G3B
      and G3C), which the test hosts' chains do not cover.

      Source:  https://repository.emsign.com/certs/crosssign/rootcscag3.crt
               (DER; converted to PEM without other changes)
      SHA-256: 7137FA71AD46BA686F746D1014AC57E8E6C6E9AF4AA2F73C0C44A5CF27DEBB59
      Valid:   2024-07-11 to 2039-07-09
      Checked: same public key as the self-signed emSign Root CS CA - G3, and
               its key verifies the signatures of both code-signing
               intermediates in impacted-intermediates.json.
