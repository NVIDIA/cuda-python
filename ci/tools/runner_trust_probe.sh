#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# runner_trust_probe.sh -- probe the OS certificate store on a Linux runner
# for the compromised eMudhra/emSign root anchor (emSign ECC Root CA - G3).
#
# Optional environment variable:
#   TRUST_PROBE_LABEL   Heading label for GITHUB_STEP_SUMMARY

set -euo pipefail

EMSIGN_G3_SHA256="86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B"
CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
TEST_ENDPOINT="https://dv-g3b-emudhra.we-have-a-problem.com/"
PROBE_LABEL="${TRUST_PROBE_LABEL:-Linux OS trust store}"

summary_file="${GITHUB_STEP_SUMMARY:-}"
append_summary() {
  if [[ -n "${summary_file}" ]]; then
    printf '%s\n' "$*" >> "${summary_file}"
  fi
}

gating_failures=0

append_summary "### ${PROBE_LABEL}"
append_summary ""
append_summary "| Check | Status | Details |"
append_summary "| --- | --- | --- |"

record_check() {
  local name="$1"
  local status="$2"
  local details="$3"
  printf '[%s] %s: %s\n' "${status}" "${name}" "${details}"
  append_summary "| ${name} | **${status}** | ${details} |"
}

# 1. Root anchor absent from CA bundle (gating)
if [[ ! -f "${CA_BUNDLE}" ]]; then
  record_check "Root anchor absent" "FAIL" "Bundle not found: \`${CA_BUNDLE}\`"
  gating_failures=$((gating_failures + 1))
elif ! command -v openssl >/dev/null 2>&1; then
  record_check "Root anchor absent" "FAIL" "\`openssl\` not found on PATH"
  gating_failures=$((gating_failures + 1))
else
  fingerprints="$(awk '
    /-----BEGIN CERTIFICATE-----/ { cert = "" }
    { cert = cert $0 "\n" }
    /-----END CERTIFICATE-----/   {
      cmd = "openssl x509 -noout -fingerprint -sha256 2>/dev/null"
      printf "%s", cert | cmd
      close(cmd)
    }
  ' "${CA_BUNDLE}")"
  cert_count="$(grep -c . <<< "${fingerprints}" || true)"

  if (( cert_count == 0 )); then
    record_check "Root anchor absent" "FAIL" "No certificates parsed from \`${CA_BUNDLE}\`"
    gating_failures=$((gating_failures + 1))
  elif grep -qiF "${EMSIGN_G3_SHA256}" <<< "${fingerprints}"; then
    record_check "Root anchor absent" "FAIL" "Found \`emSign ECC Root CA - G3\` (\`${EMSIGN_G3_SHA256}\`) in \`${CA_BUNDLE}\` (${cert_count} certs checked)"
    gating_failures=$((gating_failures + 1))
  else
    record_check "Root anchor absent" "PASS" "Absent from \`${CA_BUNDLE}\` (${cert_count} certs checked)"
  fi
fi

# 2. ca-certificates package version (informational)
ca_pkg_ver="$(dpkg-query -W -f='${Version}' ca-certificates 2>/dev/null || echo 'not-installed')"
record_check "ca-certificates version" "INFO" "\`${ca_pkg_ver}\`"

# 3. OS-store client (curl) rejects the test endpoint (only a successful TLS connection fails the job)
if ! command -v curl >/dev/null 2>&1; then
  record_check "curl rejects test endpoint" "INCONCLUSIVE" "\`curl\` not found on PATH"
else
  curl_rc=0
  curl_err="$(curl -sS --max-time 15 --cacert "${CA_BUNDLE}" -o /dev/null "${TEST_ENDPOINT}" 2>&1)" || curl_rc=$?
  curl_err_oneline="$(tr '\n' ' ' <<< "${curl_err}" | sed 's/|/\\|/g')"
  if (( curl_rc == 0 )); then
    record_check "curl rejects test endpoint" "FAIL" "Connection to \`${TEST_ENDPOINT}\` succeeded (certificate was trusted)"
    gating_failures=$((gating_failures + 1))
  elif (( curl_rc == 60 )); then
    if grep -qiE 'no alternative certificate subject name|subject alternative name|does not match' <<< "${curl_err}"; then
      record_check "curl rejects test endpoint" "FAIL" "Certificate chain was trusted; failed only hostname check (${curl_err_oneline})"
      gating_failures=$((gating_failures + 1))
    else
      record_check "curl rejects test endpoint" "PASS" "Rejected with curl exit 60 (peer certificate cannot be authenticated)"
    fi
  else
    record_check "curl rejects test endpoint" "INCONCLUSIVE" "curl exited ${curl_rc} (network/DNS or non-verification error: ${curl_err_oneline})"
  fi
fi

append_summary ""

if (( gating_failures > 0 )); then
  echo "::error::${PROBE_LABEL}: ${gating_failures} gating check(s) failed" >&2
  exit 1
fi
