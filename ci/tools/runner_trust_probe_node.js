// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// runner_trust_probe_node.js -- probe the runner agent's bundled Node.js
// version and certificate store for the eMudhra/emSign CA incident.

'use strict';

const https = require('https');

module.exports = async function runNodeTrustProbe({
  core,
  heading = 'Node runtime trust store',
  gating = true,
}) {
  const version = process.version;
  const execPath = process.execPath;
  const nodeOptions = process.env.NODE_OPTIONS || '';
  const extraCa = process.env.NODE_EXTRA_CA_CERTS || '';
  const useSystemCa = process.env.NODE_USE_SYSTEM_CA || '';

  const m = /^v(\d+)\.(\d+)\.(\d+)/.exec(version);
  const [major, minor, patch] = m ? [Number(m[1]), Number(m[2]), Number(m[3])] : [0, 0, 0];
  const versionOk = major === 24 && (minor > 22 || (minor === 22 && patch >= 1));

  const endpoint = 'https://dv-g3b-emudhra.we-have-a-problem.com/';
  const probeEndpoint = () => new Promise((resolve) => {
    const req = https.get(endpoint, { timeout: 15000 }, (res) => {
      res.resume();
      resolve({ status: 'FAIL', detail: `Connected with HTTP ${res.statusCode} (certificate was trusted)` });
    });
    req.on('timeout', () => {
      req.destroy(new Error('Request timed out'));
    });
    req.on('error', (err) => {
      const code = err.code || '';
      const msg = `${code} ${err.message || ''}`.trim();
      if (code === 'ERR_TLS_CERT_ALTNAME_INVALID') {
        resolve({ status: 'FAIL', detail: `Certificate chain was trusted; failed only hostname check (${msg})` });
      } else if (/^(CERT_|UNABLE_TO_|SELF_SIGNED_|DEPTH_ZERO_)/.test(code)) {
        resolve({ status: 'PASS', detail: `Rejected with certificate error (${msg})` });
      } else {
        resolve({ status: 'INCONCLUSIVE', detail: `Network/DNS or non-certificate error (${msg})` });
      }
    });
  });

  const tlsResult = await probeEndpoint();
  const rows = [
    ['Node version', versionOk ? '**PASS**' : '**FAIL**', `\`${version}\` (require Node 24 \`>= v24.22.1\`; \`execPath=${execPath}\`)`],
    ['Node env overrides', '**INFO**', `\`NODE_OPTIONS=${nodeOptions}\`, \`NODE_EXTRA_CA_CERTS=${extraCa}\`, \`NODE_USE_SYSTEM_CA=${useSystemCa}\``],
    ['Node rejects test endpoint', `**${tlsResult.status}**`, tlsResult.detail],
  ];

  await core.summary
    .addHeading(heading, 3)
    .addTable([
      [{ data: 'Check', header: true }, { data: 'Status', header: true }, { data: 'Details', header: true }],
      ...rows,
    ])
    .write();

  if (gating && (!versionOk || tlsResult.status === 'FAIL')) {
    core.setFailed(`Node trust probe failed (version=${version}, endpoint=${tlsResult.status})`);
  }
};
