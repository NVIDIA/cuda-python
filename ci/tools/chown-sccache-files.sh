#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Chown sccache-generated files on the host filesystem to the
# runner user, so they don't linger as root-owned and
# trigger the "Found root-owned files" abort on the next job.
#
# Runs as root inside the cibuildwheel container, against host paths
# via the /host mount. Callers may invoke it more than once (in-container
# and from a later docker exec); chown is idempotent, so no guard is
# needed.
#
# Reads paths from environment variables (set via CIBW_ENVIRONMENT_LINUX):
#   SCCACHE_STATS_FILE  -- stats file to chown (on the /host mount)
#   SCCACHE_ERROR_LOG  -- sccache server error log to chown
#   GITHUB_WORKSPACE  -- host workspace path (for uid/gid lookup)
set -euo pipefail

# Determine the (uid, gid) of the runner user from the
# /host workspace directory owner. GITHUB_WORKSPACE is
# already /host-prefixed (set in CIBW_ENVIRONMENT_LINUX).
HOST_WS="${GITHUB_WORKSPACE:-}"
if [ -d "${HOST_WS}" ]; then
  owner="$(stat -c %u "${HOST_WS}"):$(stat -c %g "${HOST_WS}")"
else
  echo "[chown-sccache] warning: ${HOST_WS:-<empty>} does not exist; falling back to current uid/gid"
  owner="$(id -u):$(id -g)"
fi

for f in "${SCCACHE_STATS_FILE:-}" "${SCCACHE_ERROR_LOG:-}"; do
  if [ -z "${f}" ] || [ ! -e "${f}" ]; then
    echo "[chown-sccache] skip: ${f:-<empty>} does not exist"
    continue
  fi
  if chown "${owner}" "${f}" 2>/dev/null; then
    new_owner=$(stat -c '%u:%g' "${f}")
    echo "[chown-sccache] ok: ${f} now owned by ${new_owner}"
  else
    echo "[chown-sccache] FAIL: could not chown ${f} (owner=${owner}, current=$(stat -c '%u:%g' "${f}" 2>/dev/null || echo '?'))"
  fi
done
