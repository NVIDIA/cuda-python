#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# install_gpu_driver.sh -- install a specific NVIDIA driver version on a
# Linux CI runner. Adapted from nv-gha-runners/vm-images PR #256
# (`nvgha-driver` CLI), trimmed and parameterised for cuda-python's CI.
#
# !!! ALPHA !!!
# Performs live modifications to the host driver stack (kernel module
# reload, package replacement, and -- inside containers -- toolkit
# bind-mount refresh) and may cause issues.
#
# Inputs (env):
#   DRIVER    Driver version, e.g. "580.65.06". Must NOT be 'latest' or
#             'earliest' -- those are runner-pre-installed and the
#             workflow is expected to skip this script for them.
#   GPU_TYPE  Lower-case GPU label from the matrix (e.g. "v100", "l4",
#             "h100"). Used only to pick the kernel module flavor
#             (Volta needs the proprietary/legacy module; everything
#             newer can use the open module).
#
# Arch is detected from `uname -m`.
#
# When the script runs inside a container (the cuda-python Linux jobs do)
# it re-execs itself on the host via `nsenter`. The job must declare
# `options: --privileged --pid=host` (the workflow only does this for
# matrix rows with a custom DRIVER). After the host-side install, the
# container's bind-mounted nvidia libs/binaries are refreshed in-place so
# the new driver is visible without restarting the container.
set -euo pipefail

: "${DRIVER:?DRIVER env var is required (e.g. 580.65.06)}"
: "${GPU_TYPE:?GPU_TYPE env var is required (e.g. l4)}"

case "$DRIVER" in
  latest|earliest)
    echo "::error::install_gpu_driver.sh must not be invoked with DRIVER=$DRIVER (runner-pre-installed)" >&2
    exit 1
    ;;
esac

VERSION="$DRIVER"

# Volta (V100) requires the legacy/proprietary kernel module; all newer
# GPUs in this matrix support the open module. Extend this if/when older
# GPUs return to the matrix.
case "$GPU_TYPE" in
  v100) KMT=proprietary ;;
  *)    KMT=open ;;
esac

case "$(uname -m)" in
  x86_64)
    ARCH_DIR=Linux-x86_64
    ARCH_SUFFIX=x86_64
    ;;
  aarch64)
    ARCH_DIR=Linux-aarch64
    ARCH_SUFFIX=aarch64
    ;;
  *)
    echo "::error::unsupported arch: $(uname -m)" >&2
    exit 1
    ;;
esac

URL="https://us.download.nvidia.com/XFree86/${ARCH_DIR}/${VERSION}/NVIDIA-Linux-${ARCH_SUFFIX}-${VERSION}.run"

# Re-elevate to root if needed (sudo is preinstalled on the runner image).
if [ "$(id -u)" != 0 ]; then
  exec sudo -E DRIVER="$DRIVER" GPU_TYPE="$GPU_TYPE" "$0" "$@"
fi

echo "install_gpu_driver.sh is ALPHA -- it performs live modifications to the host driver stack and may cause issues" >&2
echo "DRIVER=${VERSION}  GPU_TYPE=${GPU_TYPE}  KMT=${KMT}  ARCH=${ARCH_SUFFIX}" >&2
echo "URL=${URL}" >&2

# Toolkit packages we keep across the purge: dockerd's --runtime=nvidia
# resolves nvidia-container-runtime through these, and removing them
# breaks `docker exec` against any container started with that runtime.
KEEP_RE='^(nvidia-container-toolkit(-base)?|libnvidia-container1|libnvidia-container-tools)$'

in_container() {
  [ -f /.dockerenv ] || grep -qE '/(docker|kubepods|containerd)' /proc/1/cgroup 2>/dev/null
}

host_install() {
  apt-get -y install build-essential dkms "linux-headers-$(uname -r)" psmisc kmod

  systemctl stop nvidia-persistenced dcgm-exporter 2>/dev/null || true
  # if-test instead of `fuser ... || true` so a kill failure surfaces
  # (fuser exits 1 when nothing holds the device, which is the happy path).
  if fuser /dev/nvidia* >/dev/null 2>&1; then
    fuser -kv /dev/nvidia*
  fi
  sleep 1
  for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
    rmmod "$m" 2>/dev/null || true
  done

  # Purge existing nvidia/libnvidia packages, except the toolkit pieces
  # captured by KEEP_RE. Tolerate apt failures: postrm scripts can trip
  # and the .run installer is about to replace everything anyway.
  dpkg-query -W -f='${Package}\n' 'nvidia-*' 'libnvidia-*' 2>/dev/null \
    | awk -v re="$KEEP_RE" '$0 !~ re' \
    | xargs -r apt-get -y remove --purge || true

  local d
  d=$(mktemp -d)
  ( cd "$d" \
    && wget -q -O installer.run "$URL" \
    && sh installer.run --silent --dkms --no-questions \
         --accept-license --ui=none --no-cc-version-check --kernel-module-type="$KMT" )
  modprobe nvidia nvidia_uvm nvidia_modeset

  # Restore the runner image's baseline state: persistence mode ENABLED
  # plus nvidia-persistenced running. The runner-team's pre-installed
  # drivers come up with `Persistence-M: On`, but our .run install leaves
  # it Off, which breaks tests that toggle the value (cuda.core's
  # test_persistence_mode_enabled hits NVML_ERROR_UNKNOWN when setting
  # the mode to its current value on driver 610.43.02).
  #
  # `nvidia-smi -pm 1` is the load-bearing call -- it sets the kernel-
  # level persistence flag directly via NVML (equivalent to what the
  # daemon would do on startup). The systemctl block is best-effort: the
  # silent .run installer doesn't always drop the systemd unit, so we
  # daemon-reload first and tolerate failure on `start`.
  set -x
  nvidia-smi -pm 1 || true
  systemctl daemon-reload 2>/dev/null || true
  systemctl start nvidia-persistenced.service 2>/dev/null || true
  set +x
}

# Replace the toolkit's bind-mounted nvidia libs/binaries inside this
# container with copies from the host's new install. `cp` (not
# `mount --bind`) because procfs-routed binds drop the exec bit.
refresh_container_libs() {
  # Walk /proc/self/mountinfo and match the toolkit-injected nvidia
  # binds via mount point (field 5) so deleted source paths -- which
  # the kernel suffixes field 4 with " (deleted)" once the host unlinks
  # the old lib -- don't break discovery. Filters skip what we can't or
  # shouldn't refresh:
  #   $3 ~ /^0:/                tmpfs/proc/sysfs (e.g. the toolkit hook tmpfs)
  #   $5 ~ /\.json$/            vulkan/glvnd config remaps (not version-bound)
  #   $5 ~ /\/(firmware|xorg)\// firmware loads host-side; xorg unused in CUDA containers
  local mounts
  mounts=$(awk '
    $3 !~ /^0:/                     &&
    $5 !~ /\.json$/                 &&
    $5 !~ /\/(firmware|xorg)\//     &&
    $5 ~ /(nvidia|libcuda)/         { print $5 }
  ' /proc/self/mountinfo | sort -u)

  for tgt in $mounts; do
    local src="/proc/1/root$tgt"
    if [ ! -e "$src" ]; then
      # Driver swap rewrites the version suffix (libfoo.so.595.71.05 ->
      # libfoo.so.580.65.06); strip it and find the new file.
      local base
      base=$(basename "$tgt")
      base="${base%.so.*}.so"
      src=$(find "/proc/1/root$(dirname "$tgt")" -maxdepth 1 -name "${base}.*" 2>/dev/null \
            | sort -V | tail -n1)
      [ -n "$src" ] || { echo "skip $tgt: no host source" >&2; continue; }
    fi
    umount "$tgt" 2>/dev/null || true
    cp -f --remove-destination "$src" "$tgt" \
      || echo "WARN: refresh failed for $tgt (src=$src)" >&2
  done
  ldconfig
}

if [ -z "${_NVDRV_NSENTERED:-}" ] && in_container; then
  # Re-exec on the host. The runner-team's `nvgha-driver` script lives at a
  # host-side absolute path so `"$0"` survives the mount-namespace flip;
  # ours lives in the GH workspace mount (container-only), so we pipe the
  # script body in via stdin instead -- the `< "$0"` fd is opened before
  # nsenter and stays valid across the namespace switch. Env vars (DRIVER,
  # GPU_TYPE, _NVDRV_NSENTERED) are inherited by the host-side bash.
  _NVDRV_NSENTERED=1 nsenter -t 1 -m -p -n -i -u -- bash -s < "$0" \
    || { echo "::error::container needs 'options: --privileged --pid=host'" >&2; exit 1; }
  refresh_container_libs
else
  host_install
fi

nvidia-smi >/dev/null
grep -qF "$VERSION" /proc/driver/nvidia/version
