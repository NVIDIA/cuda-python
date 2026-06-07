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

  # Bring nvidia-persistenced back up. NVML state-changing calls from
  # inside the test container (e.g. nvmlDeviceSetPersistenceMode, which
  # cuda.core's test_persistence_mode_enabled exercises) talk to the
  # daemon via /run/nvidia-persistenced/socket; without a live daemon
  # they return NVML_ERROR_UNKNOWN.
  #
  # systemctl can't start the unit (the `--silent --no-questions` .run
  # installer drops /usr/bin/nvidia-persistenced but no usable systemd
  # unit), so exec the binary directly -- it self-daemonizes.
  #
  # `--user root` because the default `nvidia-persistenced` user was
  # deleted along with `nvidia-compute-utils-*` in the purge above;
  # without this flag the daemon's post-fork setuid() fails silently
  # and the process exits.
  #
  # nv-gha-runners/vm-images' `nvgha-driver` has the same latent gap;
  # its CUDA-runtime validation workload never issues an NVML SET
  # write so it hasn't surfaced there.
  /usr/bin/nvidia-persistenced --verbose --user root || true
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
  #   $3 ~ /^0:/                  tmpfs/proc/sysfs (e.g. the toolkit hook tmpfs)
  #   $5 must be under /usr/(bin|lib)  binaries + libs only -- explicitly
  #                                NOT /run/nvidia-persistenced/socket
  #                                (cp'ing the daemon's IPC socket unlinks
  #                                the container's view and turns later
  #                                NVML state-changing calls into
  #                                NVML_ERROR_UNKNOWN); NOT /dev/nvidia*
  #                                (character devices); NOT /proc/driver/nvidia
  #                                (procfs); NOT /tmp/nvidia-mps (runtime).
  #   $5 ~ /\.json$/              vulkan/glvnd config remaps (not version-bound)
  #   $5 ~ /\/(firmware|xorg)\//  firmware loads host-side; xorg unused in CUDA containers
  local mounts
  mounts=$(awk '
    $3 !~ /^0:/                     &&
    $5 ~ /^\/usr\/(bin|lib)/        &&
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
    # --preserve=mode keeps the SUID bit so the refresh doesn't silently
    # de-privilege binaries like nvidia-modprobe that ship 4755 (the
    # runner-team's nvgha-driver uses plain `cp` and has the same gap).
    cp -f --preserve=mode --remove-destination "$src" "$tgt" \
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

  # Re-bind /run/nvidia-persistenced from the host. The container's
  # original bind mount was pinned to the host's at-container-start
  # inode; the daemon stop+restart cycle recreates the dir under a
  # fresh inode, stranding the bind mount on the deleted one (socket
  # shows up with link count 0 inside the container, NVML SET calls
  # return NVML_ERROR_UNKNOWN). Re-bind from /proc/1/root so the
  # container picks up the live host dir. Needs CAP_SYS_ADMIN, which
  # the workflow grants via --privileged --pid=host on custom-DRIVER
  # rows.
  if [ -d /proc/1/root/run/nvidia-persistenced ]; then
    umount /run/nvidia-persistenced 2>/dev/null || true
    mkdir -p /run/nvidia-persistenced
    mount --bind /proc/1/root/run/nvidia-persistenced /run/nvidia-persistenced
  fi
else
  host_install
fi

nvidia-smi >/dev/null
grep -qF "$VERSION" /proc/driver/nvidia/version
