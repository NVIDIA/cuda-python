#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_ROOT="${REPO_ROOT}/.moon-out"
OUTPUT="${OUTPUT_ROOT}/docs"

if [[ -L "${OUTPUT_ROOT}" || ( -e "${OUTPUT_ROOT}" && ! -d "${OUTPUT_ROOT}" ) ]]; then
    echo "refusing to use non-directory Moon output root: ${OUTPUT_ROOT}" >&2
    exit 1
fi
if [[ -L "${OUTPUT}" || ( -e "${OUTPUT}" && ! -d "${OUTPUT}" ) ]]; then
    echo "refusing to replace non-directory Moon docs output: ${OUTPUT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"

copy_component() {
    local source=$1
    local destination=$2
    local source_root
    source_root=$(dirname -- "${source}")
    if [[ -L "${source_root}" || ! -d "${source_root}" || -L "${source}" || ! -d "${source}" ]]; then
        echo "documentation component output not found: ${source}" >&2
        exit 1
    fi
    mkdir -p "${destination}"
    cp -aL "${source}/." "${destination}/"
}

copy_component "${REPO_ROOT}/cuda_python/.moon-out/docs-ci" "${OUTPUT}"
copy_component "${REPO_ROOT}/cuda_bindings/.moon-out/docs-ci" "${OUTPUT}/cuda-bindings"
copy_component "${REPO_ROOT}/cuda_core/.moon-out/docs-ci" "${OUTPUT}/cuda-core"
copy_component "${REPO_ROOT}/cuda_pathfinder/.moon-out/docs-ci" "${OUTPUT}/cuda-pathfinder"
