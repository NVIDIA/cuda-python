#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: build_component_docs.sh COMPONENT [latest-only|moon-ci]" >&2
    exit 1
fi

COMPONENT=$1
shift

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

case "${COMPONENT}" in
    cuda-pathfinder)
        PACKAGE_DIR=cuda_pathfinder
        DISTRIBUTION=cuda-pathfinder
        VERSION_ENV=SPHINX_CUDA_PATHFINDER_VER
        VERSION_COMPONENTS=3
        DEFAULT_SPHINXOPTS="-W --keep-going -j 4 -d build/.doctrees"
        HONOR_SPHINXOPTS=1
        METADATA_FILES=(nv-versions.json)
        CLEAN_GENERATED=0
        ;;
    cuda-bindings)
        PACKAGE_DIR=cuda_bindings
        DISTRIBUTION=cuda-bindings
        VERSION_ENV=SPHINX_CUDA_BINDINGS_VER
        VERSION_COMPONENTS=3
        DEFAULT_SPHINXOPTS="-j 4 -d build/.doctrees"
        HONOR_SPHINXOPTS=1
        METADATA_FILES=(versions.json nv-versions.json)
        CLEAN_GENERATED=0
        ;;
    cuda-core)
        PACKAGE_DIR=cuda_core
        DISTRIBUTION=cuda-core
        VERSION_ENV=SPHINX_CUDA_CORE_VER
        VERSION_COMPONENTS=0
        DEFAULT_SPHINXOPTS="-W --keep-going -j 4 -d build/.doctrees"
        HONOR_SPHINXOPTS=1
        METADATA_FILES=(versions.json nv-versions.json)
        CLEAN_GENERATED=1
        ;;
    cuda-python)
        PACKAGE_DIR=cuda_python
        DISTRIBUTION=cuda-python
        VERSION_ENV=SPHINX_CUDA_PYTHON_VER
        VERSION_COMPONENTS=3
        DEFAULT_SPHINXOPTS="-j 4 -d build/.doctrees"
        # Preserve the metapackage builder's historical fixed Sphinx options.
        HONOR_SPHINXOPTS=0
        METADATA_FILES=(versions.json nv-versions.json)
        CLEAN_GENERATED=1
        ;;
    *)
        echo "unsupported documentation component: ${COMPONENT}" >&2
        exit 1
        ;;
esac

DOCS_DIR="${REPO_ROOT}/${PACKAGE_DIR}/docs"
if [[ -L "${DOCS_DIR}" || ! -d "${DOCS_DIR}" ]]; then
    echo "documentation source directory not found: ${DOCS_DIR}" >&2
    exit 1
fi
SOURCE_DIR="${DOCS_DIR}/source"
if [[ -L "${SOURCE_DIR}" || ! -d "${SOURCE_DIR}" ]]; then
    echo "documentation source directory not found: ${SOURCE_DIR}" >&2
    exit 1
fi
cd "${DOCS_DIR}"

MOON_CI=0
if [[ $# == 0 ]]; then
    LATEST_ONLY=0
elif [[ $# == 1 && $1 == latest-only ]]; then
    LATEST_ONLY=1
elif [[ $# == 1 && $1 == moon-ci ]]; then
    MOON_CI=1
    DOCS_LATEST_ONLY=${CUDA_PYTHON_DOCS_LATEST_ONLY:-true}
    case "${DOCS_LATEST_ONLY,,}" in
        1|true) LATEST_ONLY=1 ;;
        0|false) LATEST_ONLY=0 ;;
        *)
            echo "CUDA_PYTHON_DOCS_LATEST_ONLY must be true, false, 1, or 0" >&2
            exit 1
            ;;
    esac
else
    echo "usage: ./build_docs.sh [latest-only|moon-ci]" >&2
    exit 1
fi

if [[ "${LATEST_ONLY}" == 1 && -z "${BUILD_PREVIEW:-}" && -z "${BUILD_LATEST:-}" ]]; then
    export BUILD_LATEST=1
fi

VERSION_VALUE=${!VERSION_ENV-}
if [[ -z "${VERSION_VALUE}" ]]; then
    VERSION_VALUE=$(python -c \
        "from importlib.metadata import version; import sys; value = version(sys.argv[1]); count = int(sys.argv[2]); print('.'.join(value.split('.')[:count]) if count else value)" \
        "${DISTRIBUTION}" "${VERSION_COMPONENTS}")
    VERSION_VALUE=${VERSION_VALUE%%+*}
fi
case "${VERSION_VALUE}" in
    ""|.|..|latest|*/*)
        echo "${VERSION_ENV} must name a safe version directory other than latest" >&2
        exit 1
        ;;
esac
export "${VERSION_ENV}=${VERSION_VALUE}"

BUILD_DIR="${DOCS_DIR}/build"
if [[ -L "${BUILD_DIR}" || ( -e "${BUILD_DIR}" && ! -d "${BUILD_DIR}" ) ]]; then
    echo "refusing to use non-directory docs build output: ${BUILD_DIR}" >&2
    exit 1
fi
if [[ "${MOON_CI}" == 1 ]]; then
    rm -rf -- "${BUILD_DIR}"
fi

EFFECTIVE_SPHINXOPTS=${DEFAULT_SPHINXOPTS}
if [[ "${HONOR_SPHINXOPTS}" == 1 && -n "${SPHINXOPTS:-}" ]]; then
    EFFECTIVE_SPHINXOPTS=${SPHINXOPTS}
fi
SPHINXOPTS="${EFFECTIVE_SPHINXOPTS}" make html

BUILD_HTML="${BUILD_DIR}/html"
VERSION_OUTPUT="${BUILD_HTML}/${VERSION_VALUE}"
if [[ -L "${BUILD_HTML}" || ! -d "${BUILD_HTML}" ]]; then
    echo "documentation output not found: ${BUILD_HTML}" >&2
    exit 1
fi
if [[ -L "${VERSION_OUTPUT}" || ! -d "${VERSION_OUTPUT}" ]]; then
    echo "versioned documentation output not found: ${VERSION_OUTPUT}" >&2
    exit 1
fi

for metadata_file in "${METADATA_FILES[@]}"; do
    cp -- "${DOCS_DIR}/${metadata_file}" "${BUILD_HTML}/"
done
cp -- "${SOURCE_DIR}/_templates/main.html" "${BUILD_HTML}/index.html"

LATEST_OUTPUT="${BUILD_HTML}/latest"
if [[ -L "${LATEST_OUTPUT}" || ( -e "${LATEST_OUTPUT}" && ! -d "${LATEST_OUTPUT}" ) ]]; then
    echo "refusing to replace non-directory latest docs output: ${LATEST_OUTPUT}" >&2
    exit 1
fi
rm -rf -- "${LATEST_OUTPUT}"
if [[ "${LATEST_ONLY}" == 0 ]]; then
    cp -a -- "${VERSION_OUTPUT}" "${LATEST_OUTPUT}"
else
    mv -- "${VERSION_OUTPUT}" "${LATEST_OUTPUT}"
fi

cp -- "${LATEST_OUTPUT}/objects.inv" "${BUILD_HTML}/"

if [[ "${CLEAN_GENERATED}" == 1 ]]; then
    GENERATED_DIR="${SOURCE_DIR}/generated"
    if [[ -L "${GENERATED_DIR}" ]]; then
        echo "refusing to remove symlinked generated docs directory: ${GENERATED_DIR}" >&2
        exit 1
    fi
    rm -rf -- "${GENERATED_DIR}"
fi
