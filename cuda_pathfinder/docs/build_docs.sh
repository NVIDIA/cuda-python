#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

MOON_CI="0"
if [[ "$#" == "0" ]]; then
    LATEST_ONLY="0"
elif [[ "$#" == "1" && "$1" == "latest-only" ]]; then
    LATEST_ONLY="1"
elif [[ "$#" == "1" && "$1" == "moon-ci" ]]; then
    MOON_CI="1"
    DOCS_LATEST_ONLY="${CUDA_PYTHON_DOCS_LATEST_ONLY:-true}"
    case "${DOCS_LATEST_ONLY,,}" in
        1|true) LATEST_ONLY="1" ;;
        0|false) LATEST_ONLY="0" ;;
        *)
            echo "CUDA_PYTHON_DOCS_LATEST_ONLY must be true, false, 1, or 0" >&2
            exit 1
            ;;
    esac
else
    echo "usage: ./build_docs.sh [latest-only|moon-ci]"
    exit 1
fi

if [[ "${MOON_CI}" == "1" ]]; then
    if [[ -L build || ( -e build && ! -d build ) ]]; then
        echo "refusing to replace non-directory docs build output: ${SCRIPT_DIR}/build" >&2
        exit 1
    fi
    rm -rf build
fi

# SPHINX_CUDA_PATHFINDER_VER is used to create a subdir under build/html
# (the Makefile file for sphinx-build also honors it if defined).
# If there's a post release (ex: .post1) we don't want it to show up in the
# version selector or directory structure.
if [[ -z "${SPHINX_CUDA_PATHFINDER_VER}" ]]; then
    export SPHINX_CUDA_PATHFINDER_VER=$(python -c "from importlib.metadata import version; \
                                                 ver = '.'.join(str(version('cuda-pathfinder')).split('.')[:3]); \
                                                 print(ver)" \
                                      | awk -F'+' '{print $1}')
fi

if [[ "${LATEST_ONLY}" == "1" && -z "${BUILD_PREVIEW:-}" && -z "${BUILD_LATEST:-}" ]]; then
    export BUILD_LATEST=1
fi

# build the docs (in parallel)
if [[ -z "${SPHINXOPTS:-}" ]]; then
    HTML_SPHINXOPTS="-W --keep-going -j 4 -d build/.doctrees"
else
    HTML_SPHINXOPTS="${SPHINXOPTS}"
fi
SPHINXOPTS="${HTML_SPHINXOPTS}" make html

# for debugging/developing (conf.py), please comment out the above line and
# use the line below instead, as we must build in serial to avoid getting
# obsecure Sphinx errors
#SPHINXOPTS="-v" make html

# to support version dropdown menu
cp ./nv-versions.json build/html

# to have a redirection page (to the latest docs)
cp source/_templates/main.html build/html/index.html

# ensure that the latest docs is the one we built
if [[ $LATEST_ONLY == "0" ]]; then
    cp -r build/html/${SPHINX_CUDA_PATHFINDER_VER} build/html/latest
else
    mv build/html/${SPHINX_CUDA_PATHFINDER_VER} build/html/latest
fi

# ensure that the Sphinx reference uses the latest docs
cp build/html/latest/objects.inv build/html

if [[ "${MOON_CI}" == "1" ]]; then
    SOURCE="${SCRIPT_DIR}/build/html"
    OUTPUT_ROOT="${SCRIPT_DIR}/../.moon-out"
    OUTPUT="${OUTPUT_ROOT}/docs-ci"
    if [[ -L "${SOURCE}" || ! -d "${SOURCE}" ]]; then
        echo "documentation output not found: ${SOURCE}" >&2
        exit 1
    fi
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
    cp -aL "${SOURCE}/." "${OUTPUT}/"
fi
