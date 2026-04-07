#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [[ "$#" == "0" ]]; then
    LATEST_ONLY="0"
elif [[ "$#" == "1" && "$1" == "latest-only" ]]; then
    LATEST_ONLY="1"
else
    echo "usage: ./build_docs.sh [latest-only]"
    exit 1
fi

# SPHINX_CUDA_CORE_VER is used to create a subdir under build/html
# (the Makefile file for sphinx-build also honors it if defined)
if [[ -z "${SPHINX_CUDA_CORE_VER}" ]]; then
    export SPHINX_CUDA_CORE_VER=$(python -c "from importlib.metadata import version; print(version('cuda-core'))" \
                                  | awk -F'+' '{print $1}')
fi

if [[ "${LATEST_ONLY}" == "1" && -z "${BUILD_PREVIEW:-}" && -z "${BUILD_LATEST:-}" ]]; then
    export BUILD_LATEST=1
fi

# build the docs. Allow callers to override SPHINXOPTS for serial/debug runs.
if [[ -z "${SPHINXOPTS:-}" ]]; then
    SPHINXOPTS="-j 4 -d build/.doctrees"
fi
make html

# to support version dropdown menu
cp ./versions.json build/html
cp ./nv-versions.json build/html

# to have a redirection page (to the latest docs)
cp source/_templates/main.html build/html/index.html

# ensure that the latest docs is the one we built
if [[ $LATEST_ONLY == "0" ]]; then
    cp -r build/html/${SPHINX_CUDA_CORE_VER} build/html/latest
else
    mv build/html/${SPHINX_CUDA_CORE_VER} build/html/latest
fi

# ensure that the Sphinx reference uses the latest docs
cp build/html/latest/objects.inv build/html

# clean up previously auto-generated files
rm -rf source/generated/
