#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    HTML_SPHINXOPTS="-W --keep-going -j 4 -d build/.doctrees"
else
    HTML_SPHINXOPTS="${SPHINXOPTS}"
fi
SPHINXOPTS="${HTML_SPHINXOPTS}"
make html

if [[ "${DOCS_LINKCHECK:-0}" == "1" ]]; then
    if [[ -n "${CUDA_PYTHON_DOCS_GITHUB_REF:-}" ]]; then
        DOCS_EXAMPLES_REF="${CUDA_PYTHON_DOCS_GITHUB_REF}"
    elif [[ "${BUILD_PREVIEW:-0}" == "1" || "${BUILD_LATEST:-0}" == "1" ]]; then
        DOCS_EXAMPLES_REF="main"
    else
        DOCS_EXAMPLES_REF="cuda-core-v${SPHINX_CUDA_CORE_VER}"
    fi
    python ../../cuda_python/docs/check_example_links.py \
        --source-dir source \
        --examples-root cuda_core/examples \
        --expected-ref "${DOCS_EXAMPLES_REF}" \
        --placeholder cuda_core_github_ref
fi

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
