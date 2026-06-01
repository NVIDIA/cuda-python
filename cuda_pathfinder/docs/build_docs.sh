#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -ex

if [[ "$#" == "0" ]]; then
    LATEST_ONLY="0"
elif [[ "$#" == "1" && "$1" == "latest-only" ]]; then
    LATEST_ONLY="1"
else
    echo "usage: ./build_docs.sh [latest-only]"
    exit 1
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

if [[ "${DOCS_LINKCHECK:-0}" == "1" ]]; then
    if [[ -z "${LINKCHECK_SPHINXOPTS:-}" ]]; then
        LINKCHECK_SPHINXOPTS="-W --keep-going -j 4 -d build/.linkcheck-doctrees"
    fi
    SPHINXOPTS="${LINKCHECK_SPHINXOPTS}" make linkcheck BUILDDIR="build/linkcheck/${SPHINX_CUDA_PATHFINDER_VER}"
fi

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
