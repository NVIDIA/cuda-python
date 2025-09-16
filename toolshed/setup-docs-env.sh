#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Setup a local conda environment for building the sphinx docs to mirror the CI environment
# (see cuda_python/docs/environment-docs.yml).
#
# Usage:
#   ./toolshed/setup-docs-env.sh
#
# Notes:
# - Requires an existing Miniforge/Conda install and `conda` on PATH.
# - Installs the same packages as CI’s environment-docs.yml.

set -euo pipefail

ENV_NAME="cuda-python-docs"
PYVER="3.12"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# --- sanity checks -----------------------------------------------------------
if ! have_cmd conda; then
    echo "ERROR: 'conda' not found on PATH. Please ensure Miniforge is installed and initialized." >&2
    exit 1
fi

# Load conda's shell integration into this bash process
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "⚠  Environment '${ENV_NAME}' already exists → NO ACTION"
    exit 0
fi

echo "Creating environment '${ENV_NAME}'…"
# ATTENTION: This dependency list is duplicated in
#            cuda_python/docs/environment-docs.yml. Please KEEP THEM IN SYNC!
conda create -y -n "${ENV_NAME}" \
    "python=${PYVER}" \
    cython \
    myst-parser \
    numpy \
    numpydoc \
    pip \
    pydata-sphinx-theme \
    pytest \
    scipy \
    "sphinx<8.2.0" \
    sphinx-copybutton \
    myst-nb \
    enum_tools \
    sphinx-toolbox \
    pyclibrary

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install nvidia-sphinx-theme

echo
echo "✅ Environment '${ENV_NAME}' is ready."
echo
echo "Build docs with e.g.:"
echo "    conda activate ${ENV_NAME}"
echo "    cd cuda_pathfinder/"
echo "    pip install -e ."
echo "    (cd docs/ && rm -rf build source/generated && ./build_docs.sh)"
