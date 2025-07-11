#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

set -ex

# build cuda-python docs
rm -rf build
./build_docs.sh $@

# build cuda-bindings docs
CUDA_BINDINGS_PATH=build/html/cuda-bindings
mkdir -p $CUDA_BINDINGS_PATH
pushd .
cd ../../cuda_bindings/docs
rm -rf build
./build_docs.sh $@
cp -r build/html/* "$(dirs -l +1)"/$CUDA_BINDINGS_PATH
popd

# build cuda-core docs
CUDA_CORE_PATH=build/html/cuda-core
mkdir -p $CUDA_CORE_PATH
pushd .
cd ../../cuda_core/docs
rm -rf build
./build_docs.sh $@
cp -r build/html/* "$(dirs -l +1)"/$CUDA_CORE_PATH
popd
