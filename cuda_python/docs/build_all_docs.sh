#!/bin/bash

set -ex

# build cuda-python docs
./build_docs.sh

# build cuda-bindings docs
mkdir -p build/html/cuda-bindings
pushd .
cd ../../cuda_bindings/docs
./build_docs.sh
cp -r build/html/* "$(dirs +1)"/build/html/cuda-bindings
popd

# build cuda-core docs
mkdir -p build/html/cuda-core
pushd .
cd ../../cuda_core/docs
./build_docs.sh
cp -r build/html/* "$(dirs +1)"/build/html/cuda-core
popd
