#!/bin/bash

cd "$(dirname "$0")"
CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH cythonize -3 -i test_*.pyx
