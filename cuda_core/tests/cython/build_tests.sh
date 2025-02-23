#!/bin/bash

SCRIPTPATH=$(dirname $(realpath "$0"))
CPLUS_INCLUDE_PATH=$SCRIPTPATH/../../cuda/core/experimental/include:$CUDA_HOME/include:$CPLUS_INCLUDE_PATH cythonize -3 -i $(dirname "$0")/test_*.pyx
