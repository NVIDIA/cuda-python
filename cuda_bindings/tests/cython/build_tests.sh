#!/bin/bash

CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH cythonize -3 -i $(dirname "$0")/test_*.pyx
