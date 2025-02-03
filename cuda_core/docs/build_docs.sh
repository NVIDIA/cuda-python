#!/bin/bash

set -ex

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

# build the docs (in parallel)
SPHINXOPTS="-j 4 -d build/.doctrees" make html

# for debugging/developing (conf.py), please comment out the above line and
# use the line below instead, as we must build in serial to avoid getting
# obsecure Sphinx errors
#SPHINXOPTS="-v" make html

# to support version dropdown menu
cp ./versions.json build/html

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
