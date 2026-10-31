# Build the documentation

1. Install the CUDA 13 `cuda-bindings` build that you need to document.
2. Ensure the version is included in the [`nv-versions.json`](./nv-versions.json).
3. Build the docs with `./build_docs.sh`.
4. The html artifacts should be available under both `./build/html/latest` and `./build/html/<version>`.

[`cuda_python/docs/build_all_docs.sh`](../../cuda_python/docs/build_all_docs.sh)
builds this current bindings line together with the metapackage, Core, and
Pathfinder documentation. Build [CUDA 12 maintenance documentation](../../cuda_bindings_12/docs/README.md)
separately in its own environment; both roots install the same `cuda-bindings`
distribution and `cuda.bindings` namespace.

To publish the docs with the built version, it is important to note that the html files of older versions
should be kept intact, in order for the version selection (through `nv-versions.json`) to work.
