# Build the documentation

1. Install the `cuda-bindings` package of the version that we need to document.
2. Ensure the version is included in the [`nv-versions.json`](./nv-versions.json).
3. Build the docs with `./build_docs.sh`.
4. The html artifacts should be available under both `./build/html/latest` and `./build/html/<version>`.

Alternatively, we can build all the docs at once by running [`cuda_python/docs/build_all_docs.sh`](../../cuda_python/docs/build_all_docs.sh).

To publish the docs with the built version, it is important to note that the html files of older versions
should be kept intact, in order for the version selection (through `nv-versions.json`) to work.
