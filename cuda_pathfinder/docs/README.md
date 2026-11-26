# Build the documentation

1. Install the `cuda-pathfinder` package of the version that we need to document.
2. Build the docs with `./build_docs.sh`.
3. The html artifacts should be available under both `./build/html/latest` and
   `./build/html/<version>`.

A local build does not require a version-selector entry. Add the version to
[`nv-versions.json`](./nv-versions.json) on `main` only after the package is
published and its versioned documentation is deployed.

Alternatively, we can build all the docs at once by running [`cuda_python/docs/build_all_docs.sh`](../../cuda_python/docs/build_all_docs.sh).

To publish the docs with the built version, it is important to note that the html files of older versions
should be kept intact, in order for the version selection (through `nv-versions.json`) to work.
