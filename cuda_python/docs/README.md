# Build the documentation

1. Ensure the version is included in the [`versions.json`](./versions.json).
2. Build the docs with `./build_docs.sh`.
3. The html artifacts should be available under both `./build/html/latest` and `./build/html/<version>`.

Alternatively, we can build all the docs at once by running [`./build_all_docs.sh`](./build_all_docs.sh).

To publish the docs with the built version, it is important to note that the html files of older versions
should be kept intact, in order for the version selection (through `versions.json`) to work.
