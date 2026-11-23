# Build the documentation

1. Build the docs with `./build_docs.sh`.
2. The html artifacts should be available under both `./build/html/latest` and
   `./build/html/<version>`.

A local build does not require a version-selector entry. Add the version to
[`nv-versions.json`](./nv-versions.json) on `main` only after the package is
published and its versioned documentation is deployed.

[`./build_all_docs.sh`](./build_all_docs.sh) builds the metapackage, current
CUDA 13 bindings, Core, and Pathfinder documentation together. It does not
build the [CUDA 12 maintenance documentation](../../cuda_bindings_12/docs/README.md).
Build that separately in its own environment: both bindings roots install the
same `cuda-bindings` distribution and `cuda.bindings` namespace.

When building the docs, some (but not all) of the urls can be rendered/examined locally by setting the environment
variable `CUDA_PYTHON_DOMAIN` as follows:
```shell
CUDA_PYTHON_DOMAIN="http://localhost:1234/" ./build_all_docs.sh
python -m http.server -d build/html 1234
```
If the docs are built on a remote machine, you can set up the ssh tunnel in a separate terminal session
via
```shell
ssh -L 1234:localhost:1234 username@hostname
```
Then browse the built docs by visiting `http://localhost:1234/` on a local machine.

Keep older versioned HTML pages intact so existing version-selector entries
continue to work.
