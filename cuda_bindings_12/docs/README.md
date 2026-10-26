# Build the documentation

Use a separate environment from CUDA 13 documentation builds: both roots
install the same `cuda-bindings` distribution and `cuda.bindings` namespace.
The aggregate `cuda_python/docs/build_all_docs.sh` script builds the current
CUDA 13 line, so build this maintenance documentation separately.

1. Install the CUDA 12 `cuda-bindings` build that you need to document.
2. Ensure the version is included in the canonical
   [`nv-versions.json`](../../cuda_bindings_13/docs/nv-versions.json).
3. Build the docs with `./build_docs.sh`.
4. The html artifacts should be available under both `./build/html/latest` and `./build/html/<version>`.

The Pixi `docs` environment currently installs a published `12.9.*` package.
To document APIs changed in the checkout, install the matching source build
in your documentation environment before running `./build_docs.sh`.

`build/html/latest` is for local preview only. Follow the selected branch's
release procedure for publication. Maintenance publication must exclude `latest`
and the shared version-selector files so it cannot overwrite the current-major
site. Keep older versioned pages intact; do not publish the entire local build
directory over the shared site.
