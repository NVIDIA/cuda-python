This file describes `cuda_bindings_13`, the low-level CUDA host API bindings
subpackage in the `cuda-python` monorepo.

## Scope and principles

- **Role**: provide low-level, close-to-CUDA interfaces under
  `cuda.bindings.*` with broad API coverage.
- **Style**: prioritize correctness and API compatibility over convenience
  wrappers. High-level ergonomics belong in `cuda_core`, not here.
- **Cross-platform**: preserve Linux and Windows behavior unless a change is
  intentionally platform-specific.

## Package architecture

- **Public module layer**: Cython modules under `cuda/bindings/` expose user
  APIs (`driver`, `runtime`, `nvrtc`, `nvjitlink`, `nvvm`, `cufile`, etc.).
- **Internal binding layer**: `cuda/bindings/_bindings/` provides lower-level
  glue and loader helpers used by public modules.
- **Platform internals**: `cuda/bindings/_internal/` contains
  platform-specific implementation files and support code.
- **Build backend**: `build_hooks.py` drives extension configuration and
  Cythonization. Logic shared with `cuda_core` and `cuda_pathfinder`
  (toolchain selection, the compiler flag set, the Cython cache helpers, the
  rebuild stamps and the release-tag pinning of README links) lives in
  `_build_shared.py`. `cuda_bindings_13/_build_shared.py` is canonical;
  the `cuda_core` and `cuda_pathfinder` copies must each be either a symlink to
  it or a byte-for-byte identical copy.

## Generated-source workflow

- **Do not hand-edit generated binding files**: many files under
  `cuda/bindings/` (including `*.pyx` and `*.pxd`) are generated artifacts.
- **Generated files are synchronized from another repository**: changes to these
  files in this repo are expected to be overwritten by the next sync.
- **If generated output must change**: make the change at the generation source
  and sync the updated artifacts back here, rather than patching generated files
  directly in this repo.
- **Platform split files**: keep `_linux.pyx` and `_windows.pyx` variants
  aligned when behavior should be equivalent.

## Testing expectations

- **Primary tests**: `pytest tests/`
- **Cython tests**:
  - build: `tests/cython/build_tests.sh` (or platform equivalent)
  - run: `pytest tests/cython/`
- **Samples**: run pytest-based sample coverage in `tests/example_tests/`;
  sample sources and instructions live in `../samples/cuda_bindings/`.
- **Benchmarks**: see the [benchmark README](../benchmarks/cuda_bindings/README.md)
  for the Pixi tasks that run the Python and C++ benchmarks.

The `tests/legacy_api` subdirectory tests the old pre-v2 APIs of `driver`, `runtime`
and `nvrtc`.  These test files should not be added to, only updated when
necessary to fix test failures.  The canonical set of tests are those outside of
the `tests/legacy_api` subdirectory.

Tests that create memory pools must cap them: set `max_size` (v2 types) or
`maxSize` (other types) on the pool properties object to `POOL_SIZE` from
`cuda_python_test_helpers.memory`. A pool created with a size of 0 reserves
virtual address space of about twice the device memory, which fails on systems
with a limited virtual address space. `ci/tools/check_mempool_hygiene.py`
(a pre-commit hook) enforces this for `.py` tests. It cannot parse Cython, so
cap pools in `tests/cython/*.pyx` by hand. The same module holds the shared
`supports_memory_pools`, `supports_managed_memory`, and
`xfail_graph_mempool_oom` helpers; reuse them rather than defining copies.

## Build and environment notes

- `CUDA_HOME` or `CUDA_PATH` must point to a valid CUDA Toolkit for source
  builds.
- The toolkit's `cuda.h` must have the same major.minor as the generated
  sources, `CUDA_VERSION` in `cuda/bindings/cydriver.pxd`. `build_hooks.py`
  checks this before cythonize and fails with a message that names both
  versions.
- `CUDA_PYTHON_PARALLEL_LEVEL` controls build parallelism.
- Runtime behavior is affected by
  `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM` and
  `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`.

## Editing guidance

- Keep CUDA return/error semantics explicit and avoid broad fallback behavior.
- Reuse existing helper layers (`_bindings`, `_internal`, `_lib`) before adding
  new one-off utilities.
- If you add or change exported APIs, update relevant docs under
  `docs/source/module/` and tests in `tests/`.
- Prefer changes that are easy to regenerate/rebuild rather than patching
  generated output directly.
