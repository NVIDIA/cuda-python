This file describes `cuda_pathfinder`, a Python sub-package of
[cuda-python](https://github.com/NVIDIA/cuda-python). It locates and loads
NVIDIA dynamic libraries (CTK, third-party, and driver) across Linux and
Windows.

## Scope and principles

- **Language**: all implementation code in this package is pure Python.
- **Public API**: keep user-facing imports stable via `cuda.pathfinder`.
  Internal modules should stay under `cuda.pathfinder._*`.
- **Behavior**: loader behavior must remain deterministic and explicit. Avoid
  "best effort" silent fallbacks that mask why discovery/loading failed.
- **Cross-platform**: preserve Linux and Windows behavior parity unless a change
  is explicitly platform-scoped.

## Package architecture

- **Descriptor source-of-truth**: `cuda/pathfinder/_dynamic_libs/descriptor_catalog.py`
  defines canonical metadata for known libraries.
- **Registry layers**:
  - `lib_descriptor.py` builds the name-keyed runtime registry from the catalog.
  - `supported_nvidia_libs.py` keeps legacy exported tables derived from the
    catalog for compatibility.
- **Search pipeline**:
  - `search_steps.py` implements composable find steps (`site-packages`,
    `CONDA_PREFIX`, `CUDA_HOME`/`CUDA_PATH`, canary-assisted CTK root flow).
  - `search_platform.py` and `platform_loader.py` isolate OS-specific logic.
- **Load orchestration**:
  - `load_nvidia_dynamic_lib.py` coordinates find/load phases, dependency
    loading, driver-lib fast path, and cache semantics.
- **Process isolation helper**:
  - `cuda/pathfinder/_utils/spawned_process_runner.py` is used where process
    global dynamic loader state would otherwise leak across tests.

## Editing guidance

- **Edit authored descriptors, not derived tables**: when adding/changing a
  library, update `descriptor_catalog.py` first; keep derived exports in sync
  through existing conversion logic and tests.
- **Respect cache semantics**: `load_nvidia_dynamic_lib` is cached. Never add
  behavior that closes returned handles or assumes repeated fresh loads.
- **Keep error contracts intact**:
  - unknown name -> `DynamicLibUnknownError`
  - known but unsupported on this OS -> `DynamicLibNotAvailableError`
  - known/supported but not found/loadable -> `DynamicLibNotFoundError`
- **Do not hardcode host assumptions**: avoid baking in machine-local paths,
  shell-specific quoting, or environment assumptions.
- **Prefer focused abstractions**: if a change is platform-specific, route it
  through existing platform abstraction points instead of branching in many call
  sites.

## Testing expectations

- **Primary command**: run `pytest tests/` from `cuda_pathfinder/`.
- **Real-loading tests**: prefer spawned child-process tests for actual dynamic
  loading behavior; avoid in-process cross-test interference.
- **Cache-aware tests**: if a test patches internals used by
  `load_nvidia_dynamic_lib`, call `load_nvidia_dynamic_lib.cache_clear()`.
- **Negative-case names**: use obviously fake names (for example
  `"not_a_real_lib"`) in unknown/invalid-lib tests.
- **INFO summary in CI logs**: use `info_summary_append` for useful
  test-context lines visible in terminal summaries.
- **Strictness toggle**:
  `CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS` controls whether
  missing libraries are tolerated (`see_what_works`) or treated as failures
  (`all_must_work`).

## Useful commands

- Run package tests: `pytest tests/`
- Run package tests via orchestrator: `../scripts/run_tests.sh pathfinder`
- Build package docs: `cd docs && ./build_docs.sh`
