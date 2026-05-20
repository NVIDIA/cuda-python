This file describes `cuda_core`, the high-level Pythonic CUDA subpackage in the
`cuda-python` monorepo.

## Scope and principles

- **Role**: provide higher-level CUDA abstractions (`Device`, `Stream`,
  `Program`, `Linker`, memory resources, graphs) on top of `cuda.bindings`.
- **API intent**: keep interfaces Pythonic while preserving explicit CUDA
  behavior and error visibility.
- **API stability**: `cuda_core` is v1.0+; avoid breaking public APIs. Prefer
  compatibility/deprecation paths and document intentional public changes in
  docs and release notes.
- **Compatibility**: changes should remain compatible with the supported CUDA
  major-version matrix.

## Package architecture

- **Main package**: `cuda/core/` contains most Cython modules (`*.pyx`, `*.pxd`)
  implementing runtime behaviors and public objects.
- **Subsystems**:
  - memory/resource stack: `cuda/core/_memory/`
  - system-level APIs: `cuda/core/system/`
  - compile/link path: `_program.pyx`, `_linker.pyx`, `_module.pyx`
  - execution path: `_launcher.pyx`, `_launch_config.pyx`, `_stream.pyx`
- **C++ helpers**: module-specific C++ implementations live under
  `cuda/core/_cpp/`.
- **Build backend**: `build_hooks.py` handles Cython extension setup and build
  dependency wiring.

## Build and version coupling

- `build_hooks.py` determines CUDA major version from `CUDA_CORE_BUILD_MAJOR`
  or CUDA headers (`CUDA_HOME`/`CUDA_PATH`) and uses it for build decisions.
- Source builds require CUDA headers available through `CUDA_HOME` or
  `CUDA_PATH`.
- `cuda_core` expects `cuda.bindings` to be present and version-compatible.

## Testing expectations

- **Primary tests**: `pytest tests/`
- **Cython tests**:
  - build: `tests/cython/build_tests.sh` (or platform equivalent)
  - run: `pytest tests/cython/`
- **Examples**: validate affected examples in `examples/` when changing user
  workflows or public APIs.
- **Orchestrated run**: from repo root, `scripts/run_tests.sh core`.

## Runtime/build environment notes

- Runtime env vars commonly relevant:
  - `CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`
  - `CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING`
- Build env vars commonly relevant:
  - `CUDA_HOME` / `CUDA_PATH`
  - `CUDA_CORE_BUILD_MAJOR`
  - `CUDA_PYTHON_PARALLEL_LEVEL`
  - `CUDA_PYTHON_COVERAGE`

## Editing guidance

- Keep user-facing behaviors coherent with docs and examples, especially around
  stream semantics, memory ownership, and compile/link flows.
- Reuse existing shared utilities in `cuda/core/_utils/` before adding new
  helpers.
- When changing Cython signatures or cimports, verify related `.pxd` and
  call-site consistency.
- Prefer explicit error propagation over silent fallback paths.
- If you change public behavior, update tests and docs under `docs/source/`.
- For new public APIs or broad feature work, sketch the API and behavior in an
  issue/design discussion before opening a large implementation PR. Reviewers
  often block major `cuda_core` features until API shape, compatibility impact,
  examples, and docs/release-note coverage are clear.
- Feature availability checks should query CUDA driver/device capabilities
  instead of hard-coding broad platform skips. Prefer properties such as
  capability flags over assumptions like "Windows", "Linux", or "WSL".
- Preserve compatibility with the supported CUDA major-version matrix. Do not
  directly cimport newly generated binding symbols unless older supported
  CUDA-major builds are gated or have a wrapper/fallback path.
- Resource and context-manager code must preserve stream ordering, ownership,
  and exception semantics. `close()`/cleanup paths should use the stream that
  established the resource ordering, and `__exit__` should avoid masking a
  user's original exception where practical.
- Tests should cover the behavior users exercise, not just private helpers.
  Avoid large module-stubbing tests for simple implementation choices; prefer
  focused regressions around the public API or the smallest stable internal
  boundary.
