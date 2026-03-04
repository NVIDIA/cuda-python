# cuda_pathfinder agent instructions

You are working on `cuda_pathfinder`, a Python sub-package of the
[cuda-python](https://github.com/NVIDIA/cuda-python) monorepo. It finds and
loads NVIDIA dynamic libraries (CTK, third-party, and driver) across Linux and
Windows.

## Workspace

The workspace root is `cuda_pathfinder/` inside the monorepo. Use the
`working_directory` parameter on the Shell tool when you need the monorepo root
(one level up).

## Conventions

- **Python**: all source is pure Python (no Cython in this sub-package).
- **Testing**: `pytest` with `pytest-mock` (`mocker` fixture). Use
  `spawned_process_runner` for real-loading tests that need process isolation
  (dynamic linker state leaks across tests otherwise). Use the
  `info_summary_append` fixture to emit `INFO` lines visible in CI/QA logs.
- **STRICTNESS env var**: `CUDA_PATHFINDER_TEST_LOAD_NVIDIA_DYNAMIC_LIB_STRICTNESS`
  controls whether missing libs are tolerated (`see_what_works`, default) or
  fatal (`all_must_work`).
- **Formatting/linting**: rely on pre-commit (runs automatically on commit). Do
  not run formatters manually.
- **Imports**: use `from cuda.pathfinder._dynamic_libs...` for internal imports
  in tests; public API is `from cuda.pathfinder import load_nvidia_dynamic_lib`.

## Testing guidelines

- **Real tests over mocks**: mocks are fine for hard-to-reach branches (e.g.
  24-bit Python), but every loading path must also have a real-loading test that
  runs in a spawned child process. Track results with `INFO` lines so CI logs
  show what actually loaded.
- **No real lib names in negative tests**: when parametrizing unsupported /
  invalid libnames, use obviously fake names (`"bogus"`, `"not_a_real_lib"`)
  to avoid confusion when searching the codebase.
- **`functools.cache` awareness**: `load_nvidia_dynamic_lib` is cached. Tests
  that patch internals it depends on must call
  `load_nvidia_dynamic_lib.cache_clear()` first, or use a child process for
  isolation.

## Key modules

- `cuda/pathfinder/_dynamic_libs/load_nvidia_dynamic_lib.py` -- main entry
  point and dispatch logic (CTK vs driver).
- `cuda/pathfinder/_dynamic_libs/supported_nvidia_libs.py` -- canonical
  registry of sonames, DLLs, site-packages paths, and dependencies.
- `cuda/pathfinder/_dynamic_libs/find_nvidia_dynamic_lib.py` -- CTK search
  cascade (site-packages, conda, CUDA_HOME).
- `tests/child_load_nvidia_dynamic_lib_helper.py` -- lightweight helper
  imported by spawned child processes (avoids re-importing the full test
  module).

### Fix all code review findings from lib-descriptor-refactor review

**Request:** Fix all 8 findings from the external code review.

**Actions (in worktree `cuda_pathfinder_refactor`):**
1. `search_steps.py`: Restored `os.path.normpath(dirname)` in
   `_find_lib_dir_using_anchor` (regression from pre-refactor fix). Added
   `NoReturn` annotation to `raise_not_found`.
2. `search_platform.py`: Guarded `os.listdir(lib_dir)` in
   `WindowsSearchPlatform.find_in_lib_dir` with `os.path.isdir` check to
   prevent crash on missing directory.
3. `test_descriptor_catalog.py`: Rewrote tautological tests as structural
   invariant tests (uniqueness, valid names, strategy values, dep graph,
   soname/dll format, driver lib constraints). 237 new parametrized cases.
4. `platform_loader.py`: Eliminated `WindowsLoader`/`LinuxLoader` boilerplate
   classes — assign the platform module directly as `LOADER`. Removed stale
   `type: ignore`.
5. `descriptor_catalog.py`: Trimmed default-valued fields from all entries,
   added `# ---` section comments (CTK / third-party / driver).
6. `load_nvidia_dynamic_lib.py`: Fixed import layout — `TYPE_CHECKING` block
   now properly separated after unconditional imports.

All 742 tests pass, all pre-commit hooks green.
