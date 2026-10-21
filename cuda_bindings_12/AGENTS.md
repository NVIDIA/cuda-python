# CUDA 12 maintenance bindings

This root maintains the CUDA 12.9 release line. Follow the repository-wide
instructions and [maintenance guide](MAINTENANCE.md).

- Preserve the CUDA 12 generated source, build-time header parser, `.in`
  templates, legacy APIs, and namespace redirector. The CUDA 13 source layout
  and runtime extension list are different and must not replace these.
- Do not hand-edit generated bindings. Change cybind and regenerate against
  the intended CUDA 12.9 inputs, recording provenance as described in
  `../ci/README.md`.
- Source builds require CUDA 12.9 headers. `build_hooks.py` checks the first
  `cuda.h` found through `CUDA_HOME` or `CUDA_PATH` before generating files.
  Split toolkit installations remain supported through the existing path list.
- The shared toolchain and Cython cache helpers live in `_build_shared.py`,
  linked to the current bindings root. Keep each package-local helper a symlink
  or byte-for-byte copy.
- Assess handwritten fixes and build changes for both supported roots.
  Preserve intentional differences in generated APIs and packaging.
- Use the package's `test` dependency group for Python tests. Build Cython tests
  with `tests/cython/build_tests.sh` or `build_tests.bat` before running them.
