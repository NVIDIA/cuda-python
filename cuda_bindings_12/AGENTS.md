# CUDA 12 Maintenance Bindings

This root maintains CUDA 12.9 bindings under the `cuda.bindings` namespace.
Follow the repository-wide instructions, including
[Changes across bindings majors](../AGENTS.md#changes-across-bindings-majors).
Inspect the CUDA 13 counterpart during implementation and review, and record
matching changes or why they are not applicable. Preserve intentional API,
layout, and build differences rather than copying the CUDA 13 root wholesale.

## Generated and handwritten sources

- Files carrying a `CYTHON-BINDINGS-GENERATED` seal are cybind output,
  including some `.pyx`, `.pxd`, and `.in` files. Change their generation source
  and regenerate against the intended CUDA 12.9 inputs. Record the generator
  revision, toolkit inputs, and affected files in the review summary. Do not
  edit sealed files or recalculate their seals to conceal manual changes.
- Handwritten files, such as `cuda/bindings/_internal/utils.pyx`, build helpers,
  tests, and documentation, can be edited directly. Inspect the file's header
  and provenance; a filename extension alone does not identify generated code.
- Source builds also parse CUDA headers with pyclibrary and expand `.in`
  templates through `build_hooks.py` before Cythonization. Those generated
  `.pyx`, `.pxd`, and `.pxi` outputs must be regenerated from their templates
  and headers, rather than patched. Build-time expansion does not make a
  sealed input template handwritten.
- Keep the CUDA 12 header parser, templates, legacy APIs, and namespace
  redirector. The CUDA 13 extension list and generated layout are different.

## Build and test guidance

- Source builds require CUDA 12.9 headers. `build_hooks.py` checks the first
  `cuda.h` found through `CUDA_HOME` or `CUDA_PATH` before generating files.
  Split installations can still supply the existing path list.
- `_build_shared.py` links to the canonical helper in `cuda_bindings_13/`.
  Preserve the symlink or a byte-for-byte identical copy, including matching
  changes in the Core and Pathfinder helper copies.
- Use Python 3.11 or newer and a separate environment from CUDA 13 bindings.
  From this directory, install with
  `python -m pip install -v -e . --group test`, then run `python -m pytest tests/`.
  Free-threaded development also needs the `test-ft` dependency group.
- Build Cython tests with `tests/cython/build_tests.sh` or `build_tests.bat`
  before running `python -m pytest tests/cython/`. This requires the matching
  CUDA headers and compiler toolchain.
- Legacy samples remain under `examples/`; do not assume CUDA 13's sample
  layout applies to this root. Keep marked SYNCED test-bootstrap blocks equal
  across both bindings roots and Core.
