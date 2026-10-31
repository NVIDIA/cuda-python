# CUDA 12 Maintenance Bindings

This root maintains CUDA 12.9 bindings under the `cuda.bindings` namespace.
Follow the repository-wide instructions, including
[Changes across bindings majors](../AGENTS.md#changes-across-bindings-majors).
Inspect the CUDA 13 counterpart during implementation and review; preserve
intentional API, layout, and build differences.

## Generated and handwritten sources

- Files carrying a `CYTHON-BINDINGS-GENERATED` seal are generated inputs.
  Change their generation source and regenerate against CUDA 12.9 inputs.
  Record generator revision, toolkit inputs, and affected files. Do not edit
  sealed files or recalculate their seals to conceal manual changes.
- Handwritten utilities, build helpers, tests, and documentation can be edited
  directly. Inspect headers and provenance; a filename extension alone does
  not identify generated code.
- Source builds parse CUDA headers and expand `.in` templates before Cython.
  Regenerate these outputs from their templates and headers. Preserve the
  CUDA 12 parser, legacy APIs, and namespace redirector.

## Build and test guidance

- Use CUDA 12.9 headers and a separate Python environment from CUDA 13.
- `_build_shared.py` links to the canonical helper in `cuda_bindings_13/`.
  Preserve its symlink or byte-for-byte identity with the shared copies.
- Use the package's [developer commands](README.md). Build the Cython test
  extensions before running `tests/cython/`.
- Legacy samples remain under `examples/`; CUDA 13 uses a different layout.
- Run the SCM preview bootstrap only in a disposable build checkout. Never
  push its synthetic tags or run it in a shared development repository.
