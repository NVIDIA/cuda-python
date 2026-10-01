This file describes `cuda_python`, the metapackage layer in the `cuda-python`
monorepo.

## Scope

- `cuda_python` is primarily packaging and documentation glue.
- It does not host substantial runtime APIs like `cuda_core`,
  `cuda_bindings`, or `cuda_pathfinder`.

## Main files to edit

- `pyproject.toml`: project metadata and dynamic dependency declaration.
- `setup.py`: dynamic dependency pinning logic for matching `cuda-bindings`
  versions (release vs pre-release behavior).
- `docs/`: top-level docs build/aggregation scripts.

## Editing guidance

- Keep this package lightweight; prefer implementing runtime features in the
  component packages rather than here.
- Be careful when changing dependency/version logic in `setup.py`; preserve
  compatibility between metapackage versioning and subpackage constraints.

## Release coupling

- `setup.py` pins `cuda-core` to a minor series (`cuda-core~=X.Y.0`). Every
  `cuda-core` release must bump that pin, and a `cuda-python` release should
  follow right after, so that `pip install cuda-python` resolves to the new
  `cuda-core`. The `cuda-core` release checklist (`.github/RELEASE-core.md`)
  has this step.
- Users should pin `cuda-python` alone. A separate `cuda-core` pin next to it
  can make the install unresolvable after a `cuda-core` release.
- If you update docs structure, ensure `docs/build_all_docs.sh` still collects
  docs from `cuda_python`, `cuda_bindings`, `cuda_core`, and `cuda_pathfinder`.
