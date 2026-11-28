<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing CUDA bindings

## Choose the release scope and source

Develop the current and maintenance CUDA major lines together on `main`, in
the two package roots listed in [`ci/versions.yml`](../ci/versions.yml).
Release each line independently. A bindings release also needs its matching
`cuda-python` metapackage release; it does not require a release of the other
CUDA major, `cuda-core`, or `cuda-pathfinder`.

Select a tested source commit containing the intended package directory.
Record the source commit, intended tag, selected package root, release owner,
and accepted changes in the release checklist.

## Apply fixes across CUDA-major directories

CUDA 12 and CUDA 13 sources occupy separate directories on `main`. Assess
fixes in both directories and record the matching change or why it does not
apply. For handwritten changes, the contribution guide gives a
[path-relative patch example](../CONTRIBUTING.md#transferring-fixes-between-bindings-roots).
Regenerate generated files from their generation source and each directory's
intended toolkit inputs. Follow the
[shared regeneration and review guidance](../ci/README.md#maintaining-both-bindings-roots).

## Validate and publish the selected line

1. Finalize the bindings and metapackage release notes, dependency constraints,
   and selected toolkit pin in the release source tree. The registry and
   packaging metadata must agree. Run the CI-tool suite and pre-commit checks.
2. Test the candidate through a reviewed PR or manual CI dispatch on its
   branch. Normal development CI includes dependent packages and may cover
   both majors. Resolve failures in affected code before selecting the tag.
3. Create the approved immutable `vX.Y.Z` tag at the selected source commit.
   Tag-triggered CI selects that line's bindings and metapackage, validates them
   with the published Pathfinder dependency, and builds release artifacts.
   The tag does not have to be on `main`. Wait for this exact tag's CI to pass.
4. Release `cuda-bindings` before `cuda-python`. For each component, run
   **CI: Release** with `release-action=dry-run`, the tag, and the chosen
   `component`. Leave `run-id` blank to exercise automatic lookup, and leave
   `dry-run-docs-branch` blank to avoid
   deploying docs. The workflow verifies a successful tag-triggered CI run
   with the exact source SHA, validates artifacts, and builds docs.
   The `cuda-python` dry run also installs the candidate metapackage with
   dependencies from PyPI in clean environments across the tag's supported
   Python versions, including free-threaded builds, on Linux x86-64, Linux
   ARM64, and Windows x86-64. It tests both the default install and `[all]`
   using only binary wheels and runs `pip check`. Publish the matching
   bindings and any required Core or Pathfinder versions to PyPI first;
   unpublished CI dependency artifacts cannot satisfy this check.
5. Review that component's dry run, then separately approve and run
   `full-release` for the same component under the team's release procedure.
   A successful candidate rehearsal or ordinary branch CI is not a substitute
   for exact-tag CI.
   The metapackage dependency check runs again for `full-release` and must
   pass before GitHub asset uploads or TestPyPI/PyPI publication.
6. After both component releases succeed, review and publish the shared
   `vX.Y.Z` GitHub Release draft. Check its release notes, versioned
   documentation links, source archive, checksum, and wheels. These component
   workflows leave the draft for the release owner to publish.
7. [Publish both packages on conda-forge](#publish-on-conda-forge) for this
   release line.
8. Finish post-release QA and announcements and return every applicable fix
   to `main`.

For a bindings `vX.Y.Z` tag, tag CI validates that line's bindings documentation.
Validate the metapackage documentation in the separate `component=cuda-python`
release dry run. Tag CI does not publish documentation. This focused tag-CI
check is specific to the shared bindings/metapackage tags; Core and Pathfinder
release behavior is unchanged.

Release notes can land on `main` before publication without entries in the
documentation version selectors. For either bindings line, wait for successful
publication and deployment of each component's versioned documentation, then
open a PR on `main` adding the released bindings version to
[`cuda_bindings_13/docs/nv-versions.json`](../cuda_bindings_13/docs/nv-versions.json)
and its matching metapackage version to
[`cuda_python/docs/nv-versions.json`](../cuda_python/docs/nv-versions.json).
Confirm that each versioned documentation URL works before merging its entry.

Documentation deployments from `main` own these shared version selectors.
Release deployments publish versioned pages without replacing the selectors or
shared `latest` pages. An entry in the tagged source alone does not update the
live switcher. If an entry is missing after release, correct it on `main`
without changing the release tag.

The workflow revision used for a release is its **control revision**; the
tagged commit is its **source revision**. Record both and the artifact CI run
ID. Release routing uses the package layout and toolkit settings in
the release tag's `ci/versions.yml`, which must contain the
`schema_version: 2` package-root registry. Both bindings lines require
toolkit-qualified artifacts. The workflow control revision never supplies
missing tagged toolkit settings or replaces tagged release notes.

In **CI: Release**, normal bindings and metapackage releases each require
their own exact-version, nonempty release notes in the tagged source. Notes
from the workflow control revision or a different component cannot fill a gap.
Preserve their existing exemption for `.postN` releases, which do not require
new notes.

Local synthetic tags used in publication-incapable rehearsals must remain local;
they do not authorize publishing a package, creating a remote tag, or moving
an existing tag.

### Older release layouts

For tags without the package-root registry, run the release workflow from an
appropriate historical tag or maintenance branch, using that ref's own inputs.
Select a branch in Actions' **Run workflow** menu, or select a branch or tag
with the GitHub CLI's `--ref` option. Verify its release flow and artifact
availability before use. Older-tag docs rebuilds also need a caller on a
compatible ref: `build-docs.yml` is reusable and cannot be dispatched directly.

### Two tags at one commit

One tested source commit can carry tags for both bindings lines, for example
`v12.9.20` and `v13.5.4` (illustrative versions). Each tag starts its own
tag-triggered CI run, even though the source SHA is identical:

| Tag | Distributions selected for release |
| --- | --- |
| `v12.9.20` | `cuda-bindings` 12.9.20 and `cuda-python` 12.9.20 |
| `v13.5.4` | `cuda-bindings` 13.5.4 and `cuda-python` 13.5.4 |

CI explicitly selects the metapackage major with `CUDA_PYTHON_BUILD_MAJOR`
and its matching `v12.*` or `v13.*` SCM namespace. Toolkit-qualified artifact
names keep the lines separate, even at one SHA; tag CI may build additional
packages beyond the release selection.

Follow the procedure separately for each tag, invoking **CI: Release** for
both `component=cuda-bindings` and `component=cuda-python`: dry runs first,
then approved full releases. Run lookup matches the requested tag and source
SHA, and artifact validation rejects mismatched versions. Tag creation and
release order do not choose the version; releasing one line does not release
the other.

## Publish on conda-forge

The [cuda-python feedstock](https://github.com/conda-forge/cuda-python-feedstock)
builds both `cuda-bindings` and the `cuda-python` metapackage from one recipe.
Start after publishing the matching `vX.Y.Z` GitHub Release. Download its
`cuda-python-vX.Y.Z.tar.gz` release asset and its `.sha256sum` sidecar. Compute
the archive's SHA-256 and compare it with the digest in the sidecar. Use the
verified digest and the release asset URL below in the feedstock recipe.

```text
https://github.com/NVIDIA/cuda-python/releases/download/vX.Y.Z/cuda-python-vX.Y.Z.tar.gz
```

### Choose the feedstock PR

Check what has already been published and which release should go next. This
avoids duplicate updates, accidental recipe downgrades, and skipping a
release planned for conda-forge. Concretely, check the recipe's
`context.version` in `recipe/recipe.yaml` on the target feedstock branch
before requesting a bot update or opening a manual PR. If it matches the
requested version, check the existing build and verify that both packages
install from conda-forge. If the branch is already on a newer version,
check whether both requested packages are available and coordinate any
needed backfill with the feedstock maintainers. When updating to a newer
version, publish and verify any earlier planned release first.

- **CUDA 12.9** targets the feedstock's `12.9` branch. Start from that branch
  and open a manual PR from a fork with `12.9` as its base. [Conda-forge
  maintenance branch guidance](https://conda-forge.org/docs/how-to/advanced/several-versions/)
- **CUDA 13** targets the feedstock's `main` branch. Look for an existing
  version-update PR for the exact release. If there is none, open an issue in
  the feedstock with the title `@conda-forge-admin, please update version`.
  Wait for any follow-up bot commits that complete the recipe update. If the bot
  does not produce a PR for the intended version, open one from a fork against
  `main`. [Conda-forge version update command](https://conda-forge.org/docs/maintainer/infrastructure/#conda-forge-admin-please-update-version)

### Review, test, and publish

1. Review `recipe/recipe.yaml` against the release asset. Verify the PR base,
   set the release version and SHA-256, and reset the build number to `0` for
   a new version. Compare the packaged paths, license files, Python minimum,
   and dependency constraints with the tagged bindings and metapackage
   `pyproject.toml` files and `cuda_python/setup.py`. For tags with the split
   source layout, use `cuda_bindings_12` or `cuda_bindings_13` as appropriate.
   If the tagged metapackage requires `CUDA_PYTHON_BUILD_MAJOR`, set it to
   `12` or `13` for that build. Verify any newly required `cuda-core` or
   `cuda-pathfinder` versions are available from conda-forge.
2. Run `pixi run lint` in the feedstock. Run `pixi run rerender` when changes
   affect generated CI configuration or variants. Inspect any staged changes
   from the rerender and compare the resulting matrix with the target branch.
3. Wait for the feedstock's **Build conda package** checks and lint to pass at
   the final PR head. Check the logs against `.ci_support/` to verify bindings
   builds and tests across the expected platform and Python variants and the
   `noarch` metapackage build and tests. Investigate any unexplained differences
   in coverage, including test steps skipped by recipe conditions.
4. Merge the reviewed PR to start conda-forge's build and upload. Check the
   post-merge run, then verify that both exact package versions install
   together in a fresh environment from the standard `conda-forge` channel.
   Anaconda.org file listings can appear before channel repodata includes the
   builds. Record the feedstock PR, CI run, and install result in the release
   checklist.

For a smoke test, substitute the release version and a supported Python minor
for the placeholders. Pin both package versions so the solver verifies the
specific bindings and metapackage being released.

```bash
version=X.Y.Z
python_version=X.Y
pixi exec --force-reinstall --channel conda-forge \
    --spec "python=$python_version" \
    --spec "cuda-bindings==$version" --spec "cuda-python==$version" \
    python -c 'from cuda.bindings import driver, nvrtc, runtime'
```
