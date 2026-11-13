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
4. Run **CI: Release** with `release-action=dry-run`, the tag, and
   `component=cuda-bindings`; repeat for `cuda-python`. Leave `run-id` blank
   to exercise automatic lookup, and leave `dry-run-docs-branch` blank to avoid
   deploying docs. The workflow verifies a successful tag-triggered CI run
   with the exact source SHA, validates artifacts, and builds docs.
5. Review the dry runs, then separately approve and run `full-release` for
   each component under the team's release procedure. A successful candidate
   rehearsal or ordinary branch CI is not a substitute for exact-tag CI.
6. After both component releases succeed, review and publish the shared
   `vX.Y.Z` GitHub Release draft. Check its release notes, versioned
   documentation links, source archive, checksum, and wheels. These component
   workflows leave the draft for the release owner to publish.
7. Finish post-release QA and announcements and return every applicable fix
   to `main`.

For a bindings `vX.Y.Z` tag, tag CI validates that line's bindings documentation.
Validate the metapackage documentation in the separate `component=cuda-python`
release dry run. Tag CI does not publish documentation. This focused tag-CI
check is specific to the shared bindings/metapackage tags; Core and Pathfinder
release behavior is unchanged.

For either bindings line, add the released version to
[`cuda_bindings_13/docs/nv-versions.json`](../cuda_bindings_13/docs/nv-versions.json)
and its matching metapackage version to
[`cuda_python/docs/nv-versions.json`](../cuda_python/docs/nv-versions.json)
on `main`. Documentation deployments from `main` own these shared version
selectors. Release deployments publish versioned pages without replacing the
selectors or shared `latest` pages.

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
