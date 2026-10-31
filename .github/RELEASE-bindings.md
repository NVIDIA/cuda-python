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

Major-line maintenance may include selected API, Python, platform, or other
compatibility changes. Decide those changes explicitly and describe them in
the release notes. Once a patch release is frozen, admit only its approved
fixes and release prerequisites. Do not infer patch-release scope solely
from the registry's `maintenance` role.

For an ordinary release, select a tested commit on `main`. If stabilization
must continue while development advances, create a short release branch such
as `release/bindings-12.9.11` from the chosen commit. For an emergency after a
release, start from its exact tag, provided that tag contains the new package
layout. For the first release after this transition, use a validated commit
containing both roots. Keep routine development on `main`; the historical
`12.9.x` branch stays frozen except for emergency fixes using compatible
historical tooling. Reconcile those fixes into the main-tree maintenance
root. Keep the package layout and release tooling together on new release branches.

Record the source commit, intended tag, selected package root, release owner,
and accepted changes in the release checklist. A branch is a temporary scope
boundary, not a separate long-term home for shared infrastructure.

## Move fixes between integration and release branches

Prefer a reviewed fix on `main`, then backport it to the selected release
branch. Run **CI: Backport a merged PR to a release branch** from the Actions
tab on `main`, supplying the merged PR number and one existing target branch.
The workflow opens a PR; it does not merge or release the result. Inspect the
diff, request CI using the normal PR approval procedure, and review it before
merging. Workflow-file changes require a manual backport with appropriate
credentials. Conflicts and toolkit-specific generated changes require an
explicitly reviewed adaptation.

If urgency requires a fix on the release branch first, link a follow-up PR
reconciling the fix back into `main` before closing the release checklist. In either
direction, assess both bindings roots and record why a change applies to one
or both. Regenerate affected generated files from the intended toolkit and
cybind inputs instead of blindly copying generated output between majors.
See the [shared maintenance policy](../ci/README.md#maintaining-both-bindings-roots).

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
7. Finish post-release QA and announcements, return every applicable fix to
   `main`, and close the stabilization effort. Retain immutable tags and the
   branch history needed for diagnosis.

For either bindings line, add the released version to
[`cuda_bindings_13/docs/nv-versions.json`](../cuda_bindings_13/docs/nv-versions.json)
and its matching metapackage version to
[`cuda_python/docs/nv-versions.json`](../cuda_python/docs/nv-versions.json)
on `main`. Documentation deployments from `main` own these shared version
selectors. Release deployments publish versioned pages without replacing the
selectors or shared `latest` pages; updating a selector only on a stabilization
branch will not update the public selector.

The workflow revision used for a release is its **control revision**; the
tagged commit is its **source revision**. Record both and the artifact CI run
ID. Release routing uses the package layout and toolkit settings in
`ci/versions.yml` at the release tag. Retained CUDA 13 tags also support the
older `cuda.build.version` configuration, the `cuda_bindings` source directory,
and their original build matrix and metapackage artifact name. New CUDA 12
releases require the local `cuda_bindings_12` root and toolkit-qualified
artifacts. The workflow control revision never supplies missing tagged toolkit
settings or replaces tagged release notes. Older CUDA 12 branch releases use
their compatible historical tooling.

In **CI: Release**, normal bindings and metapackage releases each require
their own exact-version, nonempty release notes in the tagged source. Notes
from the workflow control revision or a different component cannot fill a gap.
Preserve their existing exemption for `.postN` releases, which do not require
new notes.

Local synthetic tags used in publication-incapable rehearsals must remain local;
they do not authorize publishing a package, creating a remote tag, or moving
an existing tag.
