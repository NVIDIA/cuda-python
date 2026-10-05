# Continuous Integration

## Checkout History

Use the default shallow checkout for jobs that only read files or test prebuilt
wheels. Source builds need complete commit ancestry and release tags so
`setuptools-scm` can derive each package's version.

For those jobs, use `fetch-depth: 0` with `filter: blob:none`. The checkout
action fetches complete ancestry for all branches and tags, while the blob
filter skips historical file contents. This avoids downloading the large
historical `gh-pages` contents discussed in issue #2197. Git retrieves file
contents as needed when checking out the requested commit or creating the
freshness check's base-commit worktree.

CI change detection uses this full-history checkout for PR mirrors, so the
actual PR base branch is available when computing its merge base. Non-PR runs
need only a shallow checkout. PR-preview cleanup checks out its script
shallowly and fetches `gh-pages` separately when needed. Pathfinder release
preparation fetches its exact tag without requesting every other tag.

Git documents blob filtering in its
[clone reference](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---filterltfilter-specgt).
The pinned checkout action's
[`README`](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/README.md#fetch-all-history-for-all-tags-and-branches)
documents `fetch-depth: 0` for fetching all history, branches, and tags.

## Repository Customizations

The workflows in this repository use the `CI_CUSTOMIZATIONS_*` namespace for
GitHub Actions configuration variables that opt an alternative synchronized
repository into repository-specific CI behavior. This keeps the workflow logic
shared without hard-coding the names of private repositories into the public
source tree.

These variables are non-secret strings configured under
**Settings > Secrets and variables > Actions > Variables**.
An unset variable, or any value other than the literal string `true`, leaves
the customization disabled. Do not store credentials or other secret values in
these variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CI_CUSTOMIZATIONS_SECURITY_SUITE_ENABLED` | Disabled | Enables the NVIDIA Security Suite after its runner, Actions variables, and OIDC/Vault authorization have been provisioned for the repository. |

The canonical `NVIDIA/cuda-python` repository does not need this variable
because its standard workflow behavior is enabled directly. Before enabling a
customization elsewhere, document the repository-specific prerequisites and
verification procedure in that repository's own documentation.

## CUDA Bindings Package-Root Registry

`ci/versions.yml` is the authoritative public registry for CUDA bindings
package roots. Each mapping key is a repository-relative package root; its
record declares an exact toolkit build/test pin and a `release_status` of
`current` or `maintenance`. Release routing uses the tag's CUDA major/minor
and the toolkit pin to select a root. Bindings and metapackage builds use
setuptools-scm's default tag parser. Each root's `git_describe_command`
selects its tag series for source builds; update its `--match` pattern when
the root moves to a new toolkit minor.

The Python helpers share registry parsing and validation, so they are modules
in the repository-private `cuda-python-ci-tools` distribution rather than
standalone scripts with repeated PEP 723 dependency metadata. Install the
package in editable mode from the repository root:

```console
python -m pip install -e ./ci
```

Use `ci.tools.bindings_config` instead of reading the YAML directly. It
validates the registry and emits normalized JSON with the configured values
plus the derived CTK target and CUDA ABI major/variant:

```console
python -m ci.tools.bindings_config
python -m ci.tools.bindings_config --package-roots
python -m ci.tools.bindings_config --release-status current
```

Install the test extra to run the CI-tool tests:

```console
python -m pip install -e './ci[test]'
python -m pytest --noconftest ci/tools/tests
```

The public wheel builder requires one `current` package root and one
`maintenance` package root with different CUDA ABI majors.
This is deliberately a two-line policy. The mapping keeps package paths out of
workflow conditionals; it does not promise arbitrary numbers of supported
lines. `cuda_bindings_12` names a stable ABI-major source tree, while `current`
and `maintenance` describe release roles that can change at the next major.

The standalone `cuda-python` source distribution contains its own small tag
selector map because it cannot import repository CI tooling. The pre-commit
metadata check keeps those selectors, bindings packaging, and the maintenance
version fallback aligned with the registry. Run it explicitly with:

```console
python -m ci.tools.bindings_config --check-package-metadata
```

## Maintaining Both Bindings Roots

`main` is the integration branch for both bindings release lines. The old
`12.9.x` branch remains available as a record of earlier releases; develop
new CUDA 12 changes in `cuda_bindings_12/` on `main`. Short release branches
may stabilize a selected version or carry urgent fixes while `main` advances.
The [bindings release guide](../.github/RELEASE-bindings.md) describes branch
creation, backport automation, independent release validation, and returning
fixes to `main`. The
[CUDA 12 maintenance guide](../cuda_bindings_12/MAINTENANCE.md) identifies the
maintenance root.

For generated bindings, change cybind first and regenerate each affected
package root against its intended toolkit inputs. Review the generated diff,
record the cybind revision, toolkit inputs, and command in the pull request,
and check modified files with
`python toolshed/check_generated_file_seals.py <paths...>`. The seal verifies
the content of an individual generated file; it does not identify the cybind
revision or prove that the two roots should have identical output. Generated
output may legitimately differ across toolkit targets.

For a handwritten bindings fix, inspect the corresponding code in the other
root. In the pull request, identify the matching change or explain concretely
why the fix does not apply there. A reviewer must verify that judgment: there
is no general deterministic check for semantic equivalence of handwritten
code across the two roots. Where files are intentionally identical, a narrowly
scoped byte-for-byte check can be added after establishing that invariant.

Major-line maintenance can include deliberately selected compatibility work
such as a new Python version or a new API supported by that toolkit. A frozen
patch release has a narrower scope: fixes and release prerequisites approved
for that release. State which policy applies in the release checklist and
notes; a maintenance label alone does not promise a fixes-only release.

Keep generator development and continuous toolkit qualification independent
of this source layout. A cybind change should be exercised against both
supported consumers before its generated output is imported. Record the
generator revision and toolkit inputs for each import, including any manual
post-generation adjustment. File seals establish content integrity, not
generation provenance. Moving QA directories or redesigning cybind asset
storage is not required to maintain both roots here.

## Changing Supported Release Lines

When support moves from CUDA 12/13 to CUDA 13/14, keep the CUDA 13 source in
its own package root while introducing a distinct CUDA 14 root. Decide the
root names as part of that change; do not replace the current root's source
until the CUDA 13 maintenance source has a home. Then:

1. Update `ci/versions.yml`: remove the retired CUDA 12 root, mark the CUDA
   13 root `maintenance`, mark the CUDA 14 root `current`, and set each exact
   toolkit build/test version.
2. Align each root's `pyproject.toml` `git_describe_command --match` selector
   and version fallback with its release family. Update other per-root packaging,
   pixi environments, documentation, and dependent package constraints for the
   supported lines.
3. Build and test both roots and their dependent packages on the supported
   platforms. Validate release selection and run publication-incapable dry
   runs for tags from each line before releasing from the new layout.

The same branch policy applies after the rollover: `main` integrates the
current and maintenance roots, short release branches stabilize releases,
and retired release branches are retained for history.
