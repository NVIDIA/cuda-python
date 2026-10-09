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

## Temporary Pixi Samples Lockfile Workaround

Pixi 0.73.0 checks PyPI dependencies that the `samples` environment explicitly
excludes, then drops its local Conda source references during a PyPI-only
refresh. The next refresh restores them, so repeated lock commands do not
converge. A general maintenance refresh cannot repair this defect.

Source-build CI temporarily uses `PIXI_FROZEN=true`: it builds the checkout's
source packages using the committed dependency solution without triggering
Pixi's workspace-wide freshness check.

The separate freshness workflow remains strict except for one exact pattern:
Pixi 0.73.0 exits 1 and removes precisely the nine `samples` source references,
with every other lockfile byte unchanged. The detector also verifies the
expected manifest overrides, local paths, platforms, and source-record
definitions. It rejects the reverse cycle, which would add missing references.
For PRs, the base must produce identical original and repaired blobs, and no
`pixi.toml`, `pyproject.toml`, or Pixi-version input may differ from the base.
Recognized cases emit a warning and restore the committed references.

The refresh workflow uses the same exact detector after updating `cuda_core`.
On a match, it restores the pre-update committed lockfile and continues
refreshing the other workspaces. This postpones `cuda_core` dependency updates
without allowing an unstable lockfile into the generated refresh PR; every
unrecognized refresh failure remains fatal.

Remove the detector and exception, and restore `PIXI_LOCKED=true`, once a
released Pixi passes repeated lock and check commands without changing either
the reduced upstream reproducer or our full workspaces.

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
