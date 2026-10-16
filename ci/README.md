# Continuous Integration

## Checkout History

Use the default shallow checkout for jobs that only read files or test prebuilt
wheels. Source builds need complete commit ancestry and release tags so
`setuptools-scm` can derive each package's version.

For those jobs, use `fetch-depth: 0` with `filter: blob:none`. The checkout
action fetches complete ancestry for all branches and tags, while the blob
filter skips historical file contents. This avoids downloading the large
historical `gh-pages` contents discussed in issue #2197. Git retrieves file
contents as needed when checking out the requested commit.

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

## Repository Links During Main Development

For development targeting `main`, keep links to the latest repository sources
on `main`, even when their targets are added or moved by the same pull request.
The link checker treats canonical
`https://github.com/NVIDIA/cuda-python/blob/main/` and `tree/main/` URLs as
references to the proposed main tree. Lychee remaps them for validation only;
source files and published documentation keep their URLs. The remaps match only
at the start of a URL, leaving URLs embedded in another URL unchanged. Relative
links and links to tags, commit SHAs, other branches, and other repositories
are checked as written.

CI remaps these links to the CI repository at the actual checked-out commit
using `--remap`, retaining GitHub's file/directory views and fragment behavior.
The initial check and all retries use the same remaps. Missing targets in that
commit still fail.

Local pre-commit runs the Python hook `ci/tools/run_lychee.py`, which computes
the checkout's file URI and passes anchored native `--remap` rules to Lychee.
This checks paths in the working tree, including uncommitted and unpushed
targets. The hook installs the official PyPI `lychee-bin==0.24.2` package in
its own Python environment and uses the platform's native wheel. Local checks
retain their existing policy of not checking fragments.

The Linux and Windows pre-commit CI jobs run a bounded smoke check of the
installed hook against local Markdown and reST fixtures. They check valid
files and directories and missing targets in paths containing spaces, and
verify that the source files are unchanged. This smoke check does not scan all
external documentation links; full authored and rendered checks run in
documentation CI.

The `check-doc-links` action is not currently present on `12.9.x`, `13.4.x`,
`12.2.x`, or `11.8.x`; merging changes into `main` does not update those
branches' workflows. A maintenance branch's `/main/` URLs may intentionally
refer to a different tree from its checkout. Before adopting this hook or
action there, choose an explicit validation policy for those links instead of
assuming that they refer to the maintenance checkout.

## Temporary Pixi Lockfile Maintenance Pause

Pixi lockfile freshness enforcement and automated refreshes are temporarily
suspended because of [Pixi #7215](https://github.com/prefix-dev/pixi/issues/7215)
and [#7000](https://github.com/prefix-dev/pixi/issues/7000). Repeated lockfile
operations can change the committed solution without converging, so a general
refresh is not a reliable repair.

The `CI: pixi lockfile freshness check` workflow still runs on every PR with its
existing `pixi lock --check (all workspaces)` check name. It succeeds with a
warning annotation and job summary explicitly reporting the suspension;
success does not mean freshness passed. The refresh workflow has no schedule.
Manual dispatch emits the same notice without solving dependencies, rewriting
lockfiles, pushing a branch, or creating a PR.

Manifest/lock consistency is not automatically enforced during this pause.
Necessary dependency changes require manual review and validation of the
affected manifests and lockfiles. Real source builds and tests continue using
`PIXI_FROZEN=true`, including nested `pixi run` calls, to build the checkout's
source packages against the committed dependency solution. The pinned Pixi
version and committed lockfiles are unchanged.

Re-enable maintenance only after a released Pixi passes both upstream issue
reproducers and repeated byte-stable update/lock/check cycles across all six
workspaces: the repository root, `cuda_pathfinder`, `cuda_bindings`, `cuda_core`,
`benchmarks/cuda_bindings`, and `benchmarks/cuda_core`. Verify both `pixi update
--no-install` and `pixi lock`, followed by repeated `pixi lock --check` commands,
converge to unchanged lockfile bytes. Then restore strict freshness checks,
scheduled refreshes, and `PIXI_LOCKED=true` in source-build CI.

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
