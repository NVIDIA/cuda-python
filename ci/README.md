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

## Documentation Link Checks

Two pre-commit hooks check links with one pinned native wheel (`lychee-bin`,
see `.pre-commit-config.yaml`) through the wrapper `ci/tools/run_lychee.py`:
`lychee` checks authored Markdown and reST (fragments unchecked), and the
manual-stage `lychee-rendered` checks the generated HTML in `artifacts/docs`,
including fragments. Developers run the same hooks locally.

CI runs a hook through `ci/tools/run_lychee_precommit.py <authored|rendered>`,
wrapped with the shared cache and report handling in
`.github/actions/lychee-links`. The driver runs the real hook and merges the
native JSON report of every invocation (pre-commit may split long file lists).
It makes up to three attempts with 60/120-second cooldowns, but only while all
remaining failures are transient HTTP trouble: timeouts, 408, 429, or 5xx.
Missing, malformed, or inconsistent reports, and scans that check no links,
fail explicitly. Per-attempt reports and a summary are kept for seven days.

| Where | What runs | Final result |
| --- | --- | --- |
| Linux/Windows pre-commit | Offline hook smoke check, then authored link warm-up from the shared cache, then the full suite once | The strict pre-commit result, including unresolved HTTP 429s |
| Full-CI documentation build | Rendered link check of the HTML it just built | Unresolved links fail, except that up to ten distinct URLs may remain at HTTP 429 as a warning |
| Nightly (`lychee.yml`) | Afresh authored and rendered checks | Unresolved links are warnings; setup, execution, and report errors fail |

The warm-up only fills the cache so the full suite, whose formatter changes must
stay failures, mostly reuses successful checks.

### Shared URL Caches

Nightly is the only producer. It skips restoring, checks afresh, and publishes
successful results; every other job restores the newest baseline and never
publishes. Failed URLs and filesystem targets are never cached. Successes keep
their original timestamps and expire after three days, so a missed nightly
refresh does not break reuse, but a link that breaks after a nightly success
can stay unnoticed for up to 72 hours.

The action keys each baseline by checker kind and a hash of
`.pre-commit-config.yaml` (hook options and wheel pin), `lychee.toml`, and
`ci/tools/run_lychee.py`, `run_lychee_precommit.py`, and `retry_lychee.py`, so
editing any of them starts a cold cache.
Authored and rendered caches stay apart because only rendered checks validate
fragments and Lychee's cache does not record which validation ran. URL results
are shared across operating systems; pre-commit's installed environments keep
their OS- and Python-specific caches.

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
