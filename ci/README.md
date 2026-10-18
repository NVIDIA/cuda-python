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

Authored Markdown and reST are checked through the same `lychee` pre-commit
hook in development and in Linux/Windows PR CI. The hook installs the official
PyPI `lychee-bin==0.24.2` native wheel in its isolated Python environment;
`ci/tools/run_lychee.py` invokes that executable and preserves its exit status.
Authored checks retain their existing policy of not validating fragments.
Repository URLs are checked as written.

### PR Checks And Cache Warming

PR CI restores successful URL checks from the shared baseline, then runs
`pre-commit run lychee --all-files` through `ci/tools/run_lychee_precommit.py`.
The helper allows up to three hook attempts, with 60/120-second cooldowns
when all remaining failures are transient HTTP failures. Successful checks
accumulate in the local cache; permanent link failures stop additional
warm-up attempts. A missing baseline simply starts with uncached checks.

After warming, `pre-commit run --all-files --show-diff-on-failure` runs once.
Its normal strict result is the final gate, including for unresolved HTTP
429s. This can make four total Lychee invocations, but successful external
checks are reused. Formatter changes remain failures because the complete
suite is not retried. A missing or invalid native report, empty check, or
hook installation/execution error fails the warm-up step.

The permanent offline smoke check runs first, isolating hook setup and shell
invocation failures from network failures. Standalone CI-tool unit tests run
in nightly CI. The full-CI documentation job checks generated HTML after its
build, avoiding a duplicate authored scan.

### Rendered Documentation

After building the documentation, full CI runs the manual hook through the
same driver:

```bash
python ci/tools/run_lychee_precommit.py rendered --policy check \
    --report /path/to/reports/lychee-rendered.json
```

The driver invokes `pre-commit run lychee-rendered --hook-stage manual
--all-files`. This hook enumerates generated HTML under `artifacts/docs`,
excluding `_static` assets, and enables full fragment validation. Generated
HTML is selected explicitly because it is not part of pre-commit's tracked
source file list. The hook uses the same pinned wheel and shared wrapper as
the authored hook, without a separate Lychee action or binary installation.

All rendered attempts use the same inputs. This mode retains up to three
attempts and the existing warning allowance for at most ten distinct URLs
that still return HTTP 429; other unresolved links fail.

### Shared URL Caches

Nightly invokes both configured hooks through the driver. Refresh mode skips
restoring yesterday's URL cache, checks afresh, and publishes the successful
results. Genuine link failures are advisory during cache warming; build,
installation, execution, and report errors remain failures. Linux pre-commit
runs on the default branch also publish their accumulated authored results. PR jobs
restore caches and do not publish them.

`ci/tools/lychee_cache.py` gives all callers the same cache policy and unique
run-ID/attempt snapshot keys. The namespace includes the hook configuration
(and native wheel pin), checker configuration, and shared helpers. Authored
and rendered caches remain separate because only rendered checks validate
fragments. Only the URL-result file is shared across operating systems;
pre-commit's installed environments retain their OS/Python-specific caches.

Successful external checks retain their original timestamps and expire after
three days, allowing reuse when a nightly refresh is missed. Nightly refreshes
still check links afresh. Failed URLs are not persisted, and filesystem targets
are checked afresh. The rendered hook uses its own cache directory, selected by
`LYCHEE_CACHE_DIR` (default `.lychee-rendered`), so its fragment-aware results
cannot mix with authored results in the checkout's `.lycheecache`. Per-pass
JSON reports and a Markdown summary are retained as CI artifacts.

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
