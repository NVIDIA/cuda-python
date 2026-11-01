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
including fragments. Developers run the same hooks locally. Public repository
`main` links are checked against the proposed checkout, as described under
[Repository Links](#repository-links).

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
stay failures, mostly reuses successful checks. With the maintainer override,
the driver's report-based result becomes the final link gate, and the full
suite skips its duplicate Lychee invocation. Every other hook still runs once.

### Maintainer Override For HTTP 429

When rate limiting persists, maintainers can apply `ci: lychee-non-blocking`
to the PR and rerun the affected Linux/Windows pre-commit jobs or rendered
documentation job. Applying or removing the label does not itself start a
new workflow run. The shared link-check action queries the PR's current labels,
so a rerun sees the current decision rather than the original event's snapshot.
The lookup makes up to three attempts, waiting 5 and 10 seconds between them.
If every attempt fails or returns an invalid response, the action warns and
keeps the caller's default policy without enabling the override: authored
checks remain strict in the full pre-commit suite, and rendered checks keep
their allowance for at most ten distinct HTTP 429 URLs. Rerun the affected job
to retry the lookup and apply the label; other pre-commit hooks still run.

The label selects the driver's `allow-429` policy for both authored and
rendered checks. Lychee still runs and retries normally; only a final report
whose remaining failures are all HTTP 429 responses from external HTTP(S)
URLs can pass with a warning. This explicit override has no URL-count limit.
The warning and retained reports identify the URLs that remain unverified.
Successful checks retain their usual cache behavior; failures are never cached.

Any other failure still blocks, including a mixture of HTTP 429s and other
errors, broken local files or fragments, timeouts, and HTTP 403, 404, or 5xx
responses. Hook setup, invalid or missing reports, and the offline smoke
checks remain blocking. Other exceptional failures require
the normal admin-merge decision.

The label applies only to PR checks. Local pre-commit runs, default-branch
checks, and nightly cache production keep their existing policies. Without
the label, authored checks remain strict and rendered checks retain their
existing allowance for at most ten distinct HTTP 429 URLs.

### Shared URL Caches

Nightly and manual refreshes skip restoring, check afresh, and publish
successful results; ordinary checks restore the newest baseline and never
publish. Failed URLs and filesystem targets are never cached. Successes keep
their original timestamps and expire after three days, so a missed nightly
refresh does not break reuse, but a link that breaks after a nightly success
can stay unnoticed for up to 72 hours.

The action keys each baseline by checker kind and a hash of
`.pre-commit-config.yaml` (hook options and wheel pin), `lychee.toml`, and
`ci/tools/run_lychee.py`, `run_lychee_precommit.py`, `retry_lychee.py`, and
`checkout_source_links.py`, so editing any of them starts a cold cache.
Authored and rendered caches stay apart because only rendered checks validate
fragments and Lychee's cache does not record which validation ran. URL results
are shared across operating systems; pre-commit's installed environments keep
their OS- and Python-specific caches.

#### Refreshing Shared Caches

To rebuild both shared caches manually, open
[CI: Lychee checks and caches](https://github.com/NVIDIA/cuda-python/actions/workflows/lychee.yml)
(`.github/workflows/lychee.yml`), select the default branch, and enable
`refresh-cache` before choosing **Run workflow**. This runs the cache refresh
without the rest of nightly CI.

Checks that restore no matching baseline add a notice and a job-summary hint.
They still check links, with potentially more runtime and network traffic.
Checker changes in a PR intentionally start with a cold cache; refreshing the
default branch cannot seed a cache for settings that exist only in that PR.
Refreshing rebuilds cached successes; broken links and upstream rate limits
still need their usual investigation.

### Repository Links

Documentation can link to a path added or moved in the same PR using its
intended public URL, for example
`https://github.com/NVIDIA/cuda-python/tree/main/new-directory`. Lychee
validates the proposed target before it exists on public `main`. Only the
validation target changes; source files and built documentation retain their URLs.

`ci/tools/checkout_source_links.py` matches the complete public repository
prefix followed by `blob/main/` or `tree/main/`:

- Authored Markdown and reST map to file URIs in the working tree, including
  uncommitted and unpushed paths. Missing files and directories fail even after
  an earlier success. These checks need no repository metadata and retain the
  existing policy of omitting fragments.
- Rendered HTML from the public repository maps source links to GitHub's file
  or directory view at the actual checked-out commit, retaining fragment checks.
  That commit must be available on GitHub.
- Rendered HTML from other repositories maps the same public source links to
  local files and directories. Only those source links omit fragments, since
  local files do not reproduce every GitHub source-view anchor. Generated
  documentation pages and their anchors still receive full validation.

Tags, explicit commits, other branches, unrelated repositories, and URLs embedded
inside other URLs retain their original targets. For rendered checks and checking
builds, CI supplies the checkout identity through `GITHUB_REPOSITORY`. Local runs
use `git config lychee.repository`, or otherwise the `upstream` remote followed by
`origin`. A checkout whose remotes only name a fork can select public-repository
validation explicitly:

```bash
git config lychee.repository NVIDIA/cuda-python
```

For rendered checks of an unpublished local commit, select your fork's identity
to validate source links against local files:

```bash
git config lychee.repository YOUR_GITHUB_USER/cuda-python
```

Use your actual fork's owner and repository. Set this before both the checking
documentation build and the rendered hook; rebuild existing checking output if
its example links contain an unpublished SHA. Clear any explicit
`CUDA_PYTHON_DOCS_GITHUB_REF` override to use this selection. Only GitHub
source-view fragments are omitted; generated HTML anchors remain checked.
CI uses `GITHUB_REPOSITORY` and retains its normal validation policy.

The checking build uses the public checkout's actual commit for generated
example links. In other repositories it uses `main`, allowing those public URLs
to receive the same local target checks without inserting another repository's
commit into public links. An explicit `CUDA_PYTHON_DOCS_GITHUB_REF` override is
retained; release builds continue to use their release tags.

These rules validate the proposed source tree. Before adopting them on a
maintenance branch, choose an explicit policy for links that intentionally point
to a different tree from the maintenance checkout.

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
version was raised to v0.82.1 and the affected certificate-related packages
(`certifi`, `pip`, `uv`, `ca-certificates`) were updated individually in the
committed lockfiles to address the eMudhra/emSign CA incident; no general
lockfile refresh was performed.

Re-enable maintenance only after a released Pixi passes both upstream issue
reproducers and repeated byte-stable update/lock/check cycles across all six
workspaces: the repository root, `cuda_pathfinder`, `cuda_bindings_13`, `cuda_core`,
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
