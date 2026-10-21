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
version and committed lockfiles are unchanged.

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

## CUDA Bindings Package-Root Registry

`ci/versions.yml` is the authoritative public registry for CUDA bindings
package roots. Each mapping key is a repository-relative package root; its
record declares an exact toolkit build/test pin and a `release_status` of
`current` or `maintenance`. Release routing uses the tag's CUDA major/minor
and the toolkit pin to select a root. Bindings and metapackage builds use
setuptools-scm's default tag parser. Each root's `git_describe_command`
selects its tag series for source builds; update its `--match` pattern when
the root moves to a new toolkit minor.

The Python helpers share registry parsing and validation. The small,
repository-private `cuda-python-ci-tools` package installs their common
dependencies and makes those helpers importable from each build environment;
it is not part of a CUDA Python wheel. Release jobs check it out under
`.ci-control/ci` so the workflow can use its own tooling while inspecting a
different tagged source tree. Install it locally from the repository root:

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
preview target aligned with the registry. Copying maintenance files does not
make their old branch an ancestor of `main`, so its later release tags are not
necessarily reachable. Until an appropriate tag is reachable, CI uses
`ci.tools.bindings_scm_version` to create a local preview tag in its disposable
checkout, then lets setuptools-scm derive the version normally. Existing
appropriate tags take precedence. The helper never pushes or moves tags;
local preview builds should use an independent clone so their temporary tags
do not affect other worktrees. Run the metadata check explicitly with:

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

When support moves from CUDA 12/13 to CUDA 13/14, retain `cuda_bindings_13/`
as the maintenance source and introduce a distinct CUDA 14 root. Then:

1. Update `ci/versions.yml`: remove the retired CUDA 12 root, mark the CUDA
   13 root `maintenance`, mark the CUDA 14 root `current`, and set each exact
   toolkit build/test version.
2. Align each root's `pyproject.toml` `git_describe_command --match` selector
   and any preview bootstrap target with its release family. Update per-root packaging,
   pixi environments, documentation, and dependent package constraints for the
   supported lines.
3. Build and test both roots and their dependent packages on the supported
   platforms. Validate release selection and run publication-incapable dry
   runs for tags from each line before releasing from the new layout.

The same branch policy applies after the rollover: `main` integrates the
current and maintenance roots, short release branches stabilize releases,
and retired release branches are retained for history.
