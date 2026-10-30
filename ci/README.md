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
reproducers and repeated byte-stable update/lock/check cycles across every
workspace reported by `python ci/tools/list_pixi_workspaces.py`, including both
bindings roots. Verify `pixi update --no-install` and `pixi lock`, followed by
repeated `pixi lock --check` commands, converge to unchanged lockfile bytes.
Then restore strict freshness checks, scheduled refreshes, and `PIXI_LOCKED=true`
in source-build CI.

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

The Python helpers share registry parsing and package consistency checks. The small,
repository-private `cuda-python-ci-tools` package installs their common
dependencies and makes those helpers importable from each build environment;
it is not part of a CUDA Python wheel. Workflows keep their control tooling separate when inspecting another
source revision. Install it locally from the repository root:

```console
python -m pip install -e ./ci
```

Use `ci.tools.bindings_config` instead of reading the YAML directly. It
reads the registry and emits normalized JSON with the configured values
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

The standalone `cuda-python` source distribution carries its selectors in
`cuda_python/pyproject.toml`, without depending on repository CI tooling. The
pre-commit metadata check keeps those selectors, bindings packaging, and Core
toolkit floors aligned with the registry:

```console
python -m ci.tools.bindings_config --check-package-metadata
```

Copying maintenance sources does not make their later branch tags ancestors of
`main`. Preview builds need an appropriate reachable milestone for their target
release. Follow the [Git-derived version lifecycle](../AGENTS.md#git-derived-version-lifecycle)
and keep temporary preview tags local to an independent disposable clone;
worktrees share tags with their repository.
CI uses `ci.tools.bindings_scm_version` in disposable build checkouts until an
appropriate maintenance milestone is reachable. It creates a local preview tag
at a validated ancestor only when required; existing suitable tags govern the
version through standard setuptools-scm. It never pushes or moves tags.
For a local maintenance preview, use an independent disposable clone (not a
linked worktree), install the CI tools, and run from its root:

```console
python -m ci.tools.bindings_scm_version --package-root cuda_bindings_12 --sha "$(git rev-parse HEAD)"
```

### SCM Milestones and Bootstrap Targets

The maintenance root's `[tool.cuda-python].scm-bootstrap-release` records the
intended preview release, for example `12.9.11`. It is not a version override.
Suitable reachable release or milestone tags take precedence through standard
`setuptools-scm` progression; do not add a synthetic tag when that history
already supplies the version.

When choosing a new target, fetch the canonical tags and coordinate with
maintainers of all release branches. Reserve the next globally unused release
number for the intended line. If an urgent release from another branch consumes
`12.9.11`, update main's preview target to the next unused release before another
preview build. Unreachable tags still reserve their release numbers. Never move
or replace such a tag to reuse its name on main. Keep the same target across
PRs and builds until release coordination calls for a change.

Temporary bootstrap tags must stay local to an independent disposable clone;
worktrees share tags. For a newly integrated maintenance root, wait until the
validated integration group is merged on `main` before creating its first
retained internal `X.Y.Za0` milestone, by explicit maintainer decision. Once that
milestone is reachable, remove the temporary bootstrap machinery and its
workflow calls as a follow-up; ordinary SCM history governs subsequent builds.
This guidance does not create or publish a milestone.

A toolkit minor rollover also changes the bindings root's `--match` selector
and the metapackage selector table. Before relying on builds from that new
series, arrange an appropriate reachable milestone, such as `v13.5.0a0` for a
validated CUDA 13.5 line. A selector change without matching history can yield
a plausible but wrong `0.1.dev...` version. The maintenance bootstrap does not
automatically establish milestones for the current-major root. Use the same
local-preview safeguards while preparing the new source and confirm the
resolved version and `git describe` output before retaining its milestone.

## Nightly Artifact Sources And Rollout

`ci-nightly.yml` downloads wheels from the latest successful main CI run and
reads the bindings registry from that run's source commit. The source registry
must be compatible with the nightly workflow; the current workflow requires
schema 2. It does not translate an older registry into the new layout.
Selected rows that install a custom driver must also include its
`DRIVER_SHA256` in the source matrix. Registry compatibility alone does not
make a pre-hardening source compatible: choose a newer successful run rather
than disabling installer verification. Rows using preinstalled drivers do not
need an installer checksum.

After multi-root CI lands on `main`, wait for the first successful, complete
main CI run containing the new registry and both bindings majors before using
it as the nightly artifact source. Until that run succeeds, auto-detection can
still select an older successful run; the nightly job then reports the
incompatible source and stops. PR CI builds and tests everything when it has
no complete, compatible artifact baseline.

To test the nightly workflow before that main run is available, supply the ID
of a successful CI run built with the compatible registry, required wheel
artifacts, and driver-checksum metadata, such as a validated PR run:

```console
gh workflow run ci-nightly.yml --repo NVIDIA/cuda-python --ref main -f 'run-id=<compatible-ci-run-id>'
```

The override selects both the wheel artifacts and their source revision. It
does not convert old artifacts or change the default selection for later
nightly runs. Once a compatible main run succeeds, ordinary nightly dispatches
can leave `run-id` empty.

## Reading CI Jobs and Summaries

A name such as `pathfinder+bindings12+core12+meta12 / py3.11, CTK12.9.1
(wheels; bindings PR)` identifies the enabled test suites, Python version,
and toolkit. `meta12` means the CUDA 12 `cuda-python` metapackage; `core12`
means Core tested with CUDA 12. The corresponding CUDA 13 suites use `13`.
`pathfinder` is shared across majors.

The two source descriptions answer different questions:

- `wheels` or `local CTK` describes how the toolkit libraries are supplied.
- `bindings PR` uses the selected root's artifact from this CI run. Unaffected
  packages can reuse exact-baseline wheels, so this label alone does not mean
  the bindings were rebuilt for the PR.
- `bindings floor` uses the oldest published bindings supported by the Core
  wheel for that CUDA major, as declared in its metadata. These rows test Core
  or the selected nightly integration against published bindings, without
  running the bindings or metapackage suites.

Open the change-detection job's **CI workplan** summary and expand its JSON.
The component build/test flags identify selected work; the baseline records
the original run and exact source SHA for reusable artifacts. Reuse requires a
successful push CI run for the exact merge-base commit on the PR's base branch,
with the complete required artifact set. If that baseline is unavailable, CI
builds and tests everything. A bindings-only change reuses the unchanged
merged Core wheel while testing Core against the affected bindings major.

The platform matrix-setup jobs' **Selected wheel test coverage** summaries
show the final suites, toolkit source, bindings root or published floor, and
GPU. Shared infrastructure changes, regular branch runs, and scheduled CI
cover both majors; a bindings tag deliberately selects one bindings/metapackage
line and published Pathfinder. Windows ARM64 omits CUDA 12 because that toolkit
has no Windows ARM64 build.

`ci/tools/select_test_matrix.py` applies the shared selection and local-wheel
coverage rules after each platform reads its source matrix and applies an
optional jq filter. Matrix data comes from the artifact source revision; the
helper comes from the workflow control revision so older artifact sources do
not need to contain newly introduced CI tooling.

## Historical Release And Documentation Reruns

Current release and documentation workflows read package paths and toolkit
settings from the schema-2 registry committed with the release tag. For tags
that predate that registry, dispatch compatible historical tooling instead.
In particular, CUDA 13.4 releases from `13.4.x` use the release workflow on
that branch, including when rebuilding their documentation.

Before dispatching historical tooling, bring its control branch up to date
with the applicable CI security fixes, including a verified patched Python
and current CA/download protections. Source-layout compatibility alone is
insufficient. The example below assumes those safeguards are in place; it
does not require changing the immutable release source.

After meeting that prerequisite, validate the existing `v13.4.4` release and
rebuild its docs with:

```console
gh workflow run release.yml --repo NVIDIA/cuda-python --ref 13.4.x \
  -f component=cuda-bindings -f release-action=dry-run \
  -f git-tag=v13.4.4 -f 'backport-git-tag=not planned'
```

Here `--ref 13.4.x` selects the workflow and its reusable documentation
workflow; `git-tag` selects the immutable release source. Leave `run-id` empty
to find that tag's successful CI run, or supply its validated artifact run ID
if needed. The artifacts must still exist. A dry run validates artifacts and
builds docs without publishing a release; its documentation is retained as a
workflow artifact. Optionally add `-f dry-run-docs-branch=gh-pages-dry-run` to
deploy the documentation to that preview branch. Run once per component when
validating more than one package; use `component=cuda-python` for the matching
metapackage documentation. Other historical release lines likewise need
tooling compatible with their tagged layout.
