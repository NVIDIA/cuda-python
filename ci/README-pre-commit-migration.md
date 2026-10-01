# Migrating from pre-commit.ci

The replacement checks are `Pre-commit (Linux)`, `Pre-commit (Windows)`, and
`Documentation links`, all from the GitHub Actions App (integration ID 15368).
They replace `pre-commit.ci - pr` from integration ID 68672.

The workflow pull requests can remain draft while their runs are reviewed.
Keep pre-commit.ci installed and required throughout that review. Do not change
the live required checks until the replacement workflows are merged and green.

## Test the draft workflows

Use the PR branch as `REF` while reviewing the implementation. These commands
run hosted checks without copying a PR into the heavyweight CUDA CI:

```sh
gh workflow run pre-commit.yml --repo NVIDIA/cuda-python --ref REF
gh workflow run pre-commit-autoupdate.yml --repo NVIDIA/cuda-python --ref REF \
  -f dry-run=true
gh workflow run lychee.yml --repo NVIDIA/cuda-python --ref REF \
  -f refresh-cache=true
gh workflow run lychee.yml --repo NVIDIA/cuda-python --ref REF \
  -f refresh-cache=false
gh run list --repo NVIDIA/cuda-python --branch REF
```

The updater dry run shows its diff without opening or updating a PR. Lychee's
refresh mode skips restoration and publishes a new snapshot even when some
links fail; only successful checks enter the persisted cache. Its normal mode
restores the newest matching snapshot and does not publish one. Branch tests
write branch-scoped caches. After merge, nightly refreshes run on `main` and
publish the baseline that all PRs can read, including fork PRs. Separate
authored/rendered namespaces include the checker version and checking policy;
documentation edits do not invalidate previously successful external checks.

For the same source-based docs build locally:

```sh
PIXI_LOCKED=true pixi run --manifest-path cuda_core/pixi.toml -e docs \
  docs-build-all-latest
```

The build verifies all three sibling libraries import from the checkout,
installs metapackage metadata without dependencies, and assembles all four
latest documentation trees under `artifacts/docs`.

## Preview the ruleset change

Run from the repository root with an authenticated GitHub CLI:

```sh
pixi exec --spec python python ci/tools/migrate_precommit_checks.py \
  --repo NVIDIA/cuda-python > /tmp/cuda-python-precommit-cutover.json
```

The default mode only reads GitHub. It discovers active rules applying to
`main`, fetches complete repository rulesets, and prints their original
configuration and the exact proposed PUT payload. Review that JSON before
cutover. The payload changes only `rules`: unrelated rules and required checks,
strictness, branch conditions, bypass actors, and enforcement remain in place.
The tool rejects inherited organization rulesets and unexpected integration IDs;
an organization ruleset's owner must handle that migration separately.

## Cut over after merge

1. Merge `.github/workflows/pre-commit.yml` and `.github/workflows/lychee.yml`.
   Check that they exist on the current default `main` branch.
2. Obtain successful, completed runs of all three replacement checks on current
   `main` or a representative open PR targeting `main`. Record the full checked
   commit SHA. This also verifies the PR event path when using a PR; a branch
   push run alone does not validate PR-specific behavior.
3. After explicit authorization for the ruleset cutover, run the preview again,
   then run the same command with `--apply --verified-sha FULL_COMMIT_SHA`.
   Omitting `--verified-sha` requires checks on the current `main` commit.
4. Confirm the active rules now require all three replacement contexts from the
   GitHub Actions App and still contain every unrelated requirement. Verify on
   a representative PR that an intentional formatting or broken-link failure
   is reported as a required failing check and blocks merging; repair it and
   verify the same check passes. Check this independently of the PR's draft
   status, which also prevents merging.
5. Only after that verification, remove this repository from the pre-commit.ci
   GitHub App installation's repository access, or disable its service through
   the repository/service settings. Preserve access for other repositories
   using the same installation. The migration script never changes App access.

`--apply` verifies the workflow files on `main`, the selected commit's checks,
their App IDs, and their successful owning workflow runs before writing. It
re-reads all affected rulesets and refuses changes made since discovery.
GitHub does not provide an atomic transaction across ruleset updates: if a
write fails partway through, inspect the emitted preview and actual rulesets
before retrying. Do not remove the service while any old required check remains.

## Validate the helper

```sh
pixi exec --spec python --spec pytest python -m pytest \
  --noconftest ci/tools/tests/test_migrate_precommit_checks.py
```

The tests mock GitHub; they cannot change repository settings.
