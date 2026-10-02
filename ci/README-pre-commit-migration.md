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
gh workflow run lychee.yml --repo NVIDIA/cuda-python --ref REF \
  -f refresh-cache=true
gh workflow run lychee.yml --repo NVIDIA/cuda-python --ref REF \
  -f refresh-cache=false
gh workflow run ci-nightly.yml --repo NVIDIA/cuda-python --ref REF \
  -f documentation-links-only=true
gh run list --repo NVIDIA/cuda-python --branch REF
```

Dependabot checks hook revisions monthly and opens update PRs with the `CI/CD`
and `dependencies` labels, without an automatic assignee or milestone. Lychee's
refresh mode skips restoration and publishes a new snapshot even when some
links fail; only successful checks enter the persisted cache. Its normal mode
restores the newest matching snapshot and does not publish one. Branch tests
write branch-scoped caches. After merge, nightly refreshes run on `main` and
publish the baseline that all PRs can read, including fork PRs. Separate
authored/rendered namespaces include the checker version and checking policy;
documentation edits do not invalidate previously successful external checks.

The nightly documentation-only mode exercises the reusable-workflow call and
status gate, plus standalone CI-tool tests. It skips wheel lookup and requires
all wheel/GPU jobs to remain skipped. The scheduled/default nightly mode still
runs the complete existing suite.

For the same source-based docs build locally:

```sh
PIXI_LOCKED=true pixi run --manifest-path cuda_core/pixi.toml -e docs \
  docs-build-all-latest
```

The build verifies all three sibling libraries import from the checkout,
installs metapackage metadata without dependencies, and assembles all four
latest documentation trees under `artifacts/docs`.
