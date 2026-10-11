---
name: create-cuda-python-pull-request
description: Create a CUDA Python pull request from an approved personal or organization-owned fork, including the GitHub CLI GraphQL fallback for renamed organization-owned forks. Use when the user directly requests creating or opening a CUDA Python pull request. Do not use for local implementation, commits, pushes, branch preparation, PR advice, or general GitHub work without that direct request.
---

# Create CUDA Python Pull Request

This skill supplies technical procedure after the user directly asks to create
a pull request. It does not define when a pull request should be created or
authorize one without that direct request. Do not infer the request from
completed work, a local commit, a push request, or the existence of a suitable
fork.

## Inspect the topology and proposed change

1. Run `git status --short --branch`, inspect the branch diff, and confirm the
   intended base branch.
2. Run `git remote -v` and resolve the complete `OWNER/REPOSITORY` names of the
   base repository and intended fork. Do not rely on remote names alone.
3. Confirm through GitHub that the push target is a fork of the base repository
   and is not the base repository itself.
4. Confirm that the intended push target complies with the repository's
   remote-write policy and the user's request.
5. Inspect the repository's pull-request template, available labels, and open
   milestones. Do not guess required metadata; ask the user when it is unclear.

## Validate and push

Run the checks appropriate to the change and review the final diff. Push the
current branch to the approved fork using the explicit remote and branch:

```bash
git push <fork-remote> <branch>
```

## Create the pull request

Prepare a complete body from the repository's pull-request template. Every
pull request must have at least one assignee, one label, and a milestone; CI
enforces this through `pr-metadata-check`.

Use `gh pr create` when it can identify the fork unambiguously. Select the base
repository and branch explicitly and supply the required metadata:

```bash
gh pr create \
  --repo <base-owner>/<base-repository> \
  --base <base-branch> \
  --head <fork-owner>:<head-branch> \
  --title "<title>" \
  --body-file <path-to-pr-body> \
  --assignee <assignee> \
  --label <label> \
  --milestone <milestone>
```

Add `--draft` when the user requests a draft pull request.

## Handle renamed organization-owned forks

[GitHub CLI issue cli/cli#10093](https://github.com/cli/cli/issues/10093)
tracks `gh pr create` support for cross-repository pull requests within one
organization. Check whether the issue has been resolved before using the
workaround.

If `gh pr create` cannot identify an organization-owned fork whose repository
name differs from the base repository, create the pull request with GitHub's
GraphQL API and pass `headRepositoryId` explicitly.

Resolve the repository node IDs:

```bash
BASE_REPO="<base-owner>/<base-repository>"
HEAD_REPO="<fork-owner>/<fork-repository>"
BASE_REPO_ID="$(gh api "repos/${BASE_REPO}" --jq '.node_id')"
HEAD_REPO_ID="$(gh api "repos/${HEAD_REPO}" --jq '.node_id')"
```

Create the pull request. Set `draft` to match the user's request.

```bash
gh api graphql \
  -f repositoryId="${BASE_REPO_ID}" \
  -f headRepositoryId="${HEAD_REPO_ID}" \
  -f baseRefName="<base-branch>" \
  -f headRefName="<head-branch>" \
  -f title="<title>" \
  -F body="@<path-to-pr-body>" \
  -F draft=false \
  -f query='
    mutation CreatePullRequest(
      $repositoryId: ID!
      $headRepositoryId: ID!
      $baseRefName: String!
      $headRefName: String!
      $title: String!
      $body: String!
      $draft: Boolean!
    ) {
      createPullRequest(input: {
        repositoryId: $repositoryId
        headRepositoryId: $headRepositoryId
        baseRefName: $baseRefName
        headRefName: $headRefName
        title: $title
        body: $body
        draft: $draft
      }) {
        pullRequest { number url }
      }
    }' \
  --jq '.data.createPullRequest.pullRequest'
```

The GraphQL API does not populate the pull-request template automatically.
After creation, add the required metadata to the returned pull-request number:

```bash
gh pr edit <pr-number> \
  --repo "${BASE_REPO}" \
  --add-assignee "<assignee>" \
  --add-label "<label>" \
  --milestone "<milestone>"
```

Verify the resulting URL, base branch, head repository and branch, draft state,
body, assignee, label, and milestone before reporting completion.
