# PR Preview Cleanup Script

## Overview

This script (`cleanup-pr-previews`) helps maintain the `gh-pages` branch by cleaning up documentation preview folders for PRs that have been closed or merged. 

## Problem

The current `doc_preview` action has some limitations that can result in stale preview folders:

1. Cleanup steps only run when the target branch is `main` (so PRs targeting feature branches don't get cleaned up)
2. Canceled/interrupted documentation jobs don't run cleanup steps
3. Various other edge cases where the cleanup logic isn't executed

This results in a mismatch between the number of `pr-XXXXX` folders in `docs/pr-preview/` and the actual number of open PRs.

## Solution

The `cleanup-pr-previews` script:

1. Fetches all `pr-XXXXX` folders from the `docs/pr-preview/` directory in the `gh-pages` branch
2. For each folder, extracts the PR number and checks its status via GitHub API
3. Identifies folders corresponding to closed/merged PRs or deleted PRs
4. Removes the stale folders and commits the changes back to `gh-pages`

## Usage

### Prerequisites

- `GH_TOKEN` environment variable with appropriate permissions
- GitHub CLI (`gh`) installed and authenticated
- `jq` installed for JSON parsing
- `git` available

### Basic Usage

```bash
# Preview what would be cleaned up (recommended first run)
./ci/cleanup-pr-previews NVIDIA/cuda-python true

# Actually perform the cleanup
./ci/cleanup-pr-previews NVIDIA/cuda-python false

# Use defaults (NVIDIA/cuda-python, actual cleanup)
./ci/cleanup-pr-previews
```

### Parameters

1. **repository** (optional): GitHub repository in `owner/repo` format. Default: `NVIDIA/cuda-python`
2. **dry-run** (optional): Set to `true` to preview changes without making them. Default: `false`

### Examples

```bash
# Preview cleanup for the main repository
./ci/cleanup-pr-previews NVIDIA/cuda-python true

# Clean up a different repository
./ci/cleanup-pr-previews myorg/my-repo false

# Show help
./ci/cleanup-pr-previews --help
```

## Sample Output

```
[INFO] Checking prerequisites...
[INFO] All prerequisites satisfied
[INFO] Fetching PR preview folders from gh-pages branch...
[INFO] Found 44 PR preview folders
[CHECK] Checking PR #415...
[REMOVE] PR #415 is closed
[CHECK] Checking PR #1021...
[KEEP] PR #1021 is still open
...

[SUMMARY]
Total PR preview folders: 44
Open PRs: 17
Folders to remove: 27

[FOLDERS TO REMOVE]
  - pr-415 (PR #415)
  - pr-435 (PR #435)
  ...

[CLEANUP] Proceeding to remove 27 folders...
[INFO] Cloning gh-pages branch to temporary directory...
[REMOVE] Removing docs/pr-preview/pr-415
...
[INFO] Committing changes...
[INFO] Pushing to gh-pages branch...
[SUCCESS] Cleanup completed! Removed 27 PR preview folders
```

## Security Considerations

- The script requires write access to the repository to modify the `gh-pages` branch
- Always run with `dry-run=true` first to verify the expected behavior
- The script clones the repository to a temporary directory which is automatically cleaned up

## Future Enhancements

This script could be integrated into a scheduled GitHub Actions workflow to run periodically (e.g., weekly) to automatically maintain the `gh-pages` branch.