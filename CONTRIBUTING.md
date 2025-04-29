# Contributing to CUDA Python

Thank you for your interest in contributing to CUDA Python! Based on the type of contribution, it will fall into two categories:

1. You want to report a bug, feature request, or documentation issue:
    - File an [issue](https://github.com/NVIDIA/cuda-python/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The NVIDIA team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to implement a feature, improvement, or bug fix:
    - Please refer to each component's guideline:
       - [`cuda.core`](https://nvidia.github.io/cuda-python/cuda-core/latest/contribute.html)
       - [`cuda.bindings`](https://nvidia.github.io/cuda-python/cuda-bindings/latest/contribute.html)

## Pre-commit
This project uses [pre-commit.ci](https://pre-commit.ci/) with GitHub Actions. All pull requests are automatically checked for pre-commit compliance, and any pre-commit failures will block merging until resolved.

To set yourself up for running pre-commit checks locally and to catch issues before pushing your changes, follow these steps:

* Install pre-commit with: `pip install pre-commit`
* You can manually check all files at any time by running: `pre-commit run --all-files`

This command runs all configured hooks (such as linters and formatters) across your repository, letting you review and address issues before committing.

**Optional: Enable automatic checks on every commit**
If you want pre-commit hooks to run automatically each time you make a commit, install the git hook with:

`pre-commit install`

This sets up a git pre-commit hook so that all configured checks will run before each commit is accepted. If any hook fails, the commit will be blocked until the issues are resolved.

**Note on workflow flexibility**
Some contributors prefer to commit intermediate or work-in-progress changes that may not pass all pre-commit checks, and only clean up their commits before pushing (for example, by squashing and running `pre-commit run --all-files` manually at the end). If this fits your workflow, you may choose not to run `pre-commit install` and instead rely on manual checks. This approach avoids disruption during iterative development, while still ensuring code quality before code is shared or merged.

Choose the setup that best fits your workflow and development style.

## Code signing

This repository implements a security check to prevent the CI system from running untrusted code. A part of the security check consists of checking if the git commits are signed. Please ensure that your commits are signed [following GitHub’s instruction](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification).
