# Contributing to CUDA Python

Thank you for your interest in contributing to CUDA Python! Based on the type of contribution, it will fall into two categories:

1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/cuda-python/issues/new)
    describing what you encountered or what you want to see changed.
    - The NVIDIA team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to implement a feature or bug-fix
    - At this time we do not accept code contributions.

## Linting

`cuda-python` uses pre-commit hooks to maintain code quality and consistency.

1. `pip install pre-commit`
2. `pre-commit install`
3. Linting will automatically run on each commit but to run manually `pre-commit run --all-files`