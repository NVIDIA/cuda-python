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

## Code signing

This repository implements a security check to prevent the CI system from running untrusted code. A part of the
security check consists of checking if the git commits are signed. See
[here](https://docs.gha-runners.nvidia.com/apps/copy-pr-bot/faqs/#why-did-i-receive-a-comment-that-my-pull-request-requires-additional-validation)
and
[here](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
for more details, including how to sign your commits.
