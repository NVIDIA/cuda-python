<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CodeRabbit Configuration

This document explains how CodeRabbit is configured for CUDA Python. For review
commands and guidance, see the
[CodeRabbit section of the contributing guide](../CONTRIBUTING.md#coderabbit).
For complete product documentation, see the
[CodeRabbit documentation](https://docs.coderabbit.ai/).

CUDA Python configures CodeRabbit through [`.coderabbit.yaml`](../.coderabbit.yaml)
in the repository root. The configuration in the pull request branch is used
for that review. Pull requests from forks are an exception: they use the target
branch's configuration, so changes to `.coderabbit.yaml` in a fork do not affect
that review.

When setting up or updating CodeRabbit:

1. Keep repository-specific settings in `.coderabbit.yaml`.
2. Use `@coderabbitai configuration` on a pull request to inspect the resolved
   configuration. This is useful when checking whether CodeRabbit is using the
   expected repository settings for that pull request.
3. Use `@coderabbitai generate configuration` to export the resolved
   configuration if a new baseline is needed. This is useful when moving
   settings into a reviewable repository configuration file.
4. Keep configuration changes small and reviewable.
5. Use `reviews.path_instructions` for path-specific review guidance.
6. Use `knowledge_base.code_guidelines.filePatterns` for CUDA Python guidance
   files that CodeRabbit should read as review context.

The CUDA Python configuration should keep comments focused on correctness, API
stability, performance, security, and other high-impact issues. Avoid enabling
features that add noisy generated comments or code by default.
